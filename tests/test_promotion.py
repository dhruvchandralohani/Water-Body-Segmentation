"""
test_promotion.py

The promotion gates decide what reaches production, so they are worth testing
harder than the thing they guard.

Driven with a stand-in MlflowClient rather than a live tracking server. The
logic being tested is entirely about which version wins and when to refuse --
none of it depends on MLflow actually storing anything, and a real server would
make a millisecond check into a slow one.
"""

import pytest

from deployment.promote_model import (
    GATE_METRIC,
    PRODUCTION_ALIAS,
    gate_metric_for_run,
    latest_version_for_run,
    registered_model_name,
)


class FakeVersion:
    """Stands in for mlflow.entities.model_registry.ModelVersion."""

    def __init__(self, name, version, run_id):
        self.name = name
        self.version = str(version)
        self.run_id = run_id


class FakeMetric:
    """Stands in for a single logged metric point."""

    def __init__(self, value):
        self.value = value


class FakeClient:
    """Minimal MlflowClient covering only what promote_model calls."""

    def __init__(self, versions=(), metrics=None, alias_version=None):
        self._versions = list(versions)
        self._metrics = metrics or {}
        self._alias_version = alias_version
        self.alias_calls = []

    def search_model_versions(self, filter_string):
        """Return versions matching a run_id='<id>' filter."""
        run_id = filter_string.split("'")[1]
        return [v for v in self._versions if v.run_id == run_id]

    def get_metric_history(self, run_id, metric):
        """Return the logged history for one metric, or empty if never logged."""
        return [FakeMetric(v) for v in self._metrics.get((run_id, metric), [])]

    def get_model_version_by_alias(self, name, alias):
        """Raise when unset, matching MLflow's behaviour rather than returning None."""
        if self._alias_version is None:
            raise RuntimeError(f"alias {alias} not set on {name}")
        return self._alias_version

    def set_registered_model_alias(self, name, alias, version):
        """Record the promotion instead of performing it."""
        self.alias_calls.append((name, alias, version))


MODEL = registered_model_name("water_body_segmentation")


def promote_with(monkeypatch, client, run_id="run-new", force=False):
    """Call promote() against a fake client, bypassing tracking-URI setup."""
    from deployment import promote_model

    monkeypatch.setattr(promote_model.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(promote_model, "MlflowClient", lambda: client)
    return promote_model.promote(run_id, "water_body_segmentation", "sqlite:///ignored", force=force)


# ---------------------------------------------------------------------------
# Version selection
# ---------------------------------------------------------------------------


def test_the_latest_version_of_a_run_is_the_one_promoted():
    """A single run registers several versions, and only the last one won it.

    train.py registers a new version every time validation improves, so a run
    that improved five times leaves five versions behind. Promoting anything but
    the highest would ship a mid-training snapshot that the run itself rejected.
    """
    client = FakeClient(versions=[
        FakeVersion(MODEL, 3, "run-a"),
        FakeVersion(MODEL, 7, "run-a"),
        FakeVersion(MODEL, 5, "run-a"),
        FakeVersion(MODEL, 9, "run-b"),
    ])
    assert latest_version_for_run(client, MODEL, "run-a").version == "7"


def test_a_run_that_registered_nothing_is_rejected():
    """Only runs that improved on validation register a version at all."""
    client = FakeClient(versions=[FakeVersion(MODEL, 1, "run-other")])
    with pytest.raises(ValueError, match="no registered version"):
        latest_version_for_run(client, MODEL, "run-none")


def test_versions_from_another_registered_model_are_ignored():
    """The grids register under their own experiment names.

    water_body_arch_benchmark_model versions must never be promotable as the
    production model just because they share a run id space.
    """
    client = FakeClient(versions=[
        FakeVersion("water_body_arch_benchmark_model", 4, "run-a"),
        FakeVersion(MODEL, 2, "run-a"),
    ])
    assert latest_version_for_run(client, MODEL, "run-a").version == "2"


# ---------------------------------------------------------------------------
# Gate 1: must have been evaluated
# ---------------------------------------------------------------------------


def test_an_unevaluated_run_cannot_be_promoted(monkeypatch):
    """val_iou alone is not grounds to ship.

    Validation is the signal used to *select* the model, so a run with no
    test_iou has been chosen but never scored on held-out data. Promoting it
    would put a number on the deployment that no one ever measured.
    """
    client = FakeClient(
        versions=[FakeVersion(MODEL, 1, "run-new")],
        metrics={("run-new", "val_iou"): [0.71]},  # selected, never evaluated
    )
    with pytest.raises(ValueError, match="never evaluated on the test set"):
        promote_with(monkeypatch, client)

    assert client.alias_calls == [], "no alias should move when a gate fails"


def test_gate_metric_reads_the_last_logged_value():
    """Evaluation logs once per run; the last value is the current answer."""
    client = FakeClient(metrics={("run-a", GATE_METRIC): [0.70, 0.74]})
    assert gate_metric_for_run(client, "run-a") == 0.74
    assert gate_metric_for_run(client, "run-missing") is None


# ---------------------------------------------------------------------------
# Gate 2: no silent regression
# ---------------------------------------------------------------------------


def test_a_worse_candidate_is_refused(monkeypatch):
    """Experimentation routinely produces worse runs; refusing is the default.

    This is the gate that makes promotion a decision rather than a rename.
    """
    client = FakeClient(
        versions=[FakeVersion(MODEL, 5, "run-new")],
        metrics={("run-new", GATE_METRIC): [0.70], ("run-old", GATE_METRIC): [0.75]},
        alias_version=FakeVersion(MODEL, 2, "run-old"),
    )
    with pytest.raises(ValueError, match="refusing to promote"):
        promote_with(monkeypatch, client)

    assert client.alias_calls == []


def test_a_worse_candidate_is_allowed_with_force(monkeypatch):
    """Rollback is a legitimate reason to promote a lower-scoring version."""
    client = FakeClient(
        versions=[FakeVersion(MODEL, 5, "run-new")],
        metrics={("run-new", GATE_METRIC): [0.70], ("run-old", GATE_METRIC): [0.75]},
        alias_version=FakeVersion(MODEL, 2, "run-old"),
    )
    result = promote_with(monkeypatch, client, force=True)

    assert client.alias_calls == [(MODEL, PRODUCTION_ALIAS, "5")]
    assert result["forced"] is True
    assert result["previous_version"] == 2


def test_a_better_candidate_is_promoted(monkeypatch):
    """The ordinary path still has to work."""
    client = FakeClient(
        versions=[FakeVersion(MODEL, 5, "run-new")],
        metrics={("run-new", GATE_METRIC): [0.78], ("run-old", GATE_METRIC): [0.75]},
        alias_version=FakeVersion(MODEL, 2, "run-old"),
    )
    result = promote_with(monkeypatch, client)

    assert client.alias_calls == [(MODEL, PRODUCTION_ALIAS, "5")]
    assert result[GATE_METRIC] == 0.78
    assert result["forced"] is False


def test_the_first_promotion_needs_no_incumbent(monkeypatch):
    """With nothing promoted there is nothing to regress against."""
    client = FakeClient(
        versions=[FakeVersion(MODEL, 1, "run-new")],
        metrics={("run-new", GATE_METRIC): [0.60]},
        alias_version=None,
    )
    result = promote_with(monkeypatch, client)

    assert client.alias_calls == [(MODEL, PRODUCTION_ALIAS, "1")]
    assert result["previous_version"] is None


def test_an_equal_score_is_allowed(monkeypatch):
    """The gate blocks regressions, not re-promotions.

    Re-exporting an identical model, or promoting a rebuild that scores the
    same, is routine and should not need --force.
    """
    client = FakeClient(
        versions=[FakeVersion(MODEL, 5, "run-new")],
        metrics={("run-new", GATE_METRIC): [0.75], ("run-old", GATE_METRIC): [0.75]},
        alias_version=FakeVersion(MODEL, 2, "run-old"),
    )
    promote_with(monkeypatch, client)
    assert client.alias_calls == [(MODEL, PRODUCTION_ALIAS, "5")]


def test_an_unevaluated_incumbent_does_not_block_promotion(monkeypatch):
    """A promotion made before the gate existed has no metric to compare against.

    Treating that as "no incumbent score" lets the first gated promotion through
    rather than deadlocking on history.
    """
    client = FakeClient(
        versions=[FakeVersion(MODEL, 5, "run-new")],
        metrics={("run-new", GATE_METRIC): [0.60]},
        alias_version=FakeVersion(MODEL, 2, "run-legacy"),
    )
    promote_with(monkeypatch, client)
    assert client.alias_calls == [(MODEL, PRODUCTION_ALIAS, "5")]


# ---------------------------------------------------------------------------
# Naming contract
# ---------------------------------------------------------------------------


def test_registered_model_name_matches_what_train_writes():
    """train.py builds this name the same way; two places, one derivation.

    If they diverge, promotion silently targets a registry entry that training
    never writes to, and every promotion fails with "no registered version".
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "training" / "train.py").read_text()
    assert 'f"{args.experiment_name}_model"' in source, (
        "train.py no longer derives the registry name as <experiment>_model; "
        "promote_model.registered_model_name must be updated to match"
    )
    assert registered_model_name("abc") == "abc_model"
    assert ast.parse(source) is not None
