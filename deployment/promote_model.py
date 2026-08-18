"""
promote_model.py

Move the `production` alias onto a chosen model version.

Without this, `export_model.py` decides what ships by calling find_best_run --
whichever run currently has the highest val_iou. That makes deployment implicit
and recomputed: a marginally better experimental run silently becomes the thing
that serves traffic, with no record of anyone deciding. Promotion turns that
into a deliberate act with an audit trail, and makes rollback a matter of moving
an alias rather than re-exporting.

Two gates, and they are the point of the script:

1. A version with no test metrics cannot be promoted. val_iou is a model
   selection signal measured on data used to select the model. Shipping on it
   alone means shipping something never scored on held-out data.
2. A version scoring worse than the incumbent cannot be promoted without
   --force. Ordinary experimentation produces occasional worse runs; the
   default should refuse them.

Usage:
    python -m deployment.promote_model --best         # promote the best evaluated run
    python -m deployment.promote_model --run-id <id>  # promote one explicitly
    python -m deployment.promote_model --show         # report the incumbent
"""

import argparse
import sys as _sys
from pathlib import Path as _Path

import mlflow
from mlflow.tracking import MlflowClient

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logger  # noqa: E402
from common.mlflow_utils import find_best_run  # noqa: E402

logger = setup_logger("promote_model", log_file="promote.log")

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
_DEFAULT_TRACKING_URI = f"sqlite:///{(_PROJECT_ROOT / 'training' / 'mlflow.db').as_posix()}"

PRODUCTION_ALIAS = "production"
GATE_METRIC = "test_iou"


def registered_model_name(experiment_name):
    """Registry name train.py writes versions under.

    Derived rather than configured, because train.py builds it the same way:
    registered_model_name=f"{experiment_name}_model". Two places computing one
    name from one input is the least bad option short of threading it through
    params.yaml.
    """
    return f"{experiment_name}_model"


def latest_version_for_run(client, model_name, run_id):
    """Find the newest registered version produced by a given run.

    A run yields several versions, not one: train.py registers a new version
    every time validation improves. The last is the checkpoint that actually won
    the run, so promoting anything earlier would ship a mid-training snapshot.

    Args:
        client: An MlflowClient.
        model_name: Registered model name.
        run_id: Run whose versions to search.

    Returns:
        The ModelVersion with the highest version number for that run.

    Raises:
        ValueError: If the run registered no versions.
    """
    versions = client.search_model_versions(f"run_id='{run_id}'")
    versions = [v for v in versions if v.name == model_name]
    if not versions:
        raise ValueError(
            f"run {run_id} has no registered version under '{model_name}'. "
            "Only runs that improved on validation register one."
        )
    return max(versions, key=lambda v: int(v.version))


def gate_metric_for_run(client, run_id):
    """Read the held-out metric a promotion is gated on.

    Args:
        client: An MlflowClient.
        run_id: Run to inspect.

    Returns:
        The metric value, or None if the run was never evaluated on the test set.
    """
    history = client.get_metric_history(run_id, GATE_METRIC)
    return history[-1].value if history else None


def current_production(client, model_name):
    """Return the version currently aliased to production, or None if unset."""
    try:
        return client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except Exception:
        # MLflow raises rather than returning None when the alias is unset, and
        # the exception type has moved between versions -- the distinction that
        # matters here is only "is there an incumbent".
        return None


def describe_production(experiment_name, tracking_uri=None):
    """Report what is currently promoted.

    Args:
        experiment_name: Experiment whose registered model to inspect.
        tracking_uri: MLflow tracking URI.

    Returns:
        A dict describing the incumbent, or None if nothing is promoted.
    """
    tracking_uri = tracking_uri or _DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = registered_model_name(experiment_name)
    incumbent = current_production(client, model_name)
    if incumbent is None:
        logger.info(f"No version of '{model_name}' is aliased to @{PRODUCTION_ALIAS}")
        return None

    metric = gate_metric_for_run(client, incumbent.run_id)
    logger.info(
        f"@{PRODUCTION_ALIAS} -> version {incumbent.version} "
        f"(run {incumbent.run_id}, {GATE_METRIC}={metric})"
    )
    return {"version": int(incumbent.version), "run_id": incumbent.run_id, GATE_METRIC: metric}


def promote(run_id, experiment_name, tracking_uri=None, force=False):
    """Move the production alias onto the version produced by `run_id`.

    Args:
        run_id: Run whose registered version should be promoted.
        experiment_name: Experiment the run belongs to.
        tracking_uri: MLflow tracking URI.
        force: Promote even when the candidate scores below the incumbent.

    Returns:
        A dict describing the promotion that was applied.

    Raises:
        ValueError: If the run has no registered version, was never evaluated on
            the test set, or scores below the incumbent without force.
    """
    tracking_uri = tracking_uri or _DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = registered_model_name(experiment_name)
    candidate = latest_version_for_run(client, model_name, run_id)
    candidate_metric = gate_metric_for_run(client, run_id)

    # Gate 1: never promote something that was never scored on held-out data.
    # val_iou is a selection signal measured on data used to select the model;
    # a run with no test_iou has not been evaluated, only chosen.
    if candidate_metric is None:
        raise ValueError(
            f"run {run_id} has no {GATE_METRIC}: it was never evaluated on the test set. "
            "Run the evaluate stage against it before promoting."
        )

    incumbent = current_production(client, model_name)
    incumbent_metric = gate_metric_for_run(client, incumbent.run_id) if incumbent else None

    # Gate 2: refuse a regression by default. Experimentation routinely produces
    # worse runs; the safe default is to say no and make the operator override.
    if incumbent_metric is not None and incumbent is not None and candidate_metric < incumbent_metric and not force:
        raise ValueError(
            f"refusing to promote: candidate {GATE_METRIC}={candidate_metric:.4f} is below "
            f"the incumbent's {incumbent_metric:.4f} (version {incumbent.version}). "
            "Pass --force if this is deliberate."
        )

    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, candidate.version)

    previous = f"version {incumbent.version}" if incumbent else "nothing"
    logger.info(
        f"@{PRODUCTION_ALIAS} moved from {previous} to version {candidate.version} "
        f"(run {run_id}, {GATE_METRIC}={candidate_metric:.4f})"
    )
    return {
        "model_name": model_name,
        "version": int(candidate.version),
        "run_id": run_id,
        GATE_METRIC: candidate_metric,
        "previous_version": int(incumbent.version) if incumbent else None,
        "forced": bool(force),
    }


def main():
    """Promote a model version, or report which one is promoted."""
    parser = argparse.ArgumentParser(
        description=f"Move the @{PRODUCTION_ALIAS} alias onto a model version."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Promote the version produced by this run.")
    group.add_argument(
        "--best",
        action="store_true",
        help="Promote the best run by val_iou. Still subject to both gates: the "
        "best-by-validation run is not automatically fit to ship.",
    )
    group.add_argument("--show", action="store_true", help="Report the current incumbent and exit.")

    parser.add_argument("--experiment-name", default="water_body_segmentation")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Promote even when the candidate scores below the incumbent.",
    )
    args = parser.parse_args()

    if args.show:
        describe_production(args.experiment_name, args.tracking_uri)
        return

    run_id = args.run_id
    if args.best:
        tracking_uri = args.tracking_uri or _DEFAULT_TRACKING_URI
        run_id, val_iou = find_best_run(args.experiment_name, tracking_uri)
        logger.info(f"Best run by val_iou: {run_id} ({val_iou:.4f})")

    promote(run_id, args.experiment_name, args.tracking_uri, force=args.force)


if __name__ == "__main__":
    main()
