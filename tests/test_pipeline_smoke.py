"""
test_pipeline_smoke.py

Runs the whole chain -- build_manifest, audit, train, evaluate, export -- on a
synthetic dataset at a size that finishes in a couple of minutes on a CPU.

This is the only test that exercises the seams between stages, which is where
the expensive failures live. Every one of these was a real break: --tracking-uri
resolving to the wrong directory, find_best_run finding nothing in a fresh
database, --metrics-json missing so `dvc metrics show` had nothing to read,
export writing where the Dockerfile does not look. Unit tests pass throughout
all of them, because each component is individually fine.

It asserts that the pipeline RUNS and produces well-formed artifacts. It asserts
nothing about model quality: one epoch on 40 synthetic images has no quality to
speak of, and a threshold on IoU here would be either vacuous or flaky.

Marked `slow`. Deselect with `-m "not slow"`.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import REPO_ROOT

pytestmark = pytest.mark.slow

# Small enough for CPU, large enough that the stratified split is feasible.
PATCH_SIZE = 64
N_PAIRS = 44


def run_stage(module, args, cwd=REPO_ROOT):
    """Invoke a pipeline stage exactly as dvc.yaml does, and fail loudly.

    Subprocesses rather than function calls: a stage's real interface is its
    command line, so this covers argument wiring and the __main__ path too.
    """
    result = subprocess.run(
        [sys.executable, "-m", module, *[str(a) for a in args]],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{module} failed with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result.stdout


@pytest.fixture(scope="module")
def raw_dataset(tmp_path_factory):
    """Synthetic images with genuinely learnable structure.

    Water regions are bright blobs on a dark background rather than pure noise.
    Not for accuracy's sake -- nothing here asserts on quality -- but because a
    target uncorrelated with the input can drive a loss to NaN, and this test
    would then fail for a reason that has nothing to do with the pipeline.
    """
    root = tmp_path_factory.mktemp("smoke")
    image_dir, mask_dir = root / "images", root / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    import cv2

    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_PAIRS):
        side = int(rng.choice([96, 128, 160]))
        mask = np.zeros((side, side), np.uint8)

        # One or two rectangular "water bodies" per image.
        for _ in range(int(rng.integers(1, 3))):
            h, w = rng.integers(side // 5, side // 2, size=2)
            r0, c0 = rng.integers(0, side - h), rng.integers(0, side - w)
            mask[r0 : r0 + h, c0 : c0 + w] = 255

        image = rng.integers(20, 70, (side, side, 3), dtype=np.uint8)
        image[mask > 0] = rng.integers(170, 240, (int((mask > 0).sum()), 3), dtype=np.uint8)

        # .jpg, not .png: build_manifest scans IMAGE_SUFFIXES, and the real
        # dataset ships JPEGs. Using another format here would test a path
        # the pipeline never takes.
        name = f"img{i:03d}.jpg"
        cv2.imwrite(str(image_dir / name), image)
        cv2.imwrite(str(mask_dir / name), mask)
        rows.append({"filename": name, "width": side, "height": side})

    pd.DataFrame(rows).to_csv(root / "expected_manifest.csv", index=False)
    return {"root": root, "image_dir": image_dir, "mask_dir": mask_dir}


@pytest.fixture(scope="module")
def pipeline(raw_dataset):
    """Run every stage in order, returning the paths each produced."""
    root = raw_dataset["root"]
    paths = {
        "root": root,
        "manifest": root / "manifest.csv",
        "corrected": root / "masks_corrected",
        "splits": root / "splits",
        "metrics": root / "metrics",
        "checkpoints": root / "checkpoints",
        "exported": root / "exported_model",
        "tracking_uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
    }

    run_stage("data_pipeline.build_manifest",
              ["--image-dir", raw_dataset["image_dir"], "--output", paths["manifest"]])

    run_stage("data_pipeline.audit", [
        "--manifest", paths["manifest"],
        "--image-dir", raw_dataset["image_dir"],
        "--mask-dir", raw_dataset["mask_dir"],
        "--corrected-mask-dir", paths["corrected"],
        "--output-dir", paths["splits"],
        "--class-balance-json", paths["metrics"] / "class_balance.json",
        "--n-buckets", 3,
    ])

    run_stage("training.train", [
        "--train-manifest", paths["splits"] / "train.csv",
        "--val-manifest", paths["splits"] / "val.csv",
        "--image-dir", raw_dataset["image_dir"],
        "--mask-dir", paths["corrected"],
        "--checkpoint-dir", paths["checkpoints"],
        "--tracking-uri", paths["tracking_uri"],
        "--experiment-name", "smoke",
        "--arch", "deeplabv3plus",
        "--encoder-name", "mobilenet_v2",
        # No download in CI, and no dependence on a pretrained checkpoint.
        "--encoder-weights", "none",
        "--patch-size", PATCH_SIZE,
        "--tile-size", PATCH_SIZE,
        "--window-min", 32,
        "--window-max", 256,
        "--patches-per-epoch", 32,
        "--batch-size", 4,
        "--epochs", 1,
        # Workers must be 0: a spawned worker re-imports the module, which on
        # Windows and in constrained CI containers is both slow and fragile.
        "--num-workers", 0,
        "--device", "cpu",
        "--decoder-atrous-rates", 2, 4, 6,
    ])

    run_stage("training.test_model", [
        "--test-manifest", paths["splits"] / "test.csv",
        "--image-dir", raw_dataset["image_dir"],
        "--mask-dir", paths["corrected"],
        "--tracking-uri", paths["tracking_uri"],
        "--experiment-name", "smoke",
        "--tile-size", PATCH_SIZE,
        "--batch-size", 4,
        "--num-workers", 0,
        "--num-images", 2,
        "--device", "cpu",
        "--output", paths["metrics"] / "predictions_preview.png",
        "--metrics-json", paths["metrics"] / "test_metrics.json",
    ])

    run_stage("deployment.export_model", [
        "--tracking-uri", paths["tracking_uri"],
        "--experiment-name", "smoke",
        "--output-dir", paths["exported"],
    ])

    return paths


def model_bundle(pipeline):
    """Locate the exported bundle by finding its MLmodel spec.

    Discovered rather than assumed: export_model calls download_artifacts with
    artifact_uri="runs:/<id>/best_model", which lands the bundle one level down
    in <output-dir>/best_model rather than in <output-dir> itself.
    """
    specs = list(pipeline["exported"].rglob("MLmodel"))
    assert len(specs) == 1, f"expected exactly one MLmodel bundle, found {specs}"
    return specs[0].parent


# ---------------------------------------------------------------------------
# Stage outputs
# ---------------------------------------------------------------------------


def test_export_layout_matches_what_serving_expects(pipeline):
    """The bundle must land at <output-dir>/best_model.

    That name is hardcoded in three places -- the artifact_uri export_model
    downloads, the path the Dockerfile COPYs, and serve.py's MODEL_PATH default
    -- and nothing else checks they agree. Rename the logged artifact and the
    Docker build breaks at COPY time, long after the export "succeeded".
    """
    assert model_bundle(pipeline).name == "best_model"
    assert model_bundle(pipeline).parent == pipeline["exported"]


def test_audit_produces_splits_and_class_balance(pipeline):
    """The three splits and the balance artifact must all exist and agree."""
    counts = {}
    for name in ("train", "val", "test"):
        frame = pd.read_csv(pipeline["splits"] / f"{name}.csv")
        assert len(frame) > 0, f"{name} split is empty"
        counts[name] = len(frame)

    with open(pipeline["metrics"] / "class_balance.json") as f:
        balance = json.load(f)

    assert balance["overall"]["n_images"] == sum(counts.values())
    assert 0.0 < balance["overall"]["foreground_fraction"] < 1.0


def test_training_registers_a_run_the_evaluator_can_find(pipeline):
    """The tracking URI, registry write and find_best_run lookup must line up.

    This seam broke before: train wrote its database somewhere evaluate did not
    look, and find_best_run returned nothing from a fresh store.
    """
    checkpoints = list(pipeline["checkpoints"].glob("*.pt"))
    assert checkpoints, "training wrote no checkpoint"
    assert (pipeline["root"] / "mlflow.db").exists(), "tracking database not created"


def test_evaluate_writes_a_metrics_json_dvc_can_read(pipeline):
    """`dvc metrics show` reads this file by name; the keys are the contract."""
    with open(pipeline["metrics"] / "test_metrics.json") as f:
        metrics = json.load(f)

    required = {
        "run_id", "threshold", "num_images",
        "test_iou", "test_dice", "test_precision", "test_recall", "test_accuracy",
        "test_per_image_iou_min", "test_per_image_iou_mean", "test_per_image_iou_max",
    }
    assert required <= set(metrics), f"missing keys: {sorted(required - set(metrics))}"

    for key in required - {"run_id"}:
        assert isinstance(metrics[key], (int, float)), f"{key} is not numeric"
    for key in ("test_iou", "test_dice", "test_precision", "test_recall", "test_accuracy"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_evaluate_writes_the_prediction_preview(pipeline):
    """The qualitative plot is a declared DVC plot; its absence fails the stage."""
    preview = pipeline["metrics"] / "predictions_preview.png"
    assert preview.exists() and preview.stat().st_size > 0


def test_metrics_json_run_id_matches_the_exported_model(pipeline):
    """Provenance: the reported numbers and the shipped weights are one run.

    Without this, `dvc metrics diff` could describe a model other than the one
    in deployment/exported_model, and nothing would say so.
    """
    with open(pipeline["metrics"] / "test_metrics.json") as f:
        run_id = json.load(f)["run_id"]

    # MLmodel, not a .json: that is the file MLflow writes the run id into, and
    # the file serve.py parses at startup to report model_version on /health.
    import yaml

    with open(model_bundle(pipeline) / "MLmodel") as f:
        mlmodel = yaml.safe_load(f)

    assert mlmodel.get("run_id") == run_id, (
        f"exported model reports run {mlmodel.get('run_id')}, metrics report {run_id}"
    )


# ---------------------------------------------------------------------------
# The exported artifact
# ---------------------------------------------------------------------------


def test_exported_model_loads_and_predicts(pipeline):
    """The artifact the Dockerfile copies must load and run standalone.

    Loaded through deployment.inference.load_model -- the exact function serve.py
    calls -- rather than through mlflow directly. Calling mlflow by hand missed
    that the production loader has to turn an absolute path into a file:// URI
    before MLflow will accept it.
    """
    import torch

    from deployment.inference import load_model

    model = load_model(model_bundle(pipeline), torch.device("cpu"))
    with torch.no_grad():
        out = model(torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE))

    assert out.shape == (1, 1, PATCH_SIZE, PATCH_SIZE)
    assert torch.isfinite(out).all()


def test_exported_model_is_deterministic(pipeline):
    """Identical input, identical output -- no dropout or BN left in train mode.

    A model exported while still in training mode gives different answers to the
    same request, which is close to impossible to diagnose from the serving side.
    """
    import torch

    from deployment.inference import load_model

    model = load_model(model_bundle(pipeline), torch.device("cpu"))
    sample = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE)

    with torch.no_grad():
        first, second = model(sample), model(sample)

    assert torch.allclose(first, second)


@pytest.mark.parametrize("relative", [True, False])
def test_model_path_is_rendered_as_a_uri_mlflow_accepts(tmp_path, relative):
    """Absolute paths must become file:// URIs before reaching MLflow.

    MLflow dispatches on URI scheme. A Windows absolute path like
    C:\\models\\exported reads "c" as the scheme, which is not a registered
    artifact repository -- the loader then fails with a message about repository
    registration that says nothing about the path. Relative paths have an empty
    scheme and are passed through unchanged.
    """
    from deployment.inference import as_model_uri

    if relative:
        # Compared as paths, not strings: str(Path(...)) normalises separators,
        # so a forward-slash literal never equals the Windows result.
        result = as_model_uri("deployment/exported_model")
        assert not result.startswith("file://")
        assert Path(result) == Path("deployment/exported_model")
    else:
        uri = as_model_uri(tmp_path)
        assert uri.startswith("file://"), uri
