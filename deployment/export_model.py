"""
export_model.py

Resolves a model from the MLflow registry once and downloads it into a
clean, standalone local folder -- weights, MLmodel spec, exact pip
requirements, everything needed to load it with zero further dependency
on the live MLflow tracking database. This is a build-time step, not
something the running inference service does itself: serve.py loads
from the exported folder, never from a live MLflow connection, so the
service's uptime is never tied to whether the training machine's
mlflow.db happens to be reachable.

Usage:
    python export_model.py --output-dir exported_model/
    python export_model.py --run-id <specific_run_id> --output-dir exported_model/
"""

import argparse
import shutil
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logger

import mlflow
from mlflow.artifacts import download_artifacts

from common.mlflow_utils import find_best_run

logger = setup_logger("export_model", log_file="export.log")

_DEFAULT_TRACKING_URI = f"sqlite:///{(_Path(__file__).resolve().parent.parent / 'training' / 'mlflow.db').as_posix()}"

_DEFAULT_OUTPUT_DIR = str(_Path(__file__).resolve().parent / "exported_model")


def export_model(output_dir, run_id=None, experiment_name="water_body_segmentation", tracking_uri=None):
    """Export the best or specified MLflow model artifact to a local directory.

    Args:
        output_dir: Destination directory for the exported model artifacts.
        run_id: Optional MLflow run ID to export. If omitted, the best run is
            resolved automatically.
        experiment_name: Name of the MLflow experiment to search when no run ID is
            provided.
        tracking_uri: MLflow tracking URI to use. Defaults to the project's local
            SQLite tracking database.

    Returns:
        A tuple containing the local export path and the MLflow run ID used.
    """
    tracking_uri = tracking_uri or _DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)

    if run_id is None:
        run_id, val_iou = find_best_run(experiment_name, tracking_uri)
        msg = f"Best run found: {run_id}"
        if val_iou is not None:
            msg += f" (val_iou={val_iou:.4f})"
        logger.info(msg)
    else:
        logger.info(f"Exporting explicitly given run: {run_id}")

    output_path = Path(output_dir)
    if output_path.exists():
        logger.info(f"Removing existing export at {output_path} before re-exporting")
        shutil.rmtree(output_path)

    local_path = download_artifacts(
        artifact_uri=f"runs:/{run_id}/best_model",
        dst_path=str(output_path),
    )
    logger.info(f"Exported model to {local_path} (run_id={run_id})")
    return local_path, run_id


def main():
    """Run the model export workflow from the command line."""
    parser = argparse.ArgumentParser(description="Export a registered model to a standalone local folder.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to deployment/exported_model relative to this project, "
        "regardless of the directory you run this script from -- matches exactly "
        "where the Dockerfile's COPY expects it. Override only for a one-off "
        "export somewhere else.",
    )
    parser.add_argument("--run-id", default=None, help="Skip auto-search and export this specific run instead.")
    parser.add_argument("--experiment-name", default="water_body_segmentation")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Defaults to training/mlflow.db relative to this project, regardless of "
        "the directory you run this script from. Override only if your database is "
        "somewhere else.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIR
    export_model(output_dir, args.run_id, args.experiment_name, args.tracking_uri)


if __name__ == "__main__":
    main()
