"""
tracking.py

MLflow experiment tracking wrapper. Logs config once per run, metrics
per epoch, and artifacts (checkpoints, sample predictions).

Uses a SQLite backend (sqlite:///mlflow.db), not the plain filesystem store.

GPU memory is tracked via MLflow's built-in log_system_metrics=True
rather than hand-rolled, so it picks up automatically on a real GPU and
degrades to "skip GPU metrics" cleanly on CPU-only runs.
"""

import hashlib
import inspect
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow.models.signature import ModelSignature
from mlflow.pytorch import log_model as _log_model_fn
from mlflow.types.schema import Schema, TensorSpec

logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)


def _log_pytorch_model(model, artifact_name, **kwargs):
    """Log a PyTorch model artifact while handling MLflow API compatibility."""
    sig = inspect.signature(_log_model_fn)

    call_kwargs = dict(kwargs)
    if "serialization_format" in call_kwargs and "serialization_format" not in sig.parameters:
        call_kwargs.pop("serialization_format")

    if "artifact_path" in sig.parameters:
        return _log_model_fn(model, artifact_path=artifact_name, **call_kwargs)
    return _log_model_fn(model, name=artifact_name, **call_kwargs)


def hash_manifest(csv_path):
    """Return a short SHA-256 hash for a manifest file.

    Args:
        csv_path: Path to the manifest file to hash.

    Returns:
        A truncated hexadecimal digest identifying the manifest contents.
    """
    with open(csv_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


class ExperimentTracker:
    """Manage MLflow experiment tracking for training runs."""

    def __init__(self, experiment_name, run_name=None, tracking_uri="sqlite:///mlflow.db"):
        """Initialize the tracker and configure the MLflow experiment.

        Args:
            experiment_name: Name of the MLflow experiment to use.
            run_name: Optional display name for the run.
            tracking_uri: MLflow tracking backend URI.
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_name = run_name
        self._epoch_start_time = None

    def __enter__(self):
        """Start an MLflow run when the tracker is entered as a context manager."""
        self._run = mlflow.start_run(run_name=self.run_name, log_system_metrics=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the active MLflow run when exiting the context manager."""
        mlflow.end_run()

    def log_config(self, data_config, model_config, loss_config, optim_config, train_manifest=None):
        """Log the training configuration as MLflow parameters.

        Args:
            data_config: Data pipeline configuration values.
            model_config: Model configuration values.
            loss_config: Loss configuration values.
            optim_config: Optimizer configuration values.
            train_manifest: Optional path to the training manifest used for hashing.
        """
        params = {}
        for prefix, cfg in [
            ("data", data_config),
            ("model", model_config),
            ("loss", loss_config),
            ("optim", optim_config),
        ]:
            for k, v in cfg.items():
                params[f"{prefix}.{k}"] = v

        if train_manifest is not None:
            params["data.train_manifest_hash"] = hash_manifest(train_manifest)

        mlflow.log_params(params)

    def start_epoch(self):
        """Call at the start of each epoch, to time it."""
        self._epoch_start_time = time.time()

    def log_epoch(self, epoch, train_loss, val_summary, lr):
        """Log training and validation metrics for a completed epoch."""
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_summary["val_loss"],
            "val_iou": val_summary["global"]["iou"],
            "val_dice": val_summary["global"]["dice"],
            "val_precision": val_summary["global"]["precision"],
            "val_recall": val_summary["global"]["recall"],
            "val_accuracy": val_summary["global"]["accuracy"],
            "val_per_image_iou_min": val_summary["per_image_iou_min"],
            "val_per_image_iou_mean": val_summary["per_image_iou_mean"],
            "val_per_image_iou_max": val_summary["per_image_iou_max"],
            "learning_rate": lr,
        }
        if self._epoch_start_time is not None:
            metrics["epoch_time_sec"] = time.time() - self._epoch_start_time

        mlflow.log_metrics(metrics, step=epoch)

    def log_checkpoint(self, model, path, is_best=False, registered_model_name=None, input_shape=None):
        """Save a checkpoint and optionally log it as the best model artifact.

        Args:
            model: PyTorch model to save.
            path: Destination filepath for the checkpoint on disk.
            is_best: Whether the checkpoint should also be logged as the best model.
            registered_model_name: Optional registered model name for MLflow.
            input_shape: Optional input shape used to create a model signature.
        """
        torch.save(model.state_dict(), path)
        mlflow.log_artifact(str(path), artifact_path="checkpoints")

        if is_best:
            signature = None
            if input_shape is not None:
                c, h, w = input_shape
                signature = ModelSignature(
                    inputs=Schema([TensorSpec(np.dtype("float32"), (-1, c, h, w))]),
                    outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 1, h, w))]),
                )

            _log_pytorch_model(
                model,
                "best_model",
                registered_model_name=registered_model_name,
                serialization_format="pickle",  # avoids pt2 trace's batch-size specialization risk
                signature=signature,
            )

    def log_sample_predictions(self, image, step=None, artifact_file="sample_predictions.png"):
        """Logs a preview image (e.g. an inspect_pipeline.py-style image/mask/overlay grid)."""
        mlflow.log_image(image, artifact_file=artifact_file, step=step)
