"""
test_model.py

Finds the best model from a tuning study (by querying MLflow for the
highest val_iou across all runs -- no need to remember a trial number
or run ID), runs it against the test set
with the same deterministic tiling + stitching used for evaluation, and
saves a visual grid: image / true mask / predicted mask / overlay, for
a handful of test images, plus the metrics for the run that was picked.

Usage:
    python test_model.py --test-manifest splits/test.csv --image-dir images/ \
        --mask-dir masks_corrected/ --num-images 4
"""

import argparse
import json
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import mlflow
from mlflow.pytorch import load_model as load_pytorch_model

from common.logging_setup import log_run_separator, setup_logger
from common.mlflow_utils import find_best_run
from data_pipeline.data import get_eval_loader
from data_pipeline.stitch import PredictionStitcher
from data_pipeline.tile_dataset import TileDataset
from data_pipeline.transforms import get_eval_transform
from training.metrics import evaluate

logger = setup_logger("test_model", log_file="testing.log")


def resize_for_display(img, max_size=800):
    """Resize an image or mask for display while preserving aspect ratio.

    Args:
        img: Image or mask array to resize.
        max_size: Maximum allowed edge length for the display output.

    Returns:
        A resized image or mask array suitable for plotting.
    """
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def main():
    """Run model evaluation on the test set and save a prediction preview image."""
    parser = argparse.ArgumentParser(
        description="Run the best tuned model against the test set and visualize predictions."
    )
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--experiment-name", default="water_body_segmentation")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--run-id", default=None, help="Skip auto-search and use this specific run instead.")
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--num-images", type=int, default=16, help="How many test images to visualize.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="predictions_preview.png")
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Also write the test metrics to this JSON path, so tools outside "
        "MLflow (DVC, CI) can read them. Ignored with --skip-metrics.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for full test-set metrics.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--skip-metrics", action="store_true", help="Skip full test-set metrics, only generate the visual preview."
    )
    parser.add_argument(
        "--skip-preview", action="store_true", help="Skip the visual preview, only compute metrics."
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.run_id:
        run_id, val_iou = args.run_id, None
    else:
        run_id, val_iou = find_best_run(args.experiment_name, args.tracking_uri)
        msg = f"Best run found: {run_id}"
        if val_iou is not None:
            msg += f" (val_iou={val_iou:.4f})"
        logger.info(msg)

    log_run_separator(logger, f"Test evaluation: run={run_id} test_manifest={args.test_manifest}")

    mlflow.set_tracking_uri(args.tracking_uri)
    model = load_pytorch_model(f"runs:/{run_id}/best_model", map_location=device)
    model.to(device)
    model.eval()

    if not args.skip_metrics:
        logger.info("Computing metrics over the full test set...")
        test_loader = get_eval_loader(
            manifest=args.test_manifest,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        summary = evaluate(model, test_loader, device, threshold=args.threshold)

        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in summary["global"].items())
        logger.info(f"Test set metrics: {metrics_str}")
        logger.info(
            f"Per-image IoU: min={summary['per_image_iou_min']:.4f} "
            f"mean={summary['per_image_iou_mean']:.4f} max={summary['per_image_iou_max']:.4f} "
            f"(n={summary['num_images']} images)"
        )

        # Built once and reused for both sinks, so the MLflow run and the JSON
        # file can never drift apart. float() strips numpy scalar types, which
        # json.dump cannot serialize.
        test_metrics = {
            "test_iou": float(summary["global"]["iou"]),
            "test_dice": float(summary["global"]["dice"]),
            "test_precision": float(summary["global"]["precision"]),
            "test_recall": float(summary["global"]["recall"]),
            "test_accuracy": float(summary["global"]["accuracy"]),
            "test_per_image_iou_min": float(summary["per_image_iou_min"]),
            "test_per_image_iou_mean": float(summary["per_image_iou_mean"]),
            "test_per_image_iou_max": float(summary["per_image_iou_max"]),
        }

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(test_metrics)
        logger.info(f"Test metrics logged to MLflow run {run_id} (visible in `mlflow ui` alongside training metrics).")

        if args.metrics_json:
            metrics_path = Path(args.metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                # Provenance: ties the numbers back to the exact evaluated run.
                "run_id": run_id,
                "threshold": args.threshold,
                "num_images": int(summary["num_images"]),
                **test_metrics,
            }
            with open(metrics_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Test metrics written to {metrics_path}")

    if args.skip_preview:
        return

    dataset = TileDataset(
        manifest=args.test_manifest,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        transform=get_eval_transform(),
    )

    filenames = list(dict.fromkeys(dataset.filenames))[: args.num_images]
    stitcher = PredictionStitcher()

    with torch.no_grad():
        for i in range(len(dataset)):
            image_idx, x0, y0, eff_tile = dataset.tile_index[i]
            filename = dataset.filenames[image_idx]
            if filename not in filenames:
                continue

            image_t, mask_t, meta = dataset[i]
            assert isinstance(image_t, torch.Tensor)
            logits = model(image_t.unsqueeze(0).to(device))
            if logits.dim() == 4 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            pred = torch.sigmoid(logits)[0].cpu().numpy()

            stitcher.add_tile(
                filename, meta["x0"], meta["y0"], meta["tile_size"],
                meta["orig_width"], meta["orig_height"], pred,
            )

    n = len(filenames)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, filename in enumerate(filenames):
        image_bgr = cv2.imread(str(dataset.image_dir / filename))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image for {filename}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        true_mask = cv2.imread(str(dataset.mask_dir / filename), cv2.IMREAD_GRAYSCALE)
        if true_mask is None:
            raise FileNotFoundError(f"Could not load mask for {filename}")
        pred_mask = stitcher.get_result(filename, threshold=args.threshold)

        image_disp = resize_for_display(image)
        true_disp = resize_for_display(true_mask)
        pred_disp = resize_for_display((pred_mask * 255).astype(np.uint8))

        overlay = image_disp.copy()
        highlight = pred_disp.astype(bool)
        overlay[highlight] = (overlay[highlight] * 0.4 + np.array([255, 60, 60]) * 0.6).astype(np.uint8)

        axes[row, 0].imshow(image_disp)
        axes[row, 1].imshow(true_disp, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(pred_disp, cmap="gray", vmin=0, vmax=255)
        axes[row, 3].imshow(overlay)
        for c in range(4):
            axes[row, c].axis("off")

        axes[row, 0].set_ylabel(filename, fontsize=8)

        if row == 0:
            axes[row, 0].set_title("image")
            axes[row, 1].set_title("true mask")
            axes[row, 2].set_title("predicted mask")
            axes[row, 3].set_title("prediction overlay")

    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved predictions preview to {args.output}")


if __name__ == "__main__":
    main()
