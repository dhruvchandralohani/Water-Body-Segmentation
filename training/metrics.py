"""
metrics.py

SegmentationMetrics: accumulates confusion-matrix counts (TP/FP/FN/TN)
per tile, both globally and grouped by source filename, without ever
assembling a full-resolution image.
"""

from collections import defaultdict

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulate confusion-matrix statistics for segmentation predictions."""

    def __init__(self, threshold=0.5, eps=1e-7):
        """Initialize the metric accumulator.

        Args:
            threshold: Probability threshold used to convert predictions to binary.
            eps: Small epsilon added to denominator terms for numerical stability.
        """
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset all accumulated counts for global and per-image metrics."""
        self._global = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self._per_image = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    def update(self, preds, targets, filenames):
        """Accumulate confusion-matrix counts from predictions and targets.

        Args:
            preds: Predicted probabilities or logits for the batch.
            targets: Ground-truth target tensors for the batch.
            filenames: Source filename for each sample in the batch.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds_bin = preds >= self.threshold
        targets_bin = targets >= 0.5

        tp = preds_bin & targets_bin
        fp = preds_bin & ~targets_bin
        fn = ~preds_bin & targets_bin
        tn = ~preds_bin & ~targets_bin

        for i, filename in enumerate(filenames):
            counts = (
                int(tp[i].sum()),
                int(fp[i].sum()),
                int(fn[i].sum()),
                int(tn[i].sum()),
            )
            for key, val in zip(("tp", "fp", "fn", "tn"), counts):
                self._global[key] += val
                self._per_image[filename][key] += val

    def _compute(self, counts):
        """Compute IoU, Dice, precision, recall, and accuracy from counts."""
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        eps = self.eps

        if tp + fp + fn == 0:
            iou = dice = 1.0
        else:
            iou = tp / (tp + fp + fn + eps)
            dice = 2 * tp / (2 * tp + fp + fn + eps)

        precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp + eps)
        recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

        return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "accuracy": accuracy}

    def global_metrics(self):
        """Return metrics aggregated across all processed samples."""
        return self._compute(self._global)

    def per_image_metrics(self):
        """Return metrics aggregated separately for each source image."""
        return {fname: self._compute(counts) for fname, counts in self._per_image.items()}

    def summary(self):
        """Global metrics plus per-image IoU min/mean/max across every filename seen so far."""
        global_m = self.global_metrics()
        per_image = self.per_image_metrics()
        ious = np.array([m["iou"] for m in per_image.values()]) if per_image else np.array([])

        return {
            "global": global_m,
            "per_image_iou_min": float(ious.min()) if len(ious) else float("nan"),
            "per_image_iou_mean": float(ious.mean()) if len(ious) else float("nan"),
            "per_image_iou_max": float(ious.max()) if len(ious) else float("nan"),
            "num_images": len(per_image),
        }


@torch.no_grad()
def evaluate(model, eval_loader, device, threshold=0.5):
    """Evaluate a model over an evaluation loader and return summary metrics.

    Args:
        model: Segmentation model to evaluate.
        eval_loader: DataLoader yielding images, masks, and metadata.
        device: Device used for model execution.
        threshold: Probability threshold used for metric binarization.

    Returns:
        A dictionary containing global metrics and per-image IoU summaries.
    """
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)

    for images, masks, meta in eval_loader:
        images = images.to(device)
        logits = model(images)
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        preds = torch.sigmoid(logits).cpu()
        metrics.update(preds, masks, meta["filename"])

    return metrics.summary()
