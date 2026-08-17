"""
loss.py

Hybrid segmentation loss: a pixel-wise term plus a region term.

    loss = pixel_weight * pixel_term + (1 - pixel_weight) * TverskyLoss(alpha, beta)

The pixel term is selectable so that "which loss works better" is a
measurable question rather than an assumption:

    bce           plain BCEWithLogitsLoss (the original setting)
    weighted_bce  BCEWithLogitsLoss with pos_weight, for class imbalance
    focal         focal loss, down-weighting easy examples

Operates on raw logits, matching model.py's activation=None output.
Always pass logits straight from the model, never pre-sigmoided probs.

CONVENTION WARNING: this file weights the Tversky index as

    TI = TP / (TP + alpha * FP + beta * FN)

Salehi et al. (2017) use the opposite assignment, with alpha on FN. So
here, penalising false negatives harder -- the recall bias you want for
thin, elongated water features -- means beta > alpha. A setting copied
from the paper without swapping will move precision and recall the wrong way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Tversky loss for segmentation logits. alpha == beta == 0.5 is plain Dice."""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """Initialize the loss with Tversky weighting and smoothing.

        Args:
            alpha: Weight applied to false positives. Raise to buy precision.
            beta: Weight applied to false negatives. Raise to buy recall.
            smooth: Smoothing constant added to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """Compute the Tversky loss from logits and target masks.

        Args:
            logits: Raw model output logits for the batch.
            targets: Binary target mask tensor with the same spatial shape.

        Returns:
            A scalar loss tensor.
        """
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1).float()

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)
        target_sum = targets.sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # An all-background target has no true positives to score, so the ratio
        # above is meaningless there. Grade those on false positives instead,
        # matching how metrics.py treats an empty mask.
        n_pixels = probs.size(1)
        empty_score = 1.0 - fp / n_pixels
        tversky = torch.where(target_sum > 0, tversky, empty_score)

        return 1.0 - tversky.mean()


# Kept so existing imports and checkpoints referring to DiceLoss still resolve.
DiceLoss = TverskyLoss


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation logits (Lin et al., 2017)."""

    def __init__(self, gamma=2.0, alpha=0.5):
        """Initialize the focal loss.

        Args:
            gamma: Focusing exponent. 0.0 reduces this to plain BCE; higher
                values suppress the contribution of already-easy pixels.
            alpha: Weight on the positive class, with (1 - alpha) on the
                negative. 0.5 is neutral and lets gamma do the work alone.
                NOTE: the paper's 0.25 DOWN-weights the foreground -- it
                counterbalances gamma under detection-scale imbalance and is
                the wrong starting point for a 1:4.4 segmentation problem.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """Compute the focal loss from logits and target masks.

        Args:
            logits: Raw model output logits for the batch.
            targets: Binary target mask tensor.

        Returns:
            A scalar loss tensor.
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * (1.0 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
        return (alpha_t * loss).mean()


class CombinedLoss(nn.Module):
    """Weighted sum of a selectable pixel-wise term and a Tversky region term."""

    PIXEL_LOSSES = ("bce", "weighted_bce", "focal")

    # Declared for type checkers only. register_buffer() assigns through
    # nn.Module.__getattr__, which is typed as returning Tensor | Module, so
    # without this annotation every read of self.pos_weight looks like it could
    # be a Module. Bare annotation, no assignment -- it creates no class
    # attribute and does not interfere with the buffer registration below.
    pos_weight: torch.Tensor

    def __init__(
        self,
        pixel_loss="bce",
        pixel_weight=0.5,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        pos_weight=1.0,
        focal_gamma=2.0,
        focal_alpha=0.5,
        smooth=1.0,
    ):
        """Initialize the hybrid loss.

        Args:
            pixel_loss: One of "bce", "weighted_bce", or "focal".
            pixel_weight: Weight on the pixel-wise term. 0.0 leaves Tversky alone.
            tversky_alpha: False-positive weight in the region term.
            tversky_beta: False-negative weight in the region term.
            pos_weight: Positive-class multiplier for "weighted_bce". Set it to
                the measured background-to-foreground pixel ratio.
            focal_gamma: Focusing exponent for "focal".
            focal_alpha: Positive-class weight for "focal".
            smooth: Smoothing constant for the region term.

        Raises:
            ValueError: If pixel_loss is not a recognised name.
        """
        super().__init__()
        if pixel_loss not in self.PIXEL_LOSSES:
            raise ValueError(f"pixel_loss must be one of {self.PIXEL_LOSSES}, got {pixel_loss!r}")

        self.pixel_loss = pixel_loss
        self.pixel_weight = pixel_weight
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def _pixel_term(self, logits, targets):
        """Dispatch to the configured pixel-wise loss."""
        if self.pixel_loss == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets)
        if self.pixel_loss == "weighted_bce":
            # .to() is a no-op when already co-located, and spares the caller
            # having to remember criterion.to(device) for this one variant.
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight.to(logits.device)
            )
        return self.focal(logits, targets)

    def forward(self, logits, targets):
        """Compute the weighted sum of the pixel-wise and region terms.

        Args:
            logits: Raw model output logits for the batch.
            targets: Binary target mask tensor.

        Returns:
            A scalar loss tensor.
        """
        targets = targets.float()

        if logits.dim() == 4 and logits.size(1) == 1 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        pixel = self._pixel_term(logits, targets)
        region = self.tversky(logits, targets)

        return self.pixel_weight * pixel + (1.0 - self.pixel_weight) * region


class BCEDiceLoss(CombinedLoss):
    """BCE + Dice, the original configuration. Retained for existing call sites."""

    def __init__(self, bce_weight=0.5, dice_alpha=0.5, dice_beta=0.5, smooth=1.0):
        """Initialize with the original argument names.

        Args:
            bce_weight: Weight assigned to the BCE term.
            dice_alpha: False-positive weight for the region term.
            dice_beta: False-negative weight for the region term.
            smooth: Smoothing constant for the region term.
        """
        super().__init__(
            pixel_loss="bce",
            pixel_weight=bce_weight,
            tversky_alpha=dice_alpha,
            tversky_beta=dice_beta,
            smooth=smooth,
        )
