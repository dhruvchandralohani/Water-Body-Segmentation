"""
loss.py

BCE + Dice hybrid loss.

Operates on raw logits, matching model.py's activation=None output:
BCEWithLogitsLoss applies sigmoid internally, and DiceLoss applies its
own sigmoid explicitly. Always pass logits straight from the model,
never pre-sigmoided probabilities.

DiceLoss is written as the Tversky generalization (alpha=beta=0.5 is
plain Dice) so shifting toward a recall bias later is a parameter change, not a rewrite.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Tversky-based Dice loss for segmentation logits."""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """Initialize the loss with Tversky weighting and smoothing.

        Args:
            alpha: Weight applied to false positives in the Tversky formulation.
            beta: Weight applied to false negatives in the Tversky formulation.
            smooth: Smoothing constant added to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """Compute the Dice-style loss from logits and target masks.

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

        n_pixels = probs.size(1)
        empty_score = 1.0 - fp / n_pixels
        tversky = torch.where(target_sum > 0, tversky, empty_score)

        return 1.0 - tversky.mean()


class BCEDiceLoss(nn.Module):
    """Combine BCE loss with the Tversky-based Dice loss."""

    def __init__(self, bce_weight=0.5, dice_alpha=0.5, dice_beta=0.5, smooth=1.0):
        """Initialize the hybrid loss with configurable BCE weighting.

        Args:
            bce_weight: Weight assigned to the BCE term.
            dice_alpha: False-positive weight for the Dice term.
            dice_beta: False-negative weight for the Dice term.
            smooth: Smoothing constant for the Dice term.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(alpha=dice_alpha, beta=dice_beta, smooth=smooth)

    def forward(self, logits, targets):
        """Compute the weighted sum of BCE and Dice losses.

        Args:
            logits: Raw model output logits for the batch.
            targets: Binary target mask tensor.

        Returns:
            A scalar loss tensor.
        """
        targets = targets.float()

        if logits.dim() == 4 and logits.size(1) == 1 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
