"""
test_loss.py

The losses are where a sign error costs a night of GPU time and produces a
result that looks merely disappointing rather than wrong.

The single most important test here is the Tversky convention. This codebase
weights the index as

    TI = TP / (TP + alpha * FP + beta * FN)

with alpha on FALSE POSITIVES. Salehi et al. (2017) put alpha on false
negatives. So a recall bias -- the setting that targets thin, elongated water
features -- means beta > alpha HERE and alpha > beta in the paper. Copying a
setting out of the literature without swapping moves precision and recall the
wrong way, and nothing about the training curve would reveal it.

Everything operates on logits, matching model.py's activation=None output.
"""


import pytest
import torch

from training.loss import BCEDiceLoss, CombinedLoss, FocalLoss, TverskyLoss


def logits_from(probabilities):
    """Convert probabilities to the logits the losses expect."""
    tensor = torch.tensor(probabilities, dtype=torch.float32)
    return torch.log(tensor / (1 - tensor))


# Under-prediction: four true positives, only one confidently found.
# This is the thin-feature failure mode in miniature -- false negatives dominate.
FN_HEAVY_PROBS = [[0.9, 0.1, 0.1, 0.1, 0.05, 0.05]]
FN_HEAVY_TARGET = [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]

# Over-prediction: everything called water, so false positives dominate.
FP_HEAVY_PROBS = [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]


# ---------------------------------------------------------------------------
# Tversky convention -- the easiest thing in the codebase to get backwards
# ---------------------------------------------------------------------------


def test_beta_above_alpha_punishes_false_negatives():
    """beta > alpha must RAISE the loss on an under-predicting model.

    Pins the convention. If someone swaps the multipliers to match the paper's
    ordering without renaming the arguments, this fails -- which is the whole
    point, because the training curve would look completely normal.
    """
    logits = logits_from(FN_HEAVY_PROBS)
    target = torch.tensor(FN_HEAVY_TARGET)

    balanced = TverskyLoss(alpha=0.5, beta=0.5)(logits, target)
    recall_biased = TverskyLoss(alpha=0.3, beta=0.7)(logits, target)
    precision_biased = TverskyLoss(alpha=0.7, beta=0.3)(logits, target)

    assert recall_biased > balanced > precision_biased


def test_alpha_above_beta_punishes_false_positives():
    """The mirror case, so the test above cannot pass by accident.

    A loss that simply increased with alpha+beta would satisfy one direction;
    only a correctly wired index satisfies both.
    """
    logits = logits_from(FP_HEAVY_PROBS)
    target = torch.tensor(FN_HEAVY_TARGET)

    recall_biased = TverskyLoss(alpha=0.3, beta=0.7)(logits, target)
    precision_biased = TverskyLoss(alpha=0.7, beta=0.3)(logits, target)

    assert precision_biased > recall_biased


def test_tversky_at_half_half_is_dice():
    """alpha == beta == 0.5 is plain Dice, which is what the project ships.

    Worth stating as a test: the class is named for Tversky but the shipped
    configuration is the degenerate case, and the knobs went unused until the
    loss ablation.
    """
    logits = logits_from([[0.8, 0.3, 0.6, 0.2]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    probs = torch.sigmoid(logits)
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    expected = 1.0 - (tp + 1.0) / (tp + 0.5 * fp + 0.5 * fn + 1.0)

    assert TverskyLoss(alpha=0.5, beta=0.5)(logits, target) == pytest.approx(expected.item(), abs=1e-6)


def test_empty_target_is_graded_on_false_positives():
    """An all-background target has no true positives to score.

    The ratio is meaningless there, so the loss falls back to a false-positive
    penalty -- matching how metrics.py treats an empty mask. Without this an
    all-background tile would report a constant loss regardless of prediction.
    """
    target = torch.zeros(1, 16)
    loss_fn = TverskyLoss()

    clean = loss_fn(logits_from([[0.01] * 16]), target)
    messy = loss_fn(logits_from([[0.99] * 16]), target)

    assert messy > clean
    assert clean == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# Focal
# ---------------------------------------------------------------------------


def test_focal_at_gamma_zero_reduces_to_scaled_bce():
    """gamma=0 removes the focusing term, leaving alpha-weighted BCE.

    A reduction test like this catches an exponent applied to the wrong factor,
    which otherwise only shows up as slightly-off convergence.
    """
    logits = logits_from([[0.8, 0.3, 0.6]])
    target = torch.tensor([[1.0, 0.0, 1.0]])

    focal = FocalLoss(gamma=0.0, alpha=0.5)(logits, target)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    assert focal == pytest.approx(0.5 * bce.item(), abs=1e-6)


def test_focal_suppresses_easy_pixels_far_more_than_hard_ones():
    """The defining property: down-weight what the model already gets right.

    Compares the ratio against plain BCE for the same pixel, so it measures the
    focusing effect itself rather than the overall scale.
    """
    target = torch.tensor([[1.0]])
    focal = FocalLoss(gamma=2.0, alpha=0.5)

    def suppression(probability):
        logits = logits_from([[probability]])
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        return bce.item() / (focal(logits, target).item() / 0.5)

    easy = suppression(0.95)
    hard = suppression(0.20)

    assert easy > 50 * hard, f"easy suppressed {easy:.0f}x, hard {hard:.1f}x"


def test_focal_alpha_shifts_weight_between_classes():
    """alpha above 0.5 favours the positive class, below it the negative.

    The paper's 0.25 DOWN-weights foreground; it counterbalances gamma under
    detection-scale imbalance and is the wrong starting point at 1:4.4, which
    is why the default here is a neutral 0.5.
    """
    positive_logits = logits_from([[0.3]])
    positive_target = torch.tensor([[1.0]])

    favours_positive = FocalLoss(gamma=2.0, alpha=0.75)(positive_logits, positive_target)
    favours_negative = FocalLoss(gamma=2.0, alpha=0.25)(positive_logits, positive_target)

    assert favours_positive > favours_negative


# ---------------------------------------------------------------------------
# CombinedLoss wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pixel_loss", ["bce", "weighted_bce", "focal"])
def test_every_pixel_term_produces_a_finite_scalar(pixel_loss):
    """All three selectable terms must return a finite scalar, not a tensor."""
    logits = logits_from([[0.8, 0.3, 0.6, 0.2]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    value = CombinedLoss(pixel_loss=pixel_loss, pos_weight=4.4)(logits, target)
    assert value.dim() == 0
    assert torch.isfinite(value)


def test_unknown_pixel_loss_is_rejected_at_construction():
    """A typo in a grid arm must fail immediately, not silently pick a default."""
    with pytest.raises(ValueError, match="pixel_loss"):
        CombinedLoss(pixel_loss="focal_dice")


def test_pixel_weight_zero_leaves_only_the_region_term():
    """pixel_weight=0 must isolate Tversky exactly, for the region-only arm."""
    logits = logits_from([[0.8, 0.3, 0.6, 0.2]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    combined = CombinedLoss(pixel_weight=0.0, tversky_alpha=0.5, tversky_beta=0.5)(logits, target)
    region_only = TverskyLoss(alpha=0.5, beta=0.5)(logits, target)

    assert combined == pytest.approx(region_only.item(), abs=1e-6)


def test_weighted_bce_penalises_missed_foreground_more_than_bce():
    """pos_weight is set from the measured background:foreground ratio of 4.4.

    Its whole purpose is to make a false negative cost more than a false
    positive, so a missed-foreground case must score higher than plain BCE.
    """
    logits = logits_from(FN_HEAVY_PROBS)
    target = torch.tensor(FN_HEAVY_TARGET)

    plain = CombinedLoss(pixel_loss="bce")(logits, target)
    weighted = CombinedLoss(pixel_loss="weighted_bce", pos_weight=4.4)(logits, target)

    assert weighted > plain


def test_pos_weight_buffer_follows_the_module_device():
    """Registered as a buffer so .to(device) moves it with the module.

    A CPU pos_weight against CUDA logits is a runtime error partway into a run,
    not at construction.
    """
    loss = CombinedLoss(pixel_loss="weighted_bce", pos_weight=4.4)
    assert "pos_weight" in dict(loss.named_buffers())


def test_bce_dice_alias_matches_the_general_class():
    """The retained alias must stay behaviourally identical to its superclass."""
    logits = logits_from([[0.8, 0.3, 0.6, 0.2]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    alias = BCEDiceLoss(bce_weight=0.5, dice_alpha=0.5, dice_beta=0.5)(logits, target)
    general = CombinedLoss(pixel_loss="bce", pixel_weight=0.5)(logits, target)

    assert alias == pytest.approx(general.item(), abs=1e-6)


# ---------------------------------------------------------------------------
# Gradient health
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pixel_loss", ["bce", "weighted_bce", "focal"])
def test_gradients_are_finite_and_nonzero(pixel_loss):
    """Every term must produce usable gradients.

    A NaN here does not raise -- it propagates into the weights and the run
    continues, producing a model that silently learned nothing.
    """
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    CombinedLoss(pixel_loss=pixel_loss, pos_weight=4.4)(logits, target).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_saturated_predictions_do_not_produce_nan():
    """Extreme logits are where a naive log() implementation breaks.

    Segmentation models do reach saturation on easy background, so this is a
    realistic input rather than an adversarial one.
    """
    logits = torch.tensor([[[[-60.0, 60.0], [60.0, -60.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

    value = CombinedLoss(pixel_loss="focal")(logits, target)
    value.backward()

    assert torch.isfinite(value)
    assert torch.isfinite(logits.grad).all()


def test_a_confident_correct_prediction_scores_near_zero():
    """Sanity anchor: the loss must actually be minimised by being right."""
    logits = torch.tensor([[[[-20.0, 20.0], [20.0, -20.0]]]])
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

    assert CombinedLoss(pixel_loss="bce")(logits, target).item() < 0.05


def test_a_three_dimensional_target_is_broadcast_to_match_logits():
    """Loaders yield (N, H, W) masks while models emit (N, 1, H, W).

    Without the unsqueeze, broadcasting silently produces an (N, N, H, W)
    tensor and a loss computed against the wrong pairing -- a bug that trains
    without complaint.
    """
    logits = torch.randn(3, 1, 4, 4)
    target = (torch.rand(3, 4, 4) > 0.5).float()

    value = CombinedLoss(pixel_loss="bce")(logits, target)
    assert value.dim() == 0
    assert torch.isfinite(value)
