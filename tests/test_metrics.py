"""
test_metrics.py

SegmentationMetrics decides every number this project reports. A defect here
does not produce a crash or a bad-looking curve -- it produces a plausible
score computed wrongly, which is the hardest kind of bug to notice.

Uses numpy arrays throughout. `update()` converts tensors when given them but
accepts arrays directly, so these tests exercise the real accumulation logic
without constructing a model or touching a GPU.

The contract worth naming up front: `update()` thresholds its input directly
against `self.threshold`, so it must receive PROBABILITIES, not logits.
Thresholding a logit at 0.5 is not the same decision as thresholding a
probability at 0.5 -- sigmoid(x) >= 0.5 is x >= 0, not x >= 0.5. Every caller
(evaluate(), validate(), test_model) applies sigmoid first.
"""

import numpy as np
import pytest

from training.metrics import SegmentationMetrics


def as_batch(array):
    """Shape a 2-D mask into the (N, H, W) batch layout update() expects."""
    return np.asarray(array, dtype=float)[None, ...]


# ---------------------------------------------------------------------------
# Correctness on hand-computable cases
# ---------------------------------------------------------------------------


def test_perfect_prediction_scores_one():
    """A prediction identical to the target gives IoU and Dice of 1."""
    target = np.array([[1, 1, 0], [0, 1, 0]])
    metrics = SegmentationMetrics()
    metrics.update(as_batch(target), as_batch(target), ["a.png"])

    result = metrics.global_metrics()
    assert result["iou"] == pytest.approx(1.0, abs=1e-6)
    assert result["dice"] == pytest.approx(1.0, abs=1e-6)
    assert result["precision"] == pytest.approx(1.0, abs=1e-6)
    assert result["recall"] == pytest.approx(1.0, abs=1e-6)


def test_completely_wrong_prediction_scores_zero():
    """A prediction that inverts the target has no true positives."""
    target = np.array([[1, 1], [0, 0]])
    metrics = SegmentationMetrics()
    metrics.update(as_batch(1 - target), as_batch(target), ["a.png"])

    result = metrics.global_metrics()
    assert result["iou"] == pytest.approx(0.0, abs=1e-6)
    assert result["dice"] == pytest.approx(0.0, abs=1e-6)


def test_known_confusion_matrix_gives_exact_iou():
    """One TP, one FP, one FN must give IoU 1/3 and Dice 1/2 exactly.

    Pinned by hand rather than by a helper, so a rewrite of _compute cannot
    quietly redefine the metric and still pass.
    """
    target = np.array([[1, 1, 0, 0]])   # two positives
    pred = np.array([[1, 0, 1, 0]])     # hits one, misses one, invents one
    metrics = SegmentationMetrics()
    metrics.update(as_batch(pred), as_batch(target), ["a.png"])

    result = metrics.global_metrics()
    assert result["iou"] == pytest.approx(1 / 3, abs=1e-5)
    assert result["dice"] == pytest.approx(0.5, abs=1e-5)
    assert result["precision"] == pytest.approx(0.5, abs=1e-5)
    assert result["recall"] == pytest.approx(0.5, abs=1e-5)
    assert result["accuracy"] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Empty masks -- 127 kept images sit under 1% water, so this branch is live
# ---------------------------------------------------------------------------


def test_empty_target_predicted_empty_scores_one():
    """No foreground and none predicted is a perfect result, not a zero.

    The naive formula gives 0/0 here. Scoring it 0.0 would silently drag down
    the per-image mean by punishing correct predictions on land-only tiles.
    """
    empty = np.zeros((4, 4))
    metrics = SegmentationMetrics()
    metrics.update(as_batch(empty), as_batch(empty), ["a.png"])

    result = metrics.global_metrics()
    assert result["iou"] == pytest.approx(1.0)
    assert result["dice"] == pytest.approx(1.0)


def test_empty_target_with_any_prediction_scores_zero():
    """Predicting water where there is none is a genuine miss, scored 0.

    Together with the previous test this pins the asymmetry: an empty target is
    only free if the prediction is empty too.
    """
    target = np.zeros((4, 4))
    pred = np.zeros((4, 4))
    pred[0, 0] = 1.0

    metrics = SegmentationMetrics()
    metrics.update(as_batch(pred), as_batch(target), ["a.png"])
    assert metrics.global_metrics()["iou"] == pytest.approx(0.0, abs=1e-6)


def test_a_per_image_iou_of_zero_is_a_real_miss_not_a_divide_by_zero():
    """Guards the interpretation of per_image_iou_min = 0.0000 in the logs.

    That value appears in every epoch of every run. It matters whether it means
    "the model failed on an image" or "the metric divided by zero on an empty
    mask", because only one of those is worth investigating.
    """
    empty = np.zeros((4, 4))
    metrics = SegmentationMetrics()
    metrics.update(as_batch(empty), as_batch(empty), ["empty.png"])
    assert metrics.summary()["per_image_iou_min"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Micro vs macro
# ---------------------------------------------------------------------------


def test_global_iou_is_pixel_weighted_not_an_average_of_images():
    """Global IoU must weight by pixels, so a large image dominates a small one.

    This is the whole reason the project reports global IoU and per-image mean
    IoU separately: image area spans three orders of magnitude here, so the two
    are different statistics and the gap between them is a real finding, not
    noise. A change making global IoU a mean over images would pass every
    single-image test above.
    """
    metrics = SegmentationMetrics()

    # Small image, perfect: 4 px, all foreground, all correct.
    small = np.ones((2, 2))
    metrics.update(as_batch(small), as_batch(small), ["small.png"])

    # Large image, half wrong: 100 px, all foreground, half predicted.
    large_target = np.ones((10, 10))
    large_pred = np.zeros((10, 10))
    large_pred[:5, :] = 1.0
    metrics.update(as_batch(large_pred), as_batch(large_target), ["large.png"])

    summary = metrics.summary()
    per_image = [1.0, 0.5]  # small perfect, large half-recalled

    # Pixel-weighted: 54 TP of 104 union.
    assert summary["global"]["iou"] == pytest.approx(54 / 104, abs=1e-4)
    assert summary["per_image_iou_mean"] == pytest.approx(np.mean(per_image), abs=1e-4)
    assert summary["global"]["iou"] != pytest.approx(summary["per_image_iou_mean"], abs=1e-3)


def test_per_image_grouping_is_by_filename(): 
    """Tiles carrying the same filename accumulate into one image, not several."""
    metrics = SegmentationMetrics()
    ones = np.ones((2, 2))
    for _ in range(3):
        metrics.update(as_batch(ones), as_batch(ones), ["same.png"])

    assert metrics.summary()["num_images"] == 1
    assert set(metrics.per_image_metrics()) == {"same.png"}


def test_tiled_accumulation_equals_whole_image_scoring():
    """Scoring an image tile by tile must equal scoring it in one piece.

    This is the class's central design claim -- confusion counts accumulate
    without ever assembling a full-resolution prediction. If it did not hold,
    every reported number would depend on the tile size used to produce it.
    """
    rng = np.random.default_rng(0)
    target = (rng.random((8, 12)) > 0.6).astype(float)
    pred = (rng.random((8, 12)) > 0.5).astype(float)

    whole = SegmentationMetrics()
    whole.update(as_batch(pred), as_batch(target), ["img.png"])

    tiled = SegmentationMetrics()
    for r0 in range(0, 8, 4):
        for c0 in range(0, 12, 4):
            tiled.update(
                as_batch(pred[r0 : r0 + 4, c0 : c0 + 4]),
                as_batch(target[r0 : r0 + 4, c0 : c0 + 4]),
                ["img.png"],
            )

    assert tiled.global_metrics()["iou"] == pytest.approx(whole.global_metrics()["iou"], abs=1e-9)
    assert tiled.summary()["num_images"] == whole.summary()["num_images"] == 1


# ---------------------------------------------------------------------------
# Thresholding contract
# ---------------------------------------------------------------------------


def test_update_thresholds_probabilities_directly():
    """update() compares its input to `threshold`, so it needs probabilities.

    Passing logits would silently reinterpret every value: sigmoid(x) >= 0.5 is
    x >= 0, not x >= 0.5, so a logit of 0.3 is a positive prediction that this
    method would score as negative.
    """
    target = np.array([[1, 1, 1, 1]])
    probs = np.array([[0.9, 0.6, 0.49, 0.1]])  # two above the 0.5 threshold

    metrics = SegmentationMetrics(threshold=0.5)
    metrics.update(as_batch(probs), as_batch(target), ["a.png"])
    assert metrics.global_metrics()["recall"] == pytest.approx(0.5, abs=1e-5)


def test_threshold_is_configurable():
    """Lowering the threshold admits more positives, raising recall."""
    target = np.array([[1, 1, 1, 1]])
    probs = np.array([[0.9, 0.6, 0.4, 0.1]])

    low = SegmentationMetrics(threshold=0.3)
    low.update(as_batch(probs), as_batch(target), ["a.png"])
    high = SegmentationMetrics(threshold=0.7)
    high.update(as_batch(probs), as_batch(target), ["a.png"])

    assert low.global_metrics()["recall"] > high.global_metrics()["recall"]


def test_reset_clears_all_accumulated_state():
    """A reused accumulator must not carry counts across evaluations."""
    metrics = SegmentationMetrics()
    ones = np.ones((2, 2))
    metrics.update(as_batch(ones), as_batch(ones), ["a.png"])
    metrics.reset()

    assert metrics.summary()["num_images"] == 0
    assert np.isnan(metrics.summary()["per_image_iou_mean"])


def test_summary_exposes_the_keys_the_pipeline_writes():
    """test_model.py writes these into metrics/test_metrics.json by name."""
    metrics = SegmentationMetrics()
    ones = np.ones((2, 2))
    metrics.update(as_batch(ones), as_batch(ones), ["a.png"])

    summary = metrics.summary()
    assert set(summary) >= {
        "global",
        "per_image_iou_min",
        "per_image_iou_mean",
        "per_image_iou_max",
        "num_images",
    }
    assert set(summary["global"]) == {"iou", "dice", "precision", "recall", "accuracy"}
