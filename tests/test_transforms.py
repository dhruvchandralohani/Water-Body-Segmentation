"""
test_transforms.py

Augmentation is the one place where a bug produces no error, no warning, and no
visible symptom -- just a model learning from mislabelled data. If a geometric
transform reaches the image but not the mask, every downstream number is
computed on scrambled supervision and nothing in the training log looks unusual.

Alignment is checked by identifying WHICH of the eight dihedral elements each of
the image and the mask received, then asserting they match. An earlier version
tracked a marker's centroid instead and gave false failures: CoarseDropout fills
holes with random_uniform noise that can outshine the marker, so
brightest-region recovery is not reliable downstream of the photometric ops.
Matching against all eight candidates is robust to brightness, contrast, noise
and occlusion, because the correct orientation still correlates far better than
the seven wrong ones.
"""

import numpy as np
import pytest

from data_pipeline.transforms import _binarize_mask, get_eval_transform, get_train_transform

SIZE = 64

# The dihedral group of a square. Direction conventions do not matter -- only
# that the set is complete, so whatever albumentations applies is somewhere in it.
D4_OPS = [
    lambda a: a,
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.rot90(a, 3),
    lambda a: np.fliplr(a),
    lambda a: np.rot90(np.fliplr(a), 1),
    lambda a: np.rot90(np.fliplr(a), 2),
    lambda a: np.rot90(np.fliplr(a), 3),
]


def reference_pair():
    """An image ramp and an asymmetric mask, both distinct under all eight elements.

    The ramp `2r + c` and the corner rectangle are chosen so no two group
    elements produce the same array -- otherwise a mismatch could hide behind a
    coincidental symmetry.
    """
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    image = np.stack([(rows * 2 + cols) % 256] * 3, axis=-1).astype(np.uint8)
    mask = (((rows < SIZE // 3) & (cols < SIZE // 2)) * 255).astype(np.uint8)
    return image, mask


def to_hw(array):
    """Reduce a transformed tensor or array to 2-D (H, W).

    Rejects a dict outright: passing the whole transform result here instead of
    one of its entries is an easy slip, and numpy silently wraps it in a 0-d
    object array rather than failing.
    """
    if isinstance(array, dict):
        raise TypeError("pass out['image'] or out['mask'], not the whole result dict")
    array = array.numpy() if hasattr(array, "numpy") else np.asarray(array)
    if array.ndim == 3:
        array = array[0] if array.shape[0] <= 4 else array[..., 0]
    return array


def identify_element(reference, observed, binary):
    """Return the index of the D4 element that best explains `observed`.

    Args:
        reference: The untransformed 2-D array.
        observed: The transformed 2-D array.
        binary: Score by IoU when True (masks), Pearson correlation otherwise.

    Returns:
        Index into D4_OPS.
    """
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if binary:
        def score(candidate):
            a, b = candidate > 0.5, observed > 0.5
            return (a & b).sum() / max(1, (a | b).sum())
    else:
        def score(candidate):
            return np.corrcoef(candidate.ravel(), observed.ravel())[0, 1]

    return int(np.argmax([score(op(reference)) for op in D4_OPS]))


def seeded(seed):
    """Seed both RNGs albumentations may draw from."""
    import random

    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Alignment -- the test that matters most
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(16))
def test_image_and_mask_receive_the_same_geometric_element(seed):
    """Whatever D4 element the image gets, the mask must get the same one.

    Sixteen seeds so every element is exercised: a single draw has a one-in-
    eight chance of the identity, which would pass no matter how broken the
    pipeline was.
    """
    seeded(seed)
    image, mask = reference_pair()
    out = get_train_transform()(image=image, mask=mask)

    image_element = identify_element(to_hw(image), to_hw(out["image"]), binary=False)
    mask_element = identify_element(to_hw(mask), to_hw(out["mask"]), binary=True)

    assert image_element == mask_element, (
        f"image received D4 element {image_element}, mask received {mask_element} -- "
        "a geometric transform reached one but not the other"
    )


def test_augmentation_reaches_most_of_the_group():
    """Guards the alignment test from passing because nothing happens at all.

    A pipeline applying no geometry would satisfy every alignment assertion
    perfectly. D4 at p=1.0 draws uniformly, so a few dozen seeds should reach
    most of the eight.
    """
    image, mask = reference_pair()
    seen = set()
    for seed in range(60):
        seeded(seed)
        out = get_train_transform()(image=image, mask=mask)
        seen.add(identify_element(to_hw(mask), to_hw(out["mask"]), binary=True))

    assert len(seen) >= 6, f"reached only {sorted(seen)} of 8 elements; is D4 at p=1.0?"


def test_d4_draws_are_roughly_uniform():
    """The reason D4(p=1.0) replaced three independent coin flips.

    Those flips reach all eight elements but weight them about 3:1 toward the
    identity and the axis-aligned ones. If the invariance is exact -- and for
    overhead imagery it is -- the correct distribution over it is uniform.
    """
    image, mask = reference_pair()
    counts = {}
    draws = 240
    for seed in range(draws):
        seeded(seed)
        out = get_train_transform()(image=image, mask=mask)
        element = identify_element(to_hw(mask), to_hw(out["mask"]), binary=True)
        counts[element] = counts.get(element, 0) + 1

    ordered = sorted(counts.values(), reverse=True)
    assert len(counts) == 8, f"only {len(counts)} of 8 elements drawn in {draws} tries"
    assert ordered[0] / ordered[-1] < 3.0, f"draws badly skewed: {ordered}"


def test_eval_transform_is_deterministic():
    """Evaluation must not augment: identical input, identical output.

    A stray random transform here would make test metrics irreproducible and
    quietly inflate or deflate them from run to run.
    """
    image, mask = reference_pair()
    transform = get_eval_transform()

    first = to_hw(transform(image=image, mask=mask)["image"])
    for _ in range(5):
        assert np.allclose(to_hw(transform(image=image, mask=mask)["image"]), first)


def test_eval_transform_applies_no_geometry():
    """The eval pipeline must leave orientation exactly as it found it."""
    image, mask = reference_pair()
    out = get_eval_transform()(image=image, mask=mask)

    assert identify_element(to_hw(mask), to_hw(out["mask"]), binary=True) == 0


# ---------------------------------------------------------------------------
# Mask integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform_factory", [get_train_transform, get_eval_transform])
def test_mask_leaves_the_pipeline_strictly_binary(transform_factory):
    """Only 0.0 and 1.0 may survive.

    Interpolation and colour ops can leave intermediate values in a mask, and
    both the losses and the metrics assume a clean binary target -- an
    in-between value is silently miscounted rather than rejected.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    mask = (rng.random((SIZE, SIZE)) > 0.5).astype(np.uint8) * 255

    out = to_hw(transform_factory()(image=image, mask=mask)["mask"])
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}, f"non-binary values: {np.unique(out)}"


def test_mask_is_never_normalized_like_an_image():
    """Normalize applies to the image only; a normalized mask would go negative.

    Sharing a normalization step between the two is an easy mistake and would
    turn every target into ImageNet-standardised noise.
    """
    image, mask = reference_pair()
    out = to_hw(get_train_transform()(image=image, mask=mask)["mask"])
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_binarize_threshold_matches_the_rest_of_the_pipeline():
    """_binarize_mask cuts at >127, the value audit and measure_sampling also use.

    Crops are resized with interpolation, so masks genuinely carry intermediate
    values here; a different threshold in one place would silently shift the
    effective foreground fraction.
    """
    values = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    assert _binarize_mask(values).tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert _binarize_mask(values).dtype == np.float32


def test_coarse_dropout_does_not_punch_holes_in_the_mask():
    """Occlusion belongs on the image only.

    If CoarseDropout also filled the mask it would erase labels the image still
    shows, teaching the model that visible water is background. An all-
    foreground mask must keep every pixel across many draws -- geometry cannot
    change the count, so any variation means the mask was occluded.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    mask = np.full((SIZE, SIZE), 255, dtype=np.uint8)

    counts = set()
    for seed in range(30):
        seeded(seed)
        out = to_hw(get_train_transform()(image=image, mask=mask)["mask"])
        counts.add(int((out > 0.5).sum()))

    assert counts == {SIZE * SIZE}, f"mask foreground count varied: {sorted(counts)}"


# ---------------------------------------------------------------------------
# Shape and dtype contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform_factory", [get_train_transform, get_eval_transform])
def test_output_shapes_and_dtypes(transform_factory):
    """ToTensorV2 must give a CHW float image and an HW mask, sizes preserved."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    mask = (rng.random((SIZE, SIZE)) > 0.5).astype(np.uint8) * 255

    out = transform_factory()(image=image, mask=mask)
    assert tuple(out["image"].shape) == (3, SIZE, SIZE)
    assert tuple(out["mask"].shape) == (SIZE, SIZE)
    assert out["image"].dtype.is_floating_point


def test_non_square_input_is_preserved():
    """Neither pipeline resizes; the sampler already fixed the size."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (48, 80, 3), dtype=np.uint8)
    mask = (rng.random((48, 80)) > 0.5).astype(np.uint8) * 255

    out = get_eval_transform()(image=image, mask=mask)
    assert tuple(out["image"].shape) == (3, 48, 80)
    assert tuple(out["mask"].shape) == (48, 80)


def test_to_hw_rejects_a_whole_result_dict():
    """Guards the helper itself: numpy would wrap a dict in a 0-d object array."""
    with pytest.raises(TypeError, match="whole result dict"):
        to_hw({"image": np.zeros((3, 4, 4)), "mask": np.zeros((4, 4))})
