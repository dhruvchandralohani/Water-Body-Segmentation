"""
measure_sampling.py

Measure what fg_bias_ratio actually does to the training distribution.

fg_bias_ratio is the probability that a patch's crop CENTER lands on a
foreground pixel. It is not a pixel-level balance target: a 256x256 window
centred on one water pixel can still be almost entirely land. So the knob
sets a center-sampling probability and the quantity it is meant to control
-- how much water the model actually sees per batch -- sits downstream of it
and has to be measured.

This samples patches at a range of ratios and reports the resulting water
fraction, so the setting can be justified against a number rather than an
assumption. No GPU and no training: it drives PatchDataset directly with
transform=None, so what it measures is the sampler, not the augmentation.

Two things to read off the output:

    mean_water_fraction   the effective class balance the model trains on.
                          Compare against the dataset-level figure from
                          metrics/class_balance.json.
    empty_patch_fraction  patches with no water at all. These contribute
                          almost nothing beyond background statistics, and
                          suppressing them is the honest argument for
                          foreground bias -- gradient signal per step, not
                          global class balance.

Usage:
    python -m data_pipeline.measure_sampling \
        --manifest data_pipeline/splits/train.csv \
        --image-dir data/raw/images --mask-dir data/processed/masks_corrected \
        --class-balance-json metrics/class_balance.json \
        --output metrics/sampling_balance.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data_pipeline.patch_dataset import PatchDataset

# Matches _binarize_mask in transforms.py. Crops are resized with
# interpolation, so a mask arrives here with intermediate values, not just
# 0 and 255 -- the threshold has to be the same one training uses or the
# measured fraction is not the fraction the model sees.
MASK_THRESHOLD = 127


def measure_ratio(
    manifest_df,
    image_dir,
    mask_dir,
    ratio,
    n_patches,
    patch_size=256,
    window_range=(64, 2048),
    seed=42,
    num_workers=4,
    batch_size=32,
):
    """Sample patches at one fg_bias_ratio and return their water fractions.

    Args:
        manifest_df: Manifest DataFrame of images to sample from.
        image_dir: Directory containing source images.
        mask_dir: Directory containing corrected masks.
        ratio: fg_bias_ratio value to measure.
        n_patches: Number of patches to draw.
        patch_size: Output patch size.
        window_range: Minimum and maximum crop window size.
        seed: Seed for the sampler. Held constant across ratios so the
            comparison isolates the ratio rather than the draw.
        num_workers: DataLoader worker count.
        batch_size: DataLoader batch size.

    Returns:
        A numpy array of per-patch foreground fractions.
    """
    dataset = PatchDataset(
        manifest=manifest_df,
        image_dir=image_dir,
        mask_dir=mask_dir,
        window_range=window_range,
        patch_size=patch_size,
        fg_bias_ratio=ratio,
        num_patches_per_epoch=n_patches,
        transform=None,  # measure the sampler, not the augmentation
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    fractions = []
    for _, masks in loader:
        arr = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
        arr = arr.reshape(arr.shape[0], -1)
        threshold = 0.5 if arr.max() <= 1 else MASK_THRESHOLD
        fractions.extend(((arr > threshold).sum(axis=1) / arr.shape[1]).tolist())

    return np.asarray(fractions, dtype=float)


def summarize(fractions):
    """Summarize a set of per-patch foreground fractions.

    Args:
        fractions: Array of per-patch foreground fractions.

    Returns:
        A dictionary of summary statistics.
    """
    return {
        "n_patches": int(fractions.size),
        "mean_water_fraction": float(fractions.mean()),
        "median_water_fraction": float(np.median(fractions)),
        "p10": float(np.percentile(fractions, 10)),
        "p90": float(np.percentile(fractions, 90)),
        "empty_patch_fraction": float((fractions == 0.0).mean()),
        "under_1pct_fraction": float((fractions < 0.01).mean()),
    }


def main():
    """Measure effective foreground balance across fg_bias_ratio settings."""
    parser = argparse.ArgumentParser(
        description="Measure the water fraction of sampled training patches per fg_bias_ratio."
    )
    parser.add_argument("--manifest", required=True, help="Split CSV to sample from (normally train.csv)")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True, help="Corrected masks, as used in training")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
        help="fg_bias_ratio values to measure. 0.0 is uniform sampling.",
    )
    parser.add_argument("--n-patches", type=int, default=2000, help="Patches drawn per ratio")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--window-min", type=int, default=64)
    parser.add_argument("--window-max", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-balance-json",
        default=None,
        help="metrics/class_balance.json, to carry the dataset-level figure into the output "
        "so the comparison is self-contained.",
    )
    parser.add_argument("--output", default=None, help="Where to write the JSON summary")
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)

    dataset_fraction = None
    if args.class_balance_json:
        with open(args.class_balance_json) as f:
            dataset_fraction = json.load(f)["overall"]["foreground_fraction"]

    results = {}
    for ratio in args.ratios:
        fractions = measure_ratio(
            manifest_df,
            args.image_dir,
            args.mask_dir,
            ratio,
            n_patches=args.n_patches,
            patch_size=args.patch_size,
            window_range=(args.window_min, args.window_max),
            seed=args.seed,
            num_workers=args.num_workers,
        )
        results[f"{ratio:g}"] = summarize(fractions)
        stats = results[f"{ratio:g}"]
        print(
            f"  fg_bias_ratio={ratio:<5g} mean={stats['mean_water_fraction']:.4f}  "
            f"median={stats['median_water_fraction']:.4f}  "
            f"empty={stats['empty_patch_fraction']:.1%}  under1%={stats['under_1pct_fraction']:.1%}"
        )

    if dataset_fraction is not None:
        print(f"\n  dataset pixel-level water fraction (uniform reference): {dataset_fraction:.4f}")

    payload = {
        "n_patches_per_ratio": args.n_patches,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "dataset_pixel_water_fraction": dataset_fraction,
        "by_ratio": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
