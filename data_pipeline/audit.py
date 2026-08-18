"""
audit.py

Dataset audit for the water body segmentation dataset. Four passes, ordered
cheapest-filter-first so that expensive work is never spent on pairs that are
about to be discarded:

1. Size filter -- drop pairs whose image is degenerate in either dimension.
   Runs off the manifest, before any file is opened.
2. All-foreground filter -- drop pairs whose mask is painted almost entirely
   white. Reads the mask only, and MUST run before pass 3: no-data correction
   repairs the black border of such a mask, dropping its measured coverage far
   below any sensible threshold and destroying the evidence of the defect.
   Order is load-bearing here, not stylistic.
3. Mask correction -- no-data (near-black) regions in the source image
   occasionally get mislabeled as foreground water. This zeroes out the mask
   wherever the image is no-data, then excludes pairs that are mostly no-data.
4. Stratified train/val/test split by image-size bucket.

The audit also quantifies pixel-level class balance over the surviving pairs
and writes it to JSON. That number is what justifies foreground-biased patch
sampling during training, so it is a pipeline artifact rather than something
recomputed by hand.

Usage:
    python audit.py --manifest filenames.csv --image-dir images/ --mask-dir masks/ \
        --corrected-mask-dir masks_corrected/ --output-dir splits/ \
        --class-balance-json metrics/class_balance.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_manifest(csv_path):
    """Load the dataset manifest from a CSV file.

    Args:
        csv_path: Path to the manifest CSV file.

    Returns:
        A pandas DataFrame containing the manifest rows.
    """
    return pd.read_csv(csv_path)


def filter_by_size(manifest_df, min_dim=32):
    """Drop pairs whose image is at or below min_dim pixels in either dimension.

    Applied to the manifest rather than to loaded images, so degenerate pairs
    are removed before any file I/O happens.

    Args:
        manifest_df: DataFrame containing filename, width, and height columns.
        min_dim: Images with min(width, height) <= this value are excluded.

    Returns:
        A tuple of the kept DataFrame and a list of exclusion record dicts.

    Raises:
        KeyError: If the manifest lacks width/height columns.
    """
    for col in ("width", "height"):
        if col not in manifest_df.columns:
            raise KeyError(f"Manifest is missing the '{col}' column; rebuild it with build_manifest.py")

    min_side = manifest_df[["width", "height"]].min(axis=1)
    too_small = min_side <= min_dim

    excluded_records = [
        {
            "filename": row["filename"],
            "excluded": True,
            "reason": f"image min dimension {int(min(row['width'], row['height']))}px <= {min_dim}px",
        }
        for _, row in manifest_df[too_small].iterrows()
    ]

    kept_df = manifest_df[~too_small].reset_index(drop=True)
    return kept_df, excluded_records


def filter_all_white_masks(manifest_df, mask_dir, all_white_thresh=0.995):
    """Drop pairs whose mask is degenerately all-foreground, before any correction.

    A mask painted white across an entire tile is an annotation defect: whatever
    the image shows, the annotator did not look. This has to run before no-data
    correction, because correction zeroes the mask over the black border and
    leaves coverage looking unremarkable -- a 40% no-data border turns a 1.00
    coverage into 0.60, which no threshold would flag.

    Reads only the mask, so it costs one grayscale decode per pair and spares
    the image read for everything it rejects.

    Args:
        manifest_df: DataFrame of pairs surviving the size filter.
        mask_dir: Directory containing the original, uncorrected masks.
        all_white_thresh: Foreground fraction at or above which the mask is
            treated as degenerate.

    Returns:
        A tuple of the kept DataFrame and a list of exclusion record dicts.
    """
    mask_dir = Path(mask_dir)
    excluded_records = []
    keep = []

    for _, row in manifest_df.iterrows():
        filename = row["filename"]
        mask = cv2.imread(str(mask_dir / filename), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            excluded_records.append(
                {"filename": filename, "excluded": True, "reason": "missing mask file"}
            )
            continue

        coverage = float((mask > 127).sum()) / mask.size
        if coverage >= all_white_thresh:
            excluded_records.append(
                {
                    "filename": filename,
                    "excluded": True,
                    "reason": "mask is all-foreground",
                    "original_coverage": coverage,
                }
            )
        else:
            keep.append(filename)

    kept_df = manifest_df[manifest_df["filename"].isin(set(keep))].reset_index(drop=True)
    return kept_df, excluded_records


def analyze_and_correct_mask(image, mask, near_black_threshold=15):
    """Identify near-black regions and correct the mask accordingly.

    Args:
        image: Source image array used to detect no-data regions.
        mask: Original segmentation mask array.
        near_black_threshold: Intensity threshold below which pixels are treated
            as no-data.

    Returns:
        A tuple containing the corrected mask and a statistics dictionary.
    """
    near_black = image.astype(np.int32).sum(axis=2) < near_black_threshold
    original_fg = mask > 127

    corrected_mask = mask.copy()
    corrected_mask[near_black] = 0
    corrected_fg = corrected_mask > 127

    total_pixels = mask.size
    stats = {
        "near_black_fraction": float(near_black.mean()),
        "original_coverage": float(original_fg.sum()) / total_pixels,
        "corrected_coverage": float(corrected_fg.sum()) / total_pixels,
        # Raw counts, not just fractions: the dataset-wide class ratio is
        # pixel-weighted, and image sizes here span three orders of magnitude,
        # so it cannot be recovered by averaging the per-image fractions.
        "total_pixels": int(total_pixels),
        "corrected_fg_pixels": int(corrected_fg.sum()),
    }
    return corrected_mask, stats


def exclusion_reason(stats, near_black_exclude_thresh=0.7):
    """Determine why a pair should be excluded after correction, if it should be.

    All-foreground masks are already gone by this point -- filter_all_white_masks
    removes them before correction runs, precisely so that this function never
    has to reason about a mask that correction has already altered.

    Args:
        stats: Dictionary of computed mask statistics for the image.
        near_black_exclude_thresh: No-data fraction above which the pair is
            excluded.

    Returns:
        A reason string, or an empty string if the pair should be kept.
    """
    if stats["near_black_fraction"] > near_black_exclude_thresh:
        return "mostly no-data image"
    return ""


def correct_and_filter(
    manifest_df,
    image_dir,
    mask_dir,
    corrected_mask_dir,
    near_black_threshold=15,
    near_black_exclude_thresh=0.7,
):
    """Correct masks for no-data pixels and filter out degenerate pairs.

    Args:
        manifest_df: DataFrame containing the dataset manifest.
        image_dir: Directory containing source images.
        mask_dir: Directory containing original masks.
        corrected_mask_dir: Directory where corrected masks are written.
        near_black_threshold: Intensity threshold used for no-data detection.
        near_black_exclude_thresh: No-data fraction threshold for exclusion.

    Returns:
        A tuple of the kept manifest DataFrame and a per-image report DataFrame.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    corrected_mask_dir = Path(corrected_mask_dir)
    corrected_mask_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in manifest_df.iterrows():
        filename = row["filename"]
        image = cv2.imread(str(image_dir / filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / filename), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            records.append({"filename": filename, "excluded": True, "reason": "missing file"})
            continue

        corrected_mask, stats = analyze_and_correct_mask(image, mask, near_black_threshold)
        reason = exclusion_reason(stats, near_black_exclude_thresh)

        # Only write masks that survive; an excluded pair has no downstream use
        # and writing it would leave orphans in the corrected-mask directory.
        if not reason:
            cv2.imwrite(str(corrected_mask_dir / filename), corrected_mask)

        records.append(
            {
                "filename": filename,
                "excluded": bool(reason),
                "reason": reason,
                **stats,
            }
        )

    report_df = pd.DataFrame(records)
    kept_df = manifest_df.merge(report_df[["filename", "excluded"]], on="filename")
    kept_df = kept_df[~kept_df["excluded"]].drop(columns=["excluded"]).reset_index(drop=True)

    return kept_df, report_df


def summarize_class_balance(report_df, split_members=None):
    """Quantify pixel-level water-vs-background balance over the kept pairs.

    Reports both the pixel-weighted (micro) ratio and the per-image (macro)
    distribution. They differ materially here because image area spans three
    orders of magnitude -- the same micro/macro distinction that separates
    global test IoU from per-image mean IoU.

    Args:
        report_df: Per-image report containing pixel counts and exclusion flags.
        split_members: Optional mapping of split name to an iterable of
            filenames, used to report per-split balance.

    Returns:
        A dictionary of class balance statistics.
    """
    if "total_pixels" not in report_df.columns:
        return {}

    kept = report_df[(~report_df["excluded"]) & report_df["total_pixels"].notna()]
    if kept.empty:
        return {}

    def _block(df):
        """Compute the pixel-weighted foreground summary for a subset."""
        total_px = float(df["total_pixels"].sum())
        fg_px = float(df["corrected_fg_pixels"].sum())
        fg_frac = fg_px / total_px if total_px else 0.0
        return {
            "n_images": int(len(df)),
            "total_pixels": int(total_px),
            "foreground_pixels": int(fg_px),
            "foreground_fraction": fg_frac,
            "background_per_foreground": (1.0 - fg_frac) / fg_frac if fg_frac else None,
        }

    coverage = kept["corrected_coverage"].astype(float)
    summary = {
        "overall": _block(kept),
        "per_image_foreground_fraction": {
            "mean": float(coverage.mean()),
            "median": float(coverage.median()),
            "p05": float(coverage.quantile(0.05)),
            "p25": float(coverage.quantile(0.25)),
            "p75": float(coverage.quantile(0.75)),
            "p95": float(coverage.quantile(0.95)),
        },
        "images_under_1pct_water": int((coverage < 0.01).sum()),
        "images_over_50pct_water": int((coverage > 0.50).sum()),
    }

    if split_members:
        summary["by_split"] = {
            name: _block(kept[kept["filename"].isin(set(files))])
            for name, files in split_members.items()
            if len(kept[kept["filename"].isin(set(files))]) > 0
        }

    return summary


def assign_size_bucket(df, n_buckets=5):
    """Assign each sample to a quantile-based size bucket.

    Args:
        df: DataFrame containing width and height columns.
        n_buckets: Number of size buckets to create.

    Returns:
        A copy of the DataFrame with a size_bucket column added.
    """
    df = df.copy()
    min_dim = df[["width", "height"]].min(axis=1)
    df["size_bucket"] = pd.qcut(min_dim, q=n_buckets, labels=False, duplicates="drop")
    return df


def split_dataset(df, val_frac=0.15, test_frac=0.15, stratify_col="size_bucket", random_state=42):
    """Split a dataset into train, validation, and test subsets.

    Args:
        df: DataFrame to split.
        val_frac: Fraction of rows assigned to validation.
        test_frac: Fraction of rows assigned to test.
        stratify_col: Column name used for stratified splitting, if present.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple of train, validation, and test DataFrames.

    Raises:
        ValueError: If the requested validation and test fractions exceed 1.0.
    """
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    strat = df[stratify_col] if stratify_col in df.columns else None
    train_df, temp_df = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=strat, random_state=random_state
    )
    assert isinstance(train_df, pd.DataFrame) and isinstance(temp_df, pd.DataFrame)

    strat_temp = temp_df[stratify_col] if stratify_col in temp_df.columns else None
    relative_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=strat_temp, random_state=random_state
    )
    assert isinstance(val_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df, val_df, test_df, output_dir):
    """Write split CSV files to disk.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        output_dir: Directory where split CSV files will be written.

    Returns:
        A tuple of paths to the written train, validation, and test CSVs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    return output_dir / "train.csv", output_dir / "val.csv", output_dir / "test.csv"


def main():
    """Run the dataset audit, mask correction, and splitting workflow from the CLI."""
    parser = argparse.ArgumentParser(description="Audit, correct, and split the water body dataset.")
    parser.add_argument("--manifest", required=True, help="Path to filename,width,height CSV")
    parser.add_argument("--image-dir", help="Directory of source images (required unless --skip-correction)")
    parser.add_argument("--mask-dir", help="Directory of original masks (required unless --skip-correction)")
    parser.add_argument(
        "--corrected-mask-dir", help="Where to write corrected masks (required unless --skip-correction)"
    )
    parser.add_argument(
        "--skip-correction",
        action="store_true",
        help="Skip the no-data mask correction pass and just size-filter and split the manifest.",
    )
    parser.add_argument("--output-dir", required=True, help="Where to write train/val/test CSVs and the report")
    parser.add_argument(
        "--class-balance-json",
        default=None,
        help="Where to write pixel-level class balance stats. Requires the correction pass.",
    )
    parser.add_argument(
        "--min-dim",
        type=int,
        default=32,
        help="Exclude pairs whose image is this many pixels or fewer in either dimension.",
    )
    parser.add_argument(
        "--all-white-thresh",
        type=float,
        default=0.995,
        help="Exclude pairs whose ORIGINAL mask is at least this fraction foreground. "
        "Checked before no-data correction, which would otherwise mask the defect.",
    )
    parser.add_argument("--near-black-threshold", type=int, default=15)
    parser.add_argument("--near-black-exclude-thresh", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--n-buckets", type=int, default=5, help="Quantile buckets over min(width, height).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest)
    n_start = len(df)

    # Pass 1: size filter, off the manifest, before any file is opened.
    kept_df, size_excluded = filter_by_size(df, min_dim=args.min_dim)
    print(f"Size filter: {len(size_excluded)} pairs excluded at min dimension <= {args.min_dim}px")

    # Explicit columns: with no size exclusions the list is empty, and a
    # column-less DataFrame would break the reason tally further down.
    report_df = pd.DataFrame(size_excluded, columns=["filename", "excluded", "reason"])  # type: ignore[call-overload]
    report_df["excluded"] = report_df["excluded"].astype(bool)

    # Pass 2: all-foreground masks, before correction can disguise them.
    if args.mask_dir:
        kept_df, all_white_excluded = filter_all_white_masks(
            kept_df, args.mask_dir, all_white_thresh=args.all_white_thresh
        )
        print(f"All-foreground filter: {len(all_white_excluded)} pairs excluded at coverage >= {args.all_white_thresh}")
        if all_white_excluded:
            report_df = pd.concat([report_df, pd.DataFrame(all_white_excluded)], ignore_index=True)

    if args.skip_correction:
        if args.class_balance_json:
            parser.error("--class-balance-json requires the mask correction pass")
    else:
        missing = [
            name
            for name, val in [
                ("--image-dir", args.image_dir),
                ("--mask-dir", args.mask_dir),
                ("--corrected-mask-dir", args.corrected_mask_dir),
            ]
            if not val
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required unless --skip-correction is set")

        # Pass 3: mask correction and no-data exclusion.
        kept_df, correction_report = correct_and_filter(
            kept_df,
            args.image_dir,
            args.mask_dir,
            args.corrected_mask_dir,
            near_black_threshold=args.near_black_threshold,
            near_black_exclude_thresh=args.near_black_exclude_thresh,
        )
        report_df = pd.concat([report_df, correction_report], ignore_index=True)

        n_corrected = int(
            (
                (correction_report["corrected_coverage"] < correction_report["original_coverage"])
                & ~correction_report["excluded"]
            ).sum()
        )
        print(f"Mask correction: {n_corrected} masks had no-data regions zeroed")

    report_df.to_csv(output_dir / "mask_correction_report.csv", index=False)

    excluded: pd.DataFrame = report_df[report_df["excluded"]]
    print(f"\nAudit: {n_start} pairs in, {len(kept_df)} kept, {len(excluded)} excluded")
    for reason, count in excluded.loc[:, "reason"].value_counts().items():
        # Size reasons embed the actual dimension, so collapse them for the tally.
        print(f"  {count:>5}  {reason}")

    # Pass 4: stratified split.
    kept_df = assign_size_bucket(kept_df, n_buckets=args.n_buckets)
    train_df, val_df, test_df = split_dataset(
        kept_df, val_frac=args.val_frac, test_frac=args.test_frac, random_state=args.seed
    )
    # size_bucket exists only to stratify the split. Writing it into the split
    # CSVs leaks an artefact of the splitting procedure into the manifests the
    # datasets consume -- harmless today because they read filename/width/height
    # and ignore the rest, but it is unintended output either way.
    helper_cols = [c for c in ("corrected_coverage", "size_bucket", "coverage_bucket", "stratum")
                   if c in train_df.columns]
    train_df, val_df, test_df = (d.drop(columns=helper_cols) for d in (train_df, val_df, test_df))
    save_splits(train_df, val_df, test_df, args.output_dir)
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    if args.class_balance_json:
        balance = summarize_class_balance(
            report_df,
            split_members={
                "train": train_df["filename"],
                "val": val_df["filename"],
                "test": test_df["filename"],
            },
        )
        balance_path = Path(args.class_balance_json)
        balance_path.parent.mkdir(parents=True, exist_ok=True)
        with open(balance_path, "w") as f:
            json.dump(balance, f, indent=2)

        overall = balance["overall"]
        per_image = balance["per_image_foreground_fraction"]
        print(
            f"\nClass balance (corrected masks, kept pairs only):\n"
            f"  pixel-weighted water fraction : {overall['foreground_fraction']:.4f} "
            f"(1 water pixel per {overall['background_per_foreground']:.1f} background)\n"
            f"  per-image mean water fraction : {per_image['mean']:.4f} "
            f"(median {per_image['median']:.4f})\n"
            f"  images under 1% water         : {balance['images_under_1pct_water']}\n"
            f"  written to {balance_path}"
        )


if __name__ == "__main__":
    main()
