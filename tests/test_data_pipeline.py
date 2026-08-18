"""
test_data_pipeline.py

The audit decides which pairs exist and which split each lands in. Everything
downstream inherits those decisions, and a defect here is invisible in the
training logs -- it just produces numbers computed on the wrong data.

Runs the real audit end to end on a synthetic dataset built in tmp_path.
No GPU, no torch, no real images.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import REPO_ROOT

from data_pipeline.audit import exclusion_reason
from data_pipeline.build_manifest import build_manifest


@pytest.fixture
def audit_result(synthetic_dataset):
    """Run the audit as a subprocess, exactly as the pipeline invokes it.

    A subprocess rather than a function call: the stage's real interface is its
    command line, so this covers argument wiring and the __main__ path too.
    """
    data = synthetic_dataset
    result = subprocess.run(
        [
            sys.executable, "-m", "data_pipeline.audit",
            "--manifest", str(data["manifest"]),
            "--image-dir", str(data["image_dir"]),
            "--mask-dir", str(data["mask_dir"]),
            "--corrected-mask-dir", str(data["corrected_mask_dir"]),
            "--output-dir", str(data["output_dir"]),
            "--class-balance-json", str(data["class_balance_json"]),
            "--n-buckets", "3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"audit failed:\n{result.stdout}\n{result.stderr}"

    report = pd.read_csv(data["output_dir"] / "mask_correction_report.csv")
    splits = {
        name: pd.read_csv(data["output_dir"] / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    return {"data": data, "report": report, "splits": splits, "stdout": result.stdout}


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def test_build_manifest_raises_when_it_finds_no_images(tmp_path):
    """An empty scan must fail here, not three stages later.

    Writing an empty CSV and exiting 0 pushes the failure downstream, where it
    surfaces as pandas reporting "No columns to parse from file" from the audit
    stage -- a message that points nowhere near the cause. The usual cause is
    images in a format build_manifest does not scan.
    """
    (tmp_path / "notes.txt").write_text("not an image")

    with pytest.raises(ValueError, match="no images with suffix"):
        build_manifest(tmp_path)


def test_build_manifest_ignores_formats_it_does_not_scan(tmp_path):
    """Only JPEGs count; anything else is skipped rather than half-read."""
    import cv2

    image = np.zeros((8, 8, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "keep.jpg"), image)
    cv2.imwrite(str(tmp_path / "ignore.png"), image)

    frame = build_manifest(tmp_path)
    assert list(frame["filename"]) == ["keep.jpg"]


def test_build_manifest_reports_a_missing_directory(tmp_path):
    """A typo'd path should say so rather than scan nothing."""
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------


def test_every_pair_gets_its_expected_verdict(audit_result):
    """Each fixture pair is kept or excluded exactly as intended."""
    report = audit_result["report"]
    excluded = set(report[report["excluded"]]["filename"])
    problems = []
    for filename, verdict in audit_result["data"]["expected"].items():
        actual = "excluded" if filename in excluded else "kept"
        if actual != verdict:
            reason = report.loc[report.filename == filename, "reason"]
            note = reason.iloc[0] if len(reason) else "-"
            problems.append(f"{filename}: expected {verdict}, got {actual} ({note})")
    assert not problems, "\n".join(problems)


def test_size_filter_boundary_is_inclusive(audit_result):
    """min_dim is `<=`: a 32px image goes, a 33px image stays.

    An off-by-one here shifts the dataset by a handful of pairs and changes
    every split, silently.
    """
    excluded = set(audit_result["report"].query("excluded")["filename"])
    assert "edge32.png" in excluded
    assert "edge33.png" not in excluded


def test_all_white_mask_over_cropped_image_is_caught(audit_result):
    """The defect no-data correction would otherwise hide.

    An all-white mask over a 40% black border is an annotation error: the black
    region cannot be water. Correction repairs that border and drops the
    measured coverage to ~0.6, so a rule applied after correction sees nothing
    unusual. This is why the all-foreground pass runs before correction.
    """
    report = audit_result["report"]
    row = report.loc[report.filename == "cropped_allwhite.png"].iloc[0]
    assert bool(row["excluded"])
    assert "all-foreground" in row["reason"]


def test_legitimate_open_water_tile_survives(audit_result):
    """A 98%-water tile with no no-data region is a real scene, not a defect.

    Guards the threshold from being lowered until the rule "does something":
    anything under ~0.98 starts deleting valid mid-lake imagery.
    """
    excluded = set(audit_result["report"].query("excluded")["filename"])
    assert "midlake.png" not in excluded


@pytest.mark.parametrize(
    "stats,expect_excluded",
    [
        ({"near_black_fraction": 0.85, "original_coverage": 0.30, "corrected_coverage": 0.10}, True),
        ({"near_black_fraction": 0.05, "original_coverage": 0.31, "corrected_coverage": 0.29}, False),
        ({"near_black_fraction": 0.00, "original_coverage": 0.98, "corrected_coverage": 0.98}, False),
    ],
    ids=["mostly-no-data", "ordinary-pair", "legit-open-water"],
)
def test_exclusion_reason_matrix(stats, expect_excluded):
    """exclusion_reason covers the post-correction cases and nothing more."""
    assert bool(exclusion_reason(stats)) is expect_excluded


def test_no_corrected_mask_is_written_for_an_excluded_pair(audit_result):
    """Excluded pairs must leave no orphan behind in the corrected-mask directory.

    Orphans are not merely untidy: the directory is a DVC output, so stale files
    change its hash and make the stage look modified when nothing did.
    """
    written = {p.name for p in audit_result["data"]["corrected_mask_dir"].iterdir()}
    excluded = set(audit_result["report"].query("excluded")["filename"])
    assert not (written & excluded), f"orphans: {sorted(written & excluded)}"

    kept = set(pd.concat(audit_result["splits"].values())["filename"])
    assert kept <= written, f"kept pairs missing a corrected mask: {sorted(kept - written)}"


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------


def test_splits_are_disjoint(audit_result):
    """No filename may appear in more than one split.

    Leakage does not fail loudly -- it inflates every reported metric and looks
    like a good result.
    """
    train, val, test = (set(audit_result["splits"][k]["filename"]) for k in ("train", "val", "test"))
    assert not (train & val), f"train/val overlap: {sorted(train & val)}"
    assert not (train & test), f"train/test overlap: {sorted(train & test)}"
    assert not (val & test), f"val/test overlap: {sorted(val & test)}"


def test_splits_cover_exactly_the_kept_pairs(audit_result):
    """The three splits partition the kept set: nothing lost, nothing invented."""
    report = audit_result["report"]
    kept = set(report[~report["excluded"]]["filename"])
    covered = set(pd.concat(audit_result["splits"].values())["filename"])
    assert covered == kept


def test_split_csvs_carry_only_manifest_columns(audit_result):
    """Helper columns used for stratification must not leak into the split files.

    The datasets read these CSVs; a stray coverage or bucket column would ride
    into training as an unintended input.
    """
    for name, frame in audit_result["splits"].items():
        assert set(frame.columns) <= {"filename", "width", "height"}, (
            f"{name}.csv has extra columns: {sorted(set(frame.columns))}"
        )


def test_split_is_deterministic_under_a_fixed_seed(synthetic_dataset, audit_result):
    """Re-running the audit with the same seed reproduces the same splits.

    `dvc.lock` claims the pipeline is reproducible; if the split wanders, that
    claim is false and no two runs are comparable.
    """
    data = synthetic_dataset
    second = data["root"] / "splits_rerun"
    result = subprocess.run(
        [
            sys.executable, "-m", "data_pipeline.audit",
            "--manifest", str(data["manifest"]),
            "--image-dir", str(data["image_dir"]),
            "--mask-dir", str(data["mask_dir"]),
            "--corrected-mask-dir", str(data["root"] / "masks_rerun"),
            "--output-dir", str(second),
            "--n-buckets", "3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for name in ("train", "val", "test"):
        first_names = list(audit_result["splits"][name]["filename"])
        rerun_names = list(pd.read_csv(second / f"{name}.csv")["filename"])
        assert first_names == rerun_names, f"{name} split changed between identical runs"


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------


def test_class_balance_reports_micro_and_macro_separately(audit_result):
    """Pixel-weighted and per-image foreground fractions are distinct statistics.

    Image area spans orders of magnitude, so averaging per-image coverage is not
    the dataset's pixel-level ratio. Both are reported because the gap is the
    same micro/macro distinction that separates global test IoU from per-image
    mean IoU.
    """
    with open(audit_result["data"]["class_balance_json"]) as f:
        balance = json.load(f)

    overall = balance["overall"]
    assert 0.0 < overall["foreground_fraction"] < 1.0
    assert overall["foreground_pixels"] < overall["total_pixels"]
    assert overall["background_per_foreground"] == pytest.approx(
        (1 - overall["foreground_fraction"]) / overall["foreground_fraction"], rel=1e-6
    )
    assert set(balance["per_image_foreground_fraction"]) >= {"mean", "median", "p05", "p95"}
    assert set(balance["by_split"]) == {"train", "val", "test"}


def test_class_balance_counts_only_kept_pairs(audit_result):
    """Excluded pairs must not contribute to the reported balance.

    They are largely all-white masks; counting them would inflate the water
    fraction with exactly the annotations the audit rejected.
    """
    with open(audit_result["data"]["class_balance_json"]) as f:
        balance = json.load(f)
    kept = (~audit_result["report"]["excluded"]).sum()
    assert balance["overall"]["n_images"] == kept

    by_split = sum(s["n_images"] for s in balance["by_split"].values())
    assert by_split == kept
