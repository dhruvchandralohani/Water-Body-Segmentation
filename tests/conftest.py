"""
conftest.py

Shared fixtures.

Nothing here touches the real dataset or a GPU. CI has neither, and tests that
depend on either are the ones that rot: they go slow, then flaky, then ignored.
Every fixture builds what it needs from scratch in a tmp_path.
"""

import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root():
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def params():
    """params.yaml, parsed."""
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def dvc_pipeline():
    """dvc.yaml, parsed."""
    with open(REPO_ROOT / "dvc.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def dvc_raw():
    """dvc.yaml as raw text, for interpolation checks that YAML parsing would hide."""
    return (REPO_ROOT / "dvc.yaml").read_text()


def declared_flags(module_path, _seen=None):
    """Collect every long flag an argparser declares, by reading the source.

    Parsed rather than imported on purpose: importing training.train pulls in
    torch and smp, which turns a config check into a slow, environment-dependent
    one. The flags are a static property of the file, so read them statically.

    Follows parser composition. tune.py declares only its own tuning flags and
    inherits the rest by calling train.py's build_argparser(), so reading a
    single file would report a parser far narrower than the one that actually
    runs.

    Args:
        module_path: Path to the Python file to inspect.
        _seen: Internal guard against import cycles.

    Returns:
        A set of flag strings including their leading dashes.
    """
    module_path = Path(module_path).resolve()
    _seen = _seen if _seen is not None else set()
    if module_path in _seen or not module_path.exists():
        return set()
    _seen.add(module_path)

    tree = ast.parse(module_path.read_text())
    flags = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("--"):
                        flags.add(arg.value)

        # `from training.train import build_argparser` -- inherit its flags.
        if isinstance(node, ast.ImportFrom) and node.module:
            inherits = any("argparser" in alias.name.lower() for alias in node.names)
            if inherits:
                source = REPO_ROOT / (node.module.replace(".", "/") + ".py")
                flags |= declared_flags(source, _seen)

    return flags


def make_pair(image_dir, mask_dir, name, image, mask, rows):
    """Write one image/mask pair to disk and record its manifest row."""
    cv2.imwrite(str(Path(image_dir) / name), image)
    cv2.imwrite(str(Path(mask_dir) / name), mask)
    rows.append({"filename": name, "width": image.shape[1], "height": image.shape[0]})


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Build a small dataset exercising every audit exclusion rule.

    Deliberately includes the cases that are easy to get wrong:

    cropped_allwhite  an all-white mask over an image with a 40% no-data border.
                      No-data correction repairs that border and drops the
                      measured coverage to ~0.6, so a rule checking the
                      CORRECTED mask misses it entirely.
    midlake           a legitimate 98%-water tile with no no-data region. Must
                      survive: it is a real open-water scene, not a defect.
    edge32 / edge33   the size-filter boundary, which is `<=` not `<`.

    Returns:
        A dict of paths plus the expected keep/exclude verdicts by filename.
    """
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.default_rng(0)
    rows = []
    expected = {}

    def rand_image(h, w):
        """Mid-brightness noise, well clear of the near-black threshold."""
        return rng.integers(60, 200, (h, w, 3), dtype=np.uint8)

    # --- size filter boundary: <= min_dim is excluded ---
    make_pair(image_dir, mask_dir, "edge32.png", rand_image(200, 32), np.zeros((200, 32), np.uint8), rows)
    expected["edge32.png"] = "excluded"
    make_pair(image_dir, mask_dir, "edge33.png", rand_image(200, 33), np.zeros((200, 33), np.uint8), rows)
    expected["edge33.png"] = "kept"
    make_pair(image_dir, mask_dir, "tiny.png", rand_image(300, 7), np.zeros((300, 7), np.uint8), rows)
    expected["tiny.png"] = "excluded"

    # --- all-white mask over a partially cropped image ---
    cropped = rand_image(200, 200)
    cropped[:80, :] = 0
    make_pair(image_dir, mask_dir, "cropped_allwhite.png", cropped, np.full((200, 200), 255, np.uint8), rows)
    expected["cropped_allwhite.png"] = "excluded"

    # --- degenerate all-white mask, no no-data at all ---
    make_pair(image_dir, mask_dir, "allwhite.png", rand_image(200, 200), np.full((200, 200), 255, np.uint8), rows)
    expected["allwhite.png"] = "excluded"

    # --- legitimate open water: 98% coverage, must be KEPT ---
    midlake = np.full((200, 200), 255, np.uint8)
    midlake.flat[: int(200 * 200 * 0.02)] = 0
    make_pair(image_dir, mask_dir, "midlake.png", rand_image(200, 200), midlake, rows)
    expected["midlake.png"] = "kept"

    # --- mostly no-data image ---
    nodata = np.zeros((200, 200, 3), np.uint8)
    nodata[:20, :] = 200
    partial = np.zeros((200, 200), np.uint8)
    partial.flat[: int(200 * 200 * 0.3)] = 255
    make_pair(image_dir, mask_dir, "nodata.png", nodata, partial, rows)
    expected["nodata.png"] = "excluded"

    # --- ordinary pairs, enough of them for a stratified split to be feasible ---
    for i in range(48):
        side = int(rng.choice([64, 96, 128, 192, 256]))
        coverage = float(rng.choice([0.02, 0.10, 0.25, 0.45]))
        mask = np.zeros((side, side), np.uint8)
        mask.flat[: int(side * side * coverage)] = 255
        name = f"ok{i:02d}.png"
        make_pair(image_dir, mask_dir, name, rand_image(side, side), mask, rows)
        expected[name] = "kept"

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    return {
        "root": tmp_path,
        "manifest": manifest,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "corrected_mask_dir": tmp_path / "masks_corrected",
        "output_dir": tmp_path / "splits",
        "class_balance_json": tmp_path / "metrics" / "class_balance.json",
        "expected": expected,
        "n_total": len(rows),
    }
