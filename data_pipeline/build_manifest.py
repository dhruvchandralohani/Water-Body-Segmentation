"""
build_manifest.py

Regenerates the filename/width/height manifest by scanning the current
image directory directly, so it always reflects what's actually on disk
(e.g. after removing outlier images) rather than trusting a possibly
stale CSV.

Usage:
    python build_manifest.py --image-dir images/ --output manifest.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def build_manifest(image_dir):
    """Build a manifest of image filenames and dimensions from a directory.

    Args:
        image_dir: Directory containing image files to inspect.

    Returns:
        A pandas DataFrame with filename, width, and height columns.
    """
    image_dir = Path(image_dir)
    rows = []
    for file_path in sorted(image_dir.iterdir()):
        if file_path.suffix.lower() in (".jpg", ".jpeg"):
            with Image.open(file_path) as img:
                width, height = img.size
            rows.append({"filename": file_path.name, "width": width, "height": height})
    return pd.DataFrame(rows)


def main():
    """Run the manifest-building workflow from the command line."""
    parser = argparse.ArgumentParser(description="Rebuild the manifest CSV from an image directory.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.is_dir() or output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "manifest.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_manifest(args.image_dir)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
