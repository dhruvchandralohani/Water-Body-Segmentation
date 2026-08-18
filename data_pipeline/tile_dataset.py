"""
tile_dataset.py

TileDataset: the val/test-time Dataset. Produces deterministic, non-random
tiles that fully cover every image (with overlap), so evaluation is
reproducible and every pixel gets scored. Each item also returns
tile position metadata so predictions can be stitched back into
full-image space afterward.

compute_tile_positions is a standalone function (not a method) so
inference.py can reuse the exact same tiling math for new images that
don't have a ground-truth mask -- TileDataset itself always expects one,
since it was built for evaluation against known masks.
"""

from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset


def compute_tile_positions(img_w, img_h, tile_size, overlap):
    """Compute tile positions that cover an image with the requested overlap.

    Args:
        img_w: Width of the source image.
        img_h: Height of the source image.
        tile_size: Desired tile edge length.
        overlap: Fractional overlap between adjacent tiles.

    Returns:
        A list of tuples of the form (x0, y0, eff_tile) describing each tile.
    """
    eff_tile = min(tile_size, img_w, img_h)
    stride = max(1, int(round(eff_tile * (1 - overlap))))

    xs = list(range(0, max(img_w - eff_tile, 0) + 1, stride))
    if not xs or xs[-1] != img_w - eff_tile:
        xs.append(max(img_w - eff_tile, 0))

    ys = list(range(0, max(img_h - eff_tile, 0) + 1, stride))
    if not ys or ys[-1] != img_h - eff_tile:
        ys.append(max(img_h - eff_tile, 0))

    return [(x, y, eff_tile) for y in ys for x in xs]


class TileDataset(Dataset):
    """Evaluation-time dataset that yields tiles covering every image deterministically."""

    def __init__(
        self,
        manifest,
        image_dir,
        mask_dir,
        tile_size=384,
        patch_size=256,
        overlap=0.25,
        transform=None,
    ):
        """Initialize the dataset from a manifest and image/mask directories.

        Args:
            manifest: DataFrame or path to a manifest CSV describing available images.
            image_dir: Directory containing source images.
            mask_dir: Directory containing segmentation masks.
            tile_size: Desired size of each evaluation tile.
            patch_size: Size used for resizing each tile before returning it.
            overlap: Fractional overlap between adjacent tiles.
            transform: Optional image/mask transform pipeline.
        """

        self.df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_size = tile_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform

        self.filenames = self.df["filename"].tolist()
        self.widths = self.df["width"].to_numpy()
        self.heights = self.df["height"].to_numpy()

        self.tile_index = self._build_tile_index()

    def _build_tile_index(self):
        """Build the internal index of image tiles for deterministic evaluation."""
        index = []
        # strict=True: widths and heights come from the same manifest, so a
        # length mismatch means the manifest is malformed -- better to raise
        # than to silently index only part of the evaluation set.
        for img_idx, (w, h) in enumerate(zip(self.widths, self.heights, strict=True)):
            for x0, y0, eff_tile in compute_tile_positions(int(w), int(h), self.tile_size, self.overlap):
                index.append((img_idx, x0, y0, eff_tile))
        return index

    def _load_pair(self, filename):
        """Load an image and its mask from disk.

        Args:
            filename: Name of the image/mask pair to load.

        Returns:
            A tuple containing the RGB image and grayscale mask.

        Raises:
            FileNotFoundError: If either file cannot be loaded.
        """
        image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image for {filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_dir / filename), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask for {filename}")

        return image, mask

    def __len__(self):
        """Return the total number of tiles produced by the dataset."""
        return len(self.tile_index)

    def __getitem__(self, idx):
        """Return a single evaluation tile, its mask, and position metadata.

        Args:
            idx: Index of the requested tile.

        Returns:
            A tuple of the resized image, resized mask, and metadata dictionary.

        Raises:
            RuntimeError: If tile loading or processing fails.
        """
        image_idx, x0, y0, eff_tile = self.tile_index[idx]
        filename = self.filenames[image_idx]
        try:
            image, mask = self._load_pair(filename)

            image_crop = image[y0 : y0 + eff_tile, x0 : x0 + eff_tile]
            mask_crop = mask[y0 : y0 + eff_tile, x0 : x0 + eff_tile]

            image_resized = cv2.resize(
                image_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR
            )
            mask_resized = cv2.resize(
                mask_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST
            )

            if self.transform is not None:
                out = self.transform(image=image_resized, mask=mask_resized)
                image_out, mask_out = out["image"], out["mask"]
            else:
                image_out, mask_out = image_resized, mask_resized
        except Exception as e:
            raise RuntimeError(f"TileDataset.__getitem__ failed on file '{filename}': {e}") from e

        meta = {
            "filename": filename,
            "x0": x0,
            "y0": y0,
            "tile_size": eff_tile,
            "orig_width": int(self.widths[image_idx]),
            "orig_height": int(self.heights[image_idx]),
        }
        return image_out, mask_out, meta
