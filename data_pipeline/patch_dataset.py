"""
patch_dataset.py

PatchDataset: the training-time Dataset. For every __getitem__ call it
picks a source image (weighted by area), draws a log-uniform window size
capped to that image's own dimensions, picks a foreground-biased crop
center, extracts and resizes the crop, then hands it to the transform
pipeline (augmentation + normalization).
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Training-time dataset that samples image patches with foreground bias."""

    def __init__(
        self,
        manifest,
        image_dir,
        mask_dir,
        window_range=(64, 2048),
        patch_size=256,
        fg_bias_ratio=0.7,
        num_patches_per_epoch=8000,
        transform=None,
        seed=None,
        fg_downsample=8,
        fg_cache_size=64,
    ):
        """Initialize the dataset from a manifest and image/mask directories.

        Args:
            manifest: DataFrame or path to a manifest CSV describing available images.
            image_dir: Directory containing source images.
            mask_dir: Directory containing segmentation masks.
            window_range: Minimum and maximum window sizes to sample.
            patch_size: Size used for resized output patches.
            fg_bias_ratio: Probability of sampling a foreground-biased center.
            num_patches_per_epoch: Number of patches exposed by the dataset per epoch.
            transform: Optional image/mask transform pipeline.
            seed: Optional random seed for deterministic sampling.
            fg_downsample: Downsampling factor used when finding foreground candidates.
            fg_cache_size: Maximum number of cached foreground candidate arrays.
        """

        self.df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.window_range = window_range
        self.patch_size = patch_size
        self.fg_bias_ratio = fg_bias_ratio
        self.num_patches_per_epoch = num_patches_per_epoch
        self.transform = transform
        self.fg_downsample = fg_downsample
        self.fg_cache_size = fg_cache_size

        self.filenames = self.df["filename"].tolist()
        self.widths = self.df["width"].to_numpy()
        self.heights = self.df["height"].to_numpy()

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._fg_cache = {}

        self.weights = self._build_image_weights()

    def _build_image_weights(self):
        """Compute area-based sampling weights for the available images.

        Returns:
            A NumPy array of probabilities used to sample images proportionally to
            their area.
        """
        areas = self.widths.astype(np.float64) * self.heights.astype(np.float64)
        return areas / areas.sum()

    def _choose_image_index(self):
        """Select an image index according to the precomputed sampling weights."""
        return self._np_rng.choice(len(self.filenames), p=self.weights)

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

    def _sample_window_size(self, img_w, img_h):
        """Sample a log-uniform window size that fits within the image.

        Args:
            img_w: Width of the source image.
            img_h: Height of the source image.

        Returns:
            A window size that is within the configured range and image bounds.
        """
        max_window = min(img_w, img_h, self.window_range[1])
        min_window = min(self.window_range[0], max_window)
        if max_window <= 1:
            return max(max_window, 1)
        log_low, log_high = math.log(min_window), math.log(max_window)
        if log_high <= log_low:
            return min_window
        window = int(round(math.exp(self._rng.uniform(log_low, log_high))))
        return max(1, min(window, max_window))

    def _get_foreground_candidates(self, filename, mask):
        """Find foreground pixel coordinates for a mask using pooled downsampling.

        Args:
            filename: Name of the image/mask pair.
            mask: Grayscale mask array.

        Returns:
            A NumPy array of foreground candidate coordinates, or None if none exist.
        """

        if filename in self._fg_cache:
            return self._fg_cache[filename]

        factor = self.fg_downsample
        h, w = mask.shape[:2]
        h_crop, w_crop = (h // factor) * factor, (w // factor) * factor

        if h_crop == 0 or w_crop == 0:
            fg = np.argwhere(mask > 0)
            candidates = fg if len(fg) > 0 else None
        else:
            pooled = (
                mask[:h_crop, :w_crop]
                .reshape(h_crop // factor, factor, w_crop // factor, factor)
                .max(axis=(1, 3))
            )
            fg = np.argwhere(pooled > 0)
            candidates = fg * factor if len(fg) > 0 else None

        if len(self._fg_cache) >= self.fg_cache_size:
            self._fg_cache.pop(next(iter(self._fg_cache)))  # evict oldest
        self._fg_cache[filename] = candidates

        return candidates

    def _sample_center(self, filename, mask, window_size):
        """Sample a patch center, favoring foreground regions when configured.

        Args:
            filename: Name of the image/mask pair.
            mask: Grayscale mask array.
            window_size: Size of the sampling window.

        Returns:
            A tuple of pixel coordinates (cx, cy) for the patch center.
        """

        h, w = mask.shape[:2]
        half = window_size // 2
        x_min, x_max = half, max(half, w - half)
        y_min, y_max = half, max(half, h - half)

        if self._rng.random() < self.fg_bias_ratio:
            candidates = self._get_foreground_candidates(filename, mask)
            if candidates is not None:
                y, x = candidates[self._rng.randrange(len(candidates))]
                jitter = self.fg_downsample
                x = int(x) + self._rng.randint(-jitter, jitter)
                y = int(y) + self._rng.randint(-jitter, jitter)
                cx = min(max(x, x_min), x_max)
                cy = min(max(y, y_min), y_max)
                return cx, cy

        cx = self._rng.randint(x_min, x_max) if x_max > x_min else x_min
        cy = self._rng.randint(y_min, y_max) if y_max > y_min else y_min
        return cx, cy

    def _crop_and_resize(self, image, mask, cx, cy, window_size):
        """Crop a window around the center and resize it to the patch size.

        Args:
            image: RGB image array.
            mask: Grayscale mask array.
            cx: X coordinate of the patch center.
            cy: Y coordinate of the patch center.
            window_size: Size of the crop window.

        Returns:
            Resized image and mask arrays for the sampled patch.
        """
        half = window_size // 2
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + window_size, y0 + window_size

        image_crop = image[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        image_resized = cv2.resize(
            image_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST
        )
        return image_resized, mask_resized

    def __len__(self):
        """Return the number of patches this dataset exposes per epoch."""
        return self.num_patches_per_epoch

    def __getitem__(self, idx):
        """Return a single sampled patch and its target mask.

        Args:
            idx: Index of the requested sample.

        Returns:
            A tuple of image and mask tensors or arrays after optional transforms.

        Raises:
            RuntimeError: If patch sampling fails for the selected image.
        """
        img_idx = self._choose_image_index()
        filename = self.filenames[img_idx]
        try:
            image, mask = self._load_pair(filename)

            h, w = image.shape[:2]
            window_size = self._sample_window_size(w, h)
            cx, cy = self._sample_center(filename, mask, window_size)
            image_crop, mask_crop = self._crop_and_resize(image, mask, cx, cy, window_size)

            if self.transform is not None:
                out = self.transform(image=image_crop, mask=mask_crop)
                return out["image"], out["mask"]

            return image_crop, mask_crop
        except Exception as e:
            raise RuntimeError(f"PatchDataset.__getitem__ failed on file '{filename}': {e}") from e
