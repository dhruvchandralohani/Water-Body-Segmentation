"""
stitch.py

PredictionStitcher: reconstructs a full-resolution prediction from the
overlapping tile predictions TileDataset produces.

Overlapping tiles are blended with a Gaussian weight map centered on each
tile, rather than simple averaging or overwriting, so tile boundaries
don't show up as visible seams in the reconstructed mask.
"""

import cv2
import numpy as np


class PredictionStitcher:
    """Blend overlapping tile predictions into a full-resolution result."""

    def __init__(self, sigma_scale=0.125):
        """Initialize the stitcher with a Gaussian blending scale.

        Args:
            sigma_scale: Scaling factor used to derive the Gaussian sigma from the
                tile size.
        """
        self.sigma_scale = sigma_scale
        self._canvases = {}
        self._weight_cache = {}

    def _gaussian_weight(self, size):
        """Create a 2D Gaussian weight map for a square tile size.

        Args:
            size: Edge length of the square tile.

        Returns:
            A 2D NumPy array containing the Gaussian weights.
        """
        if size not in self._weight_cache:
            ax = np.arange(size) - (size - 1) / 2
            sigma = max(self.sigma_scale * size, 1e-3)
            gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
            self._weight_cache[size] = np.outer(gauss_1d, gauss_1d)
        return self._weight_cache[size]

    def add_tile(self, filename, x0, y0, tile_size, orig_w, orig_h, prediction):
        """Blend a prediction tile into the canvas for the given source image.

        Args:
            filename: Identifier for the source image being reconstructed.
            x0: Horizontal offset of the tile within the full image.
            y0: Vertical offset of the tile within the full image.
            tile_size: Edge length of the square tile.
            orig_w: Full width of the reconstructed image.
            orig_h: Full height of the reconstructed image.
            prediction: Tile prediction values to blend into the canvas.
        """
        if filename not in self._canvases:
            self._canvases[filename] = {
                "sum": np.zeros((orig_h, orig_w), dtype=np.float64),
                "weight": np.zeros((orig_h, orig_w), dtype=np.float64),
            }
        canvas = self._canvases[filename]

        pred_resized = cv2.resize(
            prediction.astype(np.float32), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR
        )
        weight = self._gaussian_weight(tile_size)

        canvas["sum"][y0 : y0 + tile_size, x0 : x0 + tile_size] += pred_resized * weight
        canvas["weight"][y0 : y0 + tile_size, x0 : x0 + tile_size] += weight

    def get_result(self, filename, threshold=None):
        """Return the blended reconstruction for a source image.

        Args:
            filename: Identifier of the image whose stitched result is requested.
            threshold: Optional threshold used to convert the result to a binary
                mask.

        Returns:
            The blended prediction values or a thresholded binary mask.
        """
        canvas = self._canvases[filename]
        result = canvas["sum"] / np.clip(canvas["weight"], 1e-8, None)
        if threshold is not None:
            return (result >= threshold).astype(np.uint8)
        return result

    def has(self, filename):
        """Check whether a canvas already exists for the given filename."""
        return filename in self._canvases

    def clear(self, filename):
        """Remove the stored canvas for the given filename."""
        del self._canvases[filename]
