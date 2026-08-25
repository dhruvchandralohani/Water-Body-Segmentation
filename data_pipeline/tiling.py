"""
tiling.py

Tile geometry, separated from the dataset that uses it.

compute_tile_positions is pure arithmetic, but it lived in tile_dataset.py,
which imports torch.utils.data at module scope. That made torch a transitive
dependency of anything wanting tile positions -- including the serving path,
where the whole point of an ONNX backend is not to ship torch at all.
"""


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
