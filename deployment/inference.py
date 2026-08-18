"""
inference.py

Core inference logic: load an exported model (see export_model.py), tile
a new image (no ground-truth mask needed, unlike TileDataset), run
batched inference across tiles, stitch predictions back into a
full-resolution mask.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mlflow.pytorch import load_model as load_pytorch_model

from data_pipeline.stitch import PredictionStitcher
from data_pipeline.tile_dataset import compute_tile_positions
from data_pipeline.transforms import get_eval_transform


def as_model_uri(model_path):
    """Render a local model directory as something MLflow will accept.

    MLflow parses its argument as a URI and dispatches on the scheme. On Windows
    an absolute path like C:\\models\\exported reads "c" as the scheme, which is
    not a registered artifact repository, and the loader fails with a message
    about repository registration rather than about the path. A relative path
    has an empty scheme and works, which is why this only bites outside the
    container -- inside it the path is /app/exported_model.

    Args:
        model_path: Local filesystem path to the exported model directory.

    Returns:
        A file:// URI for absolute paths, or the path unchanged if relative.
    """
    path = Path(model_path)
    return path.resolve().as_uri() if path.is_absolute() else str(path)


def load_model(model_path, device):
    """Load a PyTorch model from an exported local model bundle.

    Args:
        model_path: Path to the exported model directory.
        device: Target torch device, such as CPU or CUDA.

    Returns:
        The loaded model moved to the requested device and switched to eval mode.
    """
    model = load_pytorch_model(as_model_uri(model_path), map_location=device)
    model.to(device)
    model.eval()
    return model


def predict_image(
    model,
    image_path,
    device,
    tile_size=384,
    overlap=0.25,
    patch_size=256,
    threshold=0.5,
    batch_size=8,
    use_fp16=False,
):
    """Run tiled inference over a single image and stitch the predictions back.

    Args:
        model: Trained segmentation model used for inference.
        image_path: Path to the input image file.
        device: Target torch device for model execution.
        tile_size: Size of each inference tile.
        overlap: Fractional overlap between adjacent tiles.
        patch_size: Size used to resize each tile before inference.
        threshold: Threshold applied to stitched probabilities to create a mask.
        batch_size: Number of tiles processed per batch.
        use_fp16: Whether to enable half-precision inference on CUDA.

    Returns:
        A tuple containing the stitched binary mask and a metrics dictionary.
    """

    timings = {}

    t0 = time.time()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    timings["load"] = time.time() - t0

    t0 = time.time()
    positions = compute_tile_positions(w, h, tile_size, overlap)
    transform = get_eval_transform()

    tiles = []
    dummy_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    for x0, y0, eff_tile in positions:
        crop = image[y0 : y0 + eff_tile, x0 : x0 + eff_tile]
        resized = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        out = transform(image=resized, mask=dummy_mask)
        tiles.append(out["image"])
    timings["tile"] = time.time() - t0

    t0 = time.time()
    stitcher = PredictionStitcher()
    key = str(image_path)

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack(tiles[i : i + batch_size]).to(device)

            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch)
            else:
                logits = model(batch)

            if logits.dim() == 4 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            preds = torch.sigmoid(logits).float().cpu().numpy()

            for j, (x0, y0, eff_tile) in enumerate(positions[i : i + batch_size]):
                stitcher.add_tile(key, x0, y0, eff_tile, w, h, preds[j])
    timings["inference"] = time.time() - t0

    t0 = time.time()
    mask = stitcher.get_result(key, threshold=threshold)
    timings["stitch"] = time.time() - t0
    timings["total"] = sum(timings.values())

    info = {
        "width": w,
        "height": h,
        "num_tiles": len(positions),
        "water_coverage_pct": float(mask.mean() * 100),
        "timings": timings,
    }
    return mask, info
