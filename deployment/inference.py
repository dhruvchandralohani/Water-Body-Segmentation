"""
inference.py

Core inference logic: load an exported model (see export_model.py), tile
a new image (no ground-truth mask needed, unlike TileDataset), run
batched inference across tiles, stitch predictions back into a
full-resolution mask.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from data_pipeline.stitch import PredictionStitcher
from data_pipeline.tiling import compute_tile_positions
from data_pipeline.transforms import get_eval_transform_numpy


def as_model_uri(model_path):
    """Render a local model directory as something MLflow will accept.

    MLflow parses its argument as a URI and dispatches on the scheme. On Windows
    an absolute path like C:\\models\\exported reads "c" as the scheme, which is
    not a registered artifact repository, and the loader fails with a message
    about repository registration rather than about the path. A relative path
    has an empty scheme and works, which is why this only bites outside the
    container -- inside it the path is /app/exported_model.

    Relative paths are returned with forward slashes, NOT via str(Path(...)).
    On Windows that renders "exported_model/best_model" as
    "exported_model\\best_model", and MLflow's validate_path_is_safe rejects any
    path containing a backslash -- so normalising separators the obvious way
    breaks the default MODEL_PATH on the platform it was meant to help.

    Args:
        model_path: Local filesystem path to the exported model directory.

    Returns:
        A file:// URI for absolute paths, or a forward-slashed relative path.
    """
    path = Path(model_path)
    return path.resolve().as_uri() if path.is_absolute() else path.as_posix()


class OnnxBackend:
    """Wraps an ONNX Runtime session in the call signature a torch module has.

    Exists so predict_image does not branch on backend. The tiling, stitching
    and thresholding are identical either way, and a second code path through
    them would be a second place for a bug to live -- while also making the two
    backends no longer comparable, since a latency difference could then come
    from either the runtime or the code around it.

    Both backends speak numpy. That direction was chosen deliberately: making
    ONNX return torch tensors would have kept torch in the serving path, which
    is most of what an ONNX backend is for.
    """

    def __init__(self, session, input_name, output_name, patch_size):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.patch_size = patch_size

    def __call__(self, batch):
        """Run one batch of NCHW float32 and return NCHW logits.

        Args:
            batch: Input array, shape (N, 3, patch_size, patch_size).

        Returns:
            Logits as an ndarray.

        Raises:
            TypeError: If the graph returned a non-dense output.
        """
        result = self.session.run([self.output_name], {self.input_name: batch})[0]
        if not isinstance(result, np.ndarray):
            raise TypeError(f"expected a dense array from the graph, got {type(result).__name__}")
        return result


class TorchBackend:
    """Wraps a PyTorch module in the same numpy call contract as OnnxBackend.

    Torch adapts to numpy rather than the reverse, so predict_image imports no
    torch at all and the two backends are exercised through identical code.
    Everything torch-specific -- device placement, no_grad, autocast -- lives
    here.
    """

    def __init__(self, module, device, use_fp16=False):
        import torch

        self.module = module
        # Accepts a string so callers never need to construct a torch.device --
        # serve.py must not import torch at all for the ONNX backend to remove
        # it from the image.
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_fp16 = use_fp16

    def __call__(self, batch):
        """Run one batch of NCHW float32 and return NCHW logits as an ndarray.

        Args:
            batch: Input array, shape (N, 3, patch_size, patch_size).

        Returns:
            Logits as an ndarray.
        """
        # Imported here rather than at module scope: with an ONNX bundle this
        # class is never constructed, and torch need not be installed at all.
        import torch

        tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            if self.use_fp16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.module(tensor)
            else:
                logits = self.module(tensor)
        return logits.float().cpu().numpy()


def load_onnx_backend(model_path, device):
    """Build an ONNX Runtime session from an exported bundle.

    Args:
        model_path: Directory containing model.onnx and onnx_metadata.json.
        device: "cuda" or "cpu", or a torch device. Used only to pick an
            execution provider.

    Returns:
        An OnnxBackend.

    Raises:
        FileNotFoundError: If the graph or its metadata is missing.
    """
    import onnxruntime as ort

    model_dir = Path(model_path)
    graph = model_dir / "model.onnx"
    meta_path = model_dir / "onnx_metadata.json"

    if not graph.exists():
        raise FileNotFoundError(f"no ONNX graph at {graph}; run deployment.export_onnx first")
    if not meta_path.exists():
        # The metadata is not optional: patch_size is baked into the graph as a
        # static dimension, and without it there is nothing to validate against.
        raise FileNotFoundError(
            f"no {meta_path.name} beside {graph.name}. It records the patch_size the "
            "graph was traced at, which the caller has to match."
        )

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if getattr(device, "type", str(device)) == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(graph), providers=providers)

    return OnnxBackend(
        session, meta["input_name"], meta["output_name"], int(meta["patch_size"])
    )


def load_pytorch_backend(model_path, device, use_fp16=False):
    """Load an MLflow PyTorch bundle and wrap it in the numpy call contract.

    A named function rather than an inline branch, mirroring load_onnx_backend.
    Both selection paths then look the same, and either can be substituted in a
    test -- patching the lazily imported mlflow loader from outside does not
    work, because mlflow resolves its flavor modules lazily and re-binds the
    original past a monkeypatch.

    Args:
        model_path: Path to the exported model directory.
        device: "cuda" or "cpu", or a torch device.
        use_fp16: Autocast on CUDA.

    Returns:
        A TorchBackend.
    """
    # Imported here, not at module scope: this function is never called for an
    # ONNX bundle, so torch and mlflow stay optional for the serving image.
    from mlflow.pytorch import load_model as load_pytorch_model

    module = load_pytorch_model(as_model_uri(model_path), map_location=device)
    module.to(device)
    module.eval()
    return TorchBackend(module, device, use_fp16=use_fp16)


def default_device():
    """Pick a device without importing torch when it is not installed.

    serve.py used torch.cuda.is_available() for this, which made torch a hard
    dependency of the serving image regardless of backend. Each runtime is asked
    only if it is already present.

    Returns:
        "cuda" or "cpu".
    """
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except ImportError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


def load_model(model_path, device, backend="auto", use_fp16=False):
    """Load the served model, as PyTorch or ONNX.

    Args:
        model_path: Path to the exported model directory.
        device: "cuda" or "cpu", or a torch device.
        backend: "pytorch", "onnx", or "auto" to prefer an ONNX graph when the
            bundle contains one and fall back to PyTorch otherwise.
        use_fp16: Autocast on CUDA. Ignored by the ONNX backend, where precision
            is a property of the exported graph rather than a runtime option.

    Returns:
        Something callable on a batch, in eval mode, on the requested device.

    Raises:
        ValueError: If backend is not a recognised value.
    """
    if backend not in ("auto", "pytorch", "onnx"):
        raise ValueError(f"backend must be 'auto', 'pytorch' or 'onnx', got {backend!r}")

    # "auto" prefers ONNX but never fails because of it: a bundle without a
    # graph serves from PyTorch rather than refusing to start. Explicit "onnx"
    # does fail, because someone who asked for it should not silently get
    # something else.
    if backend == "onnx" or (backend == "auto" and (Path(model_path) / "model.onnx").exists()):
        return load_onnx_backend(model_path, device)

    return load_pytorch_backend(model_path, device, use_fp16=use_fp16)


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
    transform = get_eval_transform_numpy()

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

    for i in range(0, len(tiles), batch_size):
        batch = np.stack(tiles[i : i + batch_size])

        logits = model(batch)

        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        # Sigmoid in numpy on the negative magnitude: exp never sees a large
        # positive argument, so a saturated logit gives 0 or 1 rather than inf.
        preds = np.where(
            logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits))
        ).astype(np.float32)

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