"""
export_onnx.py

Convert the promoted PyTorch model to ONNX.

The conversion itself is one call. Everything else in this file exists because
a bad ONNX export does not fail -- it produces a graph that loads, runs faster
than the original, and returns subtly different numbers. So the export verifies
itself before writing anything, and records enough metadata for the loader to
reject a mismatched graph rather than serve one.

Two decisions worth stating:

DYNAMIC AXES. Only the batch dimension is dynamic. The tiler resizes every tile
to patch_size before the model sees it, so height and width are genuinely fixed
-- but the last batch of a tiled image is almost always partial, so batch is
not. Leaving height and width static keeps the graph simpler and lets ORT plan
allocations, at the cost of a graph that is wrong if patch_size ever changes.
That cost is paid down by writing patch_size into the model metadata and having
the loader check it.

TOLERANCE. Agreement is checked on the thresholded MASK, not only on raw logits.
Logits differing by 1e-5 either side of the decision boundary flip pixels, and a
max-absolute-difference check passes cheerfully while IoU moves. The mask check
is the one that corresponds to what is served.

Usage:
    python -m deployment.export_onnx --model-dir deployment/exported_model/best_model
"""

import argparse
import json
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
import torch

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logger  # noqa: E402
from deployment.inference import as_model_uri  # noqa: E402

logger = setup_logger("export_onnx", log_file="export_onnx.log")

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
ONNX_FILENAME = "model.onnx"

# Opset 17 is the floor for this architecture. DeepLabV3+'s ASPP image-pooling
# branch pairs AdaptiveAvgPool2d with F.interpolate, and the resize op only
# handles a dynamic batch cleanly from opset 11; 17 is a well-supported version
# comfortably above that, and every runtime this would plausibly target
# supports it.
DEFAULT_OPSET = 17


def export(model, output_path, patch_size, opset=DEFAULT_OPSET):
    """Trace the model to ONNX with a dynamic batch axis.

    Args:
        model: A PyTorch module in eval mode.
        output_path: Where to write the .onnx file.
        patch_size: Spatial size the model is traced at, fixed in the graph.
        opset: ONNX opset version.

    Returns:
        The path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Traced at batch 2 rather than 1 on purpose: tracing at 1 lets a genuine
    # batch-dependent bug hide behind a size-1 dimension that broadcasts.
    sample = torch.randn(2, 3, patch_size, patch_size)

    torch.onnx.export(
        model,
        # A one-tuple, not a bare tensor: both work at runtime, but the
        # signature takes a tuple of positional args and a bare tensor is a
        # type error that only shows up under a checker.
        (sample,),
        str(output_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # Staying on the TorchScript exporter deliberately. PyTorch 2.9 makes
        # the torch.export-based one the default and warns about this, but the
        # two take different arguments -- dynamo uses dynamic_shapes, not
        # dynamic_axes -- so switching is a change to verify, not a flag to
        # flip. The verification below is what would catch a difference.
        dynamo=False,
    )
    logger.info(f"Wrote {output_path} (opset {opset}, patch_size {patch_size})")
    return output_path


def write_metadata(model_dir, patch_size, opset, agreement):
    """Record what the graph was traced for, so a loader can check it.

    Args:
        model_dir: Directory holding the exported bundle.
        patch_size: Spatial size baked into the graph.
        opset: Opset used.
        agreement: The verification summary from verify().

    Returns:
        Path to the metadata file.
    """
    path = Path(model_dir) / "onnx_metadata.json"
    payload = {
        "onnx_file": ONNX_FILENAME,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        # The loader compares this against the configured patch_size and
        # refuses to serve on a mismatch. Height and width are static in the
        # graph, so a mismatch is not a performance question -- it is a shape
        # error at best and a silently resized input at worst.
        "patch_size": patch_size,
        "opset": opset,
        "dynamic_axes": ["batch"],
        "verification": agreement,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def run_dense(session, sample):
    """Run the graph and return its output as a dense array.

    onnxruntime types run() as returning a union of ndarray, SparseTensor, list
    and dict, so numpy operations on the result are a type error even though
    this graph only ever emits one dense float tensor. Checking that rather
    than casting past the checker means a graph that somehow returned something
    else says so, instead of failing later inside a comparison.

    Args:
        session: An onnxruntime InferenceSession.
        sample: Input array of shape (batch, 3, H, W).

    Returns:
        The output logits as an ndarray.

    Raises:
        TypeError: If the graph returned a non-dense output.
    """
    result = session.run([OUTPUT_NAME], {INPUT_NAME: sample})[0]
    if not isinstance(result, np.ndarray):
        raise TypeError(
            f"expected a dense array from '{OUTPUT_NAME}', got {type(result).__name__}"
        )
    return result


def verify(torch_model, onnx_path, patch_size, threshold=0.5, batch_sizes=(1, 3, 8), seed=0):
    """Check the exported graph reproduces the PyTorch model.

    Compares logits AND the thresholded mask. The mask comparison is the one
    that matters: a pixel whose logit sits near the decision boundary flips on a
    difference far below any sensible logit tolerance, and that shows up as an
    IoU change while a max-abs-diff assertion passes.

    Batch sizes deliberately include values other than the traced one, and one
    that is not a divisor of anything -- a dynamic axis that silently fixed
    itself during tracing fails here rather than in production on a partial
    final batch.

    Args:
        torch_model: The source model, in eval mode.
        onnx_path: Path to the exported graph.
        patch_size: Spatial size to test at.
        threshold: Decision threshold used for the mask comparison.
        batch_sizes: Batch sizes to exercise the dynamic axis with.
        seed: RNG seed for the test inputs.

    Returns:
        A dict summarising the worst disagreement seen.

    Raises:
        RuntimeError: If onnxruntime is unavailable.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to verify an export. An unverified ONNX graph "
            "is worse than none: it runs, it is faster, and it may be wrong."
        ) from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)

    worst_logit_diff = 0.0
    worst_mask_disagreement = 0.0

    for batch in batch_sizes:
        sample = rng.standard_normal((batch, 3, patch_size, patch_size)).astype(np.float32)

        with torch.no_grad():
            torch_logits = torch_model(torch.from_numpy(sample)).numpy()
        onnx_logits = run_dense(session, sample)

        if torch_logits.shape != onnx_logits.shape:
            raise ValueError(
                f"shape mismatch at batch {batch}: torch {torch_logits.shape} vs "
                f"onnx {onnx_logits.shape} -- the dynamic axis did not survive tracing"
            )

        worst_logit_diff = max(worst_logit_diff, float(np.abs(torch_logits - onnx_logits).max()))

        torch_mask = (1 / (1 + np.exp(-torch_logits))) > threshold
        onnx_mask = (1 / (1 + np.exp(-onnx_logits))) > threshold
        worst_mask_disagreement = max(
            worst_mask_disagreement, float((torch_mask != onnx_mask).mean())
        )

    summary = {
        "max_logit_abs_diff": worst_logit_diff,
        "max_mask_disagreement_fraction": worst_mask_disagreement,
        "batch_sizes_tested": list(batch_sizes),
        "threshold": threshold,
    }
    logger.info(
        f"Verification: max logit diff {worst_logit_diff:.2e}, "
        f"mask disagreement {worst_mask_disagreement:.2%} "
        f"across batches {list(batch_sizes)}"
    )
    return summary


def main():
    """Export the promoted model to ONNX, verifying before it is kept."""
    parser = argparse.ArgumentParser(description="Export the served model to ONNX.")
    parser.add_argument(
        "--model-dir",
        default="deployment/exported_model/best_model",
        help="MLflow bundle to convert. The .onnx is written alongside it.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-logit-diff",
        type=float,
        default=1e-4,
        help="Fail if logits differ by more than this.",
    )
    parser.add_argument(
        "--max-mask-disagreement",
        type=float,
        default=1e-4,
        help="Fail if more than this fraction of thresholded pixels differ. This is "
        "the assertion that corresponds to what is actually served.",
    )
    args = parser.parse_args()

    from mlflow.pytorch import load_model as load_pytorch_model

    model_dir = Path(args.model_dir)
    logger.info(f"Loading PyTorch model from {model_dir}")
    model = load_pytorch_model(as_model_uri(model_dir), map_location=torch.device("cpu"))
    model.eval()

    onnx_path = export(model, model_dir / ONNX_FILENAME, args.patch_size, args.opset)
    summary = verify(model, onnx_path, args.patch_size, threshold=args.threshold)

    failures = []
    if summary["max_logit_abs_diff"] > args.max_logit_diff:
        failures.append(
            f"logits differ by {summary['max_logit_abs_diff']:.2e} "
            f"(limit {args.max_logit_diff:.0e})"
        )
    if summary["max_mask_disagreement_fraction"] > args.max_mask_disagreement:
        failures.append(
            f"{summary['max_mask_disagreement_fraction']:.2%} of thresholded pixels "
            f"disagree (limit {args.max_mask_disagreement:.2%})"
        )

    if failures:
        # Removed, not left on disk with a warning. A graph that failed
        # verification but is still sitting where the loader looks is exactly
        # how a wrong model reaches production.
        onnx_path.unlink(missing_ok=True)
        raise ValueError("ONNX export does not match the source model: " + "; ".join(failures))

    write_metadata(model_dir, args.patch_size, args.opset, summary)
    size_mb = onnx_path.stat().st_size / 1e6
    logger.info(f"Export verified and kept: {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()