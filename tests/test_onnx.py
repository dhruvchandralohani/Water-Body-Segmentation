"""
test_onnx.py

A bad ONNX export does not raise. It produces a graph that loads, runs faster
than the original, and returns subtly different numbers -- which is the whole
reason export_onnx.py verifies itself rather than trusting torch.onnx.export.

These tests cover the verification machinery, because that is the part standing
between a wrong graph and production. The export itself needs torch and a real
model; those tests are marked slow and skip cleanly when either is absent.
"""

import json

import numpy as np
import pytest

onnx_runtime = pytest.importorskip("onnxruntime")

from deployment.export_onnx import (  # noqa: E402
    DEFAULT_OPSET,
    INPUT_NAME,
    ONNX_FILENAME,
    OUTPUT_NAME,
    run_dense,
    write_metadata,
)

torch = pytest.importorskip("torch", reason="export tests need torch")


PATCH = 64


class TinySegModel(torch.nn.Module):
    """Stand-in with the shape contract that matters: 3 channels in, 1 out.

    Small enough to export in milliseconds. What is under test is the export
    and verification path, not the architecture -- a real DeepLabV3+ would make
    these tests slow without testing anything extra about the conversion.
    """

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.body(x)


@pytest.fixture
def exported(tmp_path):
    """Export the stand-in model once and hand back (model, path)."""
    from deployment.export_onnx import export

    model = TinySegModel().eval()
    path = export(model, tmp_path / ONNX_FILENAME, PATCH)
    return model, path


# ---------------------------------------------------------------------------
# The dynamic batch axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 3, 8, 13])
def test_the_graph_accepts_any_batch_size(exported, batch):
    """The tiler's final batch is almost always partial.

    Traced at batch 2, so 1 and 13 both exercise the dynamic axis in ways the
    trace did not see. A dynamic dimension that silently fixed itself during
    tracing fails here rather than in production on a specific image size.
    """
    _, path = exported
    session = onnx_runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    sample = np.random.default_rng(batch).standard_normal((batch, 3, PATCH, PATCH)).astype(np.float32)
    out = run_dense(session, sample)

    assert out.shape == (batch, 1, PATCH, PATCH)


def test_the_batch_axis_is_declared_dynamic(exported):
    """Asserted on the graph, not inferred from it working.

    A graph can accept several batch sizes by accident during testing while
    still carrying a fixed dimension; reading the declared shape is the direct
    check.
    """
    _, path = exported
    session = onnx_runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    shape = session.get_inputs()[0].shape
    assert isinstance(shape[0], str), f"batch dimension is fixed at {shape[0]}"
    assert shape[1:] == [3, PATCH, PATCH], "spatial dims should be static"


def test_spatial_dimensions_are_static(exported):
    """A deliberate choice, pinned so it is not changed by accident.

    Tiles are resized to patch_size before the model, so height and width are
    genuinely fixed. Keeping them static lets the runtime plan allocations. The
    cost -- a graph that is wrong if patch_size changes -- is covered by the
    metadata check, not by making these dynamic.
    """
    _, path = exported
    session = onnx_runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    wrong = np.zeros((2, 3, PATCH * 2, PATCH * 2), dtype=np.float32)
    # The specific ORT error, not a bare Exception -- that would also pass if
    # the session failed to construct, which is a different bug entirely.
    with pytest.raises(onnx_runtime.capi.onnxruntime_pybind11_state.InvalidArgument):
        session.run([OUTPUT_NAME], {INPUT_NAME: wrong})


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_a_faithful_export_verifies(exported):
    """The ordinary path: a real export should agree with its source."""
    from deployment.export_onnx import verify

    model, path = exported
    summary = verify(model, path, PATCH)

    assert summary["max_logit_abs_diff"] < 1e-4
    assert summary["max_mask_disagreement_fraction"] == 0.0


def test_verification_exercises_more_than_the_traced_batch(exported):
    """Verifying only at the traced size would miss a broken dynamic axis."""
    from deployment.export_onnx import verify

    model, path = exported
    summary = verify(model, path, PATCH)

    assert len(summary["batch_sizes_tested"]) > 1
    assert 2 not in summary["batch_sizes_tested"] or len(summary["batch_sizes_tested"]) > 2


def test_a_shape_mismatch_is_reported_not_silently_broadcast(exported, monkeypatch):
    """Different output shapes must raise, not compare via broadcasting.

    numpy would happily subtract (8,1,64,64) from (1,1,64,64) and report a tiny
    difference, so the shape check has to come first.
    """
    from deployment.export_onnx import verify

    model, path = exported

    class WrongShape(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(1, 1, PATCH, PATCH)

    with pytest.raises(ValueError, match="shape mismatch"):
        verify(WrongShape().eval(), path, PATCH, batch_sizes=(4,))


def test_mask_disagreement_catches_what_a_logit_tolerance_misses():
    """The reason verification checks the thresholded mask at all.

    A uniform 1e-4 shift in logits passes any sensible logit tolerance, and
    still flips every pixel sitting within that distance of the decision
    boundary. Segmentation logits cluster near the boundary exactly where the
    model is uncertain -- which is where the mask is most likely to matter.
    """
    # A tolerance someone would plausibly set on logits, and an export
    # difference an order of magnitude smaller than it. Comparing the shift
    # against a real tolerance is the actual claim; comparing it against itself
    # only tested float32 rounding, which is how the first version of this
    # failed at 1.0001e-4 against a 1e-4 bound.
    logit_tolerance = 1e-3
    shift = 1e-4

    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((8, 1, 128, 128)) * 0.5).astype(np.float32)
    shifted = logits + shift

    logit_diff = float(np.abs(logits - shifted).max())
    flipped = float(((1 / (1 + np.exp(-logits)) > 0.5) != (1 / (1 + np.exp(-shifted)) > 0.5)).mean())

    assert logit_diff < logit_tolerance, "the shift is well inside the logit tolerance"
    assert flipped > 0, "yet thresholded pixels changed, which is what gets served"


# ---------------------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------------------


def test_metadata_records_what_the_loader_must_check(tmp_path):
    """patch_size is the load-bearing field.

    Height and width are static in the graph, so serving with a different
    patch_size is a shape error at best. The loader compares this value and
    refuses rather than resizing silently.
    """
    summary = {"max_logit_abs_diff": 1e-7, "max_mask_disagreement_fraction": 0.0}
    path = write_metadata(tmp_path, patch_size=256, opset=DEFAULT_OPSET, agreement=summary)

    meta = json.loads(path.read_text())
    assert meta["patch_size"] == 256
    assert meta["input_name"] == INPUT_NAME
    assert meta["output_name"] == OUTPUT_NAME
    assert meta["dynamic_axes"] == ["batch"]
    # The verification result travels with the model: an operator asking "was
    # this checked, and how closely" should not have to find the build log.
    assert meta["verification"]["max_mask_disagreement_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Backend selection
#
# The adapter exists so predict_image does not branch on backend. These check
# the selection logic and the call contract, not inference itself -- that is
# covered by the equivalence tests above.
# ---------------------------------------------------------------------------


def test_auto_prefers_onnx_when_a_graph_is_present(tmp_path, exported, monkeypatch):
    """A bundle carrying a graph should serve from it without being asked."""
    from deployment import inference
    from deployment.export_onnx import write_metadata

    model, path = exported
    bundle = path.parent
    write_metadata(bundle, patch_size=PATCH, opset=DEFAULT_OPSET, agreement={})

    backend = inference.load_model(bundle, torch.device("cpu"), backend="auto")
    assert isinstance(backend, inference.OnnxBackend)


def test_auto_falls_back_to_pytorch_when_no_graph_exists(tmp_path, monkeypatch):
    """Absence of a graph must not stop the service starting.

    "auto" is a preference, not a requirement -- a bundle exported before ONNX
    was added still has to serve.

    Substitutes the loader function, not mlflow's. Patching mlflow.pytorch from
    outside does not work: it resolves its flavor modules lazily and re-binds
    the original past a monkeypatch, so the real loader runs anyway and fails on
    a directory that holds no bundle.
    """
    from deployment import inference

    called = {}

    def fake_loader(model_path, device, use_fp16=False):
        called["path"] = model_path
        return "pytorch-backend"

    monkeypatch.setattr(inference, "load_pytorch_backend", fake_loader)
    backend = inference.load_model(tmp_path, "cpu", backend="auto")

    assert backend == "pytorch-backend"
    assert called["path"] == tmp_path


def test_explicit_onnx_fails_loudly_when_the_graph_is_missing(tmp_path):
    """Asking for ONNX and silently getting PyTorch would be worse than an error.

    The difference is invisible in the response and shows up only as latency,
    which is exactly the measurement someone choosing a backend cares about.
    """
    from deployment import inference

    with pytest.raises(FileNotFoundError, match="no ONNX graph"):
        inference.load_model(tmp_path, torch.device("cpu"), backend="onnx")


def test_a_graph_without_metadata_is_rejected(tmp_path, exported):
    """patch_size is static in the graph; without metadata nothing validates it."""
    from deployment import inference

    _, path = exported
    (path.parent / "onnx_metadata.json").unlink(missing_ok=True)

    with pytest.raises(FileNotFoundError, match="onnx_metadata"):
        inference.load_model(path.parent, torch.device("cpu"), backend="onnx")


def test_an_unknown_backend_is_rejected(tmp_path):
    """A typo in config should fail at load, not pick a default."""
    from deployment import inference

    with pytest.raises(ValueError, match="backend must be"):
        inference.load_model(tmp_path, torch.device("cpu"), backend="tensorrt")


def test_both_backends_share_one_numpy_call_contract(exported):
    """NCHW float32 in, NCHW logits out -- as an ndarray, from either backend.

    numpy is the shared language on purpose: it is what lets predict_image
    import no torch, and what keeps the two backends running through identical
    tiling and stitching code. If they disagreed on types, a latency comparison
    between them would also be comparing two different code paths.
    """
    from deployment.export_onnx import write_metadata
    from deployment.inference import TorchBackend, load_model

    model, path = exported
    write_metadata(path.parent, patch_size=PATCH, opset=DEFAULT_OPSET, agreement={})

    batch = np.random.default_rng(0).standard_normal((3, 3, PATCH, PATCH)).astype(np.float32)

    onnx_backend = load_model(path.parent, torch.device("cpu"), backend="onnx")
    torch_backend = TorchBackend(model, torch.device("cpu"))

    onnx_out = onnx_backend(batch)
    torch_out = torch_backend(batch)

    for name, out in (("onnx", onnx_out), ("torch", torch_out)):
        assert isinstance(out, np.ndarray), f"{name} backend returned {type(out).__name__}"
        assert out.shape == (3, 1, PATCH, PATCH)
        assert out.dtype == np.float32

    # And they agree numerically, which is the point of having both.
    assert np.abs(onnx_out - torch_out).max() < 1e-4