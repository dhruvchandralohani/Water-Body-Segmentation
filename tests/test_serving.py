"""
test_serving.py

/predict is the endpoint that does the work, and until now nothing tested it.
CI's container check curls /health, which proves the image builds and the model
loads -- not that inference returns anything usable. A break here reaches a user
rather than a developer.

Runs against FastAPI's TestClient with a stub model injected into serve.py's
module state, so no container, no exported bundle, and no MLflow. The stub
returns a fixed pattern, which is enough: what is under test is the request
path -- decoding, temp-file handling, encoding, error mapping -- not the model.
"""

import base64
import json
import re

import cv2
import numpy as np
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


IMAGE_SIZE = 96


@pytest.fixture(scope="module")
def app_module():
    """Import serve.py once and hand back the module."""
    from deployment import serve

    return serve


@pytest.fixture
def client(app_module, tmp_path, monkeypatch):
    """A TestClient with a stub model and a temp prediction log.

    Bypasses the lifespan handler: it loads a real exported bundle, which does
    not exist in a fresh checkout and is not what these tests are about.
    """

    class StubModel:
        """Returns a fixed half-water mask regardless of input."""

        def eval(self):
            return self

        def to(self, device):
            return self

    # Derived from the response model rather than hand-written. The first
    # version listed the timing keys by hand, omitted "tile", and every
    # /predict test failed on a pydantic ValidationError -- a stub that does not
    # match the real contract tests the wrong thing, and this cannot drift from
    # TimingsResponse the way a literal can.
    timing_keys = list(app_module.TimingsResponse.model_fields)
    assert timing_keys, "TimingsResponse declares no fields; the stub would be vacuous"

    def fake_predict_image(model, path, device, **kwargs):
        """Stand in for the real tiled inference, keeping its return contract."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, f"cv2.imread returned None for {path}"
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[: height // 2, :] = 1  # top half "water"
        info = {
            "width": width,
            "height": height,
            "num_tiles": 1,
            "water_coverage_pct": 50.0,
            "timings": dict.fromkeys(timing_keys, 0.01),
        }
        return mask, info

    monkeypatch.setattr(app_module, "predict_image", fake_predict_image)
    monkeypatch.setattr(app_module, "PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setattr(app_module, "CLASS_BALANCE_PATH", str(tmp_path / "class_balance.json"))
    app_module.model_state.update(
        {"model": StubModel(), "device": "cpu", "model_name": "test-model", "model_version": "7"}
    )

    # Constructed WITHOUT the context manager on purpose. `with TestClient(app)`
    # runs the lifespan handler, which loads a real exported bundle -- absent in
    # a fresh checkout, and not what these tests are about. Plain construction
    # skips startup, leaving the stub model_state above in place.
    test_client = TestClient(app_module.app)
    test_client.log_path = tmp_path / "predictions.jsonl"
    test_client.balance_path = tmp_path / "class_balance.json"
    yield test_client


def image_bytes(width=IMAGE_SIZE, height=IMAGE_SIZE, encoding=".jpg"):
    """Encode a synthetic image the way a browser upload would arrive."""
    rng = np.random.default_rng(0)
    image = rng.integers(40, 200, (height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(encoding, image)
    assert ok
    return encoded.tobytes()


def upload(client, data=None, filename="scene.jpg", content_type="image/jpeg", **params):
    """POST one file to /predict."""
    files = {"file": (filename, data if data is not None else image_bytes(), content_type)}
    return client.post("/predict", files=files, params=params)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_json_response_carries_a_decodable_mask(client):
    """The default response must contain a real PNG, not just a base64-shaped string.

    Decoded here rather than length-checked: a truncated or mis-encoded buffer
    is still valid base64 and would pass any weaker assertion.
    """
    response = upload(client)
    assert response.status_code == 200

    payload = response.json()
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(payload["mask_png_base64"]), np.uint8), cv2.IMREAD_GRAYSCALE)

    assert mask is not None, "mask_png_base64 did not decode as an image"
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert set(np.unique(mask).tolist()) <= {0, 255}, "served mask is not binary"
    assert payload["width"] == IMAGE_SIZE and payload["height"] == IMAGE_SIZE


def test_png_format_returns_raw_bytes_not_json(client):
    """?format=png exists to skip the ~33% base64 overhead, so it must be raw."""
    response = upload(client, format="png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG signature"

    mask = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask is not None and mask.shape == (IMAGE_SIZE, IMAGE_SIZE)


def test_both_formats_return_the_same_mask(client):
    """The two representations must not diverge; only the wrapper differs."""
    as_json = upload(client).json()["mask_png_base64"]
    as_png = upload(client, format="png").content

    assert base64.b64decode(as_json) == as_png


def test_repeated_requests_return_identical_output(client):
    """Determinism, from the caller's side rather than the model's."""
    first = upload(client, format="png").content
    second = upload(client, format="png").content
    assert first == second


@pytest.mark.parametrize("encoding,content_type", [(".jpg", "image/jpeg"), (".png", "image/png")])
def test_common_upload_formats_are_accepted(client, encoding, content_type):
    """The service must not silently depend on the dataset's own format."""
    response = upload(client, data=image_bytes(encoding=encoding), filename=f"x{encoding}", content_type=content_type)
    assert response.status_code == 200


def test_non_square_input_is_preserved(client):
    """Tiling must not quietly square or pad the output."""
    response = upload(client, data=image_bytes(width=128, height=64))
    payload = response.json()
    assert (payload["width"], payload["height"]) == (128, 64)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_a_non_image_upload_is_rejected_with_4xx(client):
    """A wrong file type is the caller's error, so it must not surface as a 500."""
    response = upload(client, data=b"this is not an image", filename="notes.txt", content_type="text/plain")
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_an_unknown_format_parameter_is_rejected(client):
    """A typo'd format must fail loudly rather than falling back to a default."""
    response = upload(client, format="jpeg")
    assert response.status_code == 400
    assert "format" in response.json()["detail"].lower()


def test_a_corrupt_image_fails_without_leaking_a_temp_file(client, tmp_path, monkeypatch):
    """The `finally: os.unlink` is the only thing stopping a temp-file leak.

    Untested, a refactor that moves the unlink inside the try would leak a file
    per failed request -- invisible until a disk fills.
    """
    import tempfile

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def explode(*args, **kwargs):
        raise RuntimeError("corrupt image")

    from deployment import serve

    monkeypatch.setattr(serve, "predict_image", explode)

    response = upload(client, data=b"\xff\xd8not really a jpeg", content_type="image/jpeg")
    assert response.status_code == 500
    assert list(scratch.iterdir()) == [], "a temp file survived a failed request"


def test_predict_reports_503_when_no_model_is_loaded(client, app_module):
    """A container that started without a model must say so, not crash."""
    app_module.model_state["model"] = None
    try:
        assert upload(client).status_code == 503
    finally:
        app_module.model_state["model"] = object()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_the_loaded_model_version(client, app_module):
    """/health is the deployment's identity: which model is actually serving.

    CI only checks that this returns 200. Reporting a version means an operator
    can tell which promoted artifact is live without opening the container.
    """
    payload = client.get("/health").json()

    assert payload["model_loaded"] is True
    assert payload["status"] == "ok"
    assert payload["model_version"] == "7"
    assert payload["threshold"] == pytest.approx(app_module.THRESHOLD)


def test_inference_timings_match_the_response_model():
    """predict_image's timing keys and TimingsResponse's fields must agree exactly.

    serve.py does TimingsResponse(**info["timings"]), so a stage added to
    inference.py without updating the model raises a ValidationError on every
    request -- after inference has already run. Nothing checked this until a
    hand-written test stub happened to omit a key and failed the same way.
    """
    import ast
    from pathlib import Path

    from deployment import serve

    source = (Path(__file__).resolve().parents[1] / "deployment" / "inference.py").read_text()
    tree = ast.parse(source)
    assigned = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "timings"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }

    assert assigned == set(serve.TimingsResponse.model_fields), (
        f"inference.py writes timings {sorted(assigned)} but TimingsResponse declares "
        f"{sorted(serve.TimingsResponse.model_fields)}"
    )


# ---------------------------------------------------------------------------
# Prediction logging and drift
# ---------------------------------------------------------------------------


def test_each_prediction_appends_one_log_line(client):
    """One JSON line per served request, and nothing image-shaped in it."""
    upload(client)
    upload(client)

    lines = client.log_path.read_text().strip().splitlines()
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert record["predicted_water_fraction"] == pytest.approx(0.5, abs=0.01)
    assert record["width"] == IMAGE_SIZE and record["height"] == IMAGE_SIZE
    assert record["empty_prediction"] is False
    # Summary statistics only: no pixels, nothing reconstructable.
    assert not any(isinstance(v, (list, bytes)) for v in record.values())


def test_a_failed_prediction_is_not_logged(client, monkeypatch, app_module):
    """The log records what was served, so a 500 must leave no entry.

    Otherwise the drift signal is polluted by requests that produced no output.
    """
    def explode(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(app_module, "predict_image", explode)
    assert upload(client).status_code == 500
    assert not client.log_path.exists() or client.log_path.read_text().strip() == ""


def test_logging_failure_does_not_fail_the_request(client, app_module, monkeypatch, tmp_path):
    """A full disk must not turn a successful prediction into an error.

    The whole point of swallowing inside append() -- asserted here so a later
    refactor cannot helpfully re-raise it.
    """
    # A regular file standing in for the log's parent directory: mkdir cannot
    # succeed against it on any platform, regardless of who is running. An
    # unwritable absolute path would be permission-dependent, and this test
    # would then pass or fail based on who ran it.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setattr(app_module, "PREDICTION_LOG", str(blocker / "predictions.jsonl"))

    assert upload(client).status_code == 200


def test_drift_compares_predictions_against_the_training_reference(client):
    """The comparison that makes the log worth keeping.

    class_balance.json records the water fraction of the data the model learned
    from; a sustained gap between that and what is being predicted is the drift
    signal. Reported as a signed delta, not a verdict.
    """
    client.balance_path.write_text(json.dumps({"overall": {"foreground_fraction": 0.1843}}))
    upload(client)

    payload = client.get("/drift").json()

    assert payload["status"] == "ok"
    assert payload["n_requests"] == 1
    assert payload["reference_water_fraction"] == pytest.approx(0.1843)
    # Stub predicts 50% water against a 18.43% reference: a large positive delta.
    assert payload["water_fraction_delta"] == pytest.approx(0.5 - 0.1843, abs=0.01)


def test_drift_is_honest_when_nothing_has_been_served(client):
    """An empty log must say so rather than reporting a fabricated zero."""
    payload = client.get("/drift").json()
    assert payload["status"] == "no predictions logged yet"


def test_drift_survives_a_truncated_final_line(client):
    """A process killed mid-write leaves a partial line; it must not poison the read."""
    client.balance_path.write_text(json.dumps({"overall": {"foreground_fraction": 0.1843}}))
    upload(client)
    with open(client.log_path, "a", encoding="utf-8") as handle:
        handle.write('{"ts": 123, "predicted_wa')

    payload = client.get("/drift").json()
    assert payload["status"] == "ok"
    assert payload["n_requests"] == 1


# ---------------------------------------------------------------------------
# Prometheus metrics
#
# The whole monitoring stack reads /metrics, and until now nothing tested it.
# Counters live in a process-global registry, so these read a value before and
# after and assert the DELTA -- an absolute assertion would pass or fail based
# on which tests ran first.
# ---------------------------------------------------------------------------


def metric_value(client, name, labels=""):
    """Read one metric's current value from the /metrics endpoint.

    Args:
        client: The TestClient.
        name: Metric name.
        labels: Optional label string as it appears in the exposition, e.g.
            '{outcome="success"}'.

    Returns:
        The value as a float, or 0.0 if the series does not exist yet.
    """
    wanted = f"{name}{labels} "
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(wanted):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_endpoint_serves_prometheus_format(client):
    """Content type matters: Prometheus rejects a payload it cannot parse."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# HELP water_body_predictions_total" in response.text
    assert "# TYPE water_body_predicted_water_fraction histogram" in response.text


def test_a_prediction_increments_the_success_counter(client):
    """The counter the dashboard's request-rate panel is built on."""
    before = metric_value(client, "water_body_predictions_total", '{outcome="success"}')
    upload(client)
    after = metric_value(client, "water_body_predictions_total", '{outcome="success"}')

    assert after == before + 1


def test_a_prediction_observes_the_water_fraction(client):
    """The drift signal itself -- the reason this stack exists.

    Checks the histogram's _sum as well as its _count: a wired-up-but-always-
    zero observation would increment the count and leave the mean at zero,
    which is exactly the failure that would make the drift panel lie.
    """
    count_before = metric_value(client, "water_body_predicted_water_fraction_count")
    sum_before = metric_value(client, "water_body_predicted_water_fraction_sum")

    upload(client)

    assert metric_value(client, "water_body_predicted_water_fraction_count") == count_before + 1
    # The stub predicts exactly half the image as water.
    assert metric_value(client, "water_body_predicted_water_fraction_sum") == pytest.approx(
        sum_before + 0.5, abs=0.01
    )


def test_a_rejected_request_is_not_counted_as_a_success(client):
    """Otherwise the error-rate alert can never fire.

    A 400 must land under its own outcome label and leave success untouched --
    if failures were counted as successes the ratio stays flat no matter how
    badly the service is doing.
    """
    success_before = metric_value(client, "water_body_predictions_total", '{outcome="success"}')
    bad_before = metric_value(client, "water_body_predictions_total", '{outcome="bad_request"}')

    upload(client, data=b"not an image", filename="notes.txt", content_type="text/plain")

    assert metric_value(client, "water_body_predictions_total", '{outcome="success"}') == success_before
    assert metric_value(client, "water_body_predictions_total", '{outcome="bad_request"}') == bad_before + 1


def test_outcome_labels_stay_a_small_fixed_set(client):
    """Guards against a high-cardinality label being added later.

    Every distinct label value is a separate time series. Putting a filename,
    an error message or a request id in here would degrade Prometheus badly and
    the damage is not obvious until the instance is already struggling.
    """
    upload(client)
    upload(client, data=b"x", filename="a.txt", content_type="text/plain")

    outcomes = set(re.findall(r'water_body_predictions_total\{outcome="([^"]+)"\}', client.get("/metrics").text))
    assert outcomes <= {"success", "bad_request", "no_model", "inference_error"}, (
        f"unexpected outcome labels: {outcomes}"
    )


def test_the_training_reference_is_exported(client, app_module):
    """The gauge the drift alert divides by.

    Without it the alert expression divides by zero and never fires -- so a
    missing class_balance.json disables drift detection silently.
    """
    app_module.CLASS_BALANCE_PATH = str(client.balance_path)
    client.balance_path.write_text(json.dumps({"overall": {"foreground_fraction": 0.1843}}))
    app_module.metrics_exporter.set_training_reference(
        app_module.prediction_log.reference_water_fraction(str(client.balance_path))
    )

    assert metric_value(client, "water_body_training_water_fraction") == pytest.approx(0.1843)


def test_stage_timings_do_not_double_count_total(client):
    """'total' belongs to the latency histogram, not the per-stage one.

    Exporting it in both would make a stacked stage panel add up to twice the
    real request time.
    """
    upload(client)
    assert 'water_body_stage_duration_seconds_count{stage="total"}' not in client.get("/metrics").text
    assert 'water_body_stage_duration_seconds_count{stage="inference"}' in client.get("/metrics").text
