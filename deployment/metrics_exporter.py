"""
metrics_exporter.py

Prometheus instrumentation for the inference service.

The generic HTTP metrics here (request count, latency) are the same ones any
service exposes and are not the reason this exists. The reason is
`predicted_water_fraction`: the drift signal already recorded per-request in
prediction_log.py, but as a histogram that Prometheus scrapes.

That distinction matters because of a limitation Kubernetes introduced. The
JSONL log is written to an emptyDir -- per-pod, lost on restart -- so /drift on
a two-replica Deployment reports one pod's view of its own traffic. A scraped
histogram aggregates across every pod and survives them, which is what makes
the comparison against the training-time water fraction usable in a cluster
rather than only on a laptop.

Deliberately NOT exported: anything per-image or per-request-identifying.
Prometheus retains everything it scrapes, and a label with high cardinality
(filenames, request ids) degrades it badly. Summary distributions only.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# --- Request-level ----------------------------------------------------------

PREDICTIONS_TOTAL = Counter(
    "water_body_predictions_total",
    "Predictions served, by outcome.",
    ["outcome"],
)

PREDICTION_DURATION = Histogram(
    "water_body_prediction_duration_seconds",
    "End-to-end time to serve one prediction.",
    # Tuned to observed CPU inference on tiled full-resolution imagery, which
    # runs in single-digit seconds -- the default buckets top out at 10s and
    # would put almost everything in one bin.
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)

STAGE_DURATION = Histogram(
    "water_body_stage_duration_seconds",
    "Time per inference stage: load, tile, inference, stitch.",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

# --- The drift signal -------------------------------------------------------

PREDICTED_WATER_FRACTION = Histogram(
    "water_body_predicted_water_fraction",
    "Fraction of pixels predicted as water, per request.",
    # Linear buckets across [0, 1]: this is a proportion, not a latency, so the
    # exponential defaults would compress exactly the mid-range where drift
    # against a 0.18 training reference would show up.
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
)

EMPTY_PREDICTIONS_TOTAL = Counter(
    "water_body_empty_predictions_total",
    "Predictions containing no water at all. Tracked separately from a low "
    "fraction: a model failing closed produces exact zeros, which a mean hides.",
)

TRAINING_WATER_FRACTION = Gauge(
    "water_body_training_water_fraction",
    "Pixel-level water fraction of the training data, from class_balance.json. "
    "Constant for a given model -- the reference the served distribution is "
    "compared against.",
)

# --- Model identity ---------------------------------------------------------

MODEL_INFO = Gauge(
    "water_body_model_info",
    "Always 1. The labels carry which model is serving, so a dashboard can "
    "correlate a shift in the distribution with a rollout.",
    ["model_name", "model_version", "device", "backend"],
)

MODEL_LOADED = Gauge(
    "water_body_model_loaded",
    "1 when a model is loaded and the pod can serve, 0 otherwise.",
)


def set_model_info(model_name, model_version, device, loaded, backend="unknown"):
    """Record which model this pod is serving, and through which runtime.

    The backend label is what turns "ONNX is faster" into a measurement: the
    latency histogram is already collected, so a dashboard can split it by
    backend and show the difference rather than assert it.

    Args:
        model_name: Registered model name.
        model_version: Version string, or "unknown".
        device: Device the model runs on.
        loaded: Whether the model loaded successfully.
        backend: "onnx" or "pytorch". A fixed, tiny set -- each distinct value
            is another time series.
    """
    MODEL_INFO.labels(
        model_name=str(model_name),
        model_version=str(model_version),
        device=str(device),
        backend=str(backend),
    ).set(1)
    MODEL_LOADED.set(1 if loaded else 0)


def set_training_reference(fraction):
    """Publish the training-time water fraction as a gauge.

    Args:
        fraction: The value from class_balance.json, or None if unavailable.
    """
    if fraction is not None:
        TRAINING_WATER_FRACTION.set(float(fraction))


def observe_prediction(water_fraction, duration_s, timings=None):
    """Record one successful prediction.

    Args:
        water_fraction: Fraction of pixels predicted as water.
        duration_s: End-to-end duration in seconds.
        timings: Optional per-stage timings dict from predict_image.
    """
    PREDICTIONS_TOTAL.labels(outcome="success").inc()
    PREDICTION_DURATION.observe(duration_s)
    PREDICTED_WATER_FRACTION.observe(water_fraction)

    if water_fraction == 0.0:
        EMPTY_PREDICTIONS_TOTAL.inc()

    for stage, seconds in (timings or {}).items():
        # "total" is already PREDICTION_DURATION; exporting it twice under a
        # second name would make a dashboard double-count it.
        if stage != "total":
            STAGE_DURATION.labels(stage=stage).observe(seconds)


def observe_failure(outcome):
    """Record a prediction that did not produce a mask.

    Args:
        outcome: Short reason label -- keep the set small and fixed, since each
            distinct value is a separate time series.
    """
    PREDICTIONS_TOTAL.labels(outcome=outcome).inc()


def render():
    """Render the current metrics in Prometheus text format.

    Returns:
        A tuple of (payload bytes, content type).
    """
    return generate_latest(), CONTENT_TYPE_LATEST