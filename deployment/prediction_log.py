"""
prediction_log.py

Append one JSON line per served prediction, recording what came in and what went
out. This is the smallest honest foundation for drift detection.

The reasoning for keeping it this small: with no production traffic there is
nothing to demonstrate, and a metrics stack on a single-model service invites
"what would you alert on?" without a good answer. But there IS a good answer
available cheaply, because the training-time reference already exists.
metrics/class_balance.json records the pixel-level water fraction of the data
the model learned from. If the imagery being served shifts -- a different
sensor, a different season, a different region -- the predicted water fraction
shifts away from that reference, and comparing the two is a real signal rather
than a dashboard.

What is deliberately NOT recorded: the image itself, or anything that would let
one be reconstructed. Summary statistics only, so the log stays small enough to
keep and carries nothing that needs handling.

Logging must never break serving. Every failure here is swallowed: a full disk
or a read-only mount is not a reason to fail a prediction that already
succeeded.
"""

import json
import os
import time
from pathlib import Path

import numpy as np


def summarize_request(image_bgr, mask, duration_s, filename=None):
    """Reduce one prediction to the statistics drift would show up in.

    Args:
        image_bgr: The decoded input image, HxWx3 BGR.
        mask: The predicted binary mask, HxW, values in {0, 1}.
        duration_s: Wall-clock inference time in seconds.
        filename: Optional uploaded filename, for correlating with a source.

    Returns:
        A JSON-serialisable dict.
    """
    height, width = mask.shape[:2]
    predicted_fraction = float(np.asarray(mask).mean())

    return {
        "ts": time.time(),
        "filename": filename,
        "width": int(width),
        "height": int(height),
        # Input brightness: the cheapest proxy for a sensor or season change,
        # and it moves before the mask does.
        "mean_intensity": float(np.asarray(image_bgr).mean()),
        # The number to compare against overall.foreground_fraction in
        # metrics/class_balance.json. A sustained gap is the drift signal.
        "predicted_water_fraction": predicted_fraction,
        # An empty prediction is not the same as a low one: models that fail
        # closed produce a run of exact zeros rather than a drifting mean.
        "empty_prediction": predicted_fraction == 0.0,
        "duration_s": round(float(duration_s), 4),
    }


def append(record, log_path):
    """Append one record as a JSON line, swallowing any failure.

    Args:
        record: The dict to write.
        log_path: Destination JSONL file.

    Returns:
        True if the record was written, False if logging failed.
    """
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return True
    except Exception:
        # Never propagate. A prediction that already succeeded must not fail
        # because the log could not be written.
        return False


def reference_water_fraction(class_balance_path):
    """Read the training-time water fraction predictions are compared against.

    Args:
        class_balance_path: Path to metrics/class_balance.json.

    Returns:
        The pixel-weighted foreground fraction, or None if unavailable.
    """
    try:
        with open(class_balance_path, encoding="utf-8") as handle:
            return float(json.load(handle)["overall"]["foreground_fraction"])
    except Exception:
        return None


def summarize_log(log_path, reference=None, window=200):
    """Summarise the most recent predictions, for a drift check or a health probe.

    Deliberately a function over the file rather than a running aggregate: the
    log is the record, and anything derived from it should be recomputable from
    the record alone.

    Args:
        log_path: JSONL file written by append().
        reference: Training-time water fraction to compare against.
        window: How many of the most recent records to consider.

    Returns:
        A dict summarising the window, or None if the log is empty or unreadable.
    """
    try:
        with open(log_path, encoding="utf-8") as handle:
            lines = handle.readlines()[-window:]
    except OSError:
        return None

    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated final line is normal if the process died mid-write;
            # skip it rather than discarding the whole window.
            continue

    if not records:
        return None

    fractions = [r["predicted_water_fraction"] for r in records if "predicted_water_fraction" in r]
    summary = {
        "n_requests": len(records),
        "mean_predicted_water_fraction": float(np.mean(fractions)) if fractions else None,
        "empty_prediction_rate": sum(bool(r.get("empty_prediction")) for r in records) / len(records),
        "mean_duration_s": float(np.mean([r.get("duration_s", 0.0) for r in records])),
    }

    if reference is not None and fractions:
        summary["reference_water_fraction"] = reference
        # Reported as a signed difference rather than a pass/fail: what counts
        # as drift depends on deployment context, and inventing a threshold here
        # would be inventing the answer.
        summary["water_fraction_delta"] = summary["mean_predicted_water_fraction"] - reference

    return summary


def default_log_path():
    """Where predictions are logged unless PREDICTION_LOG overrides it."""
    return os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
