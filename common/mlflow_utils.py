"""
mlflow_utils.py

Shared MLflow helpers used by both test_model.py and export_model.py --
extracted here rather than duplicated, since both need to answer the
same question: which run actually produced the best model.
"""

import mlflow
from mlflow.tracking import MlflowClient


def find_best_run(experiment_name, tracking_uri, metric="val_iou"):
    """Return the best run ID and its best metric value for an MLflow experiment.

    Args:
        experiment_name: Name of the MLflow experiment to inspect.
        tracking_uri: MLflow tracking server URI to query.
        metric: Metric name to evaluate for each run.

    Returns:
        A tuple containing the best run ID and the highest observed metric value.

    Raises:
        ValueError: If the experiment does not exist, has no runs, or has no
            runs that logged the requested metric.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    best_run_id, best_value = None, float("-inf")
    for run in runs:
        history = client.get_metric_history(run.info.run_id, metric)
        if not history:
            continue
        peak = max(h.value for h in history)
        if peak > best_value:
            best_value = peak
            best_run_id = run.info.run_id

    if best_run_id is None:
        raise ValueError(f"No runs in experiment '{experiment_name}' have logged metric '{metric}'")

    return best_run_id, best_value
