"""Register or export the fitted Task 2 bundle without refitting anything."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import joblib
import mlflow

from .model_loader import resolve_bundle_path


def export_bundle() -> Path:
    """Copy the Task 2 bundle to the DVC-managed Task 3 model path."""

    source = resolve_bundle_path()
    destination = Path(
        os.getenv("DVC_MODEL_PATH", "Task3/models/06_inference_bundle.joblib")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def register_bundle() -> str:
    """Log the existing fitted model bundle in MLflow."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "olist-late-delivery")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    bundle_path = resolve_bundle_path()
    raw_bundle = joblib.load(bundle_path)

    with mlflow.start_run(run_name="task2-inference-bundle") as run:
        mlflow.log_param("model_name", raw_bundle["model_name"])
        mlflow.log_param("model_version", raw_bundle["model_version"])
        mlflow.log_param("threshold", raw_bundle["threshold"])
        mlflow.log_param("feature_count", len(raw_bundle["feature_names"]))
        mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")
        mlflow.set_tag("source", "Task2/notebooks/06_train_tune_evaluate.ipynb")
        mlflow.set_tag("stage", "Production")
        return run.info.run_id


if __name__ == "__main__":
    print(register_bundle())
