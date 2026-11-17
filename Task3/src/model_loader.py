"""Load and validate the fitted Task 2 inference bundle."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import joblib


REQUIRED_BUNDLE_KEYS = {
    "model",
    "preprocessor",
    "feature_names",
    "threshold",
    "model_name",
    "model_version",
}


@dataclass(frozen=True)
class InferenceBundle:
    """Validated fitted objects required for prediction."""

    model: Any
    preprocessor: Any
    feature_names: tuple[str, ...]
    threshold: float
    model_name: str
    model_version: str
    feature_metadata: dict[str, Any]


def find_project_root() -> Path:
    """Find the repository root or use ``MLOPS_PROJECT_ROOT``."""

    configured_root = os.getenv("MLOPS_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "Task2").is_dir() and (candidate / "Task3").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the repository root. Set MLOPS_PROJECT_ROOT explicitly."
    )


def resolve_bundle_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit or environment-configured model bundle path."""

    configured_path = path or os.getenv(
        "MODEL_BUNDLE_PATH",
        "Task2/artifacts/models/06_inference_bundle.joblib",
    )
    bundle_path = Path(configured_path).expanduser()
    if not bundle_path.is_absolute():
        bundle_path = find_project_root() / bundle_path
    return bundle_path.resolve()


def _validate_bundle(raw_bundle: dict[str, Any]) -> InferenceBundle:
    missing = REQUIRED_BUNDLE_KEYS.difference(raw_bundle)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Inference bundle is missing keys: {missing_names}")

    threshold = float(raw_bundle["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Inference threshold must be strictly between 0 and 1")

    feature_names = tuple(str(name) for name in raw_bundle["feature_names"])
    if not feature_names:
        raise ValueError("Inference bundle must contain at least one feature name")

    if not callable(getattr(raw_bundle["preprocessor"], "transform", None)):
        raise TypeError("Loaded preprocessor must provide transform")
    if not callable(getattr(raw_bundle["model"], "predict_proba", None)):
        raise TypeError("Loaded model must provide predict_proba")
    preprocessor = raw_bundle["preprocessor"]
    if hasattr(preprocessor, "get_feature_names_out"):
        if tuple(preprocessor.get_feature_names_out()) != feature_names:
            raise ValueError(
                "Bundle feature names do not match the fitted preprocessor"
            )
    from sklearn.utils.validation import check_is_fitted
    from sklearn.base import BaseEstimator

    for fitted_object in (raw_bundle["model"], preprocessor):
        if isinstance(fitted_object, BaseEstimator):
            check_is_fitted(fitted_object)

    model = raw_bundle["model"]
    if callable(getattr(model, "is_fitted", None)) and not model.is_fitted():
        raise ValueError("Loaded model is not fitted")
    model_feature_count = getattr(model, "n_features_in_", None)
    # CatBoost 1.2.8 can deserialize n_features_in_ as zero while retaining
    # the actual fitted feature names and trees. Validate its persisted names.
    if not model_feature_count and hasattr(model, "feature_names_"):
        model_feature_count = len(model.feature_names_)
    if model_feature_count is not None and model_feature_count != len(feature_names):
        raise ValueError("Model feature count does not match the preprocessor")
    if hasattr(model, "classes_") and set(model.classes_) != {0, 1}:
        raise ValueError("Loaded model must classify on_time=0 and late=1")

    metadata = dict(raw_bundle.get("feature_metadata", {}))
    if metadata.get("feature_policy") is not None:
        from .contract import FEATURE_POLICY, TARGET_DEFINITION, PREDICTION_STAGE

        if metadata["feature_policy"] != FEATURE_POLICY:
            raise ValueError("Unsupported feature policy")
        if (
            metadata.get("target_definition") != TARGET_DEFINITION
            or metadata.get("prediction_stage") != PREDICTION_STAGE
        ):
            raise ValueError("Bundle target/prediction-stage contract mismatch")

    return InferenceBundle(
        model=raw_bundle["model"],
        preprocessor=raw_bundle["preprocessor"],
        feature_names=feature_names,
        threshold=threshold,
        model_name=str(raw_bundle["model_name"]),
        model_version=str(raw_bundle["model_version"]),
        feature_metadata=dict(raw_bundle.get("feature_metadata", {})),
    )


def load_inference_bundle(path: str | Path | None = None) -> InferenceBundle:
    """Load a fitted bundle from disk without fitting any object."""

    source = os.getenv("MODEL_SOURCE", "local")
    if path is None and source == "mlflow":
        import mlflow

        uri = os.environ.get("MODEL_URI")
        if not uri:
            raise ValueError("MODEL_URI is required for MODEL_SOURCE=mlflow")
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
        metadata = mlflow.models.Model.load(str(downloaded / "MLmodel"))
        artifact = metadata.flavors["python_function"]["artifacts"]["bundle"]["path"]
        bundle_path = downloaded / artifact
    elif path is not None or source == "local":
        bundle_path = resolve_bundle_path(path)
    else:
        raise ValueError(f"Unsupported MODEL_SOURCE: {source}")
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Inference bundle not found: {bundle_path}")

    raw_bundle = joblib.load(bundle_path)
    if not isinstance(raw_bundle, dict):
        raise TypeError("Inference bundle must be a dictionary")

    return _validate_bundle(raw_bundle)


__all__ = [
    "InferenceBundle",
    "find_project_root",
    "load_inference_bundle",
    "resolve_bundle_path",
]
