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

    bundle_path = resolve_bundle_path(path)
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
