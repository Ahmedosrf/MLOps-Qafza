"""Prediction service built on the fitted Task 2 inference bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from time import perf_counter
from typing import Any

import pandas as pd
import math

from Task3.app.metrics import PREDICTION_COUNT

from .feature_builder import (
    build_prediction_features,
    build_corrected_prediction_features,
)
from .contract import FEATURE_POLICY, PAYMENT_FIELDS
from .model_loader import InferenceBundle
from .prediction_logger import log_prediction
from .validation import validate_prediction_frame

logger = logging.getLogger("task3.predictor")


@dataclass(frozen=True)
class PredictionResult:
    """Serializable result for one order prediction."""

    prediction: str
    is_late: int
    probability: float
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    target_definition: str | None = None
    prediction_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PredictionService:
    """Apply the fitted preprocessor and model to new order data."""

    def __init__(self, bundle: InferenceBundle) -> None:
        self.bundle = bundle

    def predict_one(self, order: dict[str, Any] | pd.DataFrame) -> PredictionResult:
        """Predict one order and return its class, probability, and metadata."""

        started = perf_counter()
        frame = self._to_frame(order)
        validate_prediction_frame(frame)
        policy = self.bundle.feature_metadata.get("feature_policy")
        if policy == FEATURE_POLICY:
            raw_features = build_corrected_prediction_features(frame)
        elif policy is None:
            if any(
                column not in frame or frame[column].isna().any()
                for column in PAYMENT_FIELDS
            ):
                raise ValueError("Legacy model bundles require all payment fields")
            raw_features = build_prediction_features(frame)
        else:
            raise RuntimeError(f"Unsupported feature policy: {policy}")
        transformed = self.bundle.preprocessor.transform(raw_features)

        expected_count = len(self.bundle.feature_names)
        actual_count = transformed.shape[1]
        if actual_count != expected_count:
            raise ValueError(
                "Transformed feature count does not match the bundle: "
                f"expected {expected_count}, got {actual_count}"
            )

        if not hasattr(self.bundle.model, "predict_proba"):
            raise TypeError("Loaded model must provide predict_proba")

        classes = list(getattr(self.bundle.model, "classes_", [0, 1]))
        if set(classes) != {0, 1}:
            raise RuntimeError("Loaded model must classify on_time=0 and late=1")
        probability = float(
            self.bundle.model.predict_proba(transformed)[0][classes.index(1)]
        )
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("Model returned an invalid probability")
        is_late = int(probability >= self.bundle.threshold)
        latency_ms = (perf_counter() - started) * 1000

        result = PredictionResult(
            prediction="late" if is_late else "on_time",
            is_late=is_late,
            probability=probability,
            threshold=self.bundle.threshold,
            model_name=self.bundle.model_name,
            model_version=self.bundle.model_version,
            latency_ms=round(latency_ms, 3),
            target_definition=self.bundle.feature_metadata.get("target_definition"),
            prediction_stage=self.bundle.feature_metadata.get("prediction_stage"),
        )
        result_dict = result.to_dict()
        PREDICTION_COUNT.labels(result.prediction, result.model_version).inc()
        logger.info("prediction_result=%s", result_dict)
        log_prediction(frame.to_dict(orient="records")[0], result_dict)
        return result

    @staticmethod
    def _to_frame(order: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(order, pd.DataFrame):
            if len(order) != 1:
                raise ValueError("predict_one expects exactly one order row")
            return order.copy()
        if isinstance(order, dict):
            return pd.DataFrame([order])
        raise TypeError("order must be a dictionary or one-row pandas DataFrame")


__all__ = ["PredictionResult", "PredictionService"]
