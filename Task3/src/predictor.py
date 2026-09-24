"""Prediction service built on the fitted Task 2 inference bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from time import perf_counter
from typing import Any

import pandas as pd

from Task3.app.metrics import PREDICTION_COUNT

from .feature_builder import build_prediction_features
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
        raw_features = build_prediction_features(frame)
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

        probability = float(self.bundle.model.predict_proba(transformed)[0][1])
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
