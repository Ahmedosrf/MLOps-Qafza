from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import pytest

from Task3.src.model_loader import load_inference_bundle
from Task3.src.predictor import PredictionService


@dataclass
class FakePreprocessor:
    def transform(self, frame):
        return np.array(
            [[float(frame.loc[0, "total_price"]), float(frame.loc[0, "num_items"])]],
            dtype=float,
        )


class FakeModel:
    def predict_proba(self, transformed):
        assert len(transformed) == 1
        return [[0.25, 0.75]]


class WrongShapePreprocessor:
    def transform(self, frame):
        return np.array([[1.0]], dtype=float)


class ModelWithoutProbability:
    def predict(self, transformed):
        return np.array([1])


VALID_ORDER = {
    "order_purchase_timestamp": "2018-01-06 10:30:00",
    "order_estimated_delivery_date": "2018-01-15",
    "order_approved_at": "2018-01-06 11:30:00",
    "customer_zip_code_prefix": 13000,
    "customer_city": "campinas",
    "customer_state": "SP",
    "num_items": 2,
    "num_distinct_products": 2,
    "total_price": 100.0,
    "total_freight": 20.0,
    "num_sellers": 1,
    "num_product_categories": 1,
    "avg_product_weight_g": 500.0,
    "avg_product_photos_qty": 3.0,
    "seller_state_count": 1,
    "payment_record_count": 1,
    "total_payment_value": 120.0,
    "max_payment_installments": 3,
    "seller_state": "RJ",
}


def bundle_dict():
    return {
        "model": FakeModel(),
        "preprocessor": FakePreprocessor(),
        "feature_names": ["numeric__total_price", "numeric__num_items"],
        "threshold": 0.30,
        "model_name": "olist-late-delivery",
        "model_version": "task2-v1",
        "feature_metadata": {"target": "is_late"},
    }


def test_loader_validates_and_loads_bundle(tmp_path):
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle_dict(), path)

    bundle = load_inference_bundle(path)

    assert bundle.threshold == 0.30
    assert bundle.model_version == "task2-v1"
    assert len(bundle.feature_names) == 2


def test_prediction_service_uses_saved_threshold():
    service = PredictionService(load_inference_bundle_from_dict(bundle_dict()))

    result = service.predict_one(VALID_ORDER)

    assert result.prediction == "late"
    assert result.is_late == 1
    assert result.probability == pytest.approx(0.75)
    assert result.threshold == pytest.approx(0.30)
    assert result.model_version == "task2-v1"
    assert result.latency_ms >= 0


def test_loader_rejects_invalid_threshold(tmp_path):
    invalid = bundle_dict()
    invalid["threshold"] = 1.0
    path = tmp_path / "invalid.joblib"
    joblib.dump(invalid, path)

    with pytest.raises(ValueError, match="threshold"):
        load_inference_bundle(path)


def test_loader_rejects_missing_bundle_keys(tmp_path):
    invalid = bundle_dict()
    del invalid["preprocessor"]
    path = tmp_path / "missing-key.joblib"
    joblib.dump(invalid, path)

    with pytest.raises(ValueError, match="preprocessor"):
        load_inference_bundle(path)


def test_prediction_service_rejects_wrong_transformed_shape():
    invalid = bundle_dict()
    invalid["preprocessor"] = WrongShapePreprocessor()
    service = PredictionService(load_inference_bundle_from_dict(invalid))

    with pytest.raises(ValueError, match="feature count"):
        service.predict_one(VALID_ORDER)


def test_prediction_service_rejects_model_without_predict_proba():
    invalid = bundle_dict()
    invalid["model"] = ModelWithoutProbability()
    service = PredictionService(load_inference_bundle_from_dict(invalid))

    with pytest.raises(TypeError, match="predict_proba"):
        service.predict_one(VALID_ORDER)


def test_prediction_service_rejects_multiple_rows_for_single_prediction():
    service = PredictionService(load_inference_bundle_from_dict(bundle_dict()))
    rows = pd.DataFrame([VALID_ORDER, VALID_ORDER])

    with pytest.raises(ValueError, match="exactly one"):
        service.predict_one(rows)


def load_inference_bundle_from_dict(raw_bundle):
    from Task3.src.model_loader import _validate_bundle

    return _validate_bundle(raw_bundle)
