"""Regression tests for day boundaries, as-of labels and v3 feature isolation."""

import numpy as np
import pandas as pd
import pytest
from Task2.src.corrected_data import (
    calendar_day_labels,
    asof_training,
    evaluation_block,
)
from Task3.src.feature_builder import build_corrected_prediction_features
from Task3.src.contract import (
    FEATURE_POLICY,
    TARGET_DEFINITION,
    PREDICTION_STAGE,
    PAYMENT_FIELDS,
)
from Task3.src.model_loader import _validate_bundle
from Task3.tests.ci_bundle import ORDER, make_bundle


def label_rows(actual):
    return pd.DataFrame(
        [
            {
                "order_id": str(index),
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-01-01 10:00",
                "order_approved_at": "2018-01-01 11:00",
                "order_estimated_delivery_date": "2018-01-10 00:00",
                "order_delivered_customer_date": value,
            }
            for index, value in enumerate(actual)
        ]
    )


def test_whole_due_date_is_on_time_but_next_midnight_is_late():
    frame = label_rows(
        [
            "2018-01-09 23:59:59",
            "2018-01-10 00:00",
            "2018-01-10 23:59:59",
            "2018-01-11 00:00",
        ]
    )
    result = calendar_day_labels(frame)
    assert result.is_late.tolist() == [0, 0, 0, 1]
    assert "is_late" not in frame


def test_timezone_is_normalized_before_calendar_comparison():
    # Jan 11 01:00 UTC is Jan 10 23:00 in Sao Paulo (summer DST).
    result = calendar_day_labels(label_rows(["2018-01-11T01:00:00Z"]))
    assert result.is_late.iloc[0] == 0


def test_invalid_approval_chronology_is_excluded():
    frame = label_rows(["2018-01-10 12:00"])
    frame["order_approved_at"] = "2018-01-01 09:00"
    assert calendar_day_labels(frame).empty


def test_training_requires_both_prediction_and_label_strictly_before_cutoff():
    frame = pd.DataFrame(
        {
            "order_id": ["known", "not_mature", "equal_cutoff", "future_order"],
            "prediction_at": pd.to_datetime(
                ["2018-01-01", "2018-01-01", "2018-01-01", "2018-02-01"]
            ),
            "label_available_at": pd.to_datetime(
                ["2018-01-15", "2018-02-02", "2018-02-01", "2018-01-31"]
            ),
        }
    )
    assert asof_training(frame, "2018-02-01").order_id.tolist() == ["known"]


def test_validation_outcomes_cannot_extend_into_test_selection_time():
    frame = pd.DataFrame(
        {
            "order_id": ["known", "future_label", "test_order"],
            "prediction_at": pd.to_datetime(["2018-04-01", "2018-04-02", "2018-06-01"]),
            "label_available_at": pd.to_datetime(
                ["2018-05-01", "2018-06-01", "2018-06-02"]
            ),
        }
    )
    assert evaluation_block(
        frame, "2018-04-01", "2018-06-01", "2018-06-01"
    ).order_id.tolist() == ["known"]


def test_v3_features_include_full_due_day_and_remove_unavailable_information():
    row = {
        **ORDER,
        "prediction_at": "2099-01-01",
        "label_available_at": "2099-01-01",
        "is_late": 1,
        "order_delivered_customer_date": "2099-01-01",
    }
    features = build_corrected_prediction_features(pd.DataFrame([row]))
    assert features.estimated_window_days.iloc[0] == pytest.approx(9.5625)
    assert features.remaining_window_days_at_approval.iloc[0] == pytest.approx(
        9.5208333333
    )
    forbidden = set(PAYMENT_FIELDS) | {
        "prediction_at",
        "label_available_at",
        "is_late",
        "purchase_year",
        "purchase_month",
        "purchase_weekofyear",
        "payment_price_difference",
        "order_delivered_customer_date",
    }
    assert forbidden.isdisjoint(features.columns)
    changed = {**row, "total_payment_value": 9999999, "payment_record_count": 999}
    pd.testing.assert_frame_equal(
        features, build_corrected_prediction_features(pd.DataFrame([changed]))
    )


def test_v3_bundle_rejects_mismatched_target_definition(tmp_path):
    bundle = make_bundle(tmp_path / "test.joblib")
    bundle["feature_metadata"]["target_definition"] = "old_timestamp_rule"
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_bundle(bundle)


def test_corrected_api_works_without_payment_fields_and_returns_contract(
    tmp_path, monkeypatch
):
    from Task3.app.main import app
    from Task3.app.dependencies import get_prediction_service, get_inference_bundle
    from fastapi.testclient import TestClient

    path = tmp_path / "bundle.joblib"
    bundle = make_bundle(path)
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(path))
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    app.dependency_overrides.clear()
    get_prediction_service.cache_clear()
    get_inference_bundle.cache_clear()
    client = TestClient(app)
    order = {key: value for key, value in ORDER.items() if key not in PAYMENT_FIELDS}
    response = client.post("/predict", json=order)
    assert response.status_code == 200
    assert response.json()["target_definition"] == TARGET_DEFINITION
    assert response.json()["prediction_stage"] == PREDICTION_STAGE
    assert client.get("/model/info").json()["target_definition"] == TARGET_DEFINITION
    assert bundle["feature_metadata"]["feature_policy"] == FEATURE_POLICY
    assert np.isfinite(response.json()["probability"])


def test_gx_contract_accepts_v3_orders_without_optional_payments():
    from Task3.src.gx_validator import validate_with_great_expectations

    row = {key: value for key, value in ORDER.items() if key not in PAYMENT_FIELDS}
    validate_with_great_expectations(pd.DataFrame([row]))
