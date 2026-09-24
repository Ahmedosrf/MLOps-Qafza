import pandas as pd
import pytest

from Task3.src.feature_builder import build_prediction_features


def sample_order_frame():
    return pd.DataFrame(
        {
            "order_id": ["order-1"],
            "customer_id": ["customer-1"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": ["2018-01-06 10:30:00"],
            "order_approved_at": ["2018-01-06 11:30:00"],
            "order_estimated_delivery_date": ["2018-01-15"],
            "order_delivered_carrier_date": ["2018-01-08"],
            "order_delivered_customer_date": ["2018-01-20"],
            "avg_review_score": [1.0],
            "review_record_count": [1],
            "total_price": [100.0],
            "total_freight": [20.0],
            "total_payment_value": [120.0],
            "num_items": [2],
            "customer_state": ["SP"],
            "seller_state": ["RJ"],
            "is_late": [1],
        }
    )


def test_build_prediction_features_creates_expected_derived_values():
    features = build_prediction_features(sample_order_frame())

    assert features.loc[0, "approval_hours"] == pytest.approx(1.0)
    assert features.loc[0, "estimated_window_days"] == pytest.approx(8.5625)
    assert features.loc[0, "freight_to_price_ratio"] == pytest.approx(0.2)
    assert features.loc[0, "price_per_item"] == pytest.approx(50.0)
    assert features.loc[0, "freight_per_item"] == pytest.approx(10.0)
    assert features.loc[0, "payment_price_difference"] == pytest.approx(20.0)
    assert features.loc[0, "is_weekend"] == 1


def test_build_prediction_features_removes_identifiers_and_leakage_columns():
    features = build_prediction_features(sample_order_frame())

    forbidden = {
        "order_id",
        "customer_id",
        "order_status",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "avg_review_score",
        "review_record_count",
        "is_late",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    }
    assert forbidden.isdisjoint(features.columns)


def test_build_prediction_features_does_not_mutate_input():
    source = sample_order_frame()
    before = source.copy(deep=True)

    build_prediction_features(source)

    pd.testing.assert_frame_equal(source, before)


def test_build_prediction_features_rejects_non_dataframe_input():
    with pytest.raises(TypeError):
        build_prediction_features({"total_price": [10]})
