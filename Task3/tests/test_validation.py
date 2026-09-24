import pandas as pd
import pytest

from Task3.src.validation import validate_prediction_frame


VALID = {
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


def test_valid_prediction_frame_passes():
    validate_prediction_frame(pd.DataFrame([VALID]))


def test_missing_required_column_is_rejected():
    data = {key: value for key, value in VALID.items() if key != "seller_state"}
    with pytest.raises(ValueError, match="columns are missing"):
        validate_prediction_frame(pd.DataFrame([data]))


def test_negative_value_is_rejected():
    data = {**VALID, "total_price": -1}
    with pytest.raises(ValueError, match="non-negative"):
        validate_prediction_frame(pd.DataFrame([data]))


def test_invalid_state_code_is_rejected():
    data = {**VALID, "customer_state": "São Paulo"}
    with pytest.raises(ValueError, match="two-letter"):
        validate_prediction_frame(pd.DataFrame([data]))
