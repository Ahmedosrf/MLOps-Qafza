"""Prediction-input validation aligned with the Task 2 feature contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "order_approved_at",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "num_items",
    "num_distinct_products",
    "total_price",
    "total_freight",
    "num_sellers",
    "num_product_categories",
    "avg_product_weight_g",
    "avg_product_photos_qty",
    "seller_state_count",
    "payment_record_count",
    "total_payment_value",
    "max_payment_installments",
    "seller_state",
}

NON_NEGATIVE_COLUMNS = {
    "customer_zip_code_prefix",
    "num_items",
    "num_distinct_products",
    "total_price",
    "total_freight",
    "num_sellers",
    "num_product_categories",
    "avg_product_weight_g",
    "avg_product_photos_qty",
    "seller_state_count",
    "payment_record_count",
    "total_payment_value",
    "max_payment_installments",
}

POSITIVE_COUNT_COLUMNS = {
    "num_items",
    "num_distinct_products",
    "num_sellers",
    "num_product_categories",
    "seller_state_count",
    "payment_record_count",
    "max_payment_installments",
}


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Raise ValueError when prediction input violates the data expectations."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"columns are missing: {sorted(missing)}")

    if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("prediction input contains missing values")

    for column in NON_NEGATIVE_COLUMNS:
        if (pd.to_numeric(frame[column], errors="coerce") < 0).any():
            raise ValueError(f"{column} must be non-negative")

    for column in POSITIVE_COUNT_COLUMNS:
        if (pd.to_numeric(frame[column], errors="coerce") <= 0).any():
            raise ValueError(f"{column} must be greater than zero")

    for column in ("customer_state", "seller_state"):
        values = frame[column].astype(str).str.strip()
        if (values.str.len() != 2).any():
            raise ValueError(f"{column} must contain two-letter state codes")

    for column in (
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_approved_at",
    ):
        if pd.to_datetime(frame[column], errors="coerce").isna().any():
            raise ValueError(f"{column} contains an invalid date")


def validation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a small summary suitable for logs and health diagnostics."""

    validate_prediction_frame(frame)
    return {"rows": int(len(frame)), "columns": int(len(frame.columns)), "valid": True}


__all__ = ["validate_prediction_frame", "validation_summary"]
