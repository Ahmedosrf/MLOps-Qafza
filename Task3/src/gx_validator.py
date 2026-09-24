"""Great Expectations adapter for prediction-time data checks."""

from __future__ import annotations

import pandas as pd

from .validation import validate_prediction_frame


def validate_with_great_expectations(frame: pd.DataFrame) -> None:
    """Run the local contract and a Great Expectations validation suite.

    The local contract provides stable error messages for the API. Great
    Expectations adds the auditable expectation objects used by data tests.
    """

    validate_prediction_frame(frame)
    try:
        import great_expectations as gx
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise RuntimeError(
            "Great Expectations is required for the data-validation command"
        ) from exc

    dataset = gx.from_pandas(frame)
    for column in frame.columns:
        dataset.expect_column_values_to_not_be_null(column)
    for column in (
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
    ):
        dataset.expect_column_values_to_be_between(column, min_value=0)

    result = dataset.validate()
    if not result["success"]:
        raise ValueError("Great Expectations validation failed")


__all__ = ["validate_with_great_expectations"]
