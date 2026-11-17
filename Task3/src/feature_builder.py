"""Build prediction-time features using the Task 2 feature policy.

This module deliberately contains feature construction only. It does not load
or fit a model, and it never uses post-delivery information as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dates import local_order_dates


POST_OUTCOME_COLUMNS = (
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "avg_review_score",
    "review_record_count",
)

IDENTIFIER_COLUMNS = (
    "is_late",
    "order_id",
    "customer_id",
    "customer_unique_id",
)

DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
)

ORIGINAL_DATE_COLUMNS = DATE_COLUMNS


def build_prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the same raw feature table used by Task 2 Notebook 5.

    Parameters
    ----------
    frame:
        One or more order rows containing the pre-delivery columns produced by
        Task 2 Notebook 1. A target column may be present for offline checks;
        it is always removed from the returned features.

    Returns
    -------
    pandas.DataFrame
        Prediction-time features before the fitted Task 2 transformer is
        applied. The input is not mutated.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    features = frame.copy()

    for column in DATE_COLUMNS:
        if column in features.columns:
            features[column] = local_order_dates(features[column])

    purchase_time = features.get("order_purchase_timestamp")
    if purchase_time is not None:
        features["purchase_year"] = purchase_time.dt.year
        features["purchase_month"] = purchase_time.dt.month
        features["purchase_dayofweek"] = purchase_time.dt.dayofweek
        features["purchase_hour"] = purchase_time.dt.hour

    if "order_approved_at" in features.columns and purchase_time is not None:
        features["approval_hours"] = (
            features["order_approved_at"] - purchase_time
        ).dt.total_seconds() / 3600

    if (
        "order_estimated_delivery_date" in features.columns
        and purchase_time is not None
    ):
        features["estimated_window_days"] = (
            features["order_estimated_delivery_date"] - purchase_time
        ).dt.total_seconds() / 86400

    if {"total_freight", "total_price"}.issubset(features.columns):
        features["freight_to_price_ratio"] = features["total_freight"] / features[
            "total_price"
        ].replace(0, np.nan)

    if {"total_price", "num_items"}.issubset(features.columns):
        features["price_per_item"] = features["total_price"] / features[
            "num_items"
        ].replace(0, np.nan)

    if {"total_freight", "num_items"}.issubset(features.columns):
        features["freight_per_item"] = features["total_freight"] / features[
            "num_items"
        ].replace(0, np.nan)

    if {"total_payment_value", "total_price"}.issubset(features.columns):
        features["payment_price_difference"] = (
            features["total_payment_value"] - features["total_price"]
        )

    if purchase_time is not None:
        features["purchase_weekofyear"] = purchase_time.dt.isocalendar().week.astype(
            "float64"
        )
        features["is_weekend"] = (purchase_time.dt.dayofweek >= 5).astype("int8")

    if {"customer_state", "seller_state"}.issubset(features.columns):
        features["same_state_route"] = (
            features["customer_state"] == features["seller_state"]
        ).astype("int8")
        features["shipping_route"] = (
            features["seller_state"].astype(str)
            + "_"
            + features["customer_state"].astype(str)
        )

    columns_to_remove = [
        *IDENTIFIER_COLUMNS,
        "order_status",
        *POST_OUTCOME_COLUMNS,
        *ORIGINAL_DATE_COLUMNS,
    ]

    return features.drop(
        columns=[column for column in columns_to_remove if column in features.columns]
    )


__all__ = [
    "DATE_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "POST_OUTCOME_COLUMNS",
    "build_prediction_features",
]
