"""Normalize timestamps to Olist's local wall clock in training and serving."""

import pandas as pd


def local_order_dates(values: pd.Series) -> pd.Series:
    """Naive dates are Sao Paulo local time; aware dates convert to that zone."""

    def normalize(value):
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("America/Sao_Paulo").tz_localize(None)
            return stamp
        except (ValueError, TypeError, OverflowError):
            return pd.NaT

    return pd.to_datetime(values.map(normalize), errors="coerce")
