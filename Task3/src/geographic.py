"""Static geographic lookups serialized with the fitted preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GeographicFeatures(TransformerMixin, BaseEstimator):
    """Estimate buyer-to-seller-state distance; never uses delivery outcomes.

    Seller-state medians approximate seller location; this is not shipping distance.
    Unknown postal prefixes remain missing and use the fitted downstream imputer.
    """

    OUTPUTS = (
        "customer_latitude",
        "customer_longitude",
        "seller_state_latitude",
        "seller_state_longitude",
        "approx_route_km",
        "window_per_1000km",
    )

    def __init__(self, zip_coordinates, state_coordinates):
        self.zip_coordinates = zip_coordinates
        self.state_coordinates = state_coordinates

    def fit(self, frame, y=None):
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        return self

    def transform(self, frame):
        check_is_fitted(self)
        result = frame.copy()
        customer = frame["customer_zip_code_prefix"].map(self.zip_coordinates)
        seller = frame["seller_state"].map(self.state_coordinates)
        for name, pairs, axis in [
            ("customer_latitude", customer, 0),
            ("customer_longitude", customer, 1),
            ("seller_state_latitude", seller, 0),
            ("seller_state_longitude", seller, 1),
        ]:
            result[name] = pairs.map(
                lambda pair: pair[axis] if isinstance(pair, tuple) else np.nan
            )
        lat1, lat2 = np.radians(result["customer_latitude"]), np.radians(
            result["seller_state_latitude"]
        )
        dlat = lat2 - lat1
        dlon = np.radians(
            result["seller_state_longitude"] - result["customer_longitude"]
        )
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        result["approx_route_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
        result["window_per_1000km"] = result["estimated_window_days"] / (
            1.0 + result["approx_route_km"] / 1000.0
        )
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.concatenate([self.feature_names_in_, self.OUTPUTS])
