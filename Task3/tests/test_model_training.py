"""Tests for threshold selection and prediction-time seller enrichment."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from Task2.src.improve_model import best_f1_threshold, seller_states
from Task3.src.feature_builder import build_prediction_features


def test_selected_threshold_maximizes_validation_f1():
    target = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0.2, 0.45, 0.3, 0.8, 0.05, 0.1])
    threshold = best_f1_threshold(target, scores)
    assert f1_score(target, scores >= threshold) == max(
        f1_score(target, scores >= candidate) for candidate in scores
    )


def test_seller_state_uses_first_item_not_file_order(tmp_path):
    pd.DataFrame(
        {
            "order_id": ["a", "a", "b"],
            "order_item_id": [2, 1, 1],
            "seller_id": ["second", "first", "second"],
        }
    ).to_csv(tmp_path / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame(
        {"seller_id": ["first", "second"], "seller_state": ["SP", "RJ"]}
    ).to_csv(tmp_path / "olist_sellers_dataset.csv", index=False)
    assert seller_states(tmp_path).to_dict() == {"a": "SP", "b": "RJ"}


def test_route_features_do_not_use_outcome_fields():
    rows = pd.DataFrame(
        {
            "customer_state": ["SP", "SP"],
            "seller_state": ["RJ", "SP"],
            "is_late": [1, 0],
            "order_delivered_customer_date": ["2099-01-01", "2099-01-01"],
        }
    )
    features = build_prediction_features(rows)
    assert features["shipping_route"].tolist() == ["RJ_SP", "SP_SP"]
    assert features["same_state_route"].tolist() == [0, 1]
    assert "is_late" not in features and "order_delivered_customer_date" not in features


def test_geographic_distance_and_unknown_prefix_are_explicit():
    from Task3.src.geographic import GeographicFeatures

    frame = pd.DataFrame(
        {
            "customer_zip_code_prefix": [10000, 99999],
            "seller_state": ["SP", "SP"],
            "estimated_window_days": [10.0, 10.0],
        }
    )
    original = frame.copy(deep=True)
    transformer = GeographicFeatures(
        {10000: (-23.5, -46.6)}, {"SP": (-23.5, -46.6)}
    ).fit(frame)
    result = transformer.transform(frame)
    assert result.loc[0, "approx_route_km"] == 0
    assert np.isnan(result.loc[1, "approx_route_km"])
    assert list(transformer.get_feature_names_out()) == list(result)
    pd.testing.assert_frame_equal(frame, original)


def test_saved_catboost_feature_count_validates_after_round_trip(tmp_path):
    import joblib
    from catboost import CatBoostClassifier
    from Task3.src.model_loader import _validate_bundle
    from Task3.tests.test_prediction_service import FakePreprocessor

    model = CatBoostClassifier(
        iterations=2, verbose=False, allow_writing_files=False, thread_count=1
    )
    model.fit(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]), [0, 1, 0, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    bundle = {
        "model": restored,
        "preprocessor": FakePreprocessor(),
        "feature_names": ["price", "items"],
        "threshold": 0.3,
        "model_name": "test",
        "model_version": "test",
    }
    assert _validate_bundle(bundle).model.is_fitted()
    bundle["feature_names"] = ["one_feature_only"]
    import pytest

    with pytest.raises(ValueError, match="feature count"):
        _validate_bundle(bundle)
