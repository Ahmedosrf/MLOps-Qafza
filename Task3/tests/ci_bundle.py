"""A fitted, synthetic sklearn bundle for CI only. Never a production model."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from Task3.src.feature_builder import build_corrected_prediction_features
from Task3.src.contract import FEATURE_POLICY, TARGET_DEFINITION, PREDICTION_STAGE

ORDER = {
    "order_purchase_timestamp": "2018-01-06 10:30:00",
    "order_approved_at": "2018-01-06 11:30:00",
    "order_estimated_delivery_date": "2018-01-15",
    "customer_zip_code_prefix": 13000,
    "customer_city": "campinas",
    "customer_state": "SP",
    "seller_state": "RJ",
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
}


def make_bundle(path):
    rows = [{**ORDER, "total_price": float(price)} for price in range(20, 220, 10)]
    features = build_corrected_prediction_features(pd.DataFrame(rows))
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [name for name in features if name not in numeric]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    x = preprocessor.fit_transform(features)
    model = LogisticRegression(random_state=42).fit(x, [0, 1] * 10)
    bundle = dict(
        model=model,
        preprocessor=preprocessor,
        feature_names=list(preprocessor.get_feature_names_out()),
        threshold=0.5,
        model_name="ci-synthetic-only",
        model_version="ci-synthetic-only",
        feature_metadata={
            "feature_policy": FEATURE_POLICY,
            "target_definition": TARGET_DEFINITION,
            "prediction_stage": PREDICTION_STAGE,
        },
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


if __name__ == "__main__":
    import sys

    make_bundle(sys.argv[1])
