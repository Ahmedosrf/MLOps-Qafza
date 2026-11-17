"""Train calendar-day classifiers with rolling as-of validation and a frozen test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from Task2.src.corrected_data import (
    prepare,
    asof_training,
    evaluation_block,
    ROLLING_FOLDS,
    TEST_START,
    TEST_END,
)
from Task2.src.improve_model import seller_states, best_f1_threshold, metrics
from Task3.src.contract import FEATURE_POLICY, TARGET_DEFINITION, PREDICTION_STAGE
from Task3.src.feature_builder import build_corrected_prediction_features
from Task3.src.geographic import GeographicFeatures
from Task3.src.model_loader import find_project_root, load_inference_bundle

CANDIDATES = [
    {"name": "logistic", "family": "logistic", "class_weight": None},
    {"name": "logistic_balanced", "family": "logistic", "class_weight": "balanced"},
    *[
        {"name": f"hgb_{leaves}", "family": "hgb", "leaves": leaves}
        for leaves in (15, 31)
    ],
    *[
        {
            "name": f"catboost_d{depth}_w{weight}",
            "family": "catboost",
            "depth": depth,
            "weight": weight,
        }
        for depth in (5, 7)
        for weight in (1, 4)
    ],
]


def geographic_reference(root: Path):
    geo = pd.read_csv(
        root / "Task1/data/olist_geolocation_dataset.csv",
        usecols=[
            "geolocation_zip_code_prefix",
            "geolocation_state",
            "geolocation_lat",
            "geolocation_lng",
        ],
    )
    geo = geo[
        geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)
    ]
    cols = ["geolocation_lat", "geolocation_lng"]
    zip_coords = {
        int(key): tuple(value)
        for key, value in geo.groupby("geolocation_zip_code_prefix")[cols]
        .median()
        .iterrows()
    }
    state_coords = {
        key: tuple(value)
        for key, value in geo.groupby("geolocation_state")[cols].median().iterrows()
    }
    return zip_coords, state_coords


def fit_candidate(candidate: dict, training: pd.DataFrame, reference):
    features = build_corrected_prediction_features(training)
    geographic = GeographicFeatures(*reference).fit(features)
    augmented = geographic.transform(features)
    numeric = augmented.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in augmented if column not in numeric]
    family = candidate["family"]
    if family == "logistic":
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
        model = LogisticRegression(
            C=1.0,
            class_weight=candidate["class_weight"],
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )
    elif family == "hgb":
        numeric_pipe = SimpleImputer(strategy="median")
        categorical_pipe = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            min_frequency=20,
            max_categories=254,
        )
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=candidate["leaves"],
            learning_rate=0.05,
            l2_regularization=10.0,
            random_state=42,
            early_stopping=False,
            categorical_features=[False] * len(numeric) + [True] * len(categorical),
        )
    else:
        from catboost import CatBoostClassifier

        numeric_pipe = SimpleImputer(strategy="median")
        categorical_pipe = SimpleImputer(strategy="constant", fill_value="__missing__")
        model = CatBoostClassifier(
            iterations=400,
            depth=candidate["depth"],
            class_weights=[1.0, candidate["weight"]],
            learning_rate=0.04,
            l2_leaf_reg=10.0,
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
            loss_function="Logloss",
            cat_features=list(range(len(numeric), len(numeric) + len(categorical))),
        )
    columns = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        sparse_threshold=1.0 if family == "logistic" else 0.0,
    )
    x = columns.fit_transform(augmented)
    model.fit(x, training.is_late.to_numpy())
    preprocessor = Pipeline([("geography", geographic), ("columns", columns)])
    return model, preprocessor


def score(model, preprocessor, data):
    features = build_corrected_prediction_features(data)
    return model.predict_proba(preprocessor.transform(features))[
        :, list(model.classes_).index(1)
    ]


def run_experiment(root: Path) -> dict:
    frame, manifest = prepare(root)
    states = seller_states(root / "Task1/data")
    frame["seller_state"] = frame.order_id.map(states)
    reference = geographic_reference(root)
    folds = [
        (asof_training(frame, start), evaluation_block(frame, start, end, TEST_START))
        for start, end in ROLLING_FOLDS
    ]
    results = []
    for candidate in CANDIDATES:
        fold_predictions, fold_targets = [], []
        for train, validation in folds:
            fitted, transformer = fit_candidate(candidate, train, reference)
            fold_predictions.append(score(fitted, transformer, validation))
            fold_targets.append(validation.is_late.to_numpy())
        predictions, target = np.concatenate(fold_predictions), np.concatenate(
            fold_targets
        )
        threshold = best_f1_threshold(target, predictions)
        result = {
            "candidate": candidate,
            "rolling_validation": metrics(target, predictions, threshold),
            "fold_metrics": [
                metrics(y, p, threshold) for y, p in zip(fold_targets, fold_predictions)
            ],
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    # Everything below is fixed before reading the test cohort for scoring.
    selected = max(
        results,
        key=lambda r: (
            r["rolling_validation"]["f1"],
            r["rolling_validation"]["average_precision"],
        ),
    )
    baseline = next(row for row in results if row["candidate"]["name"] == "logistic")
    final_training = asof_training(frame, TEST_START)
    final_model, final_preprocessor = fit_candidate(
        selected["candidate"], final_training, reference
    )
    if selected["candidate"]["name"] == "logistic":
        baseline_model, baseline_preprocessor = final_model, final_preprocessor
    else:
        baseline_model, baseline_preprocessor = fit_candidate(
            baseline["candidate"], final_training, reference
        )
    test = evaluation_block(frame, TEST_START, TEST_END)
    assert final_training.label_available_at.max() < TEST_START
    assert set(final_training.order_id).isdisjoint(test.order_id)
    probability = score(final_model, final_preprocessor, test)
    baseline_probability = score(baseline_model, baseline_preprocessor, test)
    threshold = selected["rolling_validation"]["threshold"]
    test_metrics = metrics(test.is_late.to_numpy(), probability, threshold)
    baseline_metrics = metrics(
        test.is_late.to_numpy(),
        baseline_probability,
        baseline["rolling_validation"]["threshold"],
    )
    metadata = {
        "feature_policy": FEATURE_POLICY,
        "target_definition": TARGET_DEFINITION,
        "prediction_stage": PREDICTION_STAGE,
        "time_zone": "America/Sao_Paulo",
        "deadline_policy": "end_of_estimated_local_calendar_day; next midnight exclusive",
        "payment_policy": "excluded; historical availability not verified",
        "calendar_policy": "absolute year/month/week excluded; weekday/hour retained",
        "seller_state_policy": "state of seller for smallest order_item_id",
        "final_fit_cutoff": str(TEST_START),
        "label_availability_policy": manifest["label_availability_policy"],
        "scores_calibrated": False,
        "release_status": "experimental_not_production_ready",
    }
    bundle = dict(
        model=final_model,
        preprocessor=final_preprocessor,
        feature_names=list(final_preprocessor.get_feature_names_out()),
        threshold=threshold,
        model_name="olist-late-delivery",
        model_version="task2-v3-calendar-day",
        feature_metadata=metadata,
    )
    folder = root / "Task2/artifacts/corrected"
    (folder / "models").mkdir(exist_ok=True)
    path = folder / "models/06_inference_bundle.joblib"
    joblib.dump(bundle, path)
    load_inference_bundle(path)
    baseline_bundle = {
        **bundle,
        "model": baseline_model,
        "preprocessor": baseline_preprocessor,
        "feature_names": list(baseline_preprocessor.get_feature_names_out()),
        "threshold": baseline["rolling_validation"]["threshold"],
        "model_version": "task2-v3-calendar-day-baseline",
    }
    joblib.dump(baseline_bundle, folder / "models/baseline_bundle.joblib")
    subgroup = []
    working = test.copy()
    working["probability"] = probability
    working["month"] = working.prediction_at.dt.to_period("M").astype(str)
    for name, group in working.groupby("month"):
        subgroup.append(
            {
                "month": name,
                "rows": len(group),
                **metrics(group.is_late, group.probability, threshold),
            }
        )
    calendar_days = (
        test.order_estimated_delivery_date.dt.normalize()
        - test.order_purchase_timestamp.dt.normalize()
    ).dt.days
    short = calendar_days <= 7
    short_report = {
        "rows": int(short.sum()),
        "late_count": int(test.loc[short, "is_late"].sum()),
    }
    if short.any() and test.loc[short, "is_late"].nunique() == 2:
        short_report.update(
            metrics(
                test.loc[short, "is_late"], probability[short.to_numpy()], threshold
            )
        )
    report = {
        "contract": metadata,
        "data_manifest": manifest,
        "candidates": results,
        "selected_candidate": selected["candidate"],
        "selection_rule": "rolling_validation_f1_then_average_precision",
        "baseline_test": baseline_metrics,
        "candidate_test": test_metrics,
        "naive_constant_test": metrics(
            test.is_late.to_numpy(),
            np.full(len(test), final_training.is_late.mean()),
            0.5,
        ),
        "test_by_month": subgroup,
        "test_short_promises": short_report,
        "test_is_blind": False,
        "old_timestamp_metrics_comparable": False,
        "limitations": [
            "New target and as-of split; do not compare these metrics as direct gains over v1/v2.",
            "Existing data has been inspected previously; a future untouched sample is still required.",
            "Delivered-only population and conservative label finalization; no unresolved/cancelled classification.",
            "Operational costs were not supplied; threshold maximizes rolling validation F1.",
            "Final refit uses all outcomes known before test start, including mature validation outcomes.",
            "Item/product/seller attributes are assumed available at approval; historical attribute snapshots are unavailable.",
        ],
    }
    report["bundle_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "reports/corrected_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    # Only replace the serving artifact after the experiment, validation and metadata succeed.
    canonical = root / "Task2/artifacts/models/06_inference_bundle.joblib"
    if canonical.exists():
        previous = joblib.load(canonical)
        if previous.get("model_version") != bundle["model_version"]:
            shutil.copy2(canonical, canonical.parent / "06_legacy_timestamp_v2.joblib")
    shutil.copy2(path, canonical)
    (canonical.parent / "06_model_metadata.json").write_text(
        json.dumps(
            {
                "model_version": bundle["model_version"],
                "contract": metadata,
                "selected_candidate": selected["candidate"],
                "test_metrics": test_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "selected": selected["candidate"],
                "baseline_test": baseline_metrics,
                "candidate_test": test_metrics,
                "naive_constant_test": metrics(
                    test.is_late.to_numpy(),
                    np.full(len(test), final_training.is_late.mean()),
                    0.5,
                ),
                "manifest": manifest,
            },
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    run_experiment(args.project_root or find_project_root())
