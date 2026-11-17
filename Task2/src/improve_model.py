"""Reproducible temporal model comparison; test is evaluated after selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from Task3.src.geographic import GeographicFeatures

from Task3.src.feature_builder import build_prediction_features
from Task3.src.model_loader import find_project_root


def seller_states(raw_dir: Path) -> pd.Series:
    """Use the state of the first order item, a deterministic pre-delivery field."""
    items = pd.read_csv(
        raw_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "order_item_id", "seller_id"],
    )
    sellers = pd.read_csv(
        raw_dir / "olist_sellers_dataset.csv", usecols=["seller_id", "seller_state"]
    )
    first_items = items.sort_values(["order_id", "order_item_id"]).drop_duplicates(
        "order_id"
    )
    return first_items.merge(sellers, on="seller_id", validate="many_to_one").set_index(
        "order_id"
    )["seller_state"]


def best_f1_threshold(target, probabilities):
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    index = int(np.argmax(scores))
    return float(np.clip(thresholds[index], 1e-6, 1 - 1e-6))


def metrics(target, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
    }


def train(root: Path) -> dict:
    artifacts = root / "Task2/artifacts"
    model_dir, reports = artifacts / "models", artifacts / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    train_frame = pd.read_parquet(artifacts / "tables/03_train.parquet")
    validation_frame = pd.read_parquet(artifacts / "tables/03_validation.parquet")
    states = seller_states(root / "Task1/data")
    for frame in (train_frame, validation_frame):
        frame["seller_state"] = frame["order_id"].map(states)
    x_train = build_prediction_features(train_frame)
    x_validation = build_prediction_features(validation_frame)
    geo = pd.read_csv(
        root / "Task1/data/olist_geolocation_dataset.csv",
        usecols=[
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_state",
        ],
    )
    # Static source reference, no targets or delivery timestamps are used here.
    geo = geo[
        geo["geolocation_lat"].between(-34, 6)
        & geo["geolocation_lng"].between(-74, -34)
    ]
    zip_medians = geo.groupby("geolocation_zip_code_prefix")[
        ["geolocation_lat", "geolocation_lng"]
    ].median()
    state_medians = geo.groupby("geolocation_state")[
        ["geolocation_lat", "geolocation_lng"]
    ].median()
    zip_coordinates = {int(key): tuple(value) for key, value in zip_medians.iterrows()}
    state_coordinates = {key: tuple(value) for key, value in state_medians.iterrows()}
    geography = GeographicFeatures(zip_coordinates, state_coordinates).fit(x_train)
    x_train = geography.transform(x_train)
    x_validation = geography.transform(x_validation)
    numeric = x_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [name for name in x_train if name not in numeric]
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    min_frequency=20,
                    max_categories=254,
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_validation = preprocessor.transform(x_validation)
    y_train = train_frame["is_late"].to_numpy()
    y_validation = validation_frame["is_late"].to_numpy()
    categorical_mask = [False] * len(numeric) + [True] * len(categorical)
    candidates = []
    selected = None
    best_score = (-1.0, -1.0)
    # Fixed candidate grids. Each candidate and threshold is compared on validation only.
    for leaves in (15, 31):
        for regularization in (1.0, 10.0):
            for iterations in (150, 300):
                params = dict(
                    max_leaf_nodes=leaves,
                    l2_regularization=regularization,
                    max_iter=iterations,
                    learning_rate=0.05,
                )
                started = time.perf_counter()
                model = HistGradientBoostingClassifier(
                    **params,
                    categorical_features=categorical_mask,
                    early_stopping=False,
                    random_state=42,
                )
                model.fit(transformed_train, y_train)
                probabilities = model.predict_proba(transformed_validation)[:, 1]
                threshold = best_f1_threshold(y_validation, probabilities)
                values = metrics(y_validation, probabilities, threshold)
                candidate = {
                    "params": params,
                    "validation": values,
                    "fit_seconds": round(time.perf_counter() - started, 2),
                }
                candidates.append(candidate)
                print(json.dumps(candidate), flush=True)
                score = (values["f1"], values["average_precision"])
                if score > best_score:
                    best_score = score
                    selected = (model, threshold, params, values, preprocessor)
    from catboost import CatBoostClassifier

    cat_preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    cat_train = cat_preprocessor.fit_transform(x_train)
    cat_validation = cat_preprocessor.transform(x_validation)
    cat_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    for depth in (5, 7):
        for weight in (1.0, 4.0):
            params = dict(
                iterations=1000,
                depth=depth,
                learning_rate=0.04,
                l2_leaf_reg=10.0,
                class_weights=[1.0, weight],
            )
            started = time.perf_counter()
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                thread_count=4,
                verbose=False,
                allow_writing_files=False,
                cat_features=cat_indices,
            )
            model.fit(
                cat_train,
                y_train,
                eval_set=(cat_validation, y_validation),
                early_stopping_rounds=100,
            )
            probabilities = model.predict_proba(cat_validation)[:, 1]
            threshold = best_f1_threshold(y_validation, probabilities)
            values = metrics(y_validation, probabilities, threshold)
            params["fitted_tree_count"] = model.tree_count_
            candidate = {
                "family": "catboost",
                "params": params,
                "validation": values,
                "fit_seconds": round(time.perf_counter() - started, 2),
            }
            candidates.append(candidate)
            print(json.dumps(candidate), flush=True)
            score = (values["f1"], values["average_precision"])
            if score > best_score:
                best_score = score
                selected = (model, threshold, params, values, cat_preprocessor)
    model, threshold, params, validation_metrics, columns_preprocessor = selected
    preprocessor = Pipeline(
        [("geography", geography), ("columns", columns_preprocessor)]
    )
    # Selection is frozen before opening the existing held-out test split.
    test_frame = pd.read_parquet(artifacts / "tables/03_test.parquet")
    test_frame["seller_state"] = test_frame["order_id"].map(states)
    test_features = build_prediction_features(test_frame)
    probabilities = model.predict_proba(preprocessor.transform(test_features))[:, 1]
    test_metrics = metrics(test_frame["is_late"].to_numpy(), probabilities, threshold)
    baseline_path = model_dir / "06_baseline_task2_v1.joblib"
    previous = joblib.load(
        baseline_path
        if baseline_path.exists()
        else model_dir / "06_inference_bundle.joblib"
    )
    if previous.get("model_version") != "task2-v1":
        raise ValueError(
            "Expected baseline task2-v1; restore its artifact before rerunning comparison"
        )
    baseline_features = build_prediction_features(test_frame)
    baseline_probabilities = previous["model"].predict_proba(
        previous["preprocessor"].transform(baseline_features)
    )[:, 1]
    baseline_metrics = metrics(
        test_frame["is_late"].to_numpy(), baseline_probabilities, previous["threshold"]
    )
    bundle = dict(
        model=model,
        preprocessor=preprocessor,
        feature_names=list(preprocessor.get_feature_names_out()),
        threshold=threshold,
        model_name="olist-late-delivery",
        model_version="task2-v2-" + type(model).__name__.lower(),
        feature_metadata={
            "raw_feature_columns": list(geography.feature_names_in_),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "seller_state_policy": "State of seller for smallest order_item_id",
            "fit_rule": "Fit on chronological training split only; geography is static Olist reference data",
            "selection_rule": "Maximize validation F1; break ties with average precision",
        },
    )
    result = dict(
        selected_family=type(model).__name__,
        baseline_test=baseline_metrics,
        validation_candidates=candidates,
        selected_params=params,
        validation=validation_metrics,
        test=test_metrics,
        split_rows={
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        selection_metric="validation_f1",
        feature_count=len(bundle["feature_names"]),
        python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
        limitations=[
            "Test split was previously reported for baseline and initial HGB search; this is not a new blind holdout.",
            "Threshold and hyperparameters were selected on validation only; test results did not alter selection.",
            "Observed probabilities are classifier scores; no calibration claim is made.",
        ],
    )
    # Preserve baseline before overwriting the serving artifact.
    joblib.dump(previous, model_dir / "06_baseline_task2_v1.joblib")
    joblib.dump(bundle, model_dir / "06_inference_bundle.joblib")
    joblib.dump(model, model_dir / "06_model.joblib")
    result["bundle_sha256"] = hashlib.sha256(
        (model_dir / "06_inference_bundle.joblib").read_bytes()
    ).hexdigest()
    (reports / "07_improvement_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (model_dir / "06_model_metadata.json").write_text(
        json.dumps(
            {
                "model_version": bundle["model_version"],
                "model_type": type(model).__name__,
                "threshold": threshold,
                "feature_count": len(bundle["feature_names"]),
                "test_metrics": test_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"baseline": baseline_metrics, "improved": test_metrics, "params": params},
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    train(args.project_root or find_project_root())
