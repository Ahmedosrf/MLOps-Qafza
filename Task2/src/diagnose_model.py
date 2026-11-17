"""Read-only diagnosis of a frozen Olist classifier and its temporal splits.

Reports descriptive associations. Does not fit models, calibrate scores, tune
thresholds or alter labels. Requires the Task2 artifacts and official Task1 CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from Task2.src.improve_model import metrics, seller_states
from Task3.src.feature_builder import build_prediction_features
from Task3.src.model_loader import find_project_root


def grouped_errors(frame: pd.DataFrame, column: str, min_rows: int = 100) -> list[dict]:
    """Show sufficiently supported groups; rates on small groups are unstable."""
    rows = []
    for name, group in frame.groupby(column, observed=True):
        if len(group) < min_rows:
            continue
        predicted = group["predicted_late"].to_numpy()
        target = group["is_late"].to_numpy()
        tp = int(((predicted == 1) & (target == 1)).sum())
        fp = int(((predicted == 1) & (target == 0)).sum())
        fn = int(((predicted == 0) & (target == 1)).sum())
        rows.append(
            {
                "group": str(name),
                "rows": len(group),
                "late_count": int(target.sum()),
                "late_rate": float(target.mean()),
                "alert_rate": float(predicted.mean()),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    return sorted(
        rows,
        key=lambda row: row["false_positives"] + row["false_negatives"],
        reverse=True,
    )


def numeric_psi(reference, compared, bins: int = 10) -> float:
    """Descriptive PSI using train-defined bins, not a causal/significance test."""
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    compared = pd.to_numeric(compared, errors="coerce").dropna().to_numpy()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2 or not len(compared):
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.maximum(np.histogram(reference, edges)[0] / len(reference), 1e-6)
    b = np.maximum(np.histogram(compared, edges)[0] / len(compared), 1e-6)
    return float(np.sum((b - a) * np.log(b / a)))


def diagnose(root: Path, output: Path) -> dict:
    artifacts = root / "Task2/artifacts"
    bundle_path = artifacts / "models/06_inference_bundle.joblib"
    bundle = joblib.load(bundle_path)
    states = seller_states(root / "Task1/data")
    frames, raw_features, split_results = {}, {}, {}
    reference_prevalence = None
    for name in ("train", "validation", "test"):
        frame = pd.read_parquet(artifacts / f"tables/03_{name}.parquet")
        frame["seller_state"] = frame["order_id"].map(states)
        features = build_prediction_features(frame)
        x = bundle["preprocessor"].transform(features)
        probabilities = bundle["model"].predict_proba(x)[:, 1]
        frame["probability"] = probabilities
        frame["predicted_late"] = (probabilities >= bundle["threshold"]).astype(int)
        purchase = pd.to_datetime(frame["order_purchase_timestamp"])
        frame["purchase_month"] = purchase.dt.to_period("M").astype(str)
        frame["shipping_route"] = frame["seller_state"] + "_" + frame["customer_state"]
        frame["estimated_window_bucket"] = pd.cut(
            features["estimated_window_days"], [-np.inf, 7, 14, 21, 30, np.inf]
        ).astype(str)
        target = frame["is_late"].to_numpy()
        if reference_prevalence is None:
            reference_prevalence = float(target.mean())
        values = metrics(target, probabilities, bundle["threshold"])
        values.update(
            rows=len(frame),
            late_rate=float(target.mean()),
            mean_score=float(probabilities.mean()),
            alert_rate=float(frame.predicted_late.mean()),
            brier_score=float(brier_score_loss(target, probabilities)),
            constant_train_rate_brier=float(
                brier_score_loss(target, np.full(len(frame), reference_prevalence))
            ),
            start=str(purchase.min()),
            end=str(purchase.max()),
        )
        split_results[name] = values
        frames[name], raw_features[name] = frame, features
    test = frames["test"]
    calibration = (
        test.assign(
            score_bin=pd.cut(
                test.probability,
                [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1],
                include_lowest=True,
            )
        )
        .groupby("score_bin", observed=True)
        .agg(
            rows=("is_late", "size"),
            mean_score=("probability", "mean"),
            actual_late_rate=("is_late", "mean"),
        )
        .reset_index()
    )
    calibration["score_bin"] = calibration["score_bin"].astype(str)
    calibration_gap = float(
        np.average(
            abs(calibration.mean_score - calibration.actual_late_rate),
            weights=calibration.rows,
        )
    )
    labels = {}
    sensitivity_frames = {}
    for name, frame in frames.items():
        actual = pd.to_datetime(frame.order_delivered_customer_date)
        due = pd.to_datetime(frame.order_estimated_delivery_date)
        same_day_flagged_late = (actual > due) & (
            actual.dt.normalize() == due.dt.normalize()
        )
        calendar_late = actual.dt.normalize() > due.dt.normalize()
        sensitivity = frame.copy()
        sensitivity["same_due_day_flagged_late"] = same_day_flagged_late.astype(int)
        sensitivity["calendar_late"] = calendar_late.astype(int)
        sensitivity_frames[name] = sensitivity
        labels[name] = {
            "timestamp_late_count": int(frame.is_late.sum()),
            "calendar_day_late_count": int(calendar_late.sum()),
            "same_due_day_flagged_late": int(same_day_flagged_late.sum()),
            "due_midnight_fraction": float((due == due.dt.normalize()).mean()),
            "approval_before_purchase_count": int(
                (
                    pd.to_datetime(frame.order_approved_at)
                    < pd.to_datetime(frame.order_purchase_timestamp)
                ).sum()
            ),
        }
    maturity = {}
    for source, future in (("train", "validation"), ("validation", "test")):
        cutoff = pd.to_datetime(frames[future].order_purchase_timestamp).min()
        delivery = pd.to_datetime(frames[source].order_delivered_customer_date)
        after = delivery >= cutoff
        maturity[source + "_at_" + future + "_start"] = {
            "cutoff": str(cutoff),
            "labels_finalized_at_or_after_cutoff": int(after.sum()),
            "fraction": float(after.mean()),
            "late_in_overlap": int(frames[source].loc[after, "is_late"].sum()),
        }
    columns = raw_features["train"].select_dtypes(include=np.number).columns
    shifts = [
        {
            "feature": column,
            "train_to_validation_psi": numeric_psi(
                raw_features["train"][column], raw_features["validation"][column]
            ),
            "train_to_test_psi": numeric_psi(
                raw_features["train"][column], raw_features["test"][column]
            ),
        }
        for column in columns
    ]
    shifts.sort(key=lambda row: row["train_to_test_psi"], reverse=True)
    unseen = {
        column: float(
            (
                ~raw_features["test"][column].isin(
                    set(raw_features["train"][column].dropna())
                )
            ).mean()
        )
        for column in (
            "customer_city",
            "customer_state",
            "seller_state",
            "shipping_route",
        )
    }
    original = pd.read_parquet(artifacts / "tables/01_joined_ml_table.parquet")
    for field in (
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ):
        original[field] = pd.to_datetime(original[field], errors="coerce")
    eligible = (
        (original.order_status == "delivered")
        & original.order_delivered_customer_date.notna()
        & original.order_estimated_delivery_date.notna()
    )
    feature_importances = sorted(
        [
            {"feature": name, "importance": float(value)}
            for name, value in zip(
                bundle["feature_names"], bundle["model"].feature_importances_
            )
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )
    geo = (
        bundle["preprocessor"].named_steps["geography"].transform(raw_features["test"])
    )
    sensitivity_groups = {}
    for field in ("purchase_month", "customer_state"):
        values = (
            sensitivity_frames["test"]
            .groupby(field, observed=True)
            .agg(
                rows=("is_late", "size"),
                timestamp_late=("is_late", "sum"),
                calendar_late=("calendar_late", "sum"),
                same_due_day=("same_due_day_flagged_late", "sum"),
            )
        )
        sensitivity_groups[field] = (
            values[values.rows >= 100].reset_index().to_dict(orient="records")
        )
    descriptive_numeric = {}
    for split, features in raw_features.items():
        descriptive_numeric[split] = {
            "estimated_window_min_days": float(features.estimated_window_days.min()),
            "estimated_window_max_days": float(features.estimated_window_days.max()),
            "estimated_window_median_days": float(
                features.estimated_window_days.median()
            ),
            "estimated_window_at_most_7days_rate": float(
                (features.estimated_window_days <= 7).mean()
            ),
            "freight_median": float(features.total_freight.median()),
        }
    report = {
        "model_version": bundle["model_version"],
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "policy": "Frozen bundle and threshold. No fitting, threshold selection, label change or promotion.",
        "split_metrics": split_results,
        "label_audit": labels,
        "test_label_sensitivity_groups": sensitivity_groups,
        "numeric_summaries": descriptive_numeric,
        "label_availability": maturity,
        "test_errors_by_purchase_month": grouped_errors(test, "purchase_month"),
        "test_errors_by_customer_state": grouped_errors(test, "customer_state"),
        "test_errors_by_seller_state": grouped_errors(test, "seller_state"),
        "test_errors_by_route": grouped_errors(test, "shipping_route"),
        "test_errors_by_estimated_window": grouped_errors(
            test, "estimated_window_bucket"
        ),
        "numeric_distribution_shift": shifts,
        "test_unseen_category_rates": unseen,
        "test_calibration_bins": calibration.to_dict(orient="records"),
        "test_weighted_calibration_gap": calibration_gap,
        "prediction_time_constraints": {
            "earliest_supported_stage": "after approval and availability of order items/payment aggregates",
            "approval_missing_rows": int(original.order_approved_at.isna().sum()),
            "payment_aggregate_timestamps_available": False,
            "note": "Payment rows contain no timestamps; historical prediction-time availability is unverified.",
        },
        "label_population": {
            "all_orders": len(original),
            "eligible_delivered_orders": int(eligible.sum()),
            "excluded_orders": int((~eligible).sum()),
            "excluded_by_status": original.loc[~eligible, "order_status"]
            .value_counts()
            .to_dict(),
        },
        "test_missing_customer_geo_rate": float(geo.customer_latitude.isna().mean()),
        "feature_importances": feature_importances,
        "limitations": [
            "Descriptive subgroup findings do not prove causes.",
            "Existing test data was inspected during previous experiments; not a blind confirmation.",
            "Calendar-day audit is a sensitivity check, not authorization to change business labels.",
            "Using actual delivery as label finalization is conservative; a late status can sometimes be known before final delivery.",
            "No observed business costs are supplied, so no operational threshold is prescribed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root or find_project_root()
    report = diagnose(root, args.output or root / "Task2/docs/model_diagnostics.json")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "split_metrics",
                    "label_audit",
                    "label_availability",
                    "test_unseen_category_rates",
                    "test_weighted_calibration_gap",
                )
            },
            indent=2,
        )
    )
