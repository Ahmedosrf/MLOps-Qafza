"""Calendar-day labels and conservative as-of temporal cohorts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import pandas as pd
from Task3.src.dates import local_order_dates
from Task3.src.contract import TARGET_DEFINITION, PREDICTION_STAGE

TEST_START = pd.Timestamp("2018-06-01")
TEST_END = pd.Timestamp("2018-09-01")
ROLLING_FOLDS = (
    (pd.Timestamp("2018-03-01"), pd.Timestamp("2018-04-01")),
    (pd.Timestamp("2018-04-01"), TEST_START),
)


def calendar_day_labels(joined: pd.DataFrame) -> pd.DataFrame:
    frame = joined.copy()
    dates = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ]
    for column in dates:
        frame[column] = local_order_dates(frame[column])
    valid = (
        (frame.order_status == "delivered")
        & frame[dates].notna().all(axis=1)
        & (frame.order_approved_at >= frame.order_purchase_timestamp)
        & (frame.order_delivered_customer_date >= frame.order_approved_at)
    )
    frame = frame.loc[valid].copy()
    frame["is_late"] = (
        frame.order_delivered_customer_date.dt.normalize()
        > frame.order_estimated_delivery_date.dt.normalize()
    ).astype("int8")
    frame["prediction_at"] = frame.order_approved_at
    # Conservative finalization: use actual delivery for both classes, including late.
    frame["label_available_at"] = frame.order_delivered_customer_date
    return frame.sort_values(["prediction_at", "order_id"]).reset_index(drop=True)


def asof_training(frame: pd.DataFrame, cutoff) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    return frame.loc[
        (frame.prediction_at < cutoff) & (frame.label_available_at < cutoff)
    ].copy()


def evaluation_block(
    frame: pd.DataFrame, start, end, known_before=None
) -> pd.DataFrame:
    selected = (frame.prediction_at >= pd.Timestamp(start)) & (
        frame.prediction_at < pd.Timestamp(end)
    )
    if known_before is not None:
        selected &= frame.label_available_at < pd.Timestamp(known_before)
    return frame.loc[selected].copy()


def prepare(root: Path) -> tuple[pd.DataFrame, dict]:
    joined = pd.read_parquet(root / "Task2/artifacts/tables/01_joined_ml_table.parquet")
    frame = calendar_day_labels(joined)
    folder = root / "Task2/artifacts/corrected/tables"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "02_labeled_table.parquet", index=False)
    first_start = ROLLING_FOLDS[0][0]
    # Disjoint cohorts for EDA; final serving refit uses all labels known before TEST_START.
    cohorts = {
        "train": asof_training(frame, first_start),
        "validation": evaluation_block(frame, first_start, TEST_START, TEST_START),
        "test": evaluation_block(frame, TEST_START, TEST_END),
    }
    for name, data in cohorts.items():
        data.to_parquet(folder / f"03_{name}.parquet", index=False)
    folds = []
    for start, end in ROLLING_FOLDS:
        train = asof_training(frame, start)
        validation = evaluation_block(frame, start, end, TEST_START)
        assert train.label_available_at.max() < start
        assert train.prediction_at.max() < start
        assert validation.label_available_at.max() < TEST_START
        assert set(train.order_id).isdisjoint(validation.order_id)
        folds.append(
            {
                "start": str(start),
                "end": str(end),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "purged_training_labels": int(
                    (
                        (frame.prediction_at < start)
                        & (frame.label_available_at >= start)
                    ).sum()
                ),
                "purged_validation_labels": int(
                    (
                        (frame.prediction_at >= start)
                        & (frame.prediction_at < end)
                        & (frame.label_available_at >= TEST_START)
                    ).sum()
                ),
            }
        )
    final = asof_training(frame, TEST_START)
    test = cohorts["test"]
    assert set(final.order_id).isdisjoint(test.order_id)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "Task1/data").glob("*.csv"))
    }
    joined_sha = hashlib.sha256(
        (root / "Task2/artifacts/tables/01_joined_ml_table.parquet").read_bytes()
    ).hexdigest()
    manifest = {
        "source_csv_sha256": source_hashes,
        "joined_artifact_sha256": joined_sha,
        "target_definition": TARGET_DEFINITION,
        "prediction_stage": PREDICTION_STAGE,
        "label_availability_policy": "actual_delivery_before_cutoff_for_both_classes",
        "raw_rows": len(joined),
        "eligible_rows": len(frame),
        "late_count": int(frame.is_late.sum()),
        "excluded_rows": len(joined) - len(frame),
        "folds": folds,
        "final_fit_cutoff": str(TEST_START),
        "final_fit_rows": len(final),
        "final_fit_late_count": int(final.is_late.sum()),
        "test_start": str(TEST_START),
        "test_end_exclusive": str(TEST_END),
        "test_rows": len(test),
        "test_late_count": int(test.is_late.sum()),
        "test_label_observation_end": str(test.label_available_at.max()),
        "final_fit_label_max": str(final.label_available_at.max()),
        "limitations": [
            "Delivered orders with valid chronological timestamps only; unresolved/cancelled population excluded.",
            "Test outcomes are scored retrospectively after delivery, not available for training at TEST_START.",
            "Existing June-August data was previously inspected; evaluation is exploratory, not blind.",
        ],
    }
    output = root / "Task2/artifacts/corrected/reports"
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return frame, manifest
