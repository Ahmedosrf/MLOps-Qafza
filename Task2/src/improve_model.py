"""Reproducible temporal model comparison; test is evaluated after selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    """Compatibility entry point; all new training uses the corrected v3 contract."""
    from Task2.src.corrected_training import run_experiment

    return run_experiment(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    train(args.project_root or find_project_root())
