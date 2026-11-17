# Task 2 — Calendar-day delivery prediction (v3)

The current target treats arrival anywhere on the estimated **local calendar day** as on time. Prediction is after approval. The model is experimental and its predictive performance remains weak; infrastructure checks do not establish production quality.

See [the corrected experiment](docs/CORRECTED_EXPERIMENT.md), [results](docs/corrected_results.json), and [as-of data manifest](docs/corrected_data_manifest.json).

## Run

From the repository root, in the activated Python environment:

```bash
python -m pip install -r Task3/requirements-dev.txt
python -m Task2.src.corrected_training
dvc repro Task3/dvc.yaml
```

Prerequisites: official CSVs under `Task1/data/` and `Task2/artifacts/tables/01_joined_ml_table.parquet` produced by Notebook 1. The source is https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce .

## Notebooks

| Notebook | Current behavior |
|---|---|
| 01_read_and_join_tables | Aggregate many-to-one sources and preserve one row per order |
| 02_create_labels | Calendar-day labels and chronology checks |
| 03_split_data | Disjoint EDA cohorts plus rolling as-of manifest |
| 04_eda | Explore corrected cohorts; no model selection on test |
| 05_feature_engineering | Shared corrected raw-feature policy |
| 06_train_tune_evaluate | Execute the canonical corrected rolling experiment |

The EDA training cohort is before the first rolling cutoff. The final serving fit separately uses all outcomes known before June 1, including mature validation outcomes. The code asserts train/test disjointness and label availability at cutoffs.

## Feature and target contract

- Delivery date after the estimated local date means late; the full due day is included.
- Use seller state from the item with smallest `order_item_id`.
- Untimed payment aggregates, absolute year/month/week, targets, actual delivery/reviews and label-finalization timestamps are excluded.
- Every learned imputer/encoder/scaler is fitted within its training fold; static geolocation reference dictionaries contain no target and are serialized with preprocessing.
- Static geographic distance to a seller-state median is approximate, not exact seller/road distance. Product/seller attributes are assumed known at approval; historical snapshots are unavailable.

New artifacts are under `Task2/artifacts/corrected/`. The canonical serving source remains `Task2/artifacts/models/06_inference_bundle.joblib`. Baseline and candidate now share the same corrected target and evaluation protocol. Do not directly compare their metrics with historical v1/v2 reports. The existing test period has already been examined and is not blind.

`python -m Task2.src.improve_model` is a compatibility entry point for the corrected experiment. `python -m Task2.src.diagnose_model` diagnoses the current frozen bundle and uses corrected cohorts when its feature policy is v3. Historical timestamp reports remain under docs for provenance, not current acceptance evidence.
