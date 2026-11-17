# Corrected experiment: calendar-day v3

## Contract

`is_late = 1` only when the actual delivery's **Sao Paulo local calendar date** is later than the estimated local calendar date. Arrival at 23:59 on the promised day is on time; arrival at the following midnight is late. Prediction is after approval and assumes item/product/seller data is available then. Untimed payment aggregates and absolute purchase year/month/week are excluded. The feature window includes the full promised date, with next midnight as the exclusive endpoint. Label-availability timestamps, target values and delivery/review outcomes cannot enter features.

Historical v1/v2 bundles keep their original feature behavior. The predictor chooses v3 feature construction only from validated versioned bundle metadata. Do not relabel old predictions as if they used the new target.

## As-of evaluation

- 96,395 eligible delivered orders; 6,533 calendar-day late labels. 3,046 source orders are excluded for status, missing required dates or invalid approval/delivery chronology.
- Rolling folds: March 1–April 1 and April 1–June 1, 2018. Each training pool uses only approvals and deliveries strictly before its start. Fold validation outcomes must be known before June 1.
- First fold: 53,618 training / 7,076 validation rows. Second: 60,442 / 11,180. Unavailable training outcomes purged: 3,519 and 3,778 respectively; validation outcomes unavailable by selection cutoff: 7 and 2,384.
- Candidate models and thresholds are compared using pooled forward validation F1, with average precision as tie-breaker. Per-fold metrics are retained because prevalence differs sharply. Grid: 2 logistic, 2 HGB, 4 CatBoost configurations. No test-based selection or early stopping.
- Final refit: 75,388 orders with approval and observed delivery before June 1. This may include mature validation outcomes, but no test orders.
- Test: 18,611 approvals from June 1 to September 1 exclusive, 671 positives. Test outcomes are observed retrospectively through October 17; they never enter pre-June fitting/selection.

This existing dataset was already examined during earlier work. These are exploratory retrospective results, not a fresh blind confirmation. The new target and split invalidate direct performance comparisons to the old timestamp task.

## Results on the corrected task

Selected by rolling validation: **balanced Logistic Regression**, version `task2-v3-calendar-day`, threshold `0.6356438452043396`. This threshold is an experimental F1 choice, not an operational cost decision. Scores are not calibrated probabilities.

| Metric | Retrained logistic baseline | Selected balanced logistic |
|---|---:|---:|
| precision | 0.0485 | 0.0556 |
| recall | 0.4247 | 0.3234 |
| f1 | 0.0871 | 0.0949 |
| roc_auc | 0.6338 | 0.6292 |
| average_precision | 0.0612 | 0.0603 |

The selected model produces 217 true positives, 3,685 false positives and 454 false negatives. It remains unsuitable for production decisions. Correcting the experiment is not evidence that prediction accuracy is now good. Short promises and August performance remain weak; see the JSON subgroup tables. Do not tune another threshold on these test outcomes to create an apparent gain.

## Reproduction and serving

1. Download the official Olist CSVs linked in Task1/README.md and run Task1 ingestion / Task2 Notebook 1 to produce `01_joined_ml_table.parquet`.
2. Install Task3/requirements-dev.txt and run `python -m Task2.src.corrected_training` from the repository root. Notebooks 2–6 now reference this corrected workflow. Historical artifacts remain separate; new tables/reports/models live in `Task2/artifacts/corrected/`.
3. The command replaces the canonical local inference bundle only after the experiment and bundle checks succeed. It retains the previous timestamp bundle locally as `06_legacy_timestamp_v2.joblib`. The compatibility command `python -m Task2.src.improve_model` delegates to the corrected experiment.
4. Run `dvc repro Task3/dvc.yaml`, then restart the API and verify `/model/info`: version, target definition and prediction stage must match v3. Never reuse a live process's old cached bundle after changing files.
5. Use `Olist_Calendar_Day_V3_Artifacts.zip` to restore the exact candidate, corrected baseline and DVC cache; raw data is not included. A DVC remote still needs the owner's storage configuration.

Real API verification: 100 eligible new-test orders, readiness/single/batch 200, exact parity with direct predictions, correct target/stage metadata, omitted payment fields accepted, and arbitrary optional payment values do not affect v3 scores. See `Task3/docs/v3_api_verification.json`.

The population remains delivered orders with usable observed timestamps. Unresolved/cancelled orders are excluded. Historical product/seller snapshots and operational costs are unavailable. A later untouched dataset and a stated false-alert/missed-delay cost are required before any production acceptance.
