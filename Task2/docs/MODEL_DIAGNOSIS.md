# Frozen-model diagnosis

Model: `task2-v2-catboostclassifier`. This analysis does not train models, select thresholds, alter labels or promote a version. Exact artifact SHA-256 and all supporting tables are in `model_diagnostics.json`.

## 1. Target-definition ambiguity

All estimated delivery timestamps are midnight. The current target uses `actual_delivery > estimated_delivery`, so delivery later on the promised calendar date counts as late.

| Split | Current timestamp-late labels | Same promised day | Late on a later calendar day |
|---|---:|---:|---:|
| Train | 6,096 | 806 | 5,290 |
| Validation | 773 | 149 | 624 |
| Test | 957 | 337 | 620 |

Thus 35.2% of test positives arrived on the promised date. This establishes label sensitivity, not that the labels are wrong. If the promise means the whole delivery date, these cases should be on time; if midnight is the true contractual deadline, the existing labels are consistent. Confirm the business definition before changing labels. New labels define a new target and metrics cannot be presented as a same-task improvement without stating this change.

## 2. A genuinely new delivery-window regime

Train and validation contain **zero** orders with a purchase-to-promised-delivery window of at most seven days. Test contains **1,260**, first appearing on **2018-07-26**. Of those, 395 have the existing late label and 213 were delivered on a later calendar date. At the frozen threshold, 293 late-labelled orders in this short-window group are missed: 45.9% of all 639 test false negatives.

Median promised windows shorten from 23.62 days in training to 17.59 days in test. The train-to-test PSI for this field is 0.629 (descriptive; not a significance test). This is concrete feature-support/distribution shift and is a plausible contributor to degraded generalization. It does not prove a particular model mechanism.

The test-month late rates also vary: 1.07% in its partial June block, 4.48% in July, 10.39% in its August block. Partial months and changed label sensitivity limit causal interpretation.

## 3. Retrospective split is not a strict as-of simulation

- 3,100 training orders (4.59%) were actually delivered at or after validation started on 2018-04-15.
- 1,929 validation orders (13.33%) were delivered at or after test started on 2018-06-21.

Using actual delivery as conservative label finalization, those outcomes were not all finalized at the historical boundary. Late status can sometimes be known before final delivery; these are not all proven unavailable labels. They nevertheless show that purchase-time sorting alone does not establish a fully deployable as-of backtest. Enforce label availability at each training/selection cutoff.

## 4. Calendar reliance and degradation

| Frozen-model measure | Training | Validation | Test |
|---|---:|---:|---:|
| F1 | 0.4305 | 0.2729 | 0.1703 |
| ROC-AUC | 0.8391 | 0.7902 | 0.6904 |
| Average precision | 0.4356 | 0.1992 | 0.1205 |

Purchase week, year and month together account for 38.67% of CatBoost's feature-importance measure. Importance is not causal attribution, and calendar distributions necessarily differ in a chronological split. Still, calendar ablations and forward-time validation are warranted. Unseen customer-city frequency is only 1.54% and unseen state-route frequency is 0.19% in test, so entirely new category names alone do not explain the observed degradation.

## 5. Score calibration and subgroup errors

The test mean score is 9.35%, versus an observed late rate of 6.61%. In the 0.20–0.30 score bin, 1,057 orders have mean score 24.45% but actual late rate 10.31%. Brier score is 0.06352, slightly worse than the constant training-prevalence predictor's 0.06234. Scores should not be described as validated calibrated probabilities. Calibration must be fitted on earlier data and assessed forward, never fitted on the test results reported here.

For customer state SP, recall is 20.5% across 6,750 test orders; customer state RJ produces 568 false positives across 1,625 orders. These descriptive groups motivate investigation, not state-specific rules tuned on test.

## 6. Prediction time and population

The current schema includes approval time and aggregate payment records, so it cannot represent a prediction strictly at purchase. The earliest supported stage is after approval and availability of all required aggregates. Payment records have no timestamps in this source, so their exact historical availability has not been verified. Exclude or explicitly justify unavailable aggregates in an as-of experiment.

The model labels 96,470 of 99,441 orders, primarily delivered orders. The excluded 2,971 include shipped, cancelled and unavailable orders; this is not a classifier of every possible order outcome. Do not force unresolved/cancelled orders into an on-time label.

## Next controlled experiment

1. Confirm promise semantics and the intended prediction stage. Freeze a versioned target/data contract.
2. Build rolling chronological evaluations with labels known by each cutoff, and a separate calibration/threshold selection block.
3. Compare feature ablations for absolute calendar features and unverified payment aggregates. Add mature recent-regime examples when available; do not random-split future short-window orders into past training.
4. State the operational cost of missed delays versus false alerts before choosing a threshold. Evaluate precision/recall and average precision, including short-window orders and route groups.
5. Require a later untouched time block for final confirmation. This existing test set has already been examined in several experiments; repeated improvement on it is exploratory evidence.

No model or target was changed by this diagnosis. Reproduce with `python -m Task2.src.diagnose_model` after restoring the exact improved artifact, splits and official source CSVs.
