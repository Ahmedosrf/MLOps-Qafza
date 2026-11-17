# Olist model improvement

Selected model: **CatBoostClassifier**, version `task2-v2-catboostclassifier`, 420 trees, depth 7.

Twelve final configurations (8 HGB, 4 CatBoost) were compared using validation F1; CatBoost early stopping used validation AUC. The saved threshold is 0.14328469058748036.

| Metric | Baseline | Candidate |
|---|---:|---:|
| precision | 0.0936 | 0.1145 |
| recall | 0.4681 | 0.3323 |
| f1 | 0.1560 | 0.1703 |
| roc_auc | 0.6451 | 0.6904 |
| average_precision | 0.1080 | 0.1205 |
| accuracy | 0.6649 | 0.7859 |

The same chronological split was retained: 67,529 training, 14,470 validation, 14,471 test orders. Validation/test were never added to model fitting. Static postal-prefix coordinates do not contain targets.

Features include first-item seller state, shipping route, buyer coordinates, approximate seller-state coordinates and distance. The fitted bundle embeds reference dictionaries and all preprocessing; the API does not need raw CSVs.

False positives decreased from 4,340 to 2,459, while detected late orders decreased from 448 to 318. This is a tradeoff, not an improvement on every metric. F1 increases about 9.2% relative to baseline, but remains low. Threshold changes require validation against a stated operational cost/recall target.

The existing test split had already been reported during baseline and exploratory experiments; these results are not an independent blind confirmation. Use a later untouched time window before production acceptance. Scores are not calibrated probabilities.

Run `python -m Task2.src.improve_model` after restoring the baseline and Task2 splits to reproduce the comparison. Baseline rollback artifact is retained in the improved-model package. Restart the API after selecting a bundle.

Real API verification: 100 eligible held-out orders, single/batch/readiness HTTP 200, exact parity with direct calls. See `Task3/docs/improved_olist_verification.json`.
