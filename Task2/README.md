# Task 2 — From Tables to Notebooks

Task 2 turns the PostgreSQL tables created in Task 1 into a reproducible sequence of six notebooks. Each notebook has one job, reads the artifact from the previous step, and saves artifacts for the next step.

## Structure

```text
Task2/
├── src/
│   ├── __init__.py
│   └── config.py                  # Portable project-root discovery
├── notebooks/
│   ├── 01_read_and_join_tables.ipynb
│   ├── 02_create_labels.ipynb
│   ├── 03_split_data.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_train_tune_evaluate.ipynb
├── artifacts/
│   ├── tables/
│   ├── features/
│   ├── reports/
│   ├── charts/
│   └── models/
└── README.md
```

## Notebook order

| Notebook | Responsibility | Main artifact |
|---|---|---|
| 01 | Read tables, validate grain and keys, aggregate before joining | `artifacts/tables/01_joined_ml_table.parquet` |
| 02 | Create and validate the late-delivery label | `artifacts/tables/02_labeled_table.parquet` |
| 03 | Split before deep analysis and justify the split | `artifacts/tables/03_train.parquet`, `03_validation.parquet`, `03_test.parquet` |
| 04 | Detailed EDA on training data only | `artifacts/charts/`, `artifacts/reports/04_findings.md` |
| 05 | Leakage-safe feature engineering and fitted transformers | `artifacts/features/05_X_*.npz`, `05_y_*.npy`, transformers, feature list |
| 06 | Baseline, tuning, final test evaluation | `artifacts/models/06_model.joblib`, `06_inference_bundle.joblib`, `06_model_metadata.json`, `artifacts/reports/06_results.json` |

## Rules

Do not put all work in one notebook. Do not use the test split during EDA or model selection. Fit imputers, encoders, and scalers on the training split only. The actual delivery date and post-delivery review information may be used to construct or analyze the target, but must not be used as prediction features when unavailable at prediction time.

The initial recommended problem is late-delivery binary classification. Because the late class is expected to be smaller, compare more than accuracy; report recall, precision, F1, ROC-AUC or PR-AUC, and a confusion matrix as appropriate.

## Production handoff contract

Task 2 remains the training stage; it does not serve requests or retrain from the API. Notebook 5 saves the fitted preprocessing transformer without refitting it for validation or test data. It also saves `05_preprocessor.joblib`, `05_feature_list.json`, and `05_feature_metadata.json`. Notebook 6 saves the selected classifier and creates `06_inference_bundle.joblib`, which contains the fitted model, the fitted preprocessor, the transformed feature names, the selected probability threshold, and the model version `task2-v1`.

The inference bundle is the handoff consumed by Task 3. A production service must load it and call `preprocessor.transform(...)`; it must never call `fit` or `fit_transform` on incoming requests. The selected threshold is stored in the bundle and must not be duplicated as an unexplained constant in the API.

The generated artifacts are intentionally ignored by Git because they are local outputs. They must be reproduced by running the notebooks in order, or later published to a versioned artifact store such as MLflow/DVC as part of Task 3.



## Improved temporal model

After running Notebooks 1–6 and downloading the original geolocation, items and sellers CSVs into `Task1/data/`, run from the repository root:

```bash
python -m pip install -r Task3/requirements-dev.txt
python -m Task2.src.improve_model
```

This compares native-categorical histogram gradient boosting and CatBoost, adds seller-state/route and static geographic features, and selects the model and decision threshold using validation F1. CatBoost uses validation AUC for early stopping. The chronological test split is evaluated after selection; its previously reported results mean it is not a fresh blind holdout. All transformations fitted on examples use the training split only; static geographic reference medians contain no targets. A geographic distance to a seller-state median is approximate, not an actual shipping route or exact seller position.

For orders with several sellers, `seller_state` means the state of the seller for the smallest `order_item_id`. Provide that same field in API requests. Postal-prefix and state lookup dictionaries are embedded in the fitted preprocessing bundle; inference does not need the CSVs or a network lookup. Unknown prefixes use the fitted imputer.

The command retains the original bundle as `06_baseline_task2_v1.joblib`, writes the selected bundle to the existing inference path and saves `07_improvement_results.json`. It does not refit on validation/test after selection. To reproduce the comparison, retain the baseline bundle. Notebook 6 still reproduces the original baseline; rerun the improvement command afterwards to restore the improved serving bundle.

Then export/version it with `dvc repro Task3/dvc.yaml`, restart the API to clear its process-local model cache, and verify `/model/info`. Do not silently substitute a new model file in an already-running process.


## Diagnose a frozen model before further tuning

Run `python -m Task2.src.diagnose_model` from the repository root after restoring the improved real model, splits and official CSVs. This writes `Task2/docs/model_diagnostics.json` and does not fit, tune, relabel or promote a model. `Task2/docs/MODEL_DIAGNOSIS.md` explains the observed target-definition ambiguity, delivery-window distribution shift, label-availability overlap and probability calibration limits. Treat subgroup results as descriptive, not proof of causes. Confirm the intended promise semantics before changing the target.
