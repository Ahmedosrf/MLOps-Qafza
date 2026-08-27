# Task 2 — From Tables to Notebooks

Task 2 turns the PostgreSQL tables created in Task 1 into a reproducible sequence of six notebooks. Each notebook has one job, reads the artifact from the previous step, and saves artifacts for the next step.

## Structure

```text
Task2/
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
| 06 | Baseline, tuning, final test evaluation | `artifacts/models/06_model.joblib`, `artifacts/reports/06_results.json`, `artifacts/charts/06_confusion_matrix.png` |

## Rules

Do not put all work in one notebook. Do not use the test split during EDA or model selection. Fit imputers, encoders, and scalers on the training split only. The actual delivery date and post-delivery review information may be used to construct or analyze the target, but must not be used as prediction features when unavailable at prediction time.

The initial recommended problem is late-delivery binary classification. Because the late class is expected to be smaller, compare more than accuracy; report recall, precision, F1, ROC-AUC or PR-AUC, and a confusion matrix as appropriate.
