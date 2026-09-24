# Task 3 — From Notebooks to Production

Task 3 converts the trained Olist late-delivery pipeline into a reproducible inference service. Training remains in Task 2 for this stage. The production code will load the fitted model bundle and preprocessor, validate new orders, expose predictions through FastAPI, and later add tests, containers, CI/CD, artifact tracking, and monitoring.

## Current stage

The repository structure and portable configuration are in place. The reusable `src/feature_builder.py` module mirrors Task 2 Notebook 5's prediction-time feature policy. The `src/model_loader.py` and `src/predictor.py` modules load the fitted bundle and apply the saved preprocessor and threshold without retraining. FastAPI exposes `/`, `/health`, `/model/info`, `/predict`, `/predict/batch`, and `/metrics`. The service validates inputs, writes console and rotating-file logs, stores prediction events as JSONL, and exposes Prometheus counters and latency histograms. MLflow, PostgreSQL, MinIO, DVC, Great Expectations, Docker Compose, pre-commit, and CI quality checks are configured. Training remains in Task 2; Task 3 only performs inference.

## Planned structure

```text
Task3/
├── app/                    # FastAPI application and route definitions
│   ├── schemas.py          # Request and response validation
│   ├── logging_config.py   # Safe application logging
│   └── routes/
│       ├── root.py         # Service information route
│       └── predictions.py  # Single and batch prediction routes
├── config/                 # Runtime configuration and data expectations
├── data/                   # Local validation samples only; no raw dataset
├── models/                 # Model handoff instructions; generated models stay local
├── notebooks/              # Optional comparison notebook only; no training here
├── src/                    # Inference, validation, logging, and monitoring modules
│   ├── feature_builder.py  # Task 2-compatible prediction-time features
│   ├── model_loader.py     # Fitted bundle loading and validation
│   └── predictor.py        # Prediction result contract
├── tests/                  # Unit, data, model, and API integration tests
│   ├── test_feature_builder.py
│   ├── test_prediction_service.py
│   └── test_api_integration.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Source of the model

The service will consume the bundle produced by Task 2:

```text
Task2/artifacts/models/06_inference_bundle.joblib
```

The bundle contains the fitted classifier, fitted preprocessor, feature names, threshold, metadata, and model version. The service must call `transform`, never `fit` or `fit_transform`, for incoming requests.

## Prediction request contract

`POST /predict` requires the complete order-level feature set used by Task 2. In addition to purchase and delivery-estimate timestamps, each request must provide customer location, item and product counts, price and freight totals, product aggregates, seller aggregates, and payment aggregates. The required fields are `customer_zip_code_prefix`, `customer_city`, `customer_state`, `num_items`, `num_distinct_products`, `total_price`, `total_freight`, `num_sellers`, `num_product_categories`, `avg_product_weight_g`, `avg_product_photos_qty`, `seller_state_count`, `payment_record_count`, `total_payment_value`, and `max_payment_installments`. This prevents a request with only a few convenient fields from reaching the fitted preprocessor with missing columns.

The API does not invent these aggregates. A future client or upstream data-preparation step must calculate them from the same order-level tables and aggregation rules used in Task 2 Notebook 1. Post-delivery fields and review information remain prohibited. The seller location fields include both `seller_state` (the representative seller state) and `seller_state_count` (the number of distinct seller states); both are required by the fitted preprocessor.

## Configuration principles

Machine-specific paths, credentials, and thresholds must not be embedded in application code. Runtime values are read from environment variables or `config/config.yaml`. Secrets belong in a local `.env` file, which is ignored by Git. `.env.example` documents variable names without containing real credentials.

## Verification

## Environment requirement

Use **Python 3.11, 3.12, or 3.13** for this task. The pinned scientific packages are intentionally aligned with the Task 2 training environment so that the saved scikit-learn/joblib bundle can be loaded reproducibly. The current Task 2 bundle was trained with **scikit-learn 1.8.0**, so Task 3 pins the same version. Python 3.14 may cause Pip to build older NumPy versions from source on Windows, which requires a C/C++ compiler and can fail before installation. Do not change the scikit-learn version independently of the training artifact.

On Windows, create and activate a dedicated environment with Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\Task3\requirements-dev.txt
```

If the environment was already installed with an older scikit-learn version, update it before starting Uvicorn:

```powershell
python -m pip install --upgrade --force-reinstall scikit-learn==1.8.0
```

If `py -3.11` is not available, install Python 3.11 from [python.org](https://www.python.org/downloads/) and enable the Python Launcher during installation. Then run the commands above again.

The completed task supports running tests from the `Task3` directory:

```powershell
cd C:\Users\HP\Desktop\MLOps\Task3
pytest -q
```

From the repository root, use:

```powershell
cd C:\Users\HP\Desktop\MLOps
python -m pytest -q Task3\tests
uvicorn Task3.app.main:app --reload
```

## Docker execution

Build from the repository root after Task 2 has produced the local inference bundle:

```powershell
cd C:\Users\HP\Desktop\MLOps
docker build -f Task3\Dockerfile -t olist-late-delivery-api:task3-v1 .
docker run --rm -p 8000:8000 olist-late-delivery-api:task3-v1
```

Or use Compose:

```powershell
cd C:\Users\HP\Desktop\MLOps\Task3
docker-compose up --build
```

The Compose stack contains the API, MLflow tracking server, PostgreSQL backend, MinIO artifact storage, and persistent API logs. The API image still copies the current Task 2 bundle from the local artifact path for the reproducible handoff; `src/model_registry_export.py` logs the same fitted bundle to MLflow without refitting. Raw CSV files and generated reports remain excluded from the image.

## Continuous integration

The repository workflow at `.github/workflows/ci.yml` runs Ruff, Black, and the Task 3 test suite on every push to `main` and every pull request targeting `main`. A Docker build job runs when the model artifact is present in the checkout; the current bundle is intentionally excluded from Git, so the remote build is skipped until a registry-backed model download is configured. Pre-commit applies the same Ruff and Black checks before a local commit.

## Data validation and versioning

`src/validation.py` rejects missing columns, null values, negative numeric values, invalid counts, malformed state codes, and invalid dates before the model is called. `src/gx_validator.py` runs the same contract through Great Expectations. The expectation definitions are documented in `config/great_expectations.yml`. DVC metadata is initialized at the repository root, and `Task3/dvc.yaml` describes the handoff stage that copies the Task 2 bundle into a versionable Task 3 model path. Configure a remote with `dvc remote add` before pushing the model cache to shared storage.

## MLflow and artifact storage

The Compose stack starts MLflow with PostgreSQL as its backend store and MinIO as its S3-compatible artifact store. After the stack is running and the local Task 2 bundle exists, register the fitted bundle with:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m Task3.src.model_registry_export
```

The helper records the model version, threshold, feature count, source notebook, and Production tag. It never fits a model. MLflow is available at `http://127.0.0.1:5000`, and MinIO's console is available at `http://127.0.0.1:9001`.

## Monitoring and logs

Prometheus-format metrics are available at `http://127.0.0.1:8000/metrics`. The service exposes request count, request latency, unexpected-error count, and prediction count by class and model version. API logs and prediction events are written to `Task3/logs` locally or the Compose `api_logs` volume. Alert candidates are sustained 5xx errors, p95 latency above the agreed service target, and a sudden change in the `late` prediction ratio; actual delivery outcomes should be joined later to calculate drift and realized performance.
