# Task 3 — From Notebooks to Production

Task 3 converts the trained Olist late-delivery pipeline into a reproducible inference service. Training remains in Task 2. The service loads the fitted model bundle and preprocessor, validates new orders, exposes predictions through FastAPI, and provides tests, containers, CI/CD, artifact tracking, and monitoring foundations without retraining.

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

`POST /predict` requires the complete order-level feature set used by Task 2. In addition to purchase and delivery-estimate timestamps, each request must provide customer location, item and product counts, price and freight totals, product aggregates, seller aggregates, and payment aggregates. The required fields are `customer_zip_code_prefix`, `customer_city`, `customer_state`, `seller_state`, `num_items`, `num_distinct_products`, `total_price`, `total_freight`, `num_sellers`, `num_product_categories`, `avg_product_weight_g`, `avg_product_photos_qty`, `seller_state_count`, `payment_record_count`, `total_payment_value`, and `max_payment_installments`. This prevents a request with only a few convenient fields from reaching the fitted preprocessor with missing columns.

The API does not invent these aggregates. A future client or upstream data-preparation step must calculate them from the same order-level tables and aggregation rules used in Task 2 Notebook 1. Post-delivery fields and review information remain prohibited. The seller location fields include both `seller_state` (the representative seller state) and `seller_state_count` (the number of distinct seller states); both are required by the fitted preprocessor.

## Configuration principles

Machine-specific paths, credentials, and thresholds must not be embedded in application code. Runtime values are read from environment variables or `config/config.yaml`. Secrets belong in a local `.env` file, which is ignored by Git. `.env.example` documents variable names without containing real credentials.

## Verification

Automated tests include serialized sklearn inference, readiness failures, invalid numeric values, timestamp normalization, MLflow registry round trips, and DVC publish/restore. CI always builds a container with an explicitly synthetic fitted bundle, checks its non-root user, and calls readiness and prediction endpoints. This verifies integration, not the accuracy of the real Olist model.

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

Build from the repository root after Task 2 has produced the local inference bundle. Activate the Python environment, then run `dvc repro Task3/dvc.yaml` to export the validated bundle to `Task3/models/06_inference_bundle.joblib` first:

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

The Compose stack contains the API, MLflow tracking server, PostgreSQL backend, MinIO artifact storage, and API logs. The API image copies the exported, versioned bundle from `Task3/models/06_inference_bundle.joblib`; `src/model_registry_export.py` logs the same fitted bundle to MLflow without refitting. Raw CSV files and generated reports remain excluded from the image. For a clean clone, provide the bundle through DVC or another artifact store before building the API image.

## Continuous integration

The workflow runs Ruff, Black and the test suite on pushes to `main` and pull requests. It always builds the API image using a fitted synthetic test bundle and checks the container user, `/ready`, and `/predict`. CI-generated bundles are never published as production artifacts. Real model acceptance still requires testing the version restored from the configured artifact store.

## Data validation and versioning

`src/validation.py` rejects missing fields, nulls, non-numeric and non-finite numbers, invalid counts, malformed state codes and invalid dates. Pydantic rejects non-finite floats; validation responses exclude raw input so invalid JSON numbers cannot cause a secondary 500. Naive Olist timestamps mean Sao Paulo local wall-clock time. Zoned timestamps convert to that clock before feature extraction. Task 2 Notebook 5 imports the same feature builder as serving.

DVC is initialized in `.dvc/`. From the repository root, in the activated environment:

```bash
# After generating the real Task 2 bundle:
dvc repro Task3/dvc.yaml
# Configure YOUR existing artifact store (credentials belong in --local config).
dvc remote add --local -d artifacts /path/to/shared/artifact-store
dvc push Task3/dvc.yaml
git add Task3/dvc.lock
git commit -m "Version the trained inference bundle"
```

A local filesystem remote needs no extra backend. For S3 install `dvc[s3]==3.59.1` and configure your bucket and credentials locally. In another checkout with that committed lock file and access to the same remote:

```bash
dvc remote add --local -d artifacts /path/to/shared/artifact-store
dvc pull Task3/dvc.yaml
docker build -f Task3/Dockerfile -t olist-late-delivery-api:task3-v1 .
```

The committed `Task3/dvc.lock` was generated from the real Olist bundle rebuilt from the official dataset, not from a CI fixture. See `docs/real_olist_verification.json` for dataset/model SHA-256 hashes, split sizes, test metrics and API parity checks. Pulling still needs an artifact remote populated with that bundle; source control contains checksums, not model bytes. The accompanying `Olist_Real_Model_Artifacts.zip` can instead be extracted at the repository root to restore the exact bundle and DVC cache without a remote. Run `dvc checkout Task3/dvc.yaml` after extraction. Real-data reconstruction used a temporary SQLite staging engine for Notebook 1 transformations; PostgreSQL ingestion was not rerun. The model has limited predictive performance (precision 0.0936, recall 0.4681, F1 0.156 on the held-out test set); integration success is not an accuracy claim.

## MLflow and artifact storage

The Compose stack starts MLflow with PostgreSQL as its backend store and MinIO as its S3-compatible artifact store. After the stack is running and the local Task 2 bundle exists, register the fitted bundle with:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin"
python -m Task3.src.model_registry_export
```

The helper records metadata and creates an actual registered MLflow pyfunc model version containing the fitted bundle and serving code. It does not automatically promote a version to Production. To serve a reviewed version, set `MODEL_SOURCE=mlflow`, `MODEL_URI=models:/olist-late-delivery/<version>` and the tracking/S3 connection variables, then restart the API to clear its process-local cache. Local mode remains `MODEL_SOURCE=local` with `MODEL_BUNDLE_PATH`. These variables are read from the process environment; YAML files and `.env.example` are documentation, not automatically loaded configuration. It never refits a model. MLflow is available at `http://127.0.0.1:5000`, and MinIO's console is available at `http://127.0.0.1:9001`.

## Monitoring and logs

Prometheus-format metrics are available at `http://127.0.0.1:8000/metrics`. The service exposes request count, request latency, unexpected-error count, and prediction count by class and model version. API logs and prediction events are written to `Task3/logs` locally or the Compose `api_logs` volume. Alert candidates are sustained 5xx errors, p95 latency above the agreed service target, and a sudden change in the `late` prediction ratio; actual delivery outcomes should be joined later to calculate drift and realized performance.



## Readiness and container permissions

`/live` reports process liveness. `/ready` and `/health` load and validate the cached model bundle, including fitted sklearn objects and feature order, and return 503 when it cannot be used. Compose probes `/ready` and runs as UID 10001. New log volumes inherit the image directory ownership. If an older volume was created by root, stop the API and fix ownership once using `docker compose run --rm --user 0 --no-deps --entrypoint chown api -R 10001:10001 /app/Task3/logs`, then start it normally. Do not delete an existing log volume to fix permissions. Model changes require a service restart.
