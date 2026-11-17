# Task 3 — Versioned calendar-day inference

Current bundle: `task2-v3-calendar-day`, experimental, **not production-ready**. The corrected target means delivery after the estimated local calendar date; same-day arrival is on time. Prediction is after approval. Full model/data results: [corrected experiment](../Task2/docs/CORRECTED_EXPERIMENT.md).

## Local setup

From the repository root in an activated Python 3.11–3.13 environment:

```bash
python -m pip install -r Task3/requirements-dev.txt
python -m Task2.src.corrected_training
dvc repro Task3/dvc.yaml
python -m uvicorn Task3.app.main:app --port 8000
```

Training needs the official CSVs and joined artifact; see Task2/README.md. To restore the exact trained candidate instead, extract `Olist_Calendar_Day_V3_Artifacts.zip` at the repository root and run `dvc checkout Task3/dvc.yaml`. It includes the current model and matching DVC cache; raw data is not included. A normal source clone contains no model bytes.

Environment variables are read from the process environment; YAML files and `.env.example` are documentation, not automatically loaded configuration. Default model path is `Task2/artifacts/models/06_inference_bundle.joblib`. Set `MLOPS_PROJECT_ROOT` if starting outside the repository.

## API

| Route | Meaning |
|---|---|
| `/live` | Process liveness |
| `/ready`, `/health` | Load/validate the same cached fitted bundle; 503 if unusable |
| `/model/info` | Model/version, threshold, feature count, target definition and prediction stage |
| `/predict` | Single order, score and versioned target/stage |
| `/predict/batch` | 1–100 orders, input order preserved |
| `/metrics` | Prometheus counters and latency |
| `/docs` | Request schema and examples |

The v3 API accepts omitted payment fields. If supplied they are validated but excluded from v3 features because historical availability is unverified. Legacy bundles still require payment inputs and use their historical feature behavior. The loader rejects incompatible v3 target/stage metadata.

Use the seller state of the smallest `order_item_id`, including multi-seller orders. Input dates use Sao Paulo local wall time; zoned timestamps convert to that zone. Approval cannot precede purchase. Scores are not calibrated probabilities.

## Docker and permissions

Export or restore `Task3/models/06_inference_bundle.joblib` first, then from the repository root:

```bash
docker build -f Task3/Dockerfile -t olist-api .
docker run --rm -p 8000:8000 olist-api
# Full local stack:
docker compose -f Task3/docker-compose.yml up --build
```

Compose runs the API as UID 10001 and probes `/ready`. New log volumes inherit the image directory ownership. For an older root-owned volume, stop the API and fix ownership once using `docker compose -f Task3/docker-compose.yml run --rm --user 0 --no-deps --entrypoint chown api -R 10001:10001 /app/Task3/logs`, then start normally. Do not delete existing logs to fix permissions. Restart the service whenever the bundle changes, because objects are cached per process.

## DVC

`.dvc/` is initialized. `Task3/dvc.yaml` exports the validated canonical model and tracks contract, feature and training source dependencies. The committed lock refers to the exact real v3 model.

```bash
dvc repro Task3/dvc.yaml
dvc remote add --local -d artifacts /path/to/your/shared/store
dvc push Task3/dvc.yaml
# In another checkout, configure the same remote, then:
dvc pull Task3/dvc.yaml
```

Store credentials locally. S3 remotes additionally need `dvc[s3]==3.59.1`. Source Git contains checksums, not model/cache bytes. The supplied artifact archive is an alternative to a remote for restoring this exact version.

## MLflow

Compose includes MLflow, PostgreSQL and MinIO. For registration from the host, set `MLFLOW_TRACKING_URI=http://localhost:5000`, `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`, and local MinIO access credentials (`minioadmin` defaults in this development stack), then run:

```bash
python -m Task3.src.model_registry_export
```

This creates a real registered pyfunc version containing code and fitted bundle; it does not automatically promote it. To load a reviewed version, set `MODEL_SOURCE=mlflow`, `MODEL_URI=models:/olist-late-delivery/<version>`, connection variables, then restart. To use a local file, set `MODEL_SOURCE=local` and optionally `MODEL_BUNDLE_PATH`.

## Verification

```bash
ruff check Task3 Task2/src
black --check Task3 Task2/src
python -m pytest -q Task3/tests
```

Tests cover calendar boundaries/time zones, as-of label availability, leakage exclusions, serialized inference, contract mismatch, invalid inputs, MLflow registry round trips and DVC restore. CI always builds/smoke-tests a non-root container with a clearly labelled synthetic **v3-contract** bundle. It never publishes that fixture as a production model. Real v3 API verification is recorded in `docs/v3_api_verification.json`; historical v1/v2 reports are retained separately.

Logs rotate for API messages and append prediction events as JSONL. Monitoring is descriptive; live drift/performance needs real outcomes and operational thresholds. Current predictive quality is weak and the June–August evaluation was previously inspected, so it is not blind evidence.
