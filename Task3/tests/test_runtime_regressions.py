"""Regression checks with serialized, fitted preprocessing and classification."""

import json
import os
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from Task3.app.main import app
from Task3.app.dependencies import get_prediction_service, get_inference_bundle
from Task3.src.validation import validate_prediction_frame
from Task3.src.model_loader import _validate_bundle
from Task3.tests.ci_bundle import ORDER, make_bundle


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    app.dependency_overrides.clear()
    get_prediction_service.cache_clear()
    get_inference_bundle.cache_clear()
    path = tmp_path / "06_inference_bundle.joblib"
    bundle = make_bundle(path)
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(path))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    yield TestClient(app, raise_server_exceptions=False), path, bundle


def test_ready_and_inference_use_serialized_sklearn_objects(runtime):
    client, _, bundle = runtime
    assert client.get("/ready").status_code == 200
    assert client.get("/health").json()["model_version"] == "ci-synthetic-only"
    result = client.post("/predict", json=ORDER)
    assert result.status_code == 200
    from Task3.src.feature_builder import build_prediction_features

    x = bundle["preprocessor"].transform(
        build_prediction_features(pd.DataFrame([ORDER]))
    )
    assert result.json()["probability"] == pytest.approx(
        bundle["model"].predict_proba(x)[0, 1]
    )
    assert (
        client.post("/predict/batch", json={"orders": [ORDER, ORDER]}).status_code
        == 200
    )


@pytest.mark.parametrize("kind", ["missing", "corrupt", "unfitted"])
def test_unusable_bundle_returns_503_but_process_is_live(runtime, kind):
    client, path, bundle = runtime
    if kind == "missing":
        path.unlink()
    elif kind == "corrupt":
        path.write_text("not a bundle")
    else:
        from sklearn.linear_model import LogisticRegression

        bundle["model"] = LogisticRegression()
        joblib.dump(bundle, path)
    assert client.get("/live").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "oops"])
def test_local_validation_rejects_non_finite_and_non_numeric(value):
    with pytest.raises(ValueError, match="finite numeric|missing values"):
        validate_prediction_frame(pd.DataFrame([{**ORDER, "total_price": value}]))


def test_api_rejects_infinite_json_number(runtime):
    client, _, _ = runtime
    raw = json.dumps({**ORDER, "total_price": 1}).replace(
        '"total_price": 1,', '"total_price": 1e309,'
    )
    assert (
        client.post(
            "/predict", content=raw, headers={"content-type": "application/json"}
        ).status_code
        == 422
    )


def test_mixed_timezone_has_same_prediction_as_equivalent_local_time(runtime):
    client, _, _ = runtime
    local = client.post("/predict", json=ORDER)
    aware = client.post(
        "/predict", json={**ORDER, "order_approved_at": "2018-01-06T13:30:00Z"}
    )
    assert aware.status_code == 200
    assert aware.json()["probability"] == pytest.approx(local.json()["probability"])


def test_bundle_rejects_feature_order_mismatch(runtime):
    _, _, bundle = runtime
    bundle["feature_names"] = bundle["feature_names"][::-1]
    with pytest.raises(ValueError, match="feature names"):
        _validate_bundle(bundle)


def test_log_directory_creation_failure_falls_back(tmp_path):
    # /proc cannot create a directory, including when tests run as root.
    env = {**os.environ, "LOG_FILE_PATH": "/proc/task3-invalid-directory/api.log"}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from Task3.app.logging_config import configure_logging; configure_logging()",
        ],
        env=env,
        check=True,
    )


def test_model_registry_round_trip(runtime, tmp_path, monkeypatch):
    import mlflow
    from Task3.src.model_registry_export import register_bundle
    from Task3.src.model_loader import load_inference_bundle

    tracking = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    run_id = register_bundle()
    client = mlflow.MlflowClient(tracking_uri=tracking)
    versions = client.search_model_versions("name='olist-late-delivery'")
    version = next(v.version for v in versions if v.run_id == run_id)
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setenv("MODEL_URI", f"models:/olist-late-delivery/{version}")
    loaded = load_inference_bundle()
    assert loaded.model_version == "ci-synthetic-only"
    registered = mlflow.pyfunc.load_model(f"models:/olist-late-delivery/{version}")
    result = registered.predict(pd.DataFrame([ORDER]))
    assert np.isfinite(result["probability"]).all()
