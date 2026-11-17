from fastapi.testclient import TestClient

from Task3.app.dependencies import get_inference_bundle, get_prediction_service
from Task3.app.main import app
from Task3.src.model_loader import InferenceBundle
from Task3.src.predictor import PredictionResult


class FakePreprocessor:
    pass


class FakeModel:
    pass


class FakePredictionService:
    def predict_one(self, order):
        return PredictionResult(
            prediction="late",
            is_late=1,
            probability=0.75,
            threshold=0.30,
            model_name="olist-late-delivery",
            model_version="task2-v1",
            latency_ms=1.0,
        )


def fake_bundle():
    return InferenceBundle(
        model=FakeModel(),
        preprocessor=FakePreprocessor(),
        feature_names=("feature_a", "feature_b"),
        threshold=0.30,
        model_name="olist-late-delivery",
        model_version="task2-v1",
        feature_metadata={},
    )


client = TestClient(app)
app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService()
app.dependency_overrides[get_inference_bundle] = fake_bundle


VALID_ORDER = {
    "order_purchase_timestamp": "2018-01-06 10:30:00",
    "order_estimated_delivery_date": "2018-01-15",
    "order_approved_at": "2018-01-06 11:30:00",
    "customer_zip_code_prefix": 13000,
    "customer_city": "campinas",
    "customer_state": "SP",
    "total_price": 100.0,
    "total_freight": 20.0,
    "total_payment_value": 120.0,
    "num_items": 2,
    "num_distinct_products": 2,
    "num_sellers": 1,
    "num_product_categories": 1,
    "avg_product_weight_g": 500.0,
    "avg_product_photos_qty": 3.0,
    "seller_state_count": 1,
    "payment_record_count": 1,
    "max_payment_installments": 3,
    "seller_state": "RJ",
}


def teardown_module():
    app.dependency_overrides.clear()


def test_root_route_returns_service_information():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "olist-late-delivery-api",
        "status": "running",
        "documentation": "/docs",
    }


def test_health_route_returns_service_status():
    response = client.get("/live")

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_model_info_route_returns_bundle_metadata():
    response = client.get("/model/info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "olist-late-delivery",
        "model_version": "task2-v1",
        "threshold": 0.3,
        "feature_count": 2,
    }


def test_predict_route_returns_prediction_contract():
    response = client.post("/predict", json=VALID_ORDER)

    assert response.status_code == 200
    assert response.json()["prediction"] == "late"
    assert response.json()["probability"] == 0.75
    assert response.json()["model_version"] == "task2-v1"


def test_batch_predict_preserves_order():
    response = client.post(
        "/predict/batch",
        json={"orders": [VALID_ORDER, VALID_ORDER]},
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert all(item["prediction"] == "late" for item in response.json()["predictions"])


def test_predict_rejects_invalid_payload():
    invalid_order = {**VALID_ORDER, "total_price": -1}

    response = client.post("/predict", json=invalid_order)

    assert response.status_code == 422


def test_batch_predict_rejects_empty_batch():
    response = client.post("/predict/batch", json={"orders": []})

    assert response.status_code == 422
