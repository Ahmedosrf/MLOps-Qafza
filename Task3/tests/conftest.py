"""Keep overrides isolated between tests."""

import pytest
from Task3.app.main import app
from Task3.app.dependencies import get_prediction_service, get_inference_bundle


@pytest.fixture(autouse=True)
def isolate_application_state():
    overrides = dict(app.dependency_overrides)
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(overrides)
    get_prediction_service.cache_clear()
    get_inference_bundle.cache_clear()
