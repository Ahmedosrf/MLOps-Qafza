"""FastAPI dependencies for loading the inference service."""

from functools import lru_cache

from Task3.src.model_loader import InferenceBundle, load_inference_bundle
from Task3.src.predictor import PredictionService


@lru_cache(maxsize=1)
def get_inference_bundle() -> InferenceBundle:
    """Load the fitted bundle once per application process."""

    return load_inference_bundle()


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Create one prediction service from the cached fitted bundle."""

    return PredictionService(get_inference_bundle())
