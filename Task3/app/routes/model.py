"""Model metadata route for the Task 3 service."""

from fastapi import APIRouter, Depends, HTTPException, status

from Task3.app.dependencies import get_inference_bundle
from Task3.app.schemas import ModelInfoResponse
from Task3.src.model_loader import InferenceBundle

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
def model_info(
    bundle: InferenceBundle = Depends(get_inference_bundle),
) -> ModelInfoResponse:
    """Return metadata from the loaded inference bundle."""

    if not bundle.feature_names:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Loaded model has no feature metadata",
        )

    return ModelInfoResponse(
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        threshold=bundle.threshold,
        feature_count=len(bundle.feature_names),
        target_definition=bundle.feature_metadata.get("target_definition"),
        prediction_stage=bundle.feature_metadata.get("prediction_stage"),
    )
