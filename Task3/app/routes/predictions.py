"""Single-order and batch prediction routes."""

from fastapi import APIRouter, Depends, HTTPException, status

from Task3.app.dependencies import get_prediction_service
from Task3.app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    OrderPayload,
    PredictionResponse,
)
from Task3.src.predictor import PredictionService

router = APIRouter(tags=["predictions"])


def _predict(order: OrderPayload, service: PredictionService) -> PredictionResponse:
    """Run one prediction and convert known input errors to HTTP 422."""

    try:
        result = service.predict_one(order.model_dump())
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return PredictionResponse(**result.to_dict())


@router.post("/predict", response_model=PredictionResponse)
def predict(
    order: OrderPayload,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Predict whether one order will be late."""

    return _predict(order, service)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    """Predict a bounded batch while preserving the input order."""

    predictions = []
    for index, order in enumerate(request.orders):
        try:
            predictions.append(_predict(order, service))
        except HTTPException as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"index": index, "error": exc.detail},
            ) from exc

    return BatchPredictionResponse(predictions=predictions)
