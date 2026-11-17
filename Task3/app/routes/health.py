"""Separate process liveness from validated model readiness."""

import logging
from fastapi import APIRouter, HTTPException
from Task3.app.dependencies import get_prediction_service
from Task3.app.schemas import HealthResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger("task3.health")


@router.get("/live")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Load the same validated bundle used by predictions; fail closed."""
    try:
        service = get_prediction_service()
    except Exception as exc:
        logger.warning("Model readiness failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Model unavailable") from exc
    return HealthResponse(
        status="ok",
        model_bundle_available=True,
        model_version=service.bundle.model_version,
    )
