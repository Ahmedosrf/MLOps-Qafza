"""Health route for the Task 3 service."""

from fastapi import APIRouter

from Task3.app.schemas import HealthResponse
from Task3.src.model_loader import resolve_bundle_path

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report service liveness without forcing a model load."""

    bundle_path = resolve_bundle_path()
    available = bundle_path.is_file()
    return HealthResponse(
        status="ok" if available else "degraded",
        model_bundle_available=available,
    )
