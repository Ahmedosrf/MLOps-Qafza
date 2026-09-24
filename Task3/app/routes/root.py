"""Root service information route."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["service"])


class ServiceInfoResponse(BaseModel):
    service: str
    status: str
    documentation: str


@router.get("/", response_model=ServiceInfoResponse)
def service_info() -> ServiceInfoResponse:
    """Return a small human-readable service status response."""

    return ServiceInfoResponse(
        service="olist-late-delivery-api",
        status="running",
        documentation="/docs",
    )
