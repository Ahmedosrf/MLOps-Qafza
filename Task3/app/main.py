"""FastAPI entry point for the Task 3 inference service."""

import logging
from time import perf_counter

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from Task3.app.logging_config import configure_logging
from Task3.app.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY
from Task3.app.routes.health import router as health_router
from Task3.app.routes.metrics import router as metrics_router
from Task3.app.routes.model import router as model_router
from Task3.app.routes.predictions import router as predictions_router
from Task3.app.routes.root import router as root_router

configure_logging()
logger = logging.getLogger("task3.api")

app = FastAPI(
    title="Olist Late Delivery API",
    version="task3-v1",
    description="Inference service for the Task 2 late-delivery classifier.",
)

app.include_router(health_router)
app.include_router(model_router)
app.include_router(predictions_router)
app.include_router(root_router)
app.include_router(metrics_router)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Log request method, route, status, and duration without request data."""

    started = perf_counter()
    response = await call_next(request)
    duration_ms = (perf_counter() - started) * 1000
    REQUEST_COUNT.labels(
        request.method, request.url.path, str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration_ms / 1000)
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Return a safe error message while keeping details in server logs."""

    logger.exception("Unhandled error for %s %s", request.method, request.url.path)
    ERROR_COUNT.labels(request.url.path).inc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal prediction service error"},
    )
