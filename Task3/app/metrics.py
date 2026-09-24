"""Prometheus metrics exposed by the inference API."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "task3_http_requests_total",
    "Total HTTP requests handled by the service",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "task3_http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "task3_predictions_total",
    "Total predictions by class and model version",
    ["prediction", "model_version"],
)
ERROR_COUNT = Counter(
    "task3_errors_total",
    "Total unexpected service errors",
    ["path"],
)
