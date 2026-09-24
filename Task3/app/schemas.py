"""Pydantic schemas for the Task 3 API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service liveness and model artifact availability."""

    status: Literal["ok", "degraded"]
    model_bundle_available: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Public metadata for the loaded production model."""

    model_name: str
    model_version: str
    threshold: float = Field(gt=0, lt=1)
    feature_count: int = Field(gt=0)


class OrderPayload(BaseModel):
    """Validated prediction-time fields plus permitted Task 2 extra features."""

    model_config = {"extra": "allow"}

    order_purchase_timestamp: str
    order_estimated_delivery_date: str
    order_approved_at: str
    customer_zip_code_prefix: int = Field(ge=0)
    customer_city: str
    customer_state: str
    num_items: int = Field(gt=0)
    num_distinct_products: int = Field(gt=0)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    num_sellers: int = Field(gt=0)
    num_product_categories: int = Field(gt=0)
    avg_product_weight_g: float = Field(ge=0)
    avg_product_photos_qty: float = Field(ge=0)
    seller_state_count: int = Field(gt=0)
    payment_record_count: int = Field(gt=0)
    total_payment_value: float = Field(ge=0)
    max_payment_installments: int = Field(gt=0)


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    prediction: Literal["late", "on_time"]
    is_late: int = Field(ge=0, le=1)
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(gt=0, lt=1)
    model_name: str
    model_version: str
    latency_ms: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    """Batch request with a bounded number of orders."""

    orders: list[OrderPayload] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response preserving input order."""

    predictions: list[PredictionResponse]
