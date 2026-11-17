"""Versioned target and feature contract for the corrected experiment."""

FEATURE_POLICY = "calendar-day-v3"
TARGET_DEFINITION = "late_after_estimated_calendar_day"
PREDICTION_STAGE = "after_approval"
PAYMENT_FIELDS = (
    "payment_record_count",
    "total_payment_value",
    "max_payment_installments",
)
