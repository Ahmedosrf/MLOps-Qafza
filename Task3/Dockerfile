FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_BUNDLE_PATH=/app/Task2/artifacts/models/06_inference_bundle.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

COPY Task3/requirements.txt /app/Task3/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/Task3/requirements.txt

COPY Task3 /app/Task3
COPY Task2/artifacts/models/06_inference_bundle.joblib \
     /app/Task2/artifacts/models/06_inference_bundle.joblib

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "Task3.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
