"""Append prediction events to a local JSONL file for later evaluation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def prediction_log_path() -> Path:
    return Path(os.getenv("PREDICTION_LOG_PATH", "Task3/logs/predictions.jsonl"))


def log_prediction(request_data: dict[str, Any], result: dict[str, Any]) -> None:
    """Persist one input/output event without failing the prediction request."""

    path = prediction_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": request_data,
            "output": result,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, default=str) + "\n")
    except OSError:
        # Observability must never make a valid prediction fail.
        return
