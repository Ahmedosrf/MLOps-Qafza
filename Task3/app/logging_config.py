"""Application logging configuration for console and file output."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> None:
    """Configure consistent console and rotating-file logs once per process."""

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_path = Path(os.getenv("LOG_FILE_PATH", "Task3/logs/api.log"))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError:
        # A read-only mounted volume must never prevent the API from starting.
        fallback_path = Path("/tmp/task3/api.log")
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            fallback_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
