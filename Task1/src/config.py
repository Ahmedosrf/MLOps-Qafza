"""Configuration helpers for Task 1 data ingestion.

The notebook imports these helpers so paths and database settings can be
changed through environment variables without editing the ingestion logic.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from sqlalchemy.engine import URL


@dataclass(frozen=True)
class Task1Settings:
    """Runtime settings for the Task 1 ingestion notebook."""

    project_root: Path
    data_dir: Path
    db_host: str
    db_port: int
    db_name: str
    db_user: str


def find_project_root() -> Path:
    """Find the repository root from the current working directory."""

    configured_root = os.getenv("MLOPS_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    current = Path.cwd().resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "Task1").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the project root. Set MLOPS_PROJECT_ROOT explicitly."
    )


def load_settings() -> Task1Settings:
    """Load Task 1 settings from environment variables."""

    project_root = find_project_root()
    data_dir = Path(
        os.getenv("OLIST_DATA_DIR", str(project_root / "Task1" / "data"))
    ).expanduser().resolve()

    return Task1Settings(
        project_root=project_root,
        data_dir=data_dir,
        db_host=os.getenv("OLIST_DB_HOST", "localhost"),
        db_port=int(os.getenv("OLIST_DB_PORT", "5432")),
        db_name=os.getenv("OLIST_DB_NAME", "olist"),
        db_user=os.getenv("OLIST_DB_USER", "postgres"),
    )


def build_database_url(settings: Task1Settings, password: str) -> URL:
    """Build a safely escaped SQLAlchemy PostgreSQL URL."""

    return URL.create(
        drivername="postgresql+psycopg2",
        username=settings.db_user,
        password=password,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )


__all__ = ["Task1Settings", "build_database_url", "find_project_root", "load_settings"]
