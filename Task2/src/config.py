"""Portable path configuration for the Task 2 training notebooks."""

from pathlib import Path
import os


def find_project_root() -> Path:
    """Return the repository root, optionally overridden by an environment variable."""

    configured_root = os.getenv("MLOPS_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "Task2").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the repository root. Set MLOPS_PROJECT_ROOT explicitly."
    )


__all__ = ["find_project_root"]
