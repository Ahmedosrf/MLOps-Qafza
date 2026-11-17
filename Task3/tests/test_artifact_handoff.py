"""Verify DVC reproduction and restoration without publishing user data."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from Task3.tests.ci_bundle import make_bundle


def test_dvc_publish_and_restore(tmp_path):
    root = Path(__file__).resolve().parents[2]
    repo = tmp_path / "repo"
    repo.mkdir()
    for directory in ("Task3", ".dvc"):
        shutil.copytree(
            root / directory,
            repo / directory,
            ignore=shutil.ignore_patterns("__pycache__", "logs", "*.joblib"),
        )
    (repo / "Task2/artifacts/models").mkdir(parents=True)
    make_bundle(repo / "Task2/artifacts/models/06_inference_bundle.joblib")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "DVC_NO_ANALYTICS": "true",
    }
    for variable in ("MLOPS_PROJECT_ROOT", "MODEL_BUNDLE_PATH", "MODEL_SOURCE"):
        env.pop(variable, None)

    def dvc(*args):
        subprocess.run(
            [sys.executable, "-m", "dvc", *args],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    dvc("repro", "Task3/dvc.yaml")
    dvc("remote", "add", "-d", "local", str(tmp_path / "remote"))
    dvc("push", "Task3/dvc.yaml")
    output = repo / "Task3/models/06_inference_bundle.joblib"
    expected = output.read_bytes()
    output.unlink()
    shutil.rmtree(repo / ".dvc/cache")
    dvc("pull", "Task3/dvc.yaml")
    assert output.read_bytes() == expected
    assert (repo / "Task3/dvc.lock").is_file()
