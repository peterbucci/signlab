from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTION_REVISION_PATTERN = re.compile(r"^\s*uses:\s+\S+@([^\s#]+)", re.MULTILINE)


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_python_and_uv_versions_are_pinned() -> None:
    pyproject = _read("pyproject.toml")

    assert _read(".python-version").strip() == "3.12.14"
    assert 'requires-python = ">=3.12,<3.13"' in pyproject
    assert 'required-version = "==0.12.6"' in pyproject
    assert 'requires = ["hatchling==1.32.0"]' in pyproject
    assert '"hatchling==1.32.0"' in pyproject


def test_ci_uses_immutable_actions_and_required_cross_platform_gates() -> None:
    workflow = _read(".github/workflows/ci.yml")
    revisions = ACTION_REVISION_PATTERN.findall(workflow)

    assert revisions
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in revisions)
    assert "ubuntu-24.04" in workflow
    assert "windows-latest" in workflow
    for command in (
        "uv lock --check",
        "scripts/check_repository_hygiene.py",
        "uv build --no-build-isolation",
        "scripts/verify_distribution.py dist",
        "gitleaks/gitleaks-action@",
    ):
        assert command in workflow


def test_private_and_generated_artifacts_are_ignored() -> None:
    gitignore = _read(".gitignore")

    for pattern in (
        ".env.*",
        "data/raw/",
        "artifacts/",
        "models/",
        "mlruns/",
        "*.onnx",
        "*.parquet",
        "*.sqlite",
    ):
        assert pattern in gitignore
