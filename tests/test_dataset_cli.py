from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.datasets import bundle


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def test_validate_dataset_reports_each_verification_boundary(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "private-dataset-manifest.json"
    manifest.write_bytes(b"synthetic-manifest")
    calls: list[tuple[bytes, Path, object, bool]] = []

    def validate(
        document: bytes,
        workspace_root: Path,
        *,
        split: object,
        verify_row_artifacts: bool,
    ) -> object:
        calls.append((document, workspace_root, split, verify_row_artifacts))
        return SimpleNamespace(
            data_sha256="sha256:" + "a" * 64,
            parquet_table_bytes="verified",
            semantic_integrity="verified",
            artifact_byte_integrity="not_checked",
            split_compatibility="not_checked",
            consent_authorization="not_checked",
        )

    monkeypatch.setattr(bundle, "validate_dataset_bundle", validate)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-dataset",
            str(manifest),
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert calls == [(b"synthetic-manifest", tmp_path, None, False)]
    assert result.output.splitlines() == [
        "Dataset data SHA-256: sha256:" + "a" * 64,
        "Parquet table bytes: verified",
        "Dataset semantic integrity: verified",
        "Referenced row artifacts: not checked",
        "Split compatibility: not checked",
        "Current consent authorization: not checked",
    ]
    assert manifest.name not in result.output
    assert str(tmp_path) not in result.output


def test_validate_dataset_redacts_failures(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "participant-private-manifest.json"
    manifest.write_bytes(b"participant-private-content")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise ValueError("participant-private-detail")

    monkeypatch.setattr(bundle, "validate_dataset_bundle", fail)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-dataset",
            str(manifest),
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == (
        "Dataset validation failed: manifest, table bytes, or relationships are invalid."
    )
    assert "participant-private" not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output


def test_validate_dataset_redacts_a_missing_manifest_path(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "private-participant-manifest.json"

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-dataset",
            str(missing_manifest),
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == (
        "Dataset validation failed: manifest, table bytes, or relationships are invalid."
    )
    assert missing_manifest.name not in result.output
    assert str(tmp_path) not in result.output


def test_validate_dataset_redacts_a_missing_workspace_path(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_bytes(b"synthetic-manifest")
    missing_workspace = tmp_path / "private-participant-workspace"

    def fail(
        _document: bytes,
        workspace_root: Path,
        *,
        split: object,
        verify_row_artifacts: bool,
    ) -> object:
        assert workspace_root == missing_workspace
        assert split is None
        assert verify_row_artifacts is False
        raise OSError("seeded private workspace failure")

    monkeypatch.setattr(bundle, "validate_dataset_bundle", fail)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-dataset",
            str(manifest),
            "--workspace-root",
            str(missing_workspace),
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == (
        "Dataset validation failed: manifest, table bytes, or relationships are invalid."
    )
    assert missing_workspace.name not in result.output
    assert str(tmp_path) not in result.output


def test_unrelated_cli_import_does_not_load_pyarrow() -> None:
    probe = (
        "import sys; import signlab.cli; "
        "raise SystemExit('pyarrow' if 'pyarrow' in sys.modules else 0)"
    )

    subprocess.run([sys.executable, "-c", probe], check=True)


def test_example_bundle_command_writes_then_public_validator_accepts_it(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-bundle"

    written = runner.invoke(cli.app, ["data", "write-example-dataset", str(output)])

    assert written.exit_code == 0
    assert written.output.strip() == (
        "Synthetic bundle written; Parquet table bytes and relationships verified."
    )
    manifest = output / "dataset-manifest.json"
    assert manifest.is_file()
    assert len(tuple((output / "tables").glob("*.parquet"))) == 6

    validated = runner.invoke(
        cli.app,
        [
            "data",
            "validate-dataset",
            str(manifest),
            "--workspace-root",
            str(output),
        ],
    )

    assert validated.exit_code == 0
    assert "Parquet table bytes: verified" in validated.output
    assert "Referenced row artifacts: not checked" in validated.output
    assert "Current consent authorization: not checked" in validated.output
    assert str(tmp_path) not in validated.output


def test_dataset_resource_command_validates_packaged_review_artifacts(
    runner: CliRunner,
) -> None:
    result = runner.invoke(cli.app, ["data", "validate-resources"])

    assert result.exit_code == 0
    assert result.output.strip() == "Packaged dataset and ingest schemas and examples are valid."
