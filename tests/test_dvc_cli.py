from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.reproducibility import evidence, provenance, remote, stages


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def test_reproduction_stage_command_delegates_without_printing_a_path(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        stages,
        "run_reproduction_stage",
        lambda stage, root: calls.append((stage, root)),
    )

    result = runner.invoke(cli.app, ["data", "run-reproduction-stage", "ingest"])

    assert result.exit_code == 0
    assert calls == [("ingest", tmp_path)]
    assert result.output.strip() == "Synthetic reproduction stage completed: ingest."
    assert str(tmp_path) not in result.output


def test_reproduction_stage_command_redacts_application_failure(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object) -> None:
        raise stages.ReproductionStageError("private-sentinel")

    monkeypatch.setattr(stages, "run_reproduction_stage", fail)

    result = runner.invoke(cli.app, ["data", "run-reproduction-stage", "ingest"])

    assert result.exit_code == 1
    assert result.output.strip() == "Synthetic reproduction stage failed."
    assert "private" not in result.output


def test_private_remote_command_reports_only_boolean_configuration_facts(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        remote,
        "configure_private_dvc_remote",
        lambda _root: remote.DvcRemoteConfigurationResult(
            remote_name="private",
            endpoint_configured=True,
            region_configured=False,
        ),
    )

    result = runner.invoke(cli.app, ["data", "configure-private-remote"])

    assert result.exit_code == 0
    assert result.output.strip() == (
        "Private DVC remote configured locally: endpoint override true, region override false."
    )
    assert "s3" not in result.output.casefold()


def test_private_remote_command_redacts_configuration_failure(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_root: Path) -> None:
        raise remote.DvcRemoteConfigurationError("private-sentinel")

    monkeypatch.setattr(remote, "configure_private_dvc_remote", fail)

    result = runner.invoke(cli.app, ["data", "configure-private-remote"])

    assert result.exit_code == 1
    assert result.output.strip() == "Private DVC remote configuration failed."
    assert "sentinel" not in result.output


def test_snapshot_command_captures_then_writes_ignored_evidence(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = object()
    calls: list[tuple[object, Path, str]] = []
    roles: list[str] = []
    monkeypatch.chdir(tmp_path)

    def capture(_root: Path, *, metadata_repository_role: str) -> object:
        roles.append(metadata_repository_role)
        return snapshot

    monkeypatch.setattr(evidence, "capture_dvc_snapshot", capture)
    monkeypatch.setattr(
        evidence,
        "write_dvc_snapshot",
        lambda value, root, output: calls.append((value, root, output)),
    )
    monkeypatch.setattr(provenance, "dvc_snapshot_digest", lambda _value: "sha256:" + "a" * 64)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "capture-reproduction-snapshot",
            "--repository-role",
            "protected-metadata",
        ],
    )

    assert result.exit_code == 0
    assert roles == ["protected-metadata"]
    assert calls == [(snapshot, tmp_path, "reports/reproduction/dvc-snapshot.json")]
    assert result.output.strip() == "DVC reproduction snapshot SHA-256: sha256:" + "a" * 64
    assert str(tmp_path) not in result.output


def test_snapshot_command_redacts_capture_failure(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_root: Path, *, metadata_repository_role: str) -> object:
        del metadata_repository_role
        raise evidence.DvcEvidenceError

    monkeypatch.setattr(evidence, "capture_dvc_snapshot", fail)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "capture-reproduction-snapshot",
            "--repository-role",
            "public-fixture",
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == "DVC reproduction snapshot could not be captured."
    assert "Traceback" not in result.output


def test_dataset_cli_forwards_explicit_row_artifact_verification(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from signlab.datasets import bundle

    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"fixture")
    calls: list[bool] = []

    def validate(*_args: object, **kwargs: object) -> object:
        calls.append(bool(kwargs["verify_row_artifacts"]))
        return SimpleNamespace(
            data_sha256="sha256:" + "a" * 64,
            parquet_table_bytes="verified",
            semantic_integrity="verified",
            artifact_byte_integrity="verified",
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
            "--verify-row-artifacts",
        ],
    )

    assert result.exit_code == 0
    assert calls == [True]
    assert "Referenced row artifacts: verified" in result.output
