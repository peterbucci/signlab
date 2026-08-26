from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.reproducibility import evidence, provenance, remote, stages


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def test_stage_command_runs_registered_fixture_without_printing_a_path(
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


def test_stage_command_redacts_application_failure(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-stage-sentinel"

    def fail(*_arguments: object) -> None:
        raise stages.ReproductionStageError(secret)

    monkeypatch.setattr(stages, "run_reproduction_stage", fail)

    result = runner.invoke(cli.app, ["data", "run-reproduction-stage", "ingest"])

    assert result.exit_code == 1
    assert result.output.strip() == "Synthetic reproduction stage failed."
    assert secret not in result.output


def test_private_remote_command_reports_only_non_sensitive_booleans(
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
    assert "s3://" not in result.output


def test_private_remote_command_redacts_failure(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "s3://private-bucket/private-path"

    def fail(_root: Path) -> None:
        raise remote.DvcRemoteConfigurationError(secret)

    monkeypatch.setattr(remote, "configure_private_dvc_remote", fail)

    result = runner.invoke(cli.app, ["data", "configure-private-remote"])

    assert result.exit_code == 1
    assert result.output.strip() == "Private DVC remote configuration failed."
    assert secret not in result.output


def test_snapshot_command_forwards_role_and_relative_output_then_prints_only_digest(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = object()
    captured_roles: list[str] = []
    writes: list[tuple[object, Path, str]] = []
    digest = "sha256:" + "a" * 64
    monkeypatch.chdir(tmp_path)

    def capture(_root: Path, *, metadata_repository_role: str) -> object:
        captured_roles.append(metadata_repository_role)
        return snapshot

    monkeypatch.setattr(evidence, "capture_dvc_snapshot", capture)
    monkeypatch.setattr(
        evidence,
        "write_dvc_snapshot",
        lambda value, root, output: writes.append((value, root, output)),
    )
    monkeypatch.setattr(provenance, "dvc_snapshot_digest", lambda _snapshot: digest)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "capture-reproduction-snapshot",
            "--repository-role",
            "protected-metadata",
            "--output",
            "reports/reproduction/protected.json",
        ],
    )

    assert result.exit_code == 0
    assert captured_roles == ["protected-metadata"]
    assert writes == [(snapshot, tmp_path, "reports/reproduction/protected.json")]
    assert result.output.strip() == f"DVC reproduction snapshot SHA-256: {digest}"
    assert str(tmp_path) not in result.output


def test_snapshot_command_redacts_capture_and_write_failures(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-evidence-sentinel"

    def fail(_root: Path, *, metadata_repository_role: str) -> object:
        del metadata_repository_role
        raise evidence.DvcEvidenceError(secret)

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
    assert secret not in result.output
    assert "Traceback" not in result.output
