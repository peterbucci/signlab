from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.commands import contracts as contracts_command
from signlab.contracts import pipeline
from signlab.contracts.resources import ContractResourceError, build_example_contract_chain

SUPPORTED_VERSIONS = (
    "dataset-manifest/1",
    "dataset-manifest/2",
    "model-manifest/1",
    "preprocessing-plan/1",
    "resolved-configuration/1",
    "run-record/1",
    "split-manifest/1",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def test_versions_lists_the_exact_supported_contract_versions(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["contracts", "versions"])

    assert result.exit_code == 0
    assert tuple(result.output.splitlines()) == SUPPORTED_VERSIONS


def test_validate_reads_bytes_and_prints_only_version_and_digest(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_token = "participant_private_identifier"
    path = tmp_path / "private-participant-path.json"
    payload = f'{{"secret":"{sensitive_token}"}}'.encode()
    path.write_bytes(payload)
    checked = SimpleNamespace(
        schema_version="dataset-manifest/1",
        dataset_id=sensitive_token,
    )
    calls: list[object] = []

    def fake_validate(document: object) -> object:
        calls.append(document)
        return checked

    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(pipeline, "validate_contract", fake_validate)
    monkeypatch.setattr(
        pipeline,
        "contract_digest",
        lambda document: digest if document is checked else pytest.fail("wrong contract"),
    )

    result = runner.invoke(cli.app, ["contracts", "validate", str(path)])

    assert result.exit_code == 0
    assert calls == [payload]
    assert result.output.strip() == f"dataset-manifest/1 {digest}"
    assert sensitive_token not in result.output
    assert path.name not in result.output
    assert str(tmp_path) not in result.output


def test_validate_accepts_a_real_pipeline_contract(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    contract = build_example_contract_chain()[0]
    path = tmp_path / "participant-private-path.json"
    path.write_text(contract.model_dump_json(round_trip=True), encoding="utf-8")

    result = runner.invoke(cli.app, ["contracts", "validate", str(path)])

    assert result.exit_code == 0
    assert result.output.strip() == (
        f"{contract.schema_version} {pipeline.contract_digest(contract)}"
    )
    assert contract.dataset_id not in result.output
    assert path.name not in result.output
    assert str(tmp_path) not in result.output


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_validate_unreadable_path_returns_a_stable_redacted_failure(
    runner: CliRunner,
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "participant-private-file.json"
    if kind == "directory":
        path.mkdir()

    result = runner.invoke(cli.app, ["contracts", "validate", str(path)])

    assert result.exit_code == 1
    assert result.output.strip() == ("Contract validation failed: contract file could not be read.")
    assert path.name not in result.output
    assert str(tmp_path) not in result.output


@pytest.mark.parametrize(
    "document",
    [
        b"not-json-participant-private-token",
        b'{"schema_version":"dataset-manifest/1","private":"participant-private-token"}',
        b'{"schema_version":"dataset-manifest/1","schema_version":"participant-private-token"}',
    ],
    ids=["malformed", "invalid", "duplicate"],
)
def test_validate_invalid_contracts_never_echo_input_or_path(
    runner: CliRunner,
    tmp_path: Path,
    document: bytes,
) -> None:
    path = tmp_path / "participant-private-path.json"
    path.write_bytes(document)

    result = runner.invoke(cli.app, ["contracts", "validate", str(path)])

    assert result.exit_code == 1
    assert result.output.strip() == (
        "Contract validation failed: contract is invalid or unsupported."
    )
    assert "participant-private" not in result.output
    assert path.name not in result.output
    assert str(tmp_path) not in result.output


@pytest.mark.parametrize(
    "document",
    [
        b'{"private":"participant-private-token"}',
        b'{"schema_version":42,"private":"participant-private-token"}',
        b'{"schema_version":"dataset-manifest/3","private":"participant-private-token"}',
    ],
    ids=["missing", "wrong-type", "future-version"],
)
def test_validate_unsupported_versions_provide_safe_migration_guidance(
    runner: CliRunner,
    tmp_path: Path,
    document: bytes,
) -> None:
    path = tmp_path / "participant-private-path.json"
    path.write_bytes(document)

    result = runner.invoke(cli.app, ["contracts", "validate", str(path)])

    assert result.exit_code == 1
    assert "supported: dataset-manifest/1" in result.output
    assert "model-manifest/1" in result.output
    assert "docs/contracts.md#compatibility-and-migration" in result.output
    assert "participant-private" not in result.output
    assert path.name not in result.output
    assert str(tmp_path) not in result.output


def test_validate_resources_reports_success(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def validate_resources() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        contracts_command,
        "validate_packaged_contract_resources",
        validate_resources,
    )

    result = runner.invoke(cli.app, ["contracts", "validate-resources"])

    assert result.exit_code == 0
    assert result.output.strip() == "Packaged contract schemas and examples are valid."
    assert calls == 1


def test_validate_resources_returns_a_stable_redacted_failure(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_token = "participant-private-resource"

    def fail_validation() -> None:
        raise ContractResourceError(sensitive_token)

    monkeypatch.setattr(
        contracts_command,
        "validate_packaged_contract_resources",
        fail_validation,
    )

    result = runner.invoke(cli.app, ["contracts", "validate-resources"])

    assert result.exit_code == 1
    assert result.output.strip() == "Packaged contract resource validation failed."
    assert sensitive_token not in result.output
    assert "Traceback" not in result.output


def test_contract_cli_import_does_not_load_optional_ml_runtimes() -> None:
    probe = (
        "import sys; import signlab.commands.contracts; "
        "blocked = {'mediapipe', 'mlflow', 'onnxruntime', 'torch'} & set(sys.modules); "
        "raise SystemExit(','.join(sorted(blocked)) if blocked else 0)"
    )

    subprocess.run([sys.executable, "-c", probe], check=True)


def test_validate_uses_pipeline_contract_error_without_exposing_detail(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private-contract.json"
    path.write_bytes(b"{}")

    def fail_validation(_document: object) -> pipeline.CoreContract:
        raise pipeline.PipelineContractError("participant-private-detail")

    monkeypatch.setattr(pipeline, "validate_contract", fail_validation)

    result = runner.invoke(cli.app, ["contracts", "validate", str(path)])

    assert result.exit_code == 1
    assert "participant-private-detail" not in result.output
    assert result.output.strip() == (
        "Contract validation failed: contract is invalid or unsupported."
    )
