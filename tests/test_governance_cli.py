from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.contracts.governance import validate_withdrawal_report


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def _copy_resource(relative_name: str, destination: Path) -> Path:
    destination.write_bytes(
        files("signlab.resources.governance").joinpath(*relative_name.split("/")).read_bytes()
    )
    return destination


def test_governance_help_exposes_the_complete_non_destructive_workflow(
    runner: CliRunner,
) -> None:
    result = runner.invoke(cli.app, ["governance", "--help"])

    assert result.exit_code == 0
    for command in (
        "new-participant-id",
        "policy-show",
        "readiness-check",
        "validate-consent",
        "validate-recording",
        "withdrawal-dry-run",
        "validate-withdrawal",
        "evidence-check",
    ):
        assert command in result.output
    assert "delete" not in result.output.casefold()


def test_new_participant_id_emits_random_128_bit_pseudonyms(runner: CliRunner) -> None:
    first = runner.invoke(cli.app, ["governance", "new-participant-id"])
    second = runner.invoke(cli.app, ["governance", "new-participant-id"])

    assert first.exit_code == second.exit_code == 0
    assert re.fullmatch(r"participant_[0-9a-f]{32}\n", first.output)
    assert first.output != second.output


def test_policy_show_outputs_valid_machine_readable_policy(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["governance", "policy-show"])

    assert result.exit_code == 0
    document = json.loads(result.output)
    assert document["schema_version"] == "governance-policy/1"
    assert document["deletion_includes_backups"] is True


def test_readiness_check_fails_closed_and_lists_documented_actions(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["governance", "readiness-check"])

    assert result.exit_code == 1
    assert "Collection status: blocked" in result.output
    assert "ethics_legal_institutional" in result.output
    assert "participant_contact_process" in result.output


def test_readiness_check_validates_an_external_record_against_packaged_policy(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    readiness = _copy_resource(
        "collection-readiness.template.json",
        tmp_path / "readiness.json",
    )

    result = runner.invoke(
        cli.app,
        ["governance", "readiness-check", str(readiness)],
    )

    assert result.exit_code == 1
    assert "Collection status: blocked" in result.output
    assert str(tmp_path) not in result.output

    document = json.loads(readiness.read_text(encoding="utf-8"))
    document["policy_id"] = "policy_ffffffffffffffffffffffffffffffff"
    readiness.write_text(json.dumps(document), encoding="utf-8")

    mismatched = runner.invoke(
        cli.app,
        ["governance", "readiness-check", str(readiness)],
    )

    assert mismatched.exit_code == 1
    assert "packaged policy and taxonomy" in mismatched.output
    assert "ffffffff" not in mismatched.output


def test_validate_consent_and_recording_examples(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    receipt = _copy_resource(
        "examples/consent-receipt.example.json",
        tmp_path / "receipt.json",
    )
    grant = _copy_resource(
        "examples/recording-consent-grant.example.json",
        tmp_path / "grant.json",
    )
    event_log = _copy_resource(
        "examples/consent-event-log.example.json",
        tmp_path / "event-log.json",
    )

    receipt_result = runner.invoke(
        cli.app,
        ["governance", "validate-consent", str(receipt)],
    )
    grant_result = runner.invoke(
        cli.app,
        [
            "governance",
            "validate-recording",
            str(receipt),
            str(grant),
            str(event_log),
        ],
    )

    assert receipt_result.exit_code == grant_result.exit_code == 0
    assert "internally valid" in receipt_result.output
    assert "not externally verified" in receipt_result.output
    assert "receipt-bounded" in grant_result.output
    assert "lifecycle-valid" in grant_result.output
    assert "not externally verified" in grant_result.output
    assert "participant_" not in receipt_result.output + grant_result.output


def test_withdrawal_dry_run_writes_new_deterministic_evidence_without_mutating_inputs(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    request = _copy_resource(
        "examples/withdrawal-request.example.json",
        tmp_path / "request.json",
    )
    inventory = _copy_resource(
        "examples/lineage-inventory.example.json",
        tmp_path / "inventory.json",
    )
    before = request.read_bytes(), inventory.read_bytes()
    output = tmp_path / "report.json"

    result = runner.invoke(
        cli.app,
        [
            "governance",
            "withdrawal-dry-run",
            str(request),
            str(inventory),
            "--as-of",
            "2026-08-26T14:00:00Z",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    report = validate_withdrawal_report(output.read_bytes())
    assert report.affected_asset_count == 12
    assert "storage mutations performed: false" in result.output
    assert (request.read_bytes(), inventory.read_bytes()) == before

    existing = output.read_bytes()
    second = runner.invoke(
        cli.app,
        [
            "governance",
            "withdrawal-dry-run",
            str(request),
            str(inventory),
            "--as-of",
            "2026-08-26T14:00:00Z",
            "--output",
            str(output),
        ],
    )
    assert second.exit_code == 1
    assert "was not replaced" in second.output
    assert output.read_bytes() == existing
    assert str(tmp_path) not in second.output


def test_withdrawal_markdown_and_report_validation_are_reproducible(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    request = _copy_resource(
        "examples/withdrawal-request.example.json",
        tmp_path / "request.json",
    )
    inventory = _copy_resource(
        "examples/lineage-inventory.example.json",
        tmp_path / "inventory.json",
    )
    report = _copy_resource(
        "evidence/withdrawal-dry-run-v1.json",
        tmp_path / "report.json",
    )
    markdown = tmp_path / "report.md"

    rendered = runner.invoke(
        cli.app,
        [
            "governance",
            "withdrawal-dry-run",
            str(request),
            str(inventory),
            "--as-of",
            "2026-08-26T14:00:00Z",
            "--output",
            str(markdown),
            "--format",
            "markdown",
        ],
    )
    checked = runner.invoke(
        cli.app,
        [
            "governance",
            "validate-withdrawal",
            str(request),
            str(inventory),
            str(report),
        ],
    )

    assert rendered.exit_code == checked.exit_code == 0
    evidence = markdown.read_text(encoding="utf-8")
    assert "Storage mutations performed: `false`" in evidence
    assert "Simulated non-tombstone state: `invalidated`" in evidence
    assert "Withdrawal tombstones: `retained`" in evidence
    assert "participant_" not in evidence
    assert "complete, deterministic, and read-only" in checked.output


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_external_input_failures_do_not_echo_paths_or_identifiers(
    runner: CliRunner,
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "participant-name-must-not-echo.json"
    if kind == "directory":
        path.mkdir()

    result = runner.invoke(cli.app, ["governance", "validate-consent", str(path)])

    assert result.exit_code == 1
    assert "input file could not be read" in result.output
    assert str(tmp_path) not in result.output
    assert path.name not in result.output


def test_identity_fields_and_values_are_rejected_without_echo(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    path = tmp_path / "participant-name-must-not-echo.json"
    path.write_text(
        json.dumps({"name": "Alice Private", "email": "alice@example.invalid"}),
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["governance", "validate-consent", str(path)])

    assert result.exit_code == 1
    assert "prohibited identity" in result.output
    assert "Alice" not in result.output
    assert "example.invalid" not in result.output
    assert path.name not in result.output


def test_installed_governance_evidence_check_is_self_contained(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["governance", "evidence-check"])

    assert result.exit_code == 0
    assert result.output.strip() == (
        "Packaged governance policy, schemas, examples, and dry run are valid."
    )
