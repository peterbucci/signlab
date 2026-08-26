"""CLI adapters for fail-closed participant-data governance workflows."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import typer

from signlab.contracts.governance import (
    GovernanceContractError,
    assert_receipt_grant_consistent,
    new_participant_id,
    validate_collection_readiness,
    validate_consent_event_log,
    validate_consent_receipt,
    validate_lineage_inventory,
    validate_recording_consent_grant,
    validate_withdrawal_report,
    validate_withdrawal_request,
)
from signlab.governance.resources import (
    GovernanceResourceError,
    build_collection_readiness,
    build_governance_policy,
    render_json_document,
    validate_packaged_governance_resources,
)
from signlab.governance.withdrawal import (
    WithdrawalPlanningError,
    plan_withdrawal_dry_run,
    render_withdrawal_report_markdown,
)

app = typer.Typer(
    help="Validate consent, collection readiness, and withdrawal evidence.",
    no_args_is_help=True,
)


def _read_external(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise GovernanceContractError("input file could not be read") from error


def _write_new_atomic(path: Path, content: bytes) -> None:
    """Atomically publish a complete new file without replacing existing evidence."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".signlab-governance-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise GovernanceContractError("output already exists; evidence was not replaced") from error
    except OSError as error:
        raise GovernanceContractError("output file could not be created") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fail(error: GovernanceContractError) -> None:
    typer.echo(f"Governance operation failed: {error}", err=True)
    raise typer.Exit(code=1) from error


@app.command("new-participant-id")
def new_participant_id_command() -> None:
    """Generate a random 128-bit pseudonymous participant identifier."""

    typer.echo(new_participant_id())


@app.command("policy-show")
def policy_show_command() -> None:
    """Print the packaged machine-readable engineering policy."""

    try:
        typer.echo(render_json_document(build_governance_policy()), nl=False)
    except GovernanceContractError as error:
        _fail(error)


@app.command("readiness-check")
def readiness_check_command(
    readiness_path: Annotated[
        Path | None,
        typer.Argument(
            help="Optional readiness JSON; omit to inspect the packaged blocked template.",
        ),
    ] = None,
) -> None:
    """Validate a readiness record while keeping v1 collection fail-closed."""

    try:
        policy = build_governance_policy()
        readiness = (
            build_collection_readiness()
            if readiness_path is None
            else validate_collection_readiness(_read_external(readiness_path))
        )
        if readiness.policy_id != policy.policy_id or readiness.taxonomy != policy.taxonomy:
            raise GovernanceContractError(
                "collection readiness does not reference the packaged policy and taxonomy"
            )
    except GovernanceContractError as error:
        _fail(error)
    typer.echo(f"Collection status: {readiness.status}")
    for blocker in readiness.blockers:
        typer.echo(f"- {blocker}")
    raise typer.Exit(code=1)


@app.command("validate-consent")
def validate_consent_command(
    receipt_path: Annotated[
        Path,
        typer.Argument(help="Pseudonymous consent-receipt JSON."),
    ],
) -> None:
    """Validate a consent receipt without printing its participant identifier."""

    try:
        validate_consent_receipt(_read_external(receipt_path))
    except GovernanceContractError as error:
        _fail(error)
    typer.echo(
        "Consent receipt, registered documents, and explicit scope are internally valid; "
        "identity-vault authenticity was not externally verified."
    )


@app.command("validate-recording")
def validate_recording_command(
    receipt_path: Annotated[
        Path,
        typer.Argument(help="Pseudonymous consent-receipt JSON."),
    ],
    grant_path: Annotated[
        Path,
        typer.Argument(help="Recording-level consent-grant JSON."),
    ],
    event_log_path: Annotated[
        Path,
        typer.Argument(help="Attested consent-event-log JSON complete through capture."),
    ],
) -> None:
    """Validate one grant against its receipt and complete consent lifecycle."""

    try:
        receipt = validate_consent_receipt(_read_external(receipt_path))
        grant = validate_recording_consent_grant(_read_external(grant_path))
        event_log = validate_consent_event_log(_read_external(event_log_path))
        assert_receipt_grant_consistent(receipt, grant, event_log)
    except GovernanceContractError as error:
        _fail(error)
    typer.echo(
        "Recording grant is receipt-bounded and lifecycle-valid; identity-vault and "
        "event-log attestations were not externally verified."
    )


@app.command("withdrawal-dry-run")
def withdrawal_dry_run_command(
    request_path: Annotated[
        Path,
        typer.Argument(help="Pseudonymous complete-withdrawal request JSON."),
    ],
    inventory_path: Annotated[
        Path,
        typer.Argument(help="Complete governed-asset lineage inventory JSON."),
    ],
    as_of: Annotated[
        str,
        typer.Option("--as-of", help="Explicit UTC report time (YYYY-MM-DDTHH:MM:SSZ)."),
    ],
    output_path: Annotated[
        Path,
        typer.Option("--output", help="New report file; existing files are never replaced."),
    ],
    output_format: Annotated[
        Literal["json", "markdown"],
        typer.Option("--format", help="Write canonical evidence as json or markdown."),
    ] = "json",
) -> None:
    """Trace every affected descendant without mutating data or external stores."""

    try:
        request = validate_withdrawal_request(_read_external(request_path))
        inventory = validate_lineage_inventory(_read_external(inventory_path))
        report = plan_withdrawal_dry_run(request, inventory, generated_at=as_of)
        if output_format == "json":
            content = render_json_document(report).encode("utf-8")
        else:
            content = render_withdrawal_report_markdown(report).encode("utf-8")
        _write_new_atomic(output_path, content)
    except (GovernanceContractError, WithdrawalPlanningError) as error:
        _fail(error)
    typer.echo(
        "Withdrawal dry run complete: "
        f"{report.affected_asset_count} affected assets; storage mutations performed: false."
    )


@app.command("validate-withdrawal")
def validate_withdrawal_command(
    request_path: Annotated[
        Path,
        typer.Argument(help="Pseudonymous complete-withdrawal request JSON."),
    ],
    inventory_path: Annotated[
        Path,
        typer.Argument(help="Complete governed-asset lineage inventory JSON."),
    ],
    report_path: Annotated[
        Path,
        typer.Argument(help="Withdrawal dry-run JSON to recompute and validate."),
    ],
) -> None:
    """Reject a dry-run report with omitted, extra, or changed lineage impacts."""

    try:
        request = validate_withdrawal_request(_read_external(request_path))
        inventory = validate_lineage_inventory(_read_external(inventory_path))
        report = validate_withdrawal_report(_read_external(report_path))
        expected = plan_withdrawal_dry_run(
            request,
            inventory,
            generated_at=report.generated_at,
        )
        if report != expected:
            raise GovernanceContractError(
                "withdrawal report does not match the complete deterministic closure"
            )
    except (GovernanceContractError, WithdrawalPlanningError) as error:
        _fail(error)
    typer.echo("Withdrawal report is complete, deterministic, and read-only.")


@app.command("evidence-check")
def evidence_check_command() -> None:
    """Reproduce and validate all installed governance resources and evidence."""

    try:
        validate_packaged_governance_resources()
    except (GovernanceContractError, GovernanceResourceError) as error:
        _fail(error)
    typer.echo("Packaged governance policy, schemas, examples, and dry run are valid.")
