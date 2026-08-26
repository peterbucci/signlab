"""Dataset and legacy-evidence command group."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from signlab.commands._group import create_group
from signlab.legacy.exporter import LegacyExportError, export_legacy_evidence
from signlab.legacy.validator import validate_legacy_export

app = create_group(
    help_text="Capture, import, validate, version, and split consent-approved datasets."
)


@app.command("export-legacy")
def export_legacy(
    legacy_root: Annotated[
        Path,
        typer.Option(
            "--legacy-root",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Legacy project root to inspect read-only.",
        ),
    ],
    audit_snapshot: Annotated[
        Path,
        typer.Option(
            "--audit-snapshot",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Immutable legacy-state audit snapshot.",
        ),
    ],
    public_output: Annotated[
        Path,
        typer.Option(
            "--public-output",
            resolve_path=True,
            help="Empty target for portfolio-safe evidence.",
        ),
    ],
    quarantine_output: Annotated[
        Path,
        typer.Option(
            "--quarantine-output",
            resolve_path=True,
            help="Empty ignored target for private content-addressed objects.",
        ),
    ],
) -> None:
    """Export sanitized evidence and an ignored private quarantine."""
    try:
        summary = export_legacy_evidence(
            legacy_root=legacy_root,
            audit_snapshot=audit_snapshot,
            public_output=public_output,
            quarantine_output=quarantine_output,
        )
    except LegacyExportError as error:
        typer.echo(f"Legacy export failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Legacy export complete: "
        f"{summary.runs} runs, {summary.attempts} attempts, "
        f"{summary.promoted_models} promoted models, "
        f"{summary.quarantined_segments} quarantined segments."
    )


@app.command("validate-legacy")
def validate_legacy(
    public_root: Annotated[
        Path,
        typer.Option(
            "--public-root",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Public legacy-export root.",
        ),
    ],
    quarantine_root: Annotated[
        Path | None,
        typer.Option(
            "--quarantine-root",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional private quarantine root.",
        ),
    ] = None,
) -> None:
    """Validate public evidence and optionally verify every private object."""
    try:
        summary = validate_legacy_export(
            public_root=public_root,
            quarantine_root=quarantine_root,
        )
    except LegacyExportError as error:
        typer.echo(f"Legacy export validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    private_status = " and private quarantine" if summary.quarantine_verified else ""
    typer.echo(f"Validated {summary.runs} runs and {summary.attempts} attempts{private_status}.")
