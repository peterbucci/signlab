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


@app.command("validate-resources")
def validate_dataset_resources() -> None:
    """Validate packaged table schemas, Arrow snapshots, and synthetic examples."""

    from signlab.datasets.resources import validate_packaged_dataset_resources

    try:
        validate_packaged_dataset_resources()
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Packaged dataset resource validation failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Packaged dataset schemas and examples are valid.")


@app.command("write-example-dataset")
def write_example_dataset(
    output: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            resolve_path=True,
            help="New or empty output directory for the synthetic bundle.",
        ),
    ],
) -> None:
    """Write a synthetic bundle and verify its tables and relationships."""

    from signlab.datasets.bundle import write_dataset_bundle
    from signlab.datasets.resources import build_example_dataset_bundle

    try:
        example = build_example_dataset_bundle()
        write_dataset_bundle(example.manifest, example.tables, output)
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Synthetic dataset bundle could not be written and verified.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Synthetic bundle written; Parquet table bytes and relationships verified.")


@app.command("validate-dataset")
def validate_dataset(
    manifest: Annotated[
        Path,
        typer.Argument(
            help="Table-backed dataset-manifest/2 JSON document.",
        ),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option(
            "--workspace-root",
            help="Explicit root for workspace-relative Parquet table locators.",
        ),
    ],
    split: Annotated[
        Path | None,
        typer.Option(
            "--split",
            help="Optional exact split-manifest/1 to reconcile.",
        ),
    ] = None,
) -> None:
    """Verify Parquet bytes, schemas, semantic relationships, and optional split."""

    # Keep PyArrow off the import path for unrelated CLI commands.
    from signlab.contracts.pipeline import validate_split_manifest
    from signlab.datasets.bundle import validate_dataset_bundle

    try:
        manifest_document = manifest.read_bytes()
        checked_split = validate_split_manifest(split.read_bytes()) if split is not None else None
        result = validate_dataset_bundle(
            manifest_document,
            workspace_root,
            split=checked_split,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo(
            "Dataset validation failed: manifest, table bytes, or relationships are invalid.",
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(f"Dataset data SHA-256: {result.data_sha256}")
    typer.echo(f"Parquet table bytes: {result.parquet_table_bytes.replace('_', ' ')}")
    typer.echo(f"Dataset semantic integrity: {result.semantic_integrity.replace('_', ' ')}")
    typer.echo(f"Referenced row artifacts: {result.artifact_byte_integrity.replace('_', ' ')}")
    typer.echo(f"Split compatibility: {result.split_compatibility.replace('_', ' ')}")
    typer.echo(f"Current consent authorization: {result.consent_authorization.replace('_', ' ')}")


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
