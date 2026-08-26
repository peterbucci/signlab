"""Dataset and legacy-evidence command group."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from signlab.commands._group import create_group
from signlab.legacy.exporter import LegacyExportError, export_legacy_evidence
from signlab.legacy.validator import validate_legacy_export

app = create_group(
    help_text="Capture, import, validate, version, and split consent-approved datasets."
)


@app.command("configure-private-remote")
def configure_private_remote_command() -> None:
    """Configure a credential-free private S3 DVC remote in ignored local state."""

    from signlab.reproducibility.remote import (
        DvcRemoteConfigurationError,
        configure_private_dvc_remote,
    )

    try:
        result = configure_private_dvc_remote(Path.cwd())
    except (OSError, DvcRemoteConfigurationError) as error:
        typer.echo("Private DVC remote configuration failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Private DVC remote configured locally: "
        f"endpoint override {str(result.endpoint_configured).lower()}, "
        f"region override {str(result.region_configured).lower()}."
    )


@app.command("capture-reproduction-snapshot")
def capture_reproduction_snapshot_command(
    metadata_repository_role: Annotated[
        Literal["public-fixture", "protected-metadata"],
        typer.Option(
            "--repository-role",
            help=(
                "Declare whether this checkout is the public fixture or "
                "protected metadata repository."
            ),
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="New repository-relative JSON path beneath reports/reproduction/.",
        ),
    ] = Path("reports/reproduction/dvc-snapshot.json"),
) -> None:
    """Capture clean Git/DVC identities for later experiment tracking."""

    from signlab.reproducibility.evidence import (
        DvcEvidenceError,
        capture_dvc_snapshot,
        write_dvc_snapshot,
    )
    from signlab.reproducibility.provenance import dvc_snapshot_digest

    try:
        snapshot = capture_dvc_snapshot(
            Path.cwd(),
            metadata_repository_role=metadata_repository_role,
        )
        write_dvc_snapshot(snapshot, Path.cwd(), output.as_posix())
        snapshot_sha256 = dvc_snapshot_digest(snapshot)
    except (OSError, DvcEvidenceError, ValueError) as error:
        typer.echo("DVC reproduction snapshot could not be captured.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"DVC reproduction snapshot SHA-256: {snapshot_sha256}")


@app.command("run-reproduction-stage")
def run_reproduction_stage_command(
    stage: Annotated[
        Literal["ingest", "validate", "extract", "quality", "split", "feature"],
        typer.Argument(help="Registered public-fixture stage to execute."),
    ],
) -> None:
    """Run one deterministic stage in the synthetic DVC proof graph."""

    from signlab.reproducibility.stages import ReproductionStageError, run_reproduction_stage

    try:
        run_reproduction_stage(stage, Path.cwd())
    except (OSError, ReproductionStageError) as error:
        typer.echo("Synthetic reproduction stage failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Synthetic reproduction stage completed: {stage}.")


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
    verify_row_artifacts: Annotated[
        bool,
        typer.Option(
            "--verify-row-artifacts",
            help="Stream-check every recording, materialized clip, and derived artifact.",
        ),
    ] = False,
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
            verify_row_artifacts=verify_row_artifacts,
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
