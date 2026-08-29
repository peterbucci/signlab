"""Training command group."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from signlab.commands._group import create_group
from signlab.contracts.taxonomy import (
    TaxonomyContractError,
    validate_training_taxonomy_binding,
)

app = create_group(help_text="Run reproducible training experiments from validated configurations.")


@app.command("reference-baselines")
def reference_baselines_command(
    config_path: Annotated[
        Path,
        typer.Argument(help="Checked-in reference-baseline configuration JSON."),
    ],
    corpus_root: Annotated[
        Path,
        typer.Option(help="Frozen public-split root containing portable features."),
    ],
    external_manifest: Annotated[
        Path,
        typer.Option(help="Authorized external-dataset manifest for identity checks."),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="New ignored directory for local run artifacts."),
    ],
    public_report: Annotated[
        Path | None,
        typer.Option(help="Optional new path for the sanitized Markdown report."),
    ] = None,
    tracking_uri: Annotated[
        str | None,
        typer.Option(help="Optional persistent local SQLite MLflow URI."),
    ] = None,
) -> None:
    """Run the three frozen references and open test only after model selection."""

    try:
        from signlab.experiments.baselines import (
            BaselineExperimentError,
            run_reference_baselines,
        )
        from signlab.experiments.tracking import ExperimentTrackingError
    except (ImportError, ModuleNotFoundError) as error:
        typer.echo(
            "Reference baseline run failed: install the SignLab experiments extra",
            err=True,
        )
        raise typer.Exit(code=1) from error
    try:
        result = run_reference_baselines(
            config_path,
            corpus_root=corpus_root,
            external_manifest_path=external_manifest,
            output_root=output_root,
            public_report_path=public_report,
            tracking_uri=tracking_uri,
        )
    except (BaselineExperimentError, ExperimentTrackingError) as error:
        typer.echo(f"Reference baseline run failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Reference baselines verified: "
        f"selected C={result.selected_c:g}; ledger run {result.tracking.run_id}."
    )


@app.command("validate-taxonomy")
def validate_taxonomy_command(
    path: Annotated[
        Path,
        typer.Argument(help="Training taxonomy-binding JSON to validate before training."),
    ],
) -> None:
    """Fail closed when training labels or learned negatives drift from the taxonomy."""
    try:
        binding = validate_training_taxonomy_binding(path.read_bytes())
    except OSError as error:
        typer.echo("Training taxonomy validation failed: input file could not be read", err=True)
        raise typer.Exit(code=1) from error
    except TaxonomyContractError as error:
        typer.echo(f"Training taxonomy validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Training taxonomy valid: "
        f"{binding.taxonomy.id}@{binding.taxonomy.version} with learned negative 'other'."
    )
