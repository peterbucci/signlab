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
