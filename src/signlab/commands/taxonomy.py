"""CLI adapter for inspecting and validating the gesture taxonomy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from signlab.contracts.taxonomy import (
    GestureTaxonomy,
    TaxonomyContractError,
    load_builtin_taxonomy,
    taxonomy_reference,
    validate_packaged_taxonomy_resources,
    validate_taxonomy,
)

app = typer.Typer(
    help="Inspect and validate the immutable gesture vocabulary.",
    no_args_is_help=True,
)


def _load(path: Path | None) -> tuple[str, GestureTaxonomy]:
    if path is None:
        return "built-in", load_builtin_taxonomy()
    try:
        return "external", validate_taxonomy(path.read_bytes())
    except OSError as error:
        raise TaxonomyContractError("taxonomy file could not be read") from error


@app.command("validate")
def validate_command(
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Optional taxonomy JSON; omit to validate the packaged taxonomy.",
        ),
    ] = None,
) -> None:
    """Validate a taxonomy and print its portable identity."""
    try:
        source, taxonomy = _load(path)
        reference = taxonomy_reference(taxonomy)
    except TaxonomyContractError as error:
        typer.echo(f"Taxonomy validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Taxonomy valid ({source}): {reference.id}@{reference.version} {reference.sha256}")


@app.command("validate-resources")
def validate_resources_command() -> None:
    """Validate every packaged taxonomy and generated schema resource."""
    try:
        load_builtin_taxonomy()
        validate_packaged_taxonomy_resources()
    except TaxonomyContractError as error:
        typer.echo(f"Taxonomy resource validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Packaged taxonomy and schemas are valid.")


@app.command("show")
def show_command() -> None:
    """Print the packaged taxonomy as stable, reviewable JSON."""
    try:
        taxonomy = load_builtin_taxonomy()
    except TaxonomyContractError as error:
        typer.echo(f"Taxonomy validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        json.dumps(
            taxonomy.model_dump(mode="json", round_trip=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
