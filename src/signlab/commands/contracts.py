"""Privacy-safe CLI adapter for portable pipeline contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from signlab.contracts import pipeline
from signlab.contracts.resources import (
    ContractResourceError,
    validate_packaged_contract_resources,
)

app = typer.Typer(
    help="Inspect and validate portable pipeline contracts.",
    no_args_is_help=True,
)


@app.command("versions")
def versions_command() -> None:
    """List the exact pipeline contract versions supported by this release."""
    for schema_version in sorted(pipeline.CORE_CONTRACT_MODELS):
        typer.echo(schema_version)


@app.command("validate")
def validate_command(
    path: Annotated[
        Path,
        typer.Argument(help="Pipeline contract JSON to validate."),
    ],
) -> None:
    """Validate one contract and print only its portable content identity."""
    try:
        document = path.read_bytes()
    except OSError as error:
        typer.echo("Contract validation failed: contract file could not be read.", err=True)
        raise typer.Exit(code=1) from error
    try:
        checked = pipeline.validate_contract(document)
        digest = pipeline.contract_digest(checked)
    except pipeline.ContractVersionError as error:
        supported = ", ".join(sorted(pipeline.CORE_CONTRACT_MODELS))
        typer.echo(
            "Contract validation failed: unsupported schema version; "
            f"supported: {supported}; see "
            "docs/contracts.md#compatibility-and-migration.",
            err=True,
        )
        raise typer.Exit(code=1) from error
    except pipeline.PipelineContractError as error:
        typer.echo("Contract validation failed: contract is invalid or unsupported.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{checked.schema_version} {digest}")


@app.command("validate-resources")
def validate_resources_command() -> None:
    """Validate every packaged pipeline schema and example resource."""
    try:
        validate_packaged_contract_resources()
    except ContractResourceError as error:
        typer.echo("Packaged contract resource validation failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Packaged contract schemas and examples are valid.")
