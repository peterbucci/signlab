"""Model-export command group."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from signlab.commands._group import create_group

app = create_group(help_text="Export candidate checkpoints into immutable portable bundles.")


@app.command("browser-bundle")
def browser_bundle_command(
    dossier_path: Annotated[Path, typer.Argument(help="Canonical candidate dossier.")],
    nomination_report_path: Annotated[
        Path, typer.Argument(help="Checked candidate nomination report.")
    ],
    checkpoint_path: Annotated[Path, typer.Argument(help="Exact local Keras checkpoint.")],
    repository_root: Annotated[Path, typer.Option(help="Repository evidence root.")],
    output_root: Annotated[Path, typer.Option(help="New local bundle directory.")],
) -> None:
    """Export the nominated TCN as a validated local-evaluation ONNX bundle."""

    try:
        from signlab.model_export import (
            ModelExportError,
            export_browser_candidate_bundle,
        )
    except (ImportError, ModuleNotFoundError) as error:
        typer.echo(
            "Candidate bundle export failed: install the SignLab portable-export extra.",
            err=True,
        )
        raise typer.Exit(code=1) from error
    try:
        result = export_browser_candidate_bundle(
            dossier_path,
            nomination_report_path,
            checkpoint_path,
            repository_root,
            output_root,
        )
    except ModelExportError as error:
        typer.echo(f"Candidate bundle export failed: {error.code}.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Candidate bundle exported and verified: {result.bundle_sha256}.")
