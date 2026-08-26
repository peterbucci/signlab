"""Command-line entry point for the UI-independent SignLab pipeline."""

from __future__ import annotations

from typing import Annotated

import typer

from signlab import __version__
from signlab.commands import data, doctor, evaluate, export, taxonomy, train


def _show_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Reproducible hand-gesture research and portable inference.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(data.app, name="data")
app.add_typer(train.app, name="train")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(export.app, name="export")
app.add_typer(doctor.app, name="doctor")
app.add_typer(taxonomy.app, name="taxonomy")


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            help="Show the installed SignLab version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Run a versioned SignLab pipeline command."""


def main() -> None:
    """Run the SignLab command-line application."""
    app()


if __name__ == "__main__":
    main()
