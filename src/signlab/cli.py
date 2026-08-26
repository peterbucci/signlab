"""Command-line entry point for the UI-independent SignLab pipeline."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Annotated, Any

import typer
from typer import _click
from typer.core import TyperGroup

from signlab import __version__
from signlab.commands import contracts, data, doctor, evaluate, export, governance, taxonomy, train


class PrivacySafeTyperGroup(TyperGroup):
    """Render usage failures without echoing an accidental path or identifier."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except _click.exceptions.UsageError as error:
            if not standalone_mode:
                raise _click.exceptions.UsageError(
                    "invalid command usage; run --help for accepted arguments"
                ) from error
            _click.echo(
                "Error: invalid command usage; run --help for accepted arguments.",
                file=sys.stderr,
            )
            raise SystemExit(error.exit_code) from error
        if standalone_mode:
            raise SystemExit(result if isinstance(result, int) else 0)
        return result


def _show_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


app = typer.Typer(
    cls=PrivacySafeTyperGroup,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Reproducible hand-gesture research and portable inference.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    suggest_commands=False,
)
app.add_typer(data.app, name="data")
app.add_typer(train.app, name="train")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(export.app, name="export")
app.add_typer(doctor.app, name="doctor")
app.add_typer(taxonomy.app, name="taxonomy")
app.add_typer(governance.app, name="governance")
app.add_typer(contracts.app, name="contracts")


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
