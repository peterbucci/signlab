"""Shared construction for command groups awaiting their pipeline services."""

from __future__ import annotations

import typer


def create_group(*, help_text: str) -> typer.Typer:
    """Create a discoverable group without embedding pipeline behavior in the CLI."""
    group = typer.Typer(help=help_text, no_args_is_help=True)

    @group.callback()
    def _callback() -> None:
        """Route commands to UI-independent application services."""

    return group
