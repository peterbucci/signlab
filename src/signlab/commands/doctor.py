"""Local environment diagnostics."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

import typer

from signlab import __version__


@dataclass(frozen=True)
class Diagnostic:
    """One deterministic environment check."""

    name: str
    passed: bool
    detail: str


def inspect_environment(
    *,
    python_version: tuple[int, int, int] | None = None,
    implementation: str | None = None,
    filesystem_encoding: str | None = None,
) -> tuple[Diagnostic, ...]:
    """Return checks without printing paths, credentials, or environment variables."""
    resolved_version = python_version or (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    resolved_implementation = implementation or platform.python_implementation()
    resolved_encoding = filesystem_encoding or sys.getfilesystemencoding()
    version_text = ".".join(str(part) for part in resolved_version)
    return (
        Diagnostic(
            name="Python",
            passed=resolved_version[:2] == (3, 12),
            detail=f"{version_text} (requires 3.12.x)",
        ),
        Diagnostic(
            name="Interpreter",
            passed=resolved_implementation == "CPython",
            detail=f"{resolved_implementation} (requires CPython)",
        ),
        Diagnostic(
            name="Filesystem encoding",
            passed=resolved_encoding.lower().replace("-", "") == "utf8",
            detail=resolved_encoding,
        ),
        Diagnostic(name="SignLab", passed=bool(__version__), detail=__version__),
    )


app = typer.Typer(help="Check the local SignLab environment without exposing secrets or paths.")


@app.command("check")
def check_environment() -> None:
    """Check that the active runtime satisfies the repository contract."""
    diagnostics = inspect_environment()
    for diagnostic in diagnostics:
        status = "ok" if diagnostic.passed else "error"
        typer.echo(f"[{status}] {diagnostic.name}: {diagnostic.detail}")
    if not all(diagnostic.passed for diagnostic in diagnostics):
        raise typer.Exit(code=1)
