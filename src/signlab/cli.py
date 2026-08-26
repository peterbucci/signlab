"""Command-line entry point for the UI-independent SignLab pipeline."""

from __future__ import annotations

import argparse

from signlab import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signlab",
        description="Reproducible hand-gesture recognition research pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main() -> None:
    """Run the SignLab command-line interface."""
    build_parser().parse_args()


if __name__ == "__main__":
    main()
