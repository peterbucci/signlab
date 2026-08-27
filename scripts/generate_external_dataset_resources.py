"""Regenerate or check licensed external-dataset resources."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from signlab.datasets.external_resources import generated_external_dataset_resource_texts

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIRECTORY = REPOSITORY_ROOT / "src" / "signlab" / "resources" / "external_datasets"


def write_resources(directory: Path = RESOURCE_DIRECTORY) -> tuple[Path, ...]:
    """Write every generated external-dataset resource in stable path order."""

    written: list[Path] = []
    for relative_name, content in sorted(generated_external_dataset_resource_texts().items()):
        path = directory.joinpath(*relative_name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


def _inventory(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "__init__.py"
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    }


def check_resources(directory: Path = RESOURCE_DIRECTORY) -> tuple[str, ...]:
    """Return stable names for every missing, extra, or stale resource."""

    expected = generated_external_dataset_resource_texts()
    failures = set(_inventory(directory).symmetric_difference(expected))
    for relative_name, content in expected.items():
        path = directory.joinpath(*relative_name.split("/"))
        try:
            if path.read_bytes() != content.encode("utf-8"):
                failures.add(relative_name)
        except OSError:
            failures.add(relative_name)
    return tuple(sorted(failures))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if resources differ")
    parser.add_argument("--directory", type=Path, default=RESOURCE_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write resources, or return nonzero when check mode finds drift."""

    arguments = _build_parser().parse_args(argv)
    if arguments.check:
        failures = check_resources(arguments.directory)
        for relative_name in failures:
            print(relative_name)
        return int(bool(failures))
    for path in write_resources(arguments.directory):
        try:
            print(path.relative_to(REPOSITORY_ROOT).as_posix())
        except ValueError:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
