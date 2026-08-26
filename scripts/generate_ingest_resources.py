"""Regenerate capture-sidecar and raw-manifest JSON Schemas."""

from __future__ import annotations

from pathlib import Path

from signlab.datasets.ingest_resources import generated_ingest_resource_texts

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIRECTORY = REPOSITORY_ROOT / "src" / "signlab" / "resources" / "ingest"


def write_resources(directory: Path = RESOURCE_DIRECTORY) -> tuple[Path, ...]:
    """Write every generated ingest resource in stable path order."""

    written: list[Path] = []
    for relative_name, content in sorted(generated_ingest_resource_texts().items()):
        path = directory.joinpath(*relative_name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


def main() -> None:
    """Regenerate and list reviewable public ingest resources."""

    for path in write_resources():
        print(path.relative_to(REPOSITORY_ROOT).as_posix())


if __name__ == "__main__":
    main()
