"""Regenerate governance schemas, examples, and synthetic withdrawal evidence."""

from __future__ import annotations

from pathlib import Path

from signlab.governance.resources import generated_governance_resource_texts

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIRECTORY = REPOSITORY_ROOT / "src" / "signlab" / "resources" / "governance"


def write_resources(directory: Path = RESOURCE_DIRECTORY) -> tuple[Path, ...]:
    """Write all generated governance resources in stable path order."""

    written: list[Path] = []
    for relative_name, content in sorted(generated_governance_resource_texts().items()):
        path = directory.joinpath(*relative_name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


def main() -> None:
    """Regenerate and list the reviewable governance resources."""

    for path in write_resources():
        print(path.relative_to(REPOSITORY_ROOT).as_posix())


if __name__ == "__main__":
    main()
