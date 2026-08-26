"""Regenerate committed taxonomy JSON Schemas from the Pydantic source of truth."""

from __future__ import annotations

import json
from pathlib import Path

from signlab.contracts.taxonomy import generated_json_schemas

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIRECTORY = REPOSITORY_ROOT / "src" / "signlab" / "resources" / "schemas"


def render_schema(schema: dict[str, object]) -> str:
    """Return stable, reviewable JSON for one generated schema."""
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_schemas(directory: Path = SCHEMA_DIRECTORY) -> tuple[Path, ...]:
    """Write all generated schemas and return their paths in stable order."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, schema in sorted(generated_json_schemas().items()):
        path = directory / filename
        path.write_text(render_schema(schema), encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


def main() -> None:
    """Regenerate the taxonomy schemas."""
    for path in write_schemas():
        print(path.relative_to(REPOSITORY_ROOT).as_posix())


if __name__ == "__main__":
    main()
