"""Generated JSON Schemas for capture and raw-import handoff contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel

from signlab.contracts.ingest import (
    CaptureIdentifierSetV1,
    CollectionSidecarV1,
    RawDatasetManifestV1,
)
from signlab.contracts.quarantine import QuarantineInventoryV1

RESOURCE_PACKAGE: Final = "signlab.resources.ingest"
_SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
_SCHEMA_COMMENT: Final = (
    "This JSON Schema enforces portable structure. SignLab's strict ingest validators "
    "remain authoritative for canonical hashes, ordering, state transitions, consent "
    "bindings, review projection, table relationships, and artifact-byte verification."
)

INGEST_SCHEMA_MODELS: Final[Mapping[str, type[BaseModel]]] = {
    "capture-identifier-set-1.schema.json": CaptureIdentifierSetV1,
    "collection-sidecar-1.schema.json": CollectionSidecarV1,
    "quarantine-inventory-1.schema.json": QuarantineInventoryV1,
    "raw-dataset-manifest-1.schema.json": RawDatasetManifestV1,
}
GENERATED_INGEST_RESOURCE_NAMES: Final = frozenset(
    f"schemas/{filename}" for filename in INGEST_SCHEMA_MODELS
)

# Frozen after first publication. These are hashes of the exact pretty-printed
# schema files, not contract-instance identities.
PUBLISHED_INGEST_JSON_SCHEMA_DIGESTS: Final[Mapping[str, str]] = {
    "capture-identifier-set-1.schema.json": (
        "sha256:98ea3a6477776fa277a9f4145fd2ddc56c3afa360d3721c1ed5ca5672bd0f401"
    ),
    "collection-sidecar-1.schema.json": (
        "sha256:6d97705b596b01df6a32ead0bdfce94f0a3abbb424444d15aa2008c503149e0f"
    ),
    "quarantine-inventory-1.schema.json": (
        "sha256:db1b45783eaa53ad30120b91024c125c8db3e2e6a295ed8243a910689a4fabca"
    ),
    "raw-dataset-manifest-1.schema.json": (
        "sha256:890456f079e27ed3dd52bdb248e8aa10914bf7c09a3d22a3e9ecc6f06f87dfd5"
    ),
}


class IngestResourceError(ValueError):
    """Raised when generated ingest resources are missing, invalid, or stale."""


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def _strip_nested_schema_ids(value: object, *, root: bool = False) -> None:
    if isinstance(value, dict):
        if not root:
            value.pop("$id", None)
        for nested in value.values():
            _strip_nested_schema_ids(nested)
    elif isinstance(value, list):
        for nested in value:
            _strip_nested_schema_ids(nested)


def _harden_arrays(schema: dict[str, object]) -> None:
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name in (
            "annotations",
            "assets",
            "attempts",
            "capture_checklist",
            "consent_checklist",
            "decisions",
            "occurrences",
            "participants",
            "session_plans",
            "sessions",
        ):
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict) and field_schema.get("type") == "array":
                field_schema["uniqueItems"] = True


def generated_ingest_schemas() -> dict[str, dict[str, object]]:
    """Return standalone Draft 2020-12 schemas for every public ingest root."""

    generated: dict[str, dict[str, object]] = {}
    for filename, model in INGEST_SCHEMA_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        _strip_nested_schema_ids(schema, root=True)
        schema["$id"] = f"{_SCHEMA_BASE}{filename}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = _SCHEMA_COMMENT
        _harden_arrays(schema)
        Draft202012Validator.check_schema(schema)
        generated[filename] = schema
    if set(generated) != set(INGEST_SCHEMA_MODELS):
        raise IngestResourceError("generated ingest schema registry is incomplete")
    return generated


def _render_json(value: Mapping[str, object]) -> str:
    try:
        return (
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise IngestResourceError("ingest resource is not JSON-serializable") from error


def generated_ingest_resource_texts() -> dict[str, str]:
    """Render the exact reviewable resource inventory."""

    rendered = {
        f"schemas/{filename}": _render_json(schema)
        for filename, schema in generated_ingest_schemas().items()
    }
    if set(rendered) != GENERATED_INGEST_RESOURCE_NAMES:
        raise IngestResourceError("generated ingest resource registry is incomplete")
    return rendered


def _resource_bytes(relative_name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(*relative_name.split("/")).read_bytes()
    except OSError as error:
        raise IngestResourceError("a packaged ingest resource is missing") from error


def _packaged_inventory() -> set[str]:
    root = files(RESOURCE_PACKAGE)
    inventory: set[str] = set()

    def visit(directory: Traversable, prefix: str = "") -> None:
        for child in directory.iterdir():
            relative = f"{prefix}/{child.name}" if prefix else child.name
            if child.is_dir():
                if child.name != "__pycache__":
                    visit(child, relative)
            elif child.name != "__init__.py" and not child.name.endswith(".pyc"):
                inventory.add(relative)

    visit(root)
    return inventory


def validate_packaged_ingest_resources() -> None:
    """Check exact inventory, generated-byte drift, and first-published hashes."""

    try:
        if _packaged_inventory() != GENERATED_INGEST_RESOURCE_NAMES:
            raise IngestResourceError("packaged ingest resource inventory is not exact")
        generated = generated_ingest_resource_texts()
        for relative_name, expected in generated.items():
            captured = _resource_bytes(relative_name)
            if captured != expected.encode("utf-8"):
                raise IngestResourceError("packaged ingest resource drift detected")
            filename = relative_name.rsplit("/", 1)[-1]
            digest = f"sha256:{hashlib.sha256(captured).hexdigest()}"
            if digest != PUBLISHED_INGEST_JSON_SCHEMA_DIGESTS[filename]:
                raise IngestResourceError("a published ingest schema changed in place")
    except IngestResourceError:
        raise
    except (OSError, TypeError, ValueError, SchemaError) as error:
        raise IngestResourceError("packaged ingest resources are invalid") from error


__all__ = [
    "GENERATED_INGEST_RESOURCE_NAMES",
    "INGEST_SCHEMA_MODELS",
    "PUBLISHED_INGEST_JSON_SCHEMA_DIGESTS",
    "IngestResourceError",
    "generated_ingest_resource_texts",
    "generated_ingest_schemas",
    "validate_packaged_ingest_resources",
]
