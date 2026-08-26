from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator
from scripts.generate_ingest_resources import write_resources

from signlab.datasets.ingest_resources import (
    GENERATED_INGEST_RESOURCE_NAMES,
    INGEST_SCHEMA_MODELS,
    PUBLISHED_INGEST_JSON_SCHEMA_DIGESTS,
    generated_ingest_resource_texts,
    generated_ingest_schemas,
    validate_packaged_ingest_resources,
)


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def test_ingest_schemas_are_standalone_draft_202012_documents() -> None:
    schemas = generated_ingest_schemas()

    assert set(schemas) == set(INGEST_SCHEMA_MODELS)
    for filename, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://signlab.dev/schemas/{filename}"
        assert "validators remain authoritative" in str(schema["$comment"])
        assert [node["$id"] for node in _schema_nodes(schema) if "$id" in node] == [schema["$id"]]


def test_ingest_generator_is_complete_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_ingest_resource_texts()
    first = write_resources(tmp_path)
    first_bytes = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in first}
    second = write_resources(tmp_path)

    assert first == second
    assert set(expected) == set(first_bytes) == GENERATED_INGEST_RESOURCE_NAMES
    for relative_name, text in expected.items():
        captured = tmp_path.joinpath(*relative_name.split("/")).read_bytes()
        assert captured == first_bytes[relative_name] == text.encode("utf-8")
        assert captured.endswith(b"\n")
        assert b"\r\n" not in captured
        assert json.loads(captured)


def test_packaged_ingest_schema_bytes_have_frozen_hashes() -> None:
    root = files("signlab.resources.ingest")

    assert set(PUBLISHED_INGEST_JSON_SCHEMA_DIGESTS) == set(INGEST_SCHEMA_MODELS)
    for filename, expected in PUBLISHED_INGEST_JSON_SCHEMA_DIGESTS.items():
        captured = root.joinpath("schemas", filename).read_bytes()
        assert f"sha256:{hashlib.sha256(captured).hexdigest()}" == expected
    validate_packaged_ingest_resources()
