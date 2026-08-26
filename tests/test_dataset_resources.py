from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from importlib.resources import files
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator
from scripts.generate_dataset_resources import write_resources

from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    ClipsTableV1,
    DatasetTableInput,
    DerivedArtifactsTableV1,
    TableName,
    dataset_table_digest,
    validate_dataset_table,
)
from signlab.contracts.pipeline import contract_digest
from signlab.datasets.parquet import parquet_schema_snapshot
from signlab.datasets.resources import (
    DATASET_ARROW_SCHEMA_FILENAMES,
    DATASET_MANIFEST_EXAMPLE_FILENAME,
    DATASET_TABLE_EXAMPLE_FILENAMES,
    DATASET_TABLE_SCHEMA_FILENAMES,
    GENERATED_DATASET_RESOURCE_NAMES,
    PUBLISHED_DATASET_ARROW_SCHEMA_DIGESTS,
    PUBLISHED_DATASET_JSON_SCHEMA_DIGESTS,
    PUBLISHED_DATASET_MANIFEST_DIGEST,
    PUBLISHED_DATASET_TABLE_DIGESTS,
    build_example_dataset_bundle,
    generated_dataset_resource_texts,
    generated_dataset_schemas,
    load_packaged_dataset_manifest,
    load_packaged_dataset_table,
    validate_packaged_dataset_resources,
)
from signlab.datasets.validation import validate_dataset_manifest_tables

EXPECTED_TABLE_DIGESTS: dict[TableName, str] = {
    "participants": "sha256:eb47911ecc4c60d30110210b73dcb265279fb69284fa31a25a0560a2f6a3a227",
    "sessions": "sha256:0a77852161a088baded1248c7c71945aa43569b6ef01771bb649af7602e5730a",
    "recordings": "sha256:5c79b0629d997de6bdaab75f78e2be94ed02bb1dc9e9cf3219db3d1641aca364",
    "clips": "sha256:819f4bb5f4a915deac1d344e06cc56805f787701c2d8662c486da08e46461448",
    "annotations": "sha256:dcd6919628522a36d34f0d2a2aefbc7a2c64ca5340ce7e1e173029b67dd2577b",
    "derived_artifacts": (
        "sha256:b65cc0cc4b779437780487946244ee50691c448783d7a0eac4f7d8ddad343f65"
    ),
}
EXPECTED_MANIFEST_DIGEST = "sha256:69b705dddf972e8cf4ecb5692fe7aca779f136e3e0d258a374f79156b5d2d9a0"


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def _packaged_inventory() -> set[str]:
    root = files("signlab.resources.datasets")
    inventory: set[str] = set()
    for directory_name in ("arrow", "examples", "schemas"):
        directory = root.joinpath(directory_name)
        inventory.update(
            f"{directory_name}/{child.name}" for child in directory.iterdir() if child.is_file()
        )
    return inventory


def _field_ids(fields: object) -> list[int]:
    result: list[int] = []
    if not isinstance(fields, list):
        return result
    for field in fields:
        if not isinstance(field, dict):
            continue
        result.append(cast(int, field["field_id"]))
        field_type = field.get("type")
        if isinstance(field_type, dict):
            result.extend(_field_ids(field_type.get("fields")))
            element = field_type.get("element")
            if isinstance(element, dict):
                result.extend(_field_ids([element]))
    return result


def _walk_json(value: object) -> Iterator[tuple[str | None, object]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key, nested
            yield from _walk_json(nested)
    elif isinstance(value, list):
        for nested in value:
            yield None, nested
            yield from _walk_json(nested)


def test_exact_six_table_schemas_are_standalone_draft_202012_documents() -> None:
    schemas = generated_dataset_schemas()

    assert len(schemas) == 6
    assert set(schemas) == set(DATASET_TABLE_SCHEMA_FILENAMES.values())
    for table_name, filename in DATASET_TABLE_SCHEMA_FILENAMES.items():
        schema = schemas[filename]
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://signlab.dev/schemas/{filename}"
        assert "validators remain authoritative" in str(schema["$comment"])
        assert [node["$id"] for node in _schema_nodes(schema) if "$id" in node] == [schema["$id"]]
        properties = cast(dict[str, dict[str, object]], schema["properties"])
        assert properties["schema_version"]["const"] == DATASET_TABLE_SCHEMA_VERSIONS[table_name]
        assert properties["rows"]["uniqueItems"] is True


def test_synthetic_bundle_is_cross_table_coherent_and_exhibits_chained_lineage() -> None:
    bundle = build_example_dataset_bundle()
    result = validate_dataset_manifest_tables(
        bundle.manifest,
        cast(Mapping[str, DatasetTableInput], bundle.tables),
    )

    assert set(bundle.tables) == set(DATASET_TABLE_SCHEMA_VERSIONS)
    assert {name: len(table.rows) for name, table in bundle.tables.items()} == {
        "participants": 1,
        "sessions": 1,
        "recordings": 1,
        "clips": 1,
        "annotations": 1,
        "derived_artifacts": 3,
    }
    clip = cast(ClipsTableV1, bundle.tables["clips"]).rows[0]
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"]).rows
    assert clip.artifact is not None
    assert tuple(row.derivation_kind for row in derived) == (
        "crop",
        "augmentation",
        "window",
    )
    assert derived[0].parent_artifact_ids == (clip.artifact.artifact_id,)
    assert derived[1].parent_artifact_ids == (derived[0].artifact.artifact_id,)
    assert derived[2].parent_artifact_ids == (derived[1].artifact.artifact_id,)
    assert tuple(sample.sample_id for sample in bundle.manifest.content.samples) == tuple(
        cast(str, row.sample_id) for row in derived
    )
    assert result.semantic_integrity == "verified"
    assert result.artifact_byte_integrity == "not_checked"
    assert result.split_compatibility == "not_checked"
    assert result.consent_authorization == "not_checked"


def test_table_examples_round_trip_through_schema_and_strict_contract_validation() -> None:
    bundle = build_example_dataset_bundle()
    schemas = generated_dataset_schemas()

    for table_name, expected in bundle.tables.items():
        loaded = load_packaged_dataset_table(table_name)
        payload = loaded.model_dump(mode="json", round_trip=True)
        Draft202012Validator(schemas[DATASET_TABLE_SCHEMA_FILENAMES[table_name]]).validate(payload)
        assert validate_dataset_table(json.dumps(payload)) == expected == loaded
    assert load_packaged_dataset_manifest() == bundle.manifest


def test_arrow_snapshots_are_exact_human_readable_schema_contracts() -> None:
    root = files("signlab.resources.datasets")
    for table_name, schema_version in DATASET_TABLE_SCHEMA_VERSIONS.items():
        expected = parquet_schema_snapshot(schema_version)
        packaged = json.loads(
            root.joinpath("arrow", DATASET_ARROW_SCHEMA_FILENAMES[table_name]).read_text(
                encoding="utf-8"
            )
        )
        assert packaged == expected
        assert packaged["format"] == "arrow-schema-snapshot/1"
        assert packaged["table_name"] == table_name
        assert packaged["schema_version"] == schema_version
        assert set(packaged["allowed_schema_metadata"]) == {
            "signlab:content_sha256",
            "signlab:schema_sha256",
            "signlab:schema_version",
            "signlab:table_kind",
        }
        ids = _field_ids(packaged["fields"])
        assert ids
        assert all(field_id > 0 for field_id in ids)
        assert len(ids) == len(set(ids))


def test_published_json_and_arrow_schema_bytes_have_independent_frozen_hashes() -> None:
    root = files("signlab.resources.datasets")

    assert set(PUBLISHED_DATASET_JSON_SCHEMA_DIGESTS) == set(DATASET_TABLE_SCHEMA_VERSIONS)
    assert set(PUBLISHED_DATASET_ARROW_SCHEMA_DIGESTS) == set(DATASET_TABLE_SCHEMA_VERSIONS)
    for table_name in DATASET_TABLE_SCHEMA_VERSIONS:
        schema_bytes = root.joinpath(
            "schemas", DATASET_TABLE_SCHEMA_FILENAMES[table_name]
        ).read_bytes()
        arrow_bytes = root.joinpath(
            "arrow", DATASET_ARROW_SCHEMA_FILENAMES[table_name]
        ).read_bytes()
        assert (
            f"sha256:{hashlib.sha256(schema_bytes).hexdigest()}"
            == (PUBLISHED_DATASET_JSON_SCHEMA_DIGESTS[table_name])
        )
        assert (
            f"sha256:{hashlib.sha256(arrow_bytes).hexdigest()}"
            == (PUBLISHED_DATASET_ARROW_SCHEMA_DIGESTS[table_name])
        )


def test_generator_is_complete_pretty_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_dataset_resource_texts()

    assert len(expected) == 19
    assert set(expected) == GENERATED_DATASET_RESOURCE_NAMES
    assert not any(name.endswith(".parquet") for name in expected)
    first_written = write_resources(tmp_path)
    first_bytes = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in first_written
    }
    second_written = write_resources(tmp_path)

    assert first_written == second_written
    assert set(first_bytes) == GENERATED_DATASET_RESOURCE_NAMES
    for relative_name, expected_text in expected.items():
        actual = tmp_path.joinpath(*relative_name.split("/")).read_bytes()
        assert actual == first_bytes[relative_name] == expected_text.encode()
        assert actual.endswith(b"\n")
        assert b"\r\n" not in actual
        assert json.loads(actual)


def test_public_examples_are_pseudonymous_and_have_no_machine_paths_or_pii_fields() -> None:
    root = files("signlab.resources.datasets").joinpath("examples")
    prohibited_keys = {
        "address",
        "device_serial",
        "email",
        "hostname",
        "legal_name",
        "name",
        "phone",
        "signature",
    }
    participant_pattern = re.compile(r"^participant_[0-9a-f]{32}$")
    object_path_pattern = re.compile(
        r"^objects/sha256/p-([0-9a-f]{2})/sha256-([0-9a-f]{64})/"
        r"(?:recording|clip|sample|derived_artifact)_[0-9a-f]{32}$"
    )
    table_path_pattern = re.compile(
        r"^tables/(?:participants|sessions|recordings|clips|annotations|derived_artifacts)"
        r"\.parquet$"
    )

    for child in root.iterdir():
        text = child.read_text(encoding="utf-8")
        lowered = text.casefold()
        assert "sb128" not in lowered
        assert "peter" not in lowered
        assert "bucci" not in lowered
        assert f"{chr(99)}:{chr(47)}" not in lowered
        assert f"{chr(99)}:{chr(92)}" not in lowered
        assert f"{chr(47)}users{chr(47)}" not in lowered
        assert f"{chr(47)}home{chr(47)}" not in lowered
        assert "@" not in text
        document = json.loads(text)
        for key, value in _walk_json(document):
            assert key not in prohibited_keys
            if key == "participant_id":
                assert isinstance(value, str)
                assert participant_pattern.fullmatch(value)
            if key == "path":
                assert isinstance(value, str)
                object_match = object_path_pattern.fullmatch(value)
                assert table_path_pattern.fullmatch(value) or object_match
                if object_match:
                    assert object_match.group(1) == object_match.group(2)[:2]


def test_packaged_inventory_has_no_parquet_and_no_generated_drift() -> None:
    inventory = _packaged_inventory()

    assert inventory == GENERATED_DATASET_RESOURCE_NAMES
    assert len(inventory) == 19
    assert not any(name.endswith(".parquet") for name in inventory)
    validate_packaged_dataset_resources()


def test_public_dataset_example_semantic_identities_are_immutable() -> None:
    bundle = build_example_dataset_bundle()

    assert PUBLISHED_DATASET_TABLE_DIGESTS == EXPECTED_TABLE_DIGESTS
    assert PUBLISHED_DATASET_MANIFEST_DIGEST == EXPECTED_MANIFEST_DIGEST
    assert {
        table_name: dataset_table_digest(table) for table_name, table in bundle.tables.items()
    } == EXPECTED_TABLE_DIGESTS
    assert contract_digest(bundle.manifest) == EXPECTED_MANIFEST_DIGEST
    assert contract_digest(load_packaged_dataset_manifest()) == EXPECTED_MANIFEST_DIGEST


def test_manifest_example_is_part_of_inventory_without_duplicating_core_schema() -> None:
    assert f"examples/{DATASET_MANIFEST_EXAMPLE_FILENAME}" in GENERATED_DATASET_RESOURCE_NAMES
    assert "schemas/dataset-manifest-2.schema.json" not in GENERATED_DATASET_RESOURCE_NAMES
    assert set(DATASET_TABLE_EXAMPLE_FILENAMES) == set(DATASET_TABLE_SCHEMA_VERSIONS)
    assert set(DATASET_ARROW_SCHEMA_FILENAMES) == set(DATASET_TABLE_SCHEMA_VERSIONS)
