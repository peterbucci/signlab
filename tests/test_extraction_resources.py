from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from scripts.generate_extraction_resources import check_resources, main, write_resources

from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    mediapipe_extraction_config_digest,
    validate_mediapipe_extraction_config,
)
from signlab.extraction import resources as extraction_resources
from signlab.extraction.parquet import landmark_parquet_schema_snapshot
from signlab.extraction.resources import (
    DEFAULT_CONFIG_FILENAME,
    EXTRACTION_SCHEMA_MODELS,
    GENERATED_EXTRACTION_RESOURCE_NAMES,
    LANDMARK_ARROW_SCHEMA_FILENAME,
    MODEL_LOCK_FILENAME,
    PUBLISHED_DEFAULT_CONFIG_SEMANTIC_DIGEST,
    PUBLISHED_EXTRACTION_RESOURCE_DIGESTS,
    ExtractionResourceError,
    build_default_extraction_config,
    build_mediapipe_model_lock,
    generated_extraction_resource_texts,
    generated_extraction_schemas,
    load_packaged_default_extraction_config,
    render_extraction_json,
    validate_packaged_extraction_resources,
)


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def _field_ids(fields: object) -> list[int]:
    if not isinstance(fields, list):
        return []
    result: list[int] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_id = field.get("field_id")
        if isinstance(field_id, int):
            result.append(field_id)
        field_type = field.get("type")
        if isinstance(field_type, dict):
            result.extend(_field_ids(field_type.get("fields")))
            element = field_type.get("element")
            if isinstance(element, dict):
                result.extend(_field_ids([element]))
    return result


def _packaged_inventory() -> set[str]:
    root = files("signlab.resources.extraction")
    inventory: set[str] = set()
    for directory_name in ("arrow", "config", "models", "schemas"):
        directory = root.joinpath(directory_name)
        inventory.update(
            f"{directory_name}/{child.name}" for child in directory.iterdir() if child.is_file()
        )
    return inventory


def test_extraction_schemas_are_standalone_draft_202012_documents() -> None:
    schemas = generated_extraction_schemas()

    assert len(schemas) == 3
    assert set(schemas) == set(EXTRACTION_SCHEMA_MODELS)
    for filename, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://signlab.dev/schemas/{filename}"
        assert "validators remain authoritative" in str(schema["$comment"])
        assert [node["$id"] for node in _schema_nodes(schema) if "$id" in node] == [schema["$id"]]


def test_default_config_pins_runtime_models_timestamps_tracking_and_anchors() -> None:
    config = build_default_extraction_config()
    schema = generated_extraction_schemas()["mediapipe-extraction-config-1.schema.json"]
    payload = config.model_dump(mode="json", round_trip=True)

    Draft202012Validator(schema).validate(payload)
    assert validate_mediapipe_extraction_config(payload) == config
    assert config.body_anchors == BODY_ANCHOR_NAMES
    assert (config.decoder_package, config.decoder_package_version) == ("av", "18.1.0")
    assert (config.delegate, config.running_mode, config.num_hands, config.num_poses) == (
        "CPU",
        "VIDEO",
        2,
        1,
    )
    assert (
        config.max_spatial_cost,
        config.handedness_disagreement_penalty,
        config.ambiguity_margin,
    ) == (0.25, 0.05, 1e-9)
    assert config.hand_task_asset.compatible_runtimes == ("browser", "python")
    assert config.pose_task_asset.compatible_runtimes == ("browser", "python")


def test_model_lock_is_exactly_bound_to_default_config_assets_and_packages() -> None:
    config = build_default_extraction_config()
    lock = build_mediapipe_model_lock()
    tasks = cast(list[dict[str, object]], lock["tasks"])

    assert lock["schema_version"] == "mediapipe-task-model-lock/1"
    assert lock["license"] == "Apache-2.0"
    assert lock["python_package"] == {
        "name": config.python_package,
        "version": config.python_package_version,
    }
    assert lock["browser_package"] == {
        "name": config.browser_package,
        "version": config.browser_package_version,
    }
    for task, asset in zip(
        tasks,
        (config.hand_task_asset, config.pose_task_asset),
        strict=True,
    ):
        assert task["task_kind"] == asset.task_kind
        assert task["model_id"] == asset.model_id
        assert task["revision"] == asset.model_revision
        assert task["filename"] == asset.filename
        assert task["sha256"] == asset.sha256
        assert task["size_bytes"] == asset.size_bytes
        assert task["compatible_runtimes"] == list(asset.compatible_runtimes)
        assert str(task["source_url"]).startswith("https://storage.googleapis.com/")
        assert str(task["model_card_url"]).startswith("https://storage.googleapis.com/")

    rendered = json.dumps(lock, sort_keys=True).casefold()
    assert "sb128" not in rendered
    assert f"{chr(99)}:{chr(92)}" not in rendered
    assert f"{chr(47)}users{chr(47)}" not in rendered
    assert f"{chr(47)}home{chr(47)}" not in rendered


def test_arrow_snapshot_is_exact_and_has_globally_unique_positive_field_ids() -> None:
    snapshot = landmark_parquet_schema_snapshot()
    packaged = json.loads(
        files("signlab.resources.extraction")
        .joinpath("arrow", LANDMARK_ARROW_SCHEMA_FILENAME)
        .read_text(encoding="utf-8")
    )

    assert packaged == snapshot
    assert snapshot["format"] == "arrow-schema-snapshot/1"
    assert snapshot["table_name"] == "landmark_frames"
    assert snapshot["schema_version"] == "landmark-frames-table/1"
    ids = _field_ids(snapshot["fields"])
    assert ids
    assert all(field_id > 0 for field_id in ids)
    assert len(ids) == len(set(ids))


def test_generator_and_check_mode_are_exact_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_extraction_resource_texts()
    first = write_resources(tmp_path)
    first_bytes = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in first}
    second = write_resources(tmp_path)

    assert first == second
    assert set(expected) == set(first_bytes) == GENERATED_EXTRACTION_RESOURCE_NAMES
    assert check_resources(tmp_path) == ()
    assert main(("--check", "--directory", str(tmp_path))) == 0
    for relative_name, text in expected.items():
        captured = tmp_path.joinpath(*relative_name.split("/")).read_bytes()
        assert captured == first_bytes[relative_name] == text.encode("utf-8")
        assert captured.endswith(b"\n")
        assert b"\r\n" not in captured
        assert json.loads(captured)

    stale = f"config/{DEFAULT_CONFIG_FILENAME}"
    tmp_path.joinpath(*stale.split("/")).write_text("{}\n", encoding="utf-8")
    extra = tmp_path / "schemas" / "unexpected.schema.json"
    extra.write_text("{}\n", encoding="utf-8")
    missing = f"models/{MODEL_LOCK_FILENAME}"
    tmp_path.joinpath(*missing.split("/")).unlink()

    assert check_resources(tmp_path) == (stale, missing, "schemas/unexpected.schema.json")
    assert main(("--check", "--directory", str(tmp_path))) == 1


def test_packaged_inventory_and_first_published_resource_hashes_are_exact() -> None:
    root = files("signlab.resources.extraction")

    assert _packaged_inventory() == GENERATED_EXTRACTION_RESOURCE_NAMES
    assert set(PUBLISHED_EXTRACTION_RESOURCE_DIGESTS) == GENERATED_EXTRACTION_RESOURCE_NAMES
    for relative_name, expected_digest in PUBLISHED_EXTRACTION_RESOURCE_DIGESTS.items():
        captured = root.joinpath(*relative_name.split("/")).read_bytes()
        assert f"sha256:{hashlib.sha256(captured).hexdigest()}" == expected_digest
    config = load_packaged_default_extraction_config()
    assert mediapipe_extraction_config_digest(config) == PUBLISHED_DEFAULT_CONFIG_SEMANTIC_DIGEST
    validate_packaged_extraction_resources()


def test_resource_errors_sanitize_serialization_missing_file_and_invalid_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ExtractionResourceError, match="not JSON-serializable"):
        render_extraction_json({"invalid": object()})
    with pytest.raises(ExtractionResourceError, match="resource is missing"):
        extraction_resources._resource_bytes("missing/resource.json")

    monkeypatch.setattr(extraction_resources, "_resource_bytes", lambda _name: b"{}")
    with pytest.raises(ExtractionResourceError, match="config is invalid"):
        load_packaged_default_extraction_config()


def test_packaged_validation_rejects_inventory_baseline_byte_and_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extraction_resources, "_packaged_inventory", set)
    with pytest.raises(ExtractionResourceError, match="inventory is not exact"):
        validate_packaged_extraction_resources()
    monkeypatch.undo()

    monkeypatch.setattr(extraction_resources, "PUBLISHED_EXTRACTION_RESOURCE_DIGESTS", {})
    with pytest.raises(ExtractionResourceError, match="baselines are incomplete"):
        validate_packaged_extraction_resources()
    monkeypatch.undo()

    monkeypatch.setattr(extraction_resources, "_resource_bytes", lambda _name: b"{}\n")
    with pytest.raises(ExtractionResourceError, match="resource drift"):
        validate_packaged_extraction_resources()
    monkeypatch.undo()

    wrong_digests = dict(PUBLISHED_EXTRACTION_RESOURCE_DIGESTS)
    first_name = sorted(wrong_digests)[0]
    wrong_digests[first_name] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        extraction_resources,
        "PUBLISHED_EXTRACTION_RESOURCE_DIGESTS",
        wrong_digests,
    )
    with pytest.raises(ExtractionResourceError, match="changed in place"):
        validate_packaged_extraction_resources()


def test_packaged_validation_wraps_unexpected_resource_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_inventory() -> set[str]:
        raise OSError("private path must not escape")

    monkeypatch.setattr(extraction_resources, "_packaged_inventory", unavailable_inventory)
    with pytest.raises(ExtractionResourceError, match="resources are invalid") as captured:
        validate_packaged_extraction_resources()
    assert "private path" not in str(captured.value)
