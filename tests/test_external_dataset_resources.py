from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from scripts.generate_external_dataset_resources import check_resources, main, write_resources

from signlab.contracts.external_dataset import (
    external_dataset_selection_digest,
    licensed_dataset_source_digest,
)
from signlab.datasets import external_resources
from signlab.datasets.external_resources import (
    EXTERNAL_SCHEMA_MODELS,
    GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES,
    PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS,
    PUBLISHED_POPSIGN_SELECTION_SEMANTIC_DIGEST,
    PUBLISHED_POPSIGN_SOURCE_SEMANTIC_DIGEST,
    ExternalDatasetResourceError,
    build_popsign_source,
    build_signlab_five_popsign_selection,
    external_resource_reference,
    generated_external_dataset_resource_texts,
    generated_external_dataset_schemas,
    load_popsign_source,
    load_signlab_five_popsign_selection,
    render_external_dataset_json,
    validate_packaged_external_dataset_resources,
)


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def _packaged_inventory() -> set[str]:
    root = files("signlab.resources.external_datasets")
    return {
        f"{directory.name}/{child.name}"
        for directory in root.iterdir()
        if directory.is_dir() and directory.name != "__pycache__"
        for child in directory.iterdir()
        if child.is_file() and child.name != "__init__.py" and not child.name.endswith(".pyc")
    }


def test_popsign_source_registers_official_v1_license_sensitivity_and_scope() -> None:
    source = build_popsign_source()

    assert (source.source_id, source.version, source.download_id) == (
        "popsign-asl",
        "1.0.0",
        "popsign_v1_0",
    )
    assert (source.total_videos, source.total_signs, source.total_signers) == (
        200_686,
        250,
        47,
    )
    assert source.categories == ("game", "non-game")
    assert source.splits == ("train", "val", "test")
    assert source.license.license_id == "CC-BY-4.0"
    assert source.license.attribution_required is True
    assert source.license.change_notice_required is True
    assert source.contains_identifiable_human_video is True
    assert source.provider_reports_participant_consent is True
    assert source.signlab_participant_consent_applicable is False
    assert source.publisher_checksums_available is False
    assert source.website_preview_media_permitted is False
    assert "isolated_sign_recognition" in source.suitable_uses
    assert set(source.unsuitable_uses) == {
        "continuous_sign_recognition",
        "sign_language_translation",
    }


def test_selection_maps_only_five_reviewed_targets_without_language_equivalence() -> None:
    selection = build_signlab_five_popsign_selection()

    assert selection.category == "game"
    assert selection.splits == ("train", "val", "test")
    assert tuple(
        (mapping.source_label, mapping.target_label_id) for mapping in selection.mappings
    ) == (
        ("hello", "hello"),
        ("no", "no"),
        ("please", "please"),
        ("thankyou", "thank_you"),
        ("yes", "yes"),
    )
    assert all(mapping.language_equivalence_claimed is False for mapping in selection.mappings)
    assert selection.learned_negative_included is False
    assert selection.claim_scope == "signlab_predefined_gestures_only"
    source_ref = external_resource_reference(build_popsign_source())
    selection_ref = external_resource_reference(selection)
    assert source_ref.resource_kind == "source"
    assert selection_ref.resource_kind == "selection"
    assert source_ref.sha256 == PUBLISHED_POPSIGN_SOURCE_SEMANTIC_DIGEST
    assert selection_ref.sha256 == PUBLISHED_POPSIGN_SELECTION_SEMANTIC_DIGEST


def test_external_schemas_are_standalone_closed_draft_202012_documents() -> None:
    schemas = generated_external_dataset_schemas()

    assert set(schemas) == set(EXTERNAL_SCHEMA_MODELS)
    for filename, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://signlab.dev/schemas/{filename}"
        assert "validators remain authoritative" in str(schema["$comment"])
        assert [node["$id"] for node in _schema_nodes(schema) if "$id" in node] == [schema["$id"]]
        object_nodes = [
            node
            for node in _schema_nodes(schema)
            if node.get("type") == "object" and "properties" in node
        ]
        assert object_nodes
        assert all(node.get("additionalProperties") is False for node in object_nodes)


def test_generator_and_check_mode_are_exact_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_external_dataset_resource_texts()
    first = write_resources(tmp_path)
    first_bytes = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in first}
    second = write_resources(tmp_path)

    assert first == second
    assert set(expected) == set(first_bytes) == GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES
    assert check_resources(tmp_path) == ()
    assert main(("--check", "--directory", str(tmp_path))) == 0
    for relative_name, text in expected.items():
        captured = tmp_path.joinpath(*relative_name.split("/")).read_bytes()
        assert captured == first_bytes[relative_name] == text.encode("utf-8")
        assert captured.endswith(b"\n")
        assert b"\r\n" not in captured
        assert json.loads(captured)

    stale = "registry/popsign-asl-1.0.0.json"
    tmp_path.joinpath(*stale.split("/")).write_text("{}\n", encoding="utf-8")
    extra = tmp_path / "schemas" / "unexpected.schema.json"
    extra.write_text("{}\n", encoding="utf-8")
    missing = "schemas/external-dataset-manifest-1.schema.json"
    tmp_path.joinpath(*missing.split("/")).unlink()

    assert check_resources(tmp_path) == (stale, missing, "schemas/unexpected.schema.json")
    assert main(("--check", "--directory", str(tmp_path))) == 1


def test_packaged_inventory_publication_hashes_and_semantics_are_exact() -> None:
    root = files("signlab.resources.external_datasets")

    assert _packaged_inventory() == GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES
    assert set(PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS) == (
        GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES
    )
    for relative_name, expected_digest in PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS.items():
        captured = root.joinpath(*relative_name.split("/")).read_bytes()
        assert f"sha256:{hashlib.sha256(captured).hexdigest()}" == expected_digest
    source = load_popsign_source()
    selection = load_signlab_five_popsign_selection()
    assert licensed_dataset_source_digest(source) == PUBLISHED_POPSIGN_SOURCE_SEMANTIC_DIGEST
    assert (
        external_dataset_selection_digest(selection) == PUBLISHED_POPSIGN_SELECTION_SEMANTIC_DIGEST
    )
    validate_packaged_external_dataset_resources()


def test_external_resource_failures_are_sanitized_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ExternalDatasetResourceError, match="not JSON-serializable"):
        render_external_dataset_json({"invalid": object()})
    with pytest.raises(ExternalDatasetResourceError, match="resource is missing"):
        external_resources._resource_bytes("missing/private-resource.json")

    monkeypatch.setattr(external_resources, "_resource_bytes", lambda _name: b"{}")
    with pytest.raises(ExternalDatasetResourceError, match="source is invalid"):
        load_popsign_source()


def test_packaged_validation_rejects_inventory_baseline_byte_and_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(external_resources, "_packaged_inventory", set)
    with pytest.raises(ExternalDatasetResourceError, match="inventory is not exact"):
        validate_packaged_external_dataset_resources()
    monkeypatch.undo()

    monkeypatch.setattr(external_resources, "PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS", {})
    with pytest.raises(ExternalDatasetResourceError, match="baselines are incomplete"):
        validate_packaged_external_dataset_resources()
    monkeypatch.undo()

    monkeypatch.setattr(external_resources, "_resource_bytes", lambda _name: b"{}\n")
    with pytest.raises(ExternalDatasetResourceError, match="resource drift"):
        validate_packaged_external_dataset_resources()
    monkeypatch.undo()

    wrong_digests = dict(PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS)
    first_name = sorted(wrong_digests)[0]
    wrong_digests[first_name] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        external_resources,
        "PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS",
        wrong_digests,
    )
    with pytest.raises(ExternalDatasetResourceError, match="changed in place"):
        validate_packaged_external_dataset_resources()
