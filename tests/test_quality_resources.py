from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from scripts.generate_quality_resources import check_resources, main, write_resources

from signlab.contracts.quality import (
    QUALITY_METRIC_DIRECTIONS,
    QUALITY_METRIC_NAMES,
    landmark_quality_policy_digest,
    validate_landmark_quality_policy,
)
from signlab.quality import resources as quality_resources
from signlab.quality.resources import (
    DEFAULT_POLICY_FILENAME,
    GENERATED_QUALITY_RESOURCE_NAMES,
    PUBLISHED_DEFAULT_QUALITY_POLICY_SEMANTIC_DIGEST,
    PUBLISHED_QUALITY_RESOURCE_DIGESTS,
    QUALITY_SCHEMA_MODELS,
    QualityResourceError,
    build_default_quality_policy,
    generated_quality_resource_texts,
    generated_quality_schemas,
    load_packaged_default_quality_policy,
    render_quality_json,
    validate_packaged_quality_resources,
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
    root = files("signlab.resources.quality")
    inventory: set[str] = set()
    for directory_name in ("config", "schemas"):
        directory = root.joinpath(directory_name)
        inventory.update(
            f"{directory_name}/{child.name}" for child in directory.iterdir() if child.is_file()
        )
    return inventory


def test_quality_schemas_are_standalone_closed_draft_202012_documents() -> None:
    schemas = generated_quality_schemas()

    assert len(schemas) == 2
    assert set(schemas) == set(QUALITY_SCHEMA_MODELS)
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


def test_default_policy_pins_registered_metrics_directions_and_starter_profile() -> None:
    policy = build_default_quality_policy()
    payload = policy.model_dump(mode="json", round_trip=True)
    schema = generated_quality_schemas()["landmark-quality-policy-1.schema.json"]

    Draft202012Validator(schema).validate(payload)
    assert validate_landmark_quality_policy(payload) == policy
    assert tuple(rule.metric for rule in policy.threshold_rules) == QUALITY_METRIC_NAMES
    assert tuple(rule.direction for rule in policy.threshold_rules) == tuple(
        QUALITY_METRIC_DIRECTIONS[metric] for metric in QUALITY_METRIC_NAMES
    )
    assert (
        policy.target_rate_numerator,
        policy.target_rate_denominator,
        policy.max_interpolated_missing_frames,
        policy.max_interpolation_bridge_us,
    ) == (30, 1, 2, 100_000)
    assert policy.handedness_confidence_diagnostic_ppm == 800_000
    assert (
        policy.pose_visibility_diagnostic_ppm,
        policy.pose_presence_diagnostic_ppm,
    ) == (500_000, 500_000)
    assert policy.timestamp_discontinuity_absolute_us == 100_000
    assert policy.timestamp_discontinuity_median_multiplier_ppm == 3_000_000
    assert policy.suspected_swap_cost_margin_ppm == 10_000
    assert policy.max_palm_wrist_speed_units_per_second == 12
    rules = {rule.metric: rule for rule in policy.threshold_rules}
    assert rules["expected_hand_coverage_ppm"].reject == 500_000
    assert rules["invalid_frame_fraction_ppm"].reject == 200_000
    assert rules["minimum_pose_anchor_coverage_ppm"].reject is None
    assert rules["longest_unfilled_internal_hand_gap_us"].reject is None


def test_generator_and_check_mode_are_exact_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_quality_resource_texts()
    first = write_resources(tmp_path)
    first_bytes = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in first}
    second = write_resources(tmp_path)

    assert first == second
    assert set(expected) == set(first_bytes) == GENERATED_QUALITY_RESOURCE_NAMES
    assert check_resources(tmp_path) == ()
    assert main(("--check", "--directory", str(tmp_path))) == 0
    for relative_name, text in expected.items():
        captured = tmp_path.joinpath(*relative_name.split("/")).read_bytes()
        assert captured == first_bytes[relative_name] == text.encode("utf-8")
        assert captured.endswith(b"\n")
        assert b"\r\n" not in captured
        assert json.loads(captured)

    stale = f"config/{DEFAULT_POLICY_FILENAME}"
    tmp_path.joinpath(*stale.split("/")).write_text("{}\n", encoding="utf-8")
    extra = tmp_path / "schemas" / "unexpected.schema.json"
    extra.write_text("{}\n", encoding="utf-8")
    missing = "schemas/landmark-quality-manifest-1.schema.json"
    tmp_path.joinpath(*missing.split("/")).unlink()

    assert check_resources(tmp_path) == (stale, missing, "schemas/unexpected.schema.json")
    assert main(("--check", "--directory", str(tmp_path))) == 1


def test_packaged_inventory_and_first_published_resource_hashes_are_exact() -> None:
    root = files("signlab.resources.quality")

    assert _packaged_inventory() == GENERATED_QUALITY_RESOURCE_NAMES
    assert set(PUBLISHED_QUALITY_RESOURCE_DIGESTS) == GENERATED_QUALITY_RESOURCE_NAMES
    for relative_name, expected_digest in PUBLISHED_QUALITY_RESOURCE_DIGESTS.items():
        captured = root.joinpath(*relative_name.split("/")).read_bytes()
        assert f"sha256:{hashlib.sha256(captured).hexdigest()}" == expected_digest
    policy = load_packaged_default_quality_policy()
    assert (
        landmark_quality_policy_digest(policy) == PUBLISHED_DEFAULT_QUALITY_POLICY_SEMANTIC_DIGEST
    )
    validate_packaged_quality_resources()


def test_resource_errors_sanitize_serialization_missing_file_and_invalid_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(QualityResourceError, match="not JSON-serializable"):
        render_quality_json({"invalid": object()})
    with pytest.raises(QualityResourceError, match="resource is missing"):
        quality_resources._resource_bytes("missing/resource.json")

    monkeypatch.setattr(quality_resources, "_resource_bytes", lambda _name: b"{}")
    with pytest.raises(QualityResourceError, match="policy is invalid"):
        load_packaged_default_quality_policy()


def test_packaged_validation_rejects_inventory_baseline_byte_and_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quality_resources, "_packaged_inventory", set)
    with pytest.raises(QualityResourceError, match="inventory is not exact"):
        validate_packaged_quality_resources()
    monkeypatch.undo()

    monkeypatch.setattr(quality_resources, "PUBLISHED_QUALITY_RESOURCE_DIGESTS", {})
    with pytest.raises(QualityResourceError, match="baselines are incomplete"):
        validate_packaged_quality_resources()
    monkeypatch.undo()

    monkeypatch.setattr(quality_resources, "_resource_bytes", lambda _name: b"{}\n")
    with pytest.raises(QualityResourceError, match="resource drift"):
        validate_packaged_quality_resources()
    monkeypatch.undo()

    wrong_digests = dict(PUBLISHED_QUALITY_RESOURCE_DIGESTS)
    first_name = sorted(wrong_digests)[0]
    wrong_digests[first_name] = "sha256:" + "0" * 64
    monkeypatch.setattr(quality_resources, "PUBLISHED_QUALITY_RESOURCE_DIGESTS", wrong_digests)
    with pytest.raises(QualityResourceError, match="changed in place"):
        validate_packaged_quality_resources()


def test_packaged_validation_wraps_unexpected_resource_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_inventory() -> set[str]:
        raise OSError("private path must not escape")

    monkeypatch.setattr(quality_resources, "_packaged_inventory", unavailable_inventory)
    with pytest.raises(QualityResourceError, match="resources are invalid") as captured:
        validate_packaged_quality_resources()
    assert "private path" not in str(captured.value)
