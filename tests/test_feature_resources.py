from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from scripts.generate_feature_resources import check_resources, main, write_resources

from signlab.contracts.features import (
    FEATURE_HAND_SLOTS,
    FEATURE_QUANTIZATION_SCALE,
    HAND_LANDMARK_ORDER,
    FeatureRepresentation,
    landmark_feature_plan_digest,
    registered_feature_names,
    validate_landmark_feature_plan,
)
from signlab.features import resources as feature_resources
from signlab.features.resources import (
    DEFAULT_FEATURE_PLAN_FILENAMES,
    DEFAULT_FEATURE_REPRESENTATIONS,
    FEATURE_SCHEMA_MODELS,
    GENERATED_FEATURE_RESOURCE_NAMES,
    PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS,
    PUBLISHED_FEATURE_RESOURCE_DIGESTS,
    FeatureResourceError,
    build_default_feature_plan,
    generated_feature_resource_texts,
    generated_feature_schemas,
    load_packaged_default_feature_plan,
    render_feature_json,
    validate_packaged_feature_resources,
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
    root = files("signlab.resources.features")
    inventory: set[str] = set()
    for directory_name in ("config", "schemas"):
        directory = root.joinpath(directory_name)
        inventory.update(
            f"{directory_name}/{child.name}" for child in directory.iterdir() if child.is_file()
        )
    return inventory


def test_feature_schemas_are_exact_standalone_closed_draft_202012_documents() -> None:
    schemas = generated_feature_schemas()

    assert len(schemas) == 4
    assert (
        set(schemas)
        == set(FEATURE_SCHEMA_MODELS)
        == {
            "feature-cache-key-1.schema.json",
            "feature-statistics-1.schema.json",
            "landmark-feature-plan-1.schema.json",
            "portable-feature-sequence-1.schema.json",
        }
    )
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


@pytest.mark.parametrize(
    ("representation", "expected_width"),
    [("hand_local", 126), ("body_relative", 8), ("combined", 134)],
)
def test_default_plans_pin_every_registered_rule_and_feature_name(
    representation: FeatureRepresentation,
    expected_width: int,
) -> None:
    plan = build_default_feature_plan(representation)
    payload = plan.model_dump(mode="json", round_trip=True)
    schema = generated_feature_schemas()["landmark-feature-plan-1.schema.json"]

    Draft202012Validator(schema).validate(payload)
    assert validate_landmark_feature_plan(payload) == plan
    assert plan.schema_version == "landmark-feature-plan/1"
    assert plan.plan_id == f"{representation}_64_frames"
    assert plan.version == "1.0.0"
    assert plan.representation == representation
    assert plan.compatible_runtimes == ("python", "typescript")
    assert plan.hand_slots == FEATURE_HAND_SLOTS
    assert plan.handedness_source == "mediapipe_vendor_report_corrected_by_source_mirror_state"
    assert plan.swap_rule == "preserve_slots_never_repair"
    assert plan.hand_local.model_dump(mode="python") == {
        "coordinate_space": "hand_world_xyz",
        "landmark_order": HAND_LANDMARK_ORDER,
        "center": "wrist_landmark_0",
        "scale": "wrist_to_middle_mcp_landmark_9_euclidean",
        "source_mirror_rule": "undo_world_x_when_source_mirrored",
        "anatomical_canonicalization": (
            "swap_vendor_label_when_not_mirrored_then_reflect_left_hand_x"
        ),
        "zero_scale_rule": "mask_hand_features",
    }
    assert plan.body_relative.model_dump(mode="python") == {
        "coordinate_space": "image_xy",
        "trajectory_points": ("wrist", "palm_centroid"),
        "palm_landmarks": (0, 5, 9, 17),
        "center": "shoulder_midpoint",
        "scale": "shoulder_width_xy_euclidean",
        "source_mirror_rule": "undo_image_x_when_source_mirrored",
        "missing_anchor_rule": "mask_body_keep_hand_local",
        "zero_scale_rule": "mask_body_features",
    }
    assert plan.temporal.model_dump(mode="python") == {
        "clock": "relative_timestamp_us",
        "grid_rule": "nominal_elapsed_time_append_final/1",
        "target_rate_numerator": 30,
        "target_rate_denominator": 1,
        "interpolation": "quality_report_approved_linear_coordinates_only",
        "extrapolation_allowed": False,
        "forward_fill_allowed": False,
        "derivative_rule": "backward_elapsed_time_finite_difference/1",
        "derivative_application_order": "resample_then_derive_then_select_then_pad/1",
    }
    assert plan.padding.model_dump(mode="python") == {
        "target_frame_count": 64,
        "long_sequence_rule": "uniform_endpoint_preserving_index_selection/1",
        "padding_side": "right",
        "padding_value_q": 0,
        "padding_mask_rule": "all_feature_masks_false",
        "padding_timestamp_rule": "continue_nominal_grid",
    }
    assert plan.optional.model_dump(mode="python") == {
        "include_velocity": False,
        "include_acceleration": False,
        "include_joint_angles": False,
        "include_tip_distances": False,
        "joint_angle_rule": "five_registered_pip_flexion_angles_radians/1",
        "tip_distance_rule": "five_registered_wrist_to_fingertip_distances/1",
    }
    assert plan.learned_statistics.model_dump(mode="python") == {
        "mode": "none",
        "partition_evidence": "explicit_train_membership_required",
        "masked_value_rule": "exclude_from_fit",
        "zero_count_rule": "mean_zero_scale_one",
        "zero_variance_rule": "scale_one",
    }
    assert plan.quantization_scale == FEATURE_QUANTIZATION_SCALE
    assert plan.quantization_rule == "round_half_away_from_zero/1"
    assert plan.interchange_values == "signed_integer_divided_by_quantization_scale"
    assert len(plan.feature_order) == expected_width
    assert plan.feature_order == registered_feature_names(representation, plan.optional)


def test_default_plan_registry_is_complete_and_rejects_unknown_representations() -> None:
    assert DEFAULT_FEATURE_REPRESENTATIONS == ("hand_local", "body_relative", "combined")
    assert set(DEFAULT_FEATURE_PLAN_FILENAMES) == set(DEFAULT_FEATURE_REPRESENTATIONS)
    assert tuple(DEFAULT_FEATURE_PLAN_FILENAMES.values()) == (
        "hand-local-64-1.default.json",
        "body-relative-64-1.default.json",
        "combined-64-1.default.json",
    )

    with pytest.raises(FeatureResourceError, match="unsupported"):
        build_default_feature_plan("unknown")  # type: ignore[arg-type]
    with pytest.raises(FeatureResourceError, match="unsupported"):
        load_packaged_default_feature_plan("unknown")  # type: ignore[arg-type]


def test_generator_and_check_mode_are_exact_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_feature_resource_texts()
    first = write_resources(tmp_path)
    first_bytes = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in first}
    second = write_resources(tmp_path)

    assert first == second
    assert set(expected) == set(first_bytes) == GENERATED_FEATURE_RESOURCE_NAMES
    assert check_resources(tmp_path) == ()
    assert main(("--check", "--directory", str(tmp_path))) == 0
    for relative_name, text in expected.items():
        captured = tmp_path.joinpath(*relative_name.split("/")).read_bytes()
        assert captured == first_bytes[relative_name] == text.encode("utf-8")
        assert captured.endswith(b"\n")
        assert b"\r\n" not in captured
        assert json.loads(captured)

    stale = "config/hand-local-64-1.default.json"
    tmp_path.joinpath(*stale.split("/")).write_text("{}\n", encoding="utf-8")
    extra = tmp_path / "schemas" / "unexpected.schema.json"
    extra.write_text("{}\n", encoding="utf-8")
    missing = "schemas/feature-cache-key-1.schema.json"
    tmp_path.joinpath(*missing.split("/")).unlink()

    assert check_resources(tmp_path) == (stale, missing, "schemas/unexpected.schema.json")
    assert main(("--check", "--directory", str(tmp_path))) == 1


def test_packaged_inventory_and_first_published_hashes_are_exact() -> None:
    root = files("signlab.resources.features")

    assert _packaged_inventory() == GENERATED_FEATURE_RESOURCE_NAMES
    assert set(PUBLISHED_FEATURE_RESOURCE_DIGESTS) == GENERATED_FEATURE_RESOURCE_NAMES
    assert set(PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS) == set(
        DEFAULT_FEATURE_REPRESENTATIONS
    )
    for relative_name, expected_digest in PUBLISHED_FEATURE_RESOURCE_DIGESTS.items():
        captured = root.joinpath(*relative_name.split("/")).read_bytes()
        assert f"sha256:{hashlib.sha256(captured).hexdigest()}" == expected_digest
    for representation in DEFAULT_FEATURE_REPRESENTATIONS:
        plan = load_packaged_default_feature_plan(representation)
        assert (
            landmark_feature_plan_digest(plan)
            == PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS[representation]
        )
    validate_packaged_feature_resources()


def test_resource_errors_sanitize_serialization_missing_file_and_invalid_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FeatureResourceError, match="not JSON-serializable"):
        render_feature_json({"invalid": object()})
    with pytest.raises(FeatureResourceError, match="resource is missing"):
        feature_resources._resource_bytes("missing/resource.json")

    monkeypatch.setattr(feature_resources, "_resource_bytes", lambda _name: b"{}")
    with pytest.raises(FeatureResourceError, match="plan is invalid"):
        load_packaged_default_feature_plan("hand_local")


def test_packaged_validation_rejects_inventory_baseline_byte_and_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_resources, "_packaged_inventory", set)
    with pytest.raises(FeatureResourceError, match="inventory is not exact"):
        validate_packaged_feature_resources()
    monkeypatch.undo()

    monkeypatch.setattr(feature_resources, "PUBLISHED_FEATURE_RESOURCE_DIGESTS", {})
    with pytest.raises(FeatureResourceError, match="baselines are incomplete"):
        validate_packaged_feature_resources()
    monkeypatch.undo()

    monkeypatch.setattr(feature_resources, "PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS", {})
    with pytest.raises(FeatureResourceError, match="identities are incomplete"):
        validate_packaged_feature_resources()
    monkeypatch.undo()

    monkeypatch.setattr(feature_resources, "_resource_bytes", lambda _name: b"{}\n")
    with pytest.raises(FeatureResourceError, match="resource drift"):
        validate_packaged_feature_resources()
    monkeypatch.undo()

    wrong_digests = dict(PUBLISHED_FEATURE_RESOURCE_DIGESTS)
    first_name = sorted(wrong_digests)[0]
    wrong_digests[first_name] = "sha256:" + "0" * 64
    monkeypatch.setattr(feature_resources, "PUBLISHED_FEATURE_RESOURCE_DIGESTS", wrong_digests)
    with pytest.raises(FeatureResourceError, match="changed in place"):
        validate_packaged_feature_resources()
    monkeypatch.undo()

    wrong_identities = dict(PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS)
    wrong_identities["hand_local"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        feature_resources,
        "PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS",
        wrong_identities,
    )
    with pytest.raises(FeatureResourceError, match="identity changed"):
        validate_packaged_feature_resources()


def test_packaged_validation_wraps_unexpected_resource_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_inventory() -> set[str]:
        raise OSError("private path must not escape")

    monkeypatch.setattr(feature_resources, "_packaged_inventory", unavailable_inventory)
    with pytest.raises(FeatureResourceError, match="resources are invalid") as captured:
        validate_packaged_feature_resources()
    assert "private path" not in str(captured.value)
