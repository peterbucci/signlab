"""Generated public resources for portable landmark feature representations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel

from signlab.contracts.features import (
    FEATURE_HAND_SLOTS,
    FEATURE_QUANTIZATION_SCALE,
    HAND_LANDMARK_ORDER,
    BodyRelativeRuleV1,
    FeatureCacheKeyV1,
    FeatureRepresentation,
    FeatureStatisticsV1,
    HandLocalRuleV1,
    LandmarkFeaturePlanV1,
    LearnedStatisticsRuleV1,
    OptionalFeatureRuleV1,
    PaddingFeatureRuleV1,
    PortableFeatureSequenceV1,
    TemporalFeatureRuleV1,
    landmark_feature_plan_digest,
    registered_feature_names,
    validate_landmark_feature_plan,
)

RESOURCE_PACKAGE: Final = "signlab.resources.features"
_SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
_SCHEMA_COMMENT: Final = (
    "This JSON Schema enforces portable landmark-feature structure. SignLab's "
    "strict feature validators remain authoritative for canonical hashes, registered "
    "feature ordering, cross-runtime quantization, fixed shapes and masks, training-only "
    "statistics, and cache-key bindings."
)

DEFAULT_FEATURE_REPRESENTATIONS: Final[tuple[FeatureRepresentation, ...]] = (
    "hand_local",
    "body_relative",
    "combined",
)
DEFAULT_FEATURE_PLAN_FILENAMES: Final[Mapping[FeatureRepresentation, str]] = MappingProxyType(
    {
        "hand_local": "hand-local-64-1.default.json",
        "body_relative": "body-relative-64-1.default.json",
        "combined": "combined-64-1.default.json",
    }
)
FEATURE_SCHEMA_MODELS: Final[Mapping[str, type[BaseModel]]] = MappingProxyType(
    {
        "feature-cache-key-1.schema.json": FeatureCacheKeyV1,
        "feature-statistics-1.schema.json": FeatureStatisticsV1,
        "landmark-feature-plan-1.schema.json": LandmarkFeaturePlanV1,
        "portable-feature-sequence-1.schema.json": PortableFeatureSequenceV1,
    }
)
GENERATED_FEATURE_RESOURCE_NAMES: Final = frozenset(
    {
        *(f"schemas/{filename}" for filename in FEATURE_SCHEMA_MODELS),
        *(
            f"config/{DEFAULT_FEATURE_PLAN_FILENAMES[representation]}"
            for representation in DEFAULT_FEATURE_REPRESENTATIONS
        ),
    }
)

# Frozen after first publication. These values hash exact pretty-printed UTF-8
# resource bytes independently of their generators and semantic plan identities.
PUBLISHED_FEATURE_RESOURCE_DIGESTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "config/body-relative-64-1.default.json": (
            "sha256:288971cffe3eba68ce2360b6da1a0b159253e8e9578b058e9fb1c7184c13f93f"
        ),
        "config/combined-64-1.default.json": (
            "sha256:4e5f464e322e9c9a3011159918d5c6e4a2945a0319c278633bcb4b3098d6be61"
        ),
        "config/hand-local-64-1.default.json": (
            "sha256:e036ac13411efc5bba2c7a2f189968febf8526ee946d1b66f26bd967a788fe40"
        ),
        "schemas/feature-cache-key-1.schema.json": (
            "sha256:93710f36afa5ab1a41ff95cc48df5c9959993f5cac701129cc25a75b74fd6534"
        ),
        "schemas/feature-statistics-1.schema.json": (
            "sha256:da3bee7e6cbf8df68c972155457567879fa95e6da035645392d111a84b6b252d"
        ),
        "schemas/landmark-feature-plan-1.schema.json": (
            "sha256:8a3b04ce6c2404ee2c37902171dd1b05d60396c799a5c2a7e2f2a40ec5f1577a"
        ),
        "schemas/portable-feature-sequence-1.schema.json": (
            "sha256:b392a22d9066403afab94fb55628e0a85f8f10d86228c99a8d0554d38b8620e8"
        ),
    }
)
PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS: Final[Mapping[FeatureRepresentation, str]] = (
    MappingProxyType(
        {
            "hand_local": (
                "sha256:1c62d2738ce0609168967b675fa0dcd1797f8fbe881cd9b5c775d4e2a83e4a3e"
            ),
            "body_relative": (
                "sha256:31df715e8e522d256865c7da063cbac749cc10df75f82423d46d9270307f62b1"
            ),
            "combined": ("sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6"),
        }
    )
)


class FeatureResourceError(ValueError):
    """Raised when packaged feature resources are missing, stale, or invalid."""


def _hand_local_rule() -> HandLocalRuleV1:
    return HandLocalRuleV1(
        coordinate_space="hand_world_xyz",
        landmark_order=HAND_LANDMARK_ORDER,
        center="wrist_landmark_0",
        scale="wrist_to_middle_mcp_landmark_9_euclidean",
        source_mirror_rule="undo_world_x_when_source_mirrored",
        anatomical_canonicalization=(
            "swap_vendor_label_when_not_mirrored_then_reflect_left_hand_x"
        ),
        zero_scale_rule="mask_hand_features",
    )


def _body_relative_rule() -> BodyRelativeRuleV1:
    return BodyRelativeRuleV1(
        coordinate_space="image_xy",
        trajectory_points=("wrist", "palm_centroid"),
        palm_landmarks=(0, 5, 9, 17),
        center="shoulder_midpoint",
        scale="shoulder_width_xy_euclidean",
        source_mirror_rule="undo_image_x_when_source_mirrored",
        missing_anchor_rule="mask_body_keep_hand_local",
        zero_scale_rule="mask_body_features",
    )


def _temporal_rule() -> TemporalFeatureRuleV1:
    return TemporalFeatureRuleV1(
        clock="relative_timestamp_us",
        grid_rule="nominal_elapsed_time_append_final/1",
        target_rate_numerator=30,
        target_rate_denominator=1,
        interpolation="quality_report_approved_linear_coordinates_only",
        extrapolation_allowed=False,
        forward_fill_allowed=False,
        derivative_rule="backward_elapsed_time_finite_difference/1",
        derivative_application_order="resample_then_derive_then_select_then_pad/1",
    )


def _padding_rule() -> PaddingFeatureRuleV1:
    return PaddingFeatureRuleV1(
        target_frame_count=64,
        long_sequence_rule="uniform_endpoint_preserving_index_selection/1",
        padding_side="right",
        padding_value_q=0,
        padding_mask_rule="all_feature_masks_false",
        padding_timestamp_rule="continue_nominal_grid",
    )


def _optional_rule() -> OptionalFeatureRuleV1:
    return OptionalFeatureRuleV1(
        include_velocity=False,
        include_acceleration=False,
        include_joint_angles=False,
        include_tip_distances=False,
        joint_angle_rule="five_registered_pip_flexion_angles_radians/1",
        tip_distance_rule="five_registered_wrist_to_fingertip_distances/1",
    )


def _learned_statistics_rule() -> LearnedStatisticsRuleV1:
    return LearnedStatisticsRuleV1(
        mode="none",
        partition_evidence="explicit_train_membership_required",
        masked_value_rule="exclude_from_fit",
        zero_count_rule="mean_zero_scale_one",
        zero_variance_rule="scale_one",
    )


def build_default_feature_plan(representation: FeatureRepresentation) -> LandmarkFeaturePlanV1:
    """Build one registered, cross-runtime, fixed-64-frame feature plan."""

    if representation not in DEFAULT_FEATURE_REPRESENTATIONS:
        raise FeatureResourceError("unsupported default feature representation")
    optional = _optional_rule()
    return LandmarkFeaturePlanV1(
        schema_version="landmark-feature-plan/1",
        plan_id=f"{representation}_64_frames",
        version="1.0.0",
        representation=representation,
        compatible_runtimes=("python", "typescript"),
        hand_slots=FEATURE_HAND_SLOTS,
        handedness_source="mediapipe_vendor_report_corrected_by_source_mirror_state",
        swap_rule="preserve_slots_never_repair",
        hand_local=_hand_local_rule(),
        body_relative=_body_relative_rule(),
        temporal=_temporal_rule(),
        padding=_padding_rule(),
        optional=optional,
        learned_statistics=_learned_statistics_rule(),
        quantization_scale=FEATURE_QUANTIZATION_SCALE,
        quantization_rule="round_half_away_from_zero/1",
        interchange_values="signed_integer_divided_by_quantization_scale",
        feature_order=registered_feature_names(representation, optional),
    )


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


def _harden_unique_arrays(schema: dict[str, object]) -> None:
    unique_fields = {
        "feature_names",
        "feature_order",
        "selected_source_indices",
        "training_sequence_sha256",
    }
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name in unique_fields:
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict) and field_schema.get("type") == "array":
                field_schema["uniqueItems"] = True


def generated_feature_schemas() -> dict[str, dict[str, object]]:
    """Return standalone Draft 2020-12 schemas for feature handoffs."""

    generated: dict[str, dict[str, object]] = {}
    for filename, model in FEATURE_SCHEMA_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        _strip_nested_schema_ids(schema, root=True)
        schema["$id"] = f"{_SCHEMA_BASE}{filename}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = _SCHEMA_COMMENT
        _harden_unique_arrays(schema)
        Draft202012Validator.check_schema(schema)
        generated[filename] = schema
    if set(generated) != set(FEATURE_SCHEMA_MODELS):
        raise FeatureResourceError("generated feature schema registry is incomplete")
    return generated


def render_feature_json(value: BaseModel | Mapping[str, object]) -> str:
    """Render deterministic reviewable JSON with one trailing newline."""

    payload: object = (
        value.model_dump(mode="json", round_trip=True)
        if isinstance(value, BaseModel)
        else dict(value)
    )
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise FeatureResourceError("feature resource is not JSON-serializable") from error


def generated_feature_resource_texts() -> dict[str, str]:
    """Render the exact default-plan and feature-schema inventory."""

    rendered = {
        f"schemas/{filename}": render_feature_json(schema)
        for filename, schema in generated_feature_schemas().items()
    }
    for representation in DEFAULT_FEATURE_REPRESENTATIONS:
        filename = DEFAULT_FEATURE_PLAN_FILENAMES[representation]
        rendered[f"config/{filename}"] = render_feature_json(
            build_default_feature_plan(representation)
        )
    if set(rendered) != GENERATED_FEATURE_RESOURCE_NAMES:
        raise FeatureResourceError("generated feature resource registry is incomplete")
    return rendered


def _resource_bytes(relative_name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(*relative_name.split("/")).read_bytes()
    except OSError as error:
        raise FeatureResourceError("a packaged feature resource is missing") from error


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


def load_packaged_default_feature_plan(
    representation: FeatureRepresentation,
) -> LandmarkFeaturePlanV1:
    """Load and strictly validate one packaged default feature plan."""

    try:
        filename = DEFAULT_FEATURE_PLAN_FILENAMES[representation]
        return validate_landmark_feature_plan(_resource_bytes(f"config/{filename}"))
    except KeyError as error:
        raise FeatureResourceError("unsupported default feature representation") from error
    except (TypeError, ValueError) as error:
        raise FeatureResourceError("packaged feature plan is invalid") from error


def validate_packaged_feature_resources() -> None:
    """Check exact inventory, generated bytes, frozen hashes, and plan identities."""

    try:
        if _packaged_inventory() != GENERATED_FEATURE_RESOURCE_NAMES:
            raise FeatureResourceError("packaged feature resource inventory is not exact")
        if set(PUBLISHED_FEATURE_RESOURCE_DIGESTS) != GENERATED_FEATURE_RESOURCE_NAMES:
            raise FeatureResourceError("published feature resource baselines are incomplete")
        if set(PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS) != set(
            DEFAULT_FEATURE_REPRESENTATIONS
        ):
            raise FeatureResourceError("published feature plan identities are incomplete")

        generated = generated_feature_resource_texts()
        for relative_name, expected_text in generated.items():
            captured = _resource_bytes(relative_name)
            if captured != expected_text.encode("utf-8"):
                raise FeatureResourceError("packaged feature resource drift detected")
            digest = f"sha256:{hashlib.sha256(captured).hexdigest()}"
            if digest != PUBLISHED_FEATURE_RESOURCE_DIGESTS[relative_name]:
                raise FeatureResourceError("a published feature resource changed in place")

        plan_schema = generated_feature_schemas()["landmark-feature-plan-1.schema.json"]
        for representation in DEFAULT_FEATURE_REPRESENTATIONS:
            plan = load_packaged_default_feature_plan(representation)
            Draft202012Validator(plan_schema).validate(
                plan.model_dump(mode="json", round_trip=True)
            )
            if (
                landmark_feature_plan_digest(plan)
                != PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS[representation]
            ):
                raise FeatureResourceError("a published feature plan identity changed")
    except FeatureResourceError:
        raise
    except (
        JsonSchemaValidationError,
        OSError,
        SchemaError,
        TypeError,
        ValueError,
    ) as error:
        raise FeatureResourceError("packaged feature resources are invalid") from error


__all__ = [
    "DEFAULT_FEATURE_PLAN_FILENAMES",
    "DEFAULT_FEATURE_REPRESENTATIONS",
    "FEATURE_SCHEMA_MODELS",
    "GENERATED_FEATURE_RESOURCE_NAMES",
    "PUBLISHED_DEFAULT_FEATURE_PLAN_SEMANTIC_DIGESTS",
    "PUBLISHED_FEATURE_RESOURCE_DIGESTS",
    "FeatureResourceError",
    "build_default_feature_plan",
    "generated_feature_resource_texts",
    "generated_feature_schemas",
    "load_packaged_default_feature_plan",
    "render_feature_json",
    "validate_packaged_feature_resources",
]
