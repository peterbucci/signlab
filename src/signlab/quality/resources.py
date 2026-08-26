"""Generated public resources for landmark-quality policy and reports."""

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

from signlab.contracts.quality import (
    LandmarkQualityManifestV1,
    LandmarkQualityPolicyV1,
    QualityThresholdRuleV1,
    landmark_quality_policy_digest,
    validate_landmark_quality_policy,
)

RESOURCE_PACKAGE: Final = "signlab.resources.quality"
_SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
_SCHEMA_COMMENT: Final = (
    "This JSON Schema enforces portable landmark-quality structure. SignLab's "
    "strict quality validators remain authoritative for canonical hashes, metric "
    "directions, severity ordering, gap decisions, weighted aggregates, elapsed-time "
    "grids, and extraction bindings."
)

DEFAULT_POLICY_FILENAME: Final = "landmark-quality-policy-1.default.json"
QUALITY_SCHEMA_MODELS: Final[Mapping[str, type[BaseModel]]] = MappingProxyType(
    {
        "landmark-quality-manifest-1.schema.json": LandmarkQualityManifestV1,
        "landmark-quality-policy-1.schema.json": LandmarkQualityPolicyV1,
    }
)
GENERATED_QUALITY_RESOURCE_NAMES: Final = frozenset(
    {
        *(f"schemas/{filename}" for filename in QUALITY_SCHEMA_MODELS),
        f"config/{DEFAULT_POLICY_FILENAME}",
    }
)

# Frozen after first publication. These values hash exact pretty-printed UTF-8
# resource bytes, independently of their generators and semantic identities.
PUBLISHED_QUALITY_RESOURCE_DIGESTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "config/landmark-quality-policy-1.default.json": (
            "sha256:754bb5bce4f204b732f4bf555ca1e6780055807abe643563b5e175bb92c05114"
        ),
        "schemas/landmark-quality-manifest-1.schema.json": (
            "sha256:541bf0363fd2c30bf003cb83467729d62731ee47671835a7b1abceabc8395fd4"
        ),
        "schemas/landmark-quality-policy-1.schema.json": (
            "sha256:1090f4527d791e6c69fef46f2b482ef8d152fc697079b6f12b02408f209e06a1"
        ),
    }
)
PUBLISHED_DEFAULT_QUALITY_POLICY_SEMANTIC_DIGEST: Final = (
    "sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545"
)


class QualityResourceError(ValueError):
    """Raised when packaged quality resources are missing, stale, or invalid."""


def _rule(
    metric: str,
    direction: str,
    *,
    warning: int | None,
    quarantine: int | None,
    reject: int | None,
) -> QualityThresholdRuleV1:
    return QualityThresholdRuleV1.model_validate(
        {
            "schema_version": "quality-threshold-rule/1",
            "rule_id": metric,
            "metric": metric,
            "direction": direction,
            "warning": warning,
            "quarantine": quarantine,
            "reject": reject,
        },
        strict=True,
    )


def build_default_quality_policy() -> LandmarkQualityPolicyV1:
    """Build the reviewable pilot-screening policy used by public fixtures."""

    return LandmarkQualityPolicyV1(
        schema_version="landmark-quality-policy/1",
        policy_id="landmark_quality_pilot",
        version="1.0.0",
        expected_hand_cardinality_rule="recording_handedness_unknown_means_one",
        target_rate_numerator=30,
        target_rate_denominator=1,
        resampling_rule="nominal_elapsed_time_append_final/1",
        interpolation_method="linear_coordinates_only",
        extrapolation_allowed=False,
        max_interpolated_missing_frames=2,
        max_interpolation_bridge_us=100_000,
        handedness_confidence_diagnostic_ppm=800_000,
        pose_visibility_diagnostic_ppm=500_000,
        pose_presence_diagnostic_ppm=500_000,
        timestamp_discontinuity_absolute_us=100_000,
        timestamp_discontinuity_median_multiplier_ppm=3_000_000,
        suspected_swap_cost_margin_ppm=10_000,
        max_palm_wrist_speed_units_per_second=12,
        threshold_rules=(
            _rule(
                "expected_hand_coverage_ppm",
                "lower_is_worse",
                warning=950_000,
                quarantine=800_000,
                reject=500_000,
            ),
            _rule(
                "invalid_frame_fraction_ppm",
                "higher_is_worse",
                warning=10_000,
                quarantine=50_000,
                reject=200_000,
            ),
            _rule(
                "longest_unfilled_internal_hand_gap_us",
                "higher_is_worse",
                warning=100_000,
                quarantine=500_000,
                reject=None,
            ),
            _rule(
                "low_handedness_confidence_fraction_ppm",
                "higher_is_worse",
                warning=100_000,
                quarantine=300_000,
                reject=None,
            ),
            _rule(
                "minimum_pose_anchor_coverage_ppm",
                "lower_is_worse",
                warning=900_000,
                quarantine=700_000,
                reject=None,
            ),
            _rule(
                "suspected_hand_swap_count",
                "higher_is_worse",
                warning=0,
                quarantine=2,
                reject=None,
            ),
            _rule(
                "temporal_discontinuity_count",
                "higher_is_worse",
                warning=0,
                quarantine=2,
                reject=None,
            ),
            _rule(
                "timestamp_discontinuity_count",
                "higher_is_worse",
                warning=0,
                quarantine=2,
                reject=None,
            ),
        ),
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
        "findings",
        "gaps",
        "reasons",
        "sequence_reports",
        "threshold_rules",
    }
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name in unique_fields:
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict) and field_schema.get("type") == "array":
                field_schema["uniqueItems"] = True


def generated_quality_schemas() -> dict[str, dict[str, object]]:
    """Return standalone Draft 2020-12 schemas for quality handoffs."""

    generated: dict[str, dict[str, object]] = {}
    for filename, model in QUALITY_SCHEMA_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        _strip_nested_schema_ids(schema, root=True)
        schema["$id"] = f"{_SCHEMA_BASE}{filename}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = _SCHEMA_COMMENT
        _harden_unique_arrays(schema)
        Draft202012Validator.check_schema(schema)
        generated[filename] = schema
    if set(generated) != set(QUALITY_SCHEMA_MODELS):
        raise QualityResourceError("generated quality schema registry is incomplete")
    return generated


def render_quality_json(value: BaseModel | Mapping[str, object]) -> str:
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
        raise QualityResourceError("quality resource is not JSON-serializable") from error


def generated_quality_resource_texts() -> dict[str, str]:
    """Render the exact policy and schema inventory."""

    rendered = {
        f"schemas/{filename}": render_quality_json(schema)
        for filename, schema in generated_quality_schemas().items()
    }
    rendered[f"config/{DEFAULT_POLICY_FILENAME}"] = render_quality_json(
        build_default_quality_policy()
    )
    if set(rendered) != GENERATED_QUALITY_RESOURCE_NAMES:
        raise QualityResourceError("generated quality resource registry is incomplete")
    return rendered


def _resource_bytes(relative_name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(*relative_name.split("/")).read_bytes()
    except OSError as error:
        raise QualityResourceError("a packaged quality resource is missing") from error


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


def load_packaged_default_quality_policy() -> LandmarkQualityPolicyV1:
    """Load and strictly validate the packaged default quality policy."""

    try:
        return validate_landmark_quality_policy(
            _resource_bytes(f"config/{DEFAULT_POLICY_FILENAME}")
        )
    except (TypeError, ValueError) as error:
        raise QualityResourceError("packaged quality policy is invalid") from error


def validate_packaged_quality_resources() -> None:
    """Check exact inventory, generated bytes, frozen hashes, and policy identity."""

    try:
        if _packaged_inventory() != GENERATED_QUALITY_RESOURCE_NAMES:
            raise QualityResourceError("packaged quality resource inventory is not exact")
        if set(PUBLISHED_QUALITY_RESOURCE_DIGESTS) != GENERATED_QUALITY_RESOURCE_NAMES:
            raise QualityResourceError("published quality resource baselines are incomplete")

        generated = generated_quality_resource_texts()
        for relative_name, expected_text in generated.items():
            captured = _resource_bytes(relative_name)
            if captured != expected_text.encode("utf-8"):
                raise QualityResourceError("packaged quality resource drift detected")
            digest = f"sha256:{hashlib.sha256(captured).hexdigest()}"
            if digest != PUBLISHED_QUALITY_RESOURCE_DIGESTS[relative_name]:
                raise QualityResourceError("a published quality resource changed in place")

        policy = load_packaged_default_quality_policy()
        policy_payload = policy.model_dump(mode="json", round_trip=True)
        policy_schema = generated_quality_schemas()["landmark-quality-policy-1.schema.json"]
        Draft202012Validator(policy_schema).validate(policy_payload)
        if (
            landmark_quality_policy_digest(policy)
            != PUBLISHED_DEFAULT_QUALITY_POLICY_SEMANTIC_DIGEST
        ):
            raise QualityResourceError("the published quality policy identity changed")
    except QualityResourceError:
        raise
    except (
        JsonSchemaValidationError,
        OSError,
        SchemaError,
        TypeError,
        ValueError,
    ) as error:
        raise QualityResourceError("packaged quality resources are invalid") from error


__all__ = [
    "DEFAULT_POLICY_FILENAME",
    "GENERATED_QUALITY_RESOURCE_NAMES",
    "PUBLISHED_DEFAULT_QUALITY_POLICY_SEMANTIC_DIGEST",
    "PUBLISHED_QUALITY_RESOURCE_DIGESTS",
    "QUALITY_SCHEMA_MODELS",
    "QualityResourceError",
    "build_default_quality_policy",
    "generated_quality_resource_texts",
    "generated_quality_schemas",
    "load_packaged_default_quality_policy",
    "render_quality_json",
    "validate_packaged_quality_resources",
]
