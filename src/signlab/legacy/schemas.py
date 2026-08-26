"""JSON Schema 2020-12 contracts for the legacy export."""

from __future__ import annotations

from typing import Final, cast

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]

SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"
FORMAT_VERSION: Final = 1
PUBLIC_KIND: Final = "signlab.legacy-public-export"
QUARANTINE_KIND: Final = "signlab.legacy-quarantine"
DATA_ROLE: Final = "development-only"


def _object_schema(
    properties: JsonObject,
    required: list[str],
    *,
    title: str,
) -> JsonObject:
    return {
        "title": title,
        "type": "object",
        "properties": properties,
        "required": cast(JsonValue, required),
        "additionalProperties": False,
    }


def _root_schema(properties: JsonObject, required: list[str], *, title: str) -> JsonObject:
    schema = _object_schema(properties, required, title=title)
    schema["$schema"] = SCHEMA_DIALECT
    return schema


SHA256_SCHEMA: JsonObject = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
NONNEGATIVE_INTEGER_SCHEMA: JsonObject = {"type": "integer", "minimum": 0}
NULLABLE_NONNEGATIVE_INTEGER_SCHEMA: JsonObject = {
    "type": ["integer", "null"],
    "minimum": 0,
}
NULLABLE_NUMBER_SCHEMA: JsonObject = {"type": ["number", "null"]}
QUARANTINE_URI_SCHEMA: JsonObject = {
    "type": "string",
    "pattern": "^quarantine://sha256/[0-9a-f]{64}\\.(json|keras|npz)$",
}
PORTABLE_LOCATOR_SCHEMA: JsonObject = {
    "anyOf": [
        {
            "type": "string",
            "maxLength": 4096,
            "not": {
                "pattern": "(?:^[A-Za-z]:[\\\\/]|^/|^[Ff][Ii][Ll][Ee]:///|^\\\\\\\\)",
            },
        },
        {"type": "null"},
    ]
}

PORTABLE_COMPONENT_SCHEMA: JsonObject = _object_schema(
    {
        "path": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_.-]*(/[a-z0-9_.-]+)*$"},
        "bytes": NONNEGATIVE_INTEGER_SCHEMA,
        "sha256": SHA256_SCHEMA,
        "records": NONNEGATIVE_INTEGER_SCHEMA,
    },
    ["path", "bytes", "sha256"],
    title="Portable component",
)

PUBLIC_COUNT_KEYS = (
    "runs",
    "runs_succeeded",
    "runs_failed",
    "runs_running",
    "preprocessing_plans",
    "promoted_models",
    "attempts",
    "annotations",
    "annotated_attempts",
    "unannotated_attempts",
    "sessions",
    "detections",
    "segments",
    "referenced_segments",
    "orphan_segments",
)
PUBLIC_COUNTS_SCHEMA: JsonObject = _object_schema(
    {key: NONNEGATIVE_INTEGER_SCHEMA for key in PUBLIC_COUNT_KEYS},
    list(PUBLIC_COUNT_KEYS),
    title="Public export counts",
)

PRIVATE_COUNT_KEYS = (
    "attempts",
    "annotations",
    "annotated_attempts",
    "unannotated_attempts",
    "sessions",
    "detections",
    "segments",
    "referenced_segments",
    "orphan_segments",
    "promoted_models",
    "quarantine_objects",
)
PRIVATE_COUNTS_SCHEMA: JsonObject = _object_schema(
    {key: NONNEGATIVE_INTEGER_SCHEMA for key in PRIVATE_COUNT_KEYS},
    list(PRIVATE_COUNT_KEYS),
    title="Private quarantine counts",
)

PUBLIC_MANIFEST_SCHEMA: JsonObject = _root_schema(
    {
        "schema_version": {"const": FORMAT_VERSION},
        "kind": {"const": PUBLIC_KIND},
        "source": _object_schema(
            {
                "head_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "tree_object": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "audit_sha256": SHA256_SCHEMA,
            },
            ["head_commit", "tree_object", "audit_sha256"],
            title="Legacy source anchor",
        ),
        "policy": _object_schema(
            {
                "data_role": {"const": DATA_ROLE},
                "eligible_for_locked_test": {"const": False},
                "contains_private_artifacts": {"const": False},
                "durability": {"const": "local-only-pending-private-remote"},
            },
            [
                "data_role",
                "eligible_for_locked_test",
                "contains_private_artifacts",
                "durability",
            ],
            title="Legacy evidence policy",
        ),
        "counts": PUBLIC_COUNTS_SCHEMA,
        "components": {"type": "array", "items": PORTABLE_COMPONENT_SCHEMA},
    },
    ["schema_version", "kind", "source", "policy", "counts", "components"],
    title="SignLab public legacy export manifest",
)

ARTIFACT_LOCATOR_SCHEMA: JsonObject = _object_schema(
    {
        "registered_locator": PORTABLE_LOCATOR_SCHEMA,
        "resolved_locator": PORTABLE_LOCATOR_SCHEMA,
        "availability": {
            "enum": ["available", "ambiguous", "missing", "not-recorded"],
        },
    },
    ["registered_locator", "resolved_locator", "availability"],
    title="Historical artifact locator",
)

RUN_CONFIGURATION_SCHEMA: JsonObject = _object_schema(
    {
        "model_settings": {"type": ["object", "null"]},
        "global_settings": {"type": ["object", "null"]},
        "data_locator": PORTABLE_LOCATOR_SCHEMA,
        "sequence_length": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
        "test_fraction": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "validation_fraction": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
        },
        "batch_size": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
        "epochs": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
    },
    [
        "model_settings",
        "global_settings",
        "data_locator",
        "sequence_length",
        "test_fraction",
        "validation_fraction",
        "batch_size",
        "epochs",
    ],
    title="Historical run configuration",
)

RUN_METRICS_SCHEMA: JsonObject = _object_schema(
    {
        "quick": {"type": ["object", "null"]},
        "test_loss": NULLABLE_NUMBER_SCHEMA,
        "test_accuracy": NULLABLE_NUMBER_SCHEMA,
        "samples_train": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
        "samples_validation": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
        "samples_test": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
    },
    [
        "quick",
        "test_loss",
        "test_accuracy",
        "samples_train",
        "samples_validation",
        "samples_test",
    ],
    title="Historical run metrics",
)

RUN_ARTIFACT_KEYS = (
    "directory",
    "model",
    "label_map",
    "configuration",
    "history",
    "predictions",
    "metrics",
)
RUN_ARTIFACTS_SCHEMA: JsonObject = _object_schema(
    {role: ARTIFACT_LOCATOR_SCHEMA for role in RUN_ARTIFACT_KEYS},
    list(RUN_ARTIFACT_KEYS),
    title="Historical run artifacts",
)

RUN_VALIDITY_SCHEMA: JsonObject = _object_schema(
    {
        "data_role": {"const": DATA_ROLE},
        "eligible_for_locked_test": {"const": False},
        "notes": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "legacy_notes_present": {"type": "boolean"},
        "legacy_error_present": {"type": "boolean"},
    },
    [
        "data_role",
        "eligible_for_locked_test",
        "notes",
        "legacy_notes_present",
        "legacy_error_present",
    ],
    title="Historical run validity",
)

RUN_RECORD_KEYS = (
    "record_id",
    "run_name",
    "source_run_id",
    "started_at",
    "finished_at",
    "status",
    "model_key",
    "configuration",
    "metrics",
    "artifacts",
    "validity",
)
RUN_RECORD_SCHEMA: JsonObject = _object_schema(
    {
        "record_id": {"type": "string", "pattern": "^run-[0-9]{6}$"},
        "run_name": {"type": "string", "minLength": 1, "maxLength": 160},
        "source_run_id": {"type": "string", "minLength": 1, "maxLength": 321},
        "started_at": {"type": "string", "minLength": 1, "maxLength": 64},
        "finished_at": {"type": ["string", "null"], "maxLength": 64},
        "status": {"enum": ["succeeded", "failed", "running"]},
        "model_key": {"type": "string", "minLength": 1, "maxLength": 160},
        "configuration": RUN_CONFIGURATION_SCHEMA,
        "metrics": RUN_METRICS_SCHEMA,
        "artifacts": RUN_ARTIFACTS_SCHEMA,
        "validity": RUN_VALIDITY_SCHEMA,
    },
    list(RUN_RECORD_KEYS),
    title="Historical run record",
)

RUN_SHARD_SCHEMA: JsonObject = _root_schema(
    {
        "schema_version": {"const": FORMAT_VERSION},
        "kind": {"const": "signlab.legacy-run-index"},
        "shard": NONNEGATIVE_INTEGER_SCHEMA,
        "records": {"type": "array", "items": RUN_RECORD_SCHEMA},
    },
    ["schema_version", "kind", "shard", "records"],
    title="Legacy run-index shard",
)

PROMOTED_VALIDITY_SCHEMA: JsonObject = _object_schema(
    {
        "data_role": {"const": DATA_ROLE},
        "eligible_for_locked_test": {"const": False},
        "notes": {"type": "array", "minItems": 1, "items": {"type": "string"}},
    },
    ["data_role", "eligible_for_locked_test", "notes"],
    title="Promoted artifact validity",
)

PROMOTED_MODEL_SCHEMA: JsonObject = _object_schema(
    {
        "bytes": NONNEGATIVE_INTEGER_SCHEMA,
        "sha256": SHA256_SCHEMA,
        "quarantine_uri": QUARANTINE_URI_SCHEMA,
        "storage_status": {"const": "local-quarantine"},
    },
    ["bytes", "sha256", "quarantine_uri", "storage_status"],
    title="Promoted model identity",
)

PROMOTED_LABEL_MAP_SCHEMA: JsonObject = _object_schema(
    {
        "bytes": NONNEGATIVE_INTEGER_SCHEMA,
        "sha256": SHA256_SCHEMA,
        "quarantine_uri": QUARANTINE_URI_SCHEMA,
        "labels": {
            "type": "object",
            "propertyNames": {"pattern": "^[0-9]+$"},
            "additionalProperties": {
                "enum": ["hello", "no", "please", "thank you", "yes"],
            },
        },
    },
    ["bytes", "sha256", "quarantine_uri", "labels"],
    title="Promoted label-map identity",
)

PROMOTED_ARTIFACT_ITEM_SCHEMA: JsonObject = _object_schema(
    {
        "run_id": {"type": "string", "minLength": 1, "maxLength": 160},
        "model_key": {"type": "string", "minLength": 1, "maxLength": 160},
        "preprocessing_plan": {"type": "string", "minLength": 1, "maxLength": 160},
        "model": PROMOTED_MODEL_SCHEMA,
        "label_map": PROMOTED_LABEL_MAP_SCHEMA,
        "validity": PROMOTED_VALIDITY_SCHEMA,
    },
    ["run_id", "model_key", "preprocessing_plan", "model", "label_map", "validity"],
    title="Promoted legacy artifact",
)

PROMOTED_ARTIFACT_SCHEMA: JsonObject = _root_schema(
    {
        "schema_version": {"const": FORMAT_VERSION},
        "kind": {"const": "signlab.legacy-promoted-artifacts"},
        "artifacts": {"type": "array", "items": PROMOTED_ARTIFACT_ITEM_SCHEMA},
    },
    ["schema_version", "kind", "artifacts"],
    title="Promoted legacy artifacts",
)

PLAN_STEP_SCHEMA: JsonObject = _object_schema(
    {
        "key": {"type": "string", "minLength": 1},
        "params": {"type": "object"},
    },
    ["key", "params"],
    title="Historical preprocessing step",
)

PLAN_SCHEMA: JsonObject = _object_schema(
    {
        "name": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "created_at": {"type": "string", "minLength": 1},
        "updated_at": {"type": "string", "minLength": 1},
        "steps": {"type": "array", "minItems": 1, "items": PLAN_STEP_SCHEMA},
    },
    ["name", "description", "created_at", "updated_at", "steps"],
    title="Historical preprocessing plan",
)

PREPROCESSING_PLAN_SCHEMA: JsonObject = _root_schema(
    {
        "schema_version": {"const": FORMAT_VERSION},
        "kind": {"const": "signlab.legacy-preprocessing-plans"},
        "source_sha256": SHA256_SCHEMA,
        "quarantine_uri": QUARANTINE_URI_SCHEMA,
        "plans": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": PLAN_SCHEMA,
        },
    },
    ["schema_version", "kind", "source_sha256", "quarantine_uri", "plans"],
    title="Historical preprocessing plans",
)

QUARANTINE_RECEIPT_POLICY_SCHEMA: JsonObject = _object_schema(
    {
        "data_role": {"const": DATA_ROLE},
        "eligible_for_locked_test": {"const": False},
        "publishable": {"const": False},
        "durability": {"const": "local-only-pending-private-remote"},
        "contains_individual_object_hashes": {"const": False},
    },
    [
        "data_role",
        "eligible_for_locked_test",
        "publishable",
        "durability",
        "contains_individual_object_hashes",
    ],
    title="Public quarantine-receipt policy",
)

QUARANTINE_RECEIPT_SCHEMA: JsonObject = _root_schema(
    {
        "schema_version": {"const": FORMAT_VERSION},
        "kind": {"const": "signlab.legacy-quarantine-receipt"},
        "policy": QUARANTINE_RECEIPT_POLICY_SCHEMA,
        "counts": PRIVATE_COUNTS_SCHEMA,
        "components": {"type": "array", "items": PORTABLE_COMPONENT_SCHEMA},
    },
    ["schema_version", "kind", "policy", "counts", "components"],
    title="Private quarantine receipt",
)

QUARANTINE_POLICY_SCHEMA: JsonObject = _object_schema(
    {
        "data_role": {"const": DATA_ROLE},
        "eligible_for_locked_test": {"const": False},
        "contains_private_artifacts": {"const": True},
        "publishable": {"const": False},
        "identifiers": {"const": "pseudonymized-ordinal"},
        "timestamps": {"const": "live-records-relative-offsets-only"},
    },
    [
        "data_role",
        "eligible_for_locked_test",
        "contains_private_artifacts",
        "publishable",
        "identifiers",
        "timestamps",
    ],
    title="Private quarantine policy",
)

QUARANTINE_OBJECT_SCHEMA: JsonObject = _object_schema(
    {
        "uri": QUARANTINE_URI_SCHEMA,
        "storage_key": {
            "type": "string",
            "pattern": "^objects/sha256/[0-9a-f]{64}\\.(json|keras|npz)$",
        },
        "bytes": NONNEGATIVE_INTEGER_SCHEMA,
        "sha256": SHA256_SCHEMA,
        "roles": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    },
    ["uri", "storage_key", "bytes", "sha256", "roles"],
    title="Content-addressed quarantine object",
)

QUARANTINE_MANIFEST_SCHEMA: JsonObject = _root_schema(
    {
        "schema_version": {"const": FORMAT_VERSION},
        "kind": {"const": QUARANTINE_KIND},
        "policy": QUARANTINE_POLICY_SCHEMA,
        "counts": PRIVATE_COUNTS_SCHEMA,
        "record_components": {"type": "array", "items": PORTABLE_COMPONENT_SCHEMA},
        "objects": {"type": "array", "items": QUARANTINE_OBJECT_SCHEMA},
    },
    ["schema_version", "kind", "policy", "counts", "record_components", "objects"],
    title="Private legacy quarantine manifest",
)

ATTEMPT_SCHEMA: JsonObject = _root_schema(
    {
        "record_id": {"type": "string", "pattern": "^attempt-[0-9]{6}$"},
        "data_role": {"const": DATA_ROLE},
        "start_offset_ms": NONNEGATIVE_INTEGER_SCHEMA,
        "duration_ms": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
        "detected": {"type": "boolean"},
        "intended_label": {
            "anyOf": [
                {"enum": ["hello", "no", "please", "thank you", "yes"]},
                {"type": "null"},
            ]
        },
        "predicted_label": {
            "enum": ["NONE", "hello", "no", "nothing", "please", "thank you", "yes"],
        },
        "correct": {"type": ["boolean", "null"]},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "latency_ms": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
        "segment_frames": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
        "model_run_id": {"type": "string", "minLength": 1},
        "segmentation": {"type": "object"},
        "segment_uri": {"anyOf": [QUARANTINE_URI_SCHEMA, {"type": "null"}]},
    },
    [
        "record_id",
        "data_role",
        "start_offset_ms",
        "duration_ms",
        "detected",
        "intended_label",
        "predicted_label",
        "correct",
        "confidence",
        "latency_ms",
        "segment_frames",
        "model_run_id",
        "segmentation",
        "segment_uri",
    ],
    title="Sanitized live attempt",
)

ANNOTATION_SCHEMA: JsonObject = _root_schema(
    {
        "record_id": {"type": "string", "pattern": "^annotation-[0-9]{6}$"},
        "attempt_ref": {"type": "string", "pattern": "^attempt-[0-9]{6}$"},
        "data_role": {"const": DATA_ROLE},
        "offset_ms": NONNEGATIVE_INTEGER_SCHEMA,
        "feedback_type": {"enum": ["confirm", "missed", "wrong"]},
        "corrected_label": {
            "anyOf": [
                {"enum": ["hello", "no", "please", "thank you", "yes"]},
                {"type": "null"},
            ]
        },
        "freeform_note_present": {"type": "boolean"},
    },
    [
        "record_id",
        "attempt_ref",
        "data_role",
        "offset_ms",
        "feedback_type",
        "corrected_label",
        "freeform_note_present",
    ],
    title="Sanitized feedback annotation",
)

SESSION_SCHEMA: JsonObject = _root_schema(
    {
        "record_id": {"type": "string", "pattern": "^session-[0-9]{4}$"},
        "data_role": {"const": DATA_ROLE},
        "model_run_id": {"type": "string", "minLength": 1},
        "start_offset_ms": NONNEGATIVE_INTEGER_SCHEMA,
        "duration_ms": NULLABLE_NONNEGATIVE_INTEGER_SCHEMA,
    },
    ["record_id", "data_role", "model_run_id", "start_offset_ms", "duration_ms"],
    title="Sanitized legacy feedback session",
)

DETECTION_SCHEMA: JsonObject = _root_schema(
    {
        "record_id": {"type": "string", "pattern": "^detection-[0-9]{6}$"},
        "session_ref": {"type": "string", "pattern": "^session-[0-9]{4}$"},
        "data_role": {"const": DATA_ROLE},
        "offset_ms": NONNEGATIVE_INTEGER_SCHEMA,
        "predicted_label": {
            "anyOf": [
                {"enum": ["hello", "no", "please", "thank you", "yes"]},
                {"type": "null"},
            ]
        },
        "actual_label": {
            "anyOf": [
                {"enum": ["hello", "no", "please", "thank you", "yes"]},
                {"type": "null"},
            ]
        },
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "correct": {"type": ["boolean", "null"]},
    },
    [
        "record_id",
        "session_ref",
        "data_role",
        "offset_ms",
        "predicted_label",
        "actual_label",
        "confidence",
        "correct",
    ],
    title="Sanitized legacy detection",
)

SCHEMAS: Final[dict[str, JsonObject]] = {
    "annotation.schema.json": ANNOTATION_SCHEMA,
    "detection.schema.json": DETECTION_SCHEMA,
    "live-attempt.schema.json": ATTEMPT_SCHEMA,
    "preprocessing-plans.schema.json": PREPROCESSING_PLAN_SCHEMA,
    "promoted-artifacts.schema.json": PROMOTED_ARTIFACT_SCHEMA,
    "public-manifest.schema.json": PUBLIC_MANIFEST_SCHEMA,
    "quarantine-manifest.schema.json": QUARANTINE_MANIFEST_SCHEMA,
    "quarantine-receipt.schema.json": QUARANTINE_RECEIPT_SCHEMA,
    "run-index.schema.json": RUN_SHARD_SCHEMA,
    "session.schema.json": SESSION_SCHEMA,
}
