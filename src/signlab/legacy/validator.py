"""Standalone validation for public legacy evidence and private quarantine bundles."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from signlab.legacy.exporter import SENSITIVE_KEYS, LegacyExportError
from signlab.legacy.schemas import (
    DATA_ROLE,
    FORMAT_VERSION,
    PUBLIC_KIND,
    QUARANTINE_KIND,
    SCHEMAS,
)

type JsonObject = dict[str, object]

MACHINE_PATH_PATTERNS = (
    re.compile(rb"(?i)(?<![a-z0-9])[a-z]:[\\/]"),
    re.compile(rb"(?i)(?:^|[\s\"'=(])/(?:users|home)/[^/\s]+(?:/|\\)"),
    re.compile(rb"(?i)(?:\"|:\s*)/(?!/)[^\"\r\n]*"),
    re.compile(rb"(?i)file:///(?:[a-z]:|users/|home/)"),
    re.compile(rb"\\\\[^\\\s]+\\[^\\\s]+"),
)
RAW_UUID_PATTERN = re.compile(
    rb"(?i)(?<![0-9a-f])(?:[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})(?![0-9a-f])"
)
ABSOLUTE_TIMESTAMP_PATTERN = re.compile(rb"20[0-9]{2}-[0-9]{2}-[0-9]{2}T")
FORBIDDEN_RECORD_KEYS = SENSITIVE_KEYS | {
    "absolute_path",
    "attempt_id",
    "error",
    "freeform_note",
}


@dataclass(frozen=True)
class ValidationSummary:
    """Counts proven by a successful export validation."""

    runs: int
    attempts: int
    annotations: int
    detections: int
    sessions: int
    quarantine_verified: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> JsonObject:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LegacyExportError("A legacy-export JSON component is unreadable.") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LegacyExportError("A legacy-export JSON component is not an object.")
    return cast(JsonObject, value)


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LegacyExportError(f"The legacy export has an invalid {label} object.")
    return cast(JsonObject, value)


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise LegacyExportError(f"The legacy export has an invalid {label} array.")
    return cast(list[object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise LegacyExportError(f"The legacy export has unsupported {label} fields.")


def _safe_relative(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise LegacyExportError("A component path is not a string.")
    portable = PurePosixPath(value)
    if portable.is_absolute() or not portable.parts or ".." in portable.parts:
        raise LegacyExportError("A component path is not portable.")
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(portable.as_posix())).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise LegacyExportError("A component path escapes its export root.") from error
    return candidate


def _validate_component(root: Path, raw: object) -> tuple[str, int | None]:
    component = _object(raw, "component")
    allowed = {"path", "bytes", "sha256", "records"}
    if not set(component).issubset(allowed) or not {"path", "bytes", "sha256"}.issubset(component):
        raise LegacyExportError("A legacy-export component has unsupported fields.")
    path = _safe_relative(root, component["path"])
    expected_bytes = component["bytes"]
    expected_sha = component["sha256"]
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise LegacyExportError("A component byte count is invalid.")
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise LegacyExportError("A component digest is invalid.")
    if not path.is_file():
        raise LegacyExportError("A declared legacy-export component is missing.")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
        raise LegacyExportError("A legacy-export component failed its integrity check.")
    records = component.get("records")
    if records is not None and (not isinstance(records, int) or records < 0):
        raise LegacyExportError("A component record count is invalid.")
    return cast(str, component["path"]), records


def _scan_portability(path: Path, *, reject_timestamps: bool) -> None:
    content = path.read_bytes()
    if any(pattern.search(content) for pattern in MACHINE_PATH_PATTERNS):
        raise LegacyExportError("A legacy export exposes an absolute machine path.")
    if RAW_UUID_PATTERN.search(content):
        raise LegacyExportError("A legacy export exposes a raw UUID-like identifier.")
    if reject_timestamps and ABSOLUTE_TIMESTAMP_PATTERN.search(content):
        raise LegacyExportError("A quarantine record exposes an absolute timestamp.")


def _walk_keys(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _has_forbidden_record_key(value: object) -> bool:
    return bool(FORBIDDEN_RECORD_KEYS & {key.casefold() for key in _walk_keys(value)})


def _schema_is_closed(value: object) -> bool:
    if isinstance(value, dict):
        if (
            value.get("type") == "object"
            and "properties" in value
            and value.get("additionalProperties") is not False
        ):
            return False
        return all(_schema_is_closed(child) for child in value.values())
    if isinstance(value, list):
        return all(_schema_is_closed(child) for child in value)
    return True


def _validate_schema_instance(value: object, schema_name: str, label: str) -> None:
    schema = SCHEMAS[schema_name]
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise LegacyExportError(f"The legacy export has an invalid {label} contract.") from error


def _read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            value: Any = json.loads(line)
            if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
                raise LegacyExportError("A quarantine JSON Lines record is not an object.")
            records.append(cast(JsonObject, value))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LegacyExportError("A quarantine JSON Lines component is unreadable.") from error
    return records


def _validate_public(
    public_root: Path,
) -> tuple[JsonObject, JsonObject, JsonObject, JsonObject, int, dict[str, str]]:
    manifest = _read_object(public_root / "manifest.json")
    _validate_schema_instance(manifest, "public-manifest.schema.json", "public manifest")
    _exact_keys(
        manifest,
        {"schema_version", "kind", "source", "policy", "counts", "components"},
        "public manifest",
    )
    if manifest["schema_version"] != FORMAT_VERSION or manifest["kind"] != PUBLIC_KIND:
        raise LegacyExportError("The public legacy-export format is unsupported.")
    policy = _object(manifest["policy"], "public policy")
    if policy.get("data_role") != DATA_ROLE or policy.get("eligible_for_locked_test") is not False:
        raise LegacyExportError("The public legacy evidence has an unsafe data policy.")
    if policy.get("contains_private_artifacts") is not False:
        raise LegacyExportError("The public legacy evidence claims private artifact content.")

    component_paths: set[str] = set()
    component_records: dict[str, int | None] = {}
    public_documents: list[JsonObject] = [manifest]
    run_ids: list[object] = []
    source_runs: dict[str, str] = {}
    shard_numbers: list[object] = []
    run_count = 0
    status_counts: dict[str, int] = {"succeeded": 0, "failed": 0, "running": 0}
    for raw_component in _array(manifest["components"], "public components"):
        relative, declared_records = _validate_component(public_root, raw_component)
        if relative in component_paths:
            raise LegacyExportError("The public manifest declares a component twice.")
        component_paths.add(relative)
        component_records[relative] = declared_records
        component_path = public_root / relative
        _scan_portability(component_path, reject_timestamps=False)
        if relative.startswith("runs/"):
            shard = _read_object(component_path)
            public_documents.append(shard)
            _validate_schema_instance(shard, "run-index.schema.json", "run index")
            _exact_keys(shard, {"schema_version", "kind", "shard", "records"}, "run shard")
            if shard["schema_version"] != FORMAT_VERSION or shard["kind"] != (
                "signlab.legacy-run-index"
            ):
                raise LegacyExportError("A run-index shard has an unsupported format.")
            shard_number = shard["shard"]
            if (
                not isinstance(shard_number, int)
                or relative != f"runs/runs-{shard_number:03d}.json"
            ):
                raise LegacyExportError("A run-index shard has an inconsistent identity.")
            shard_numbers.append(shard_number)
            records = _array(shard["records"], "run records")
            if declared_records != len(records):
                raise LegacyExportError("A run shard count does not match its manifest.")
            for raw_record in records:
                record = _object(raw_record, "run record")
                _exact_keys(
                    record,
                    {
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
                    },
                    "run record",
                )
                status = record["status"]
                if not isinstance(status, str):
                    raise LegacyExportError("A run status is invalid.")
                status_counts[status] = status_counts.get(status, 0) + 1
                run_ids.append(record["record_id"])
                source_run_id = record["source_run_id"]
                if not isinstance(source_run_id, str):
                    raise LegacyExportError("A source run identity is invalid.")
                model_key = record["model_key"]
                if not isinstance(model_key, str):
                    raise LegacyExportError("A run model key is invalid.")
                if source_run_id in source_runs:
                    raise LegacyExportError("The run-index source identities are not unique.")
                source_runs[source_run_id] = model_key
            run_count += len(records)

    if shard_numbers != list(range(len(shard_numbers))):
        raise LegacyExportError("The run-index shard sequence is incomplete or unordered.")
    expected_run_ids = [f"run-{index:06d}" for index in range(1, run_count + 1)]
    if run_ids != expected_run_ids:
        raise LegacyExportError("The run-index record sequence is incomplete or unordered.")
    actual_files = {
        path.relative_to(public_root).as_posix()
        for path in public_root.rglob("*")
        if path.is_file()
    }
    if actual_files != component_paths | {"manifest.json"}:
        raise LegacyExportError("The public export has undeclared or missing files.")
    _scan_portability(public_root / "manifest.json", reject_timestamps=False)
    for name, expected_schema in SCHEMAS.items():
        schema_path = public_root / "schemas" / name
        schema = _read_object(schema_path)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise LegacyExportError("A legacy-export schema is invalid.") from error
        if schema != expected_schema or not _schema_is_closed(schema):
            raise LegacyExportError("A legacy-export schema is open or has drifted.")

    counts = _object(manifest["counts"], "public counts")
    expected_status_counts = {
        "runs_succeeded": status_counts.get("succeeded", 0),
        "runs_failed": status_counts.get("failed", 0),
        "runs_running": status_counts.get("running", 0),
    }
    if counts.get("runs") != run_count or any(
        counts.get(key) != value for key, value in expected_status_counts.items()
    ):
        raise LegacyExportError("The public run counts do not reconcile.")
    receipt = _read_object(public_root / "quarantine-receipt.json")
    public_documents.append(receipt)
    _validate_schema_instance(receipt, "quarantine-receipt.schema.json", "quarantine receipt")
    if receipt.get("kind") != "signlab.legacy-quarantine-receipt":
        raise LegacyExportError("The quarantine receipt has an unsupported format.")
    promoted = _read_object(public_root / "promoted-artifacts.json")
    public_documents.append(promoted)
    _validate_schema_instance(
        promoted,
        "promoted-artifacts.schema.json",
        "promoted artifacts",
    )
    plans = _read_object(public_root / "preprocessing-plans.json")
    public_documents.append(plans)
    _validate_schema_instance(
        plans,
        "preprocessing-plans.schema.json",
        "preprocessing plans",
    )
    plans_uri = plans.get("quarantine_uri")
    plans_digest = plans.get("source_sha256")
    if not isinstance(plans_uri, str) or not isinstance(plans_digest, str):
        raise LegacyExportError("The preprocessing-plan identity is invalid.")
    if plans_uri != f"quarantine://sha256/{plans_digest}.json":
        raise LegacyExportError("The preprocessing-plan URI does not match its digest.")
    plan_names = set(_object(plans["plans"], "preprocessing plans"))
    promoted_artifacts = _array(promoted["artifacts"], "promoted artifacts")
    promoted_run_ids: set[str] = set()
    promoted_model_keys: set[str] = set()
    for raw_artifact in promoted_artifacts:
        artifact = _object(raw_artifact, "promoted artifact")
        run_id = artifact.get("run_id")
        model_key = artifact.get("model_key")
        if not isinstance(run_id, str) or not isinstance(model_key, str):
            raise LegacyExportError("A promoted model identity is invalid.")
        if run_id in promoted_run_ids or model_key in promoted_model_keys:
            raise LegacyExportError("A promoted model identity is duplicated.")
        promoted_run_ids.add(run_id)
        promoted_model_keys.add(model_key)
        if source_runs.get(run_id) != model_key:
            raise LegacyExportError("A promoted model does not match its indexed run.")
        if artifact.get("preprocessing_plan") not in plan_names:
            raise LegacyExportError("A promoted model references an unknown preprocessing plan.")
        for role in ("model", "label_map"):
            identity = _object(artifact[role], f"promoted {role}")
            digest = identity.get("sha256")
            uri = identity.get("quarantine_uri")
            if not isinstance(digest, str) or not isinstance(uri, str):
                raise LegacyExportError("A promoted artifact identity is invalid.")
            suffix = "keras" if role == "model" else "json"
            if uri != f"quarantine://sha256/{digest}.{suffix}":
                raise LegacyExportError("A promoted artifact URI does not match its digest.")
    if component_records.get("promoted-artifacts.json") != len(promoted_artifacts):
        raise LegacyExportError("The promoted-artifact component count does not reconcile.")
    if component_records.get("preprocessing-plans.json") != len(plan_names):
        raise LegacyExportError("The preprocessing-plan component count does not reconcile.")
    if counts.get("promoted_models") != len(promoted_artifacts) or counts.get(
        "preprocessing_plans"
    ) != len(plan_names):
        raise LegacyExportError("The public artifact counts do not reconcile.")

    receipt_counts = _object(receipt["counts"], "receipt counts")
    for key, value in receipt_counts.items():
        if key != "quarantine_objects" and counts.get(key) != value:
            raise LegacyExportError("The public manifest and quarantine receipt disagree.")
    receipt_paths = {
        _object(raw, "receipt component").get("path")
        for raw in _array(receipt["components"], "receipt components")
    }
    expected_receipt_paths = {
        "manifest.json",
        "records/attempts.jsonl",
        "records/annotations.jsonl",
        "records/sessions.jsonl",
        "records/detections.jsonl",
    }
    if receipt_paths != expected_receipt_paths:
        raise LegacyExportError("The quarantine receipt component set is incomplete.")
    for document in public_documents:
        if _has_forbidden_record_key(document):
            raise LegacyExportError("The public export exposes a forbidden private field.")
    return manifest, receipt, promoted, plans, run_count, source_runs


def _validate_record_keys(record: JsonObject, kind: str) -> None:
    expected = {
        "attempt": {
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
        },
        "annotation": {
            "record_id",
            "attempt_ref",
            "data_role",
            "offset_ms",
            "feedback_type",
            "corrected_label",
            "freeform_note_present",
        },
        "session": {
            "record_id",
            "data_role",
            "model_run_id",
            "start_offset_ms",
            "duration_ms",
        },
        "detection": {
            "record_id",
            "session_ref",
            "data_role",
            "offset_ms",
            "predicted_label",
            "actual_label",
            "confidence",
            "correct",
        },
    }[kind]
    _exact_keys(record, expected, kind)
    if record.get("data_role") != DATA_ROLE:
        raise LegacyExportError("A quarantine record has an unsafe data role.")


def _validate_quarantine(
    quarantine_root: Path,
    receipt: JsonObject,
    promoted: JsonObject,
    plans: JsonObject,
    source_runs: dict[str, str],
) -> tuple[int, int, int, int]:
    _validate_schema_instance(receipt, "quarantine-receipt.schema.json", "quarantine receipt")
    receipt_components = _array(receipt.get("components"), "receipt components")
    receipt_by_path: dict[str, JsonObject] = {}
    for raw_component in receipt_components:
        relative, _ = _validate_component(quarantine_root, raw_component)
        component = _object(raw_component, "receipt component")
        if relative in receipt_by_path:
            raise LegacyExportError("The quarantine receipt declares a component twice.")
        receipt_by_path[relative] = component
    manifest = _read_object(quarantine_root / "manifest.json")
    _validate_schema_instance(
        manifest,
        "quarantine-manifest.schema.json",
        "quarantine manifest",
    )
    _exact_keys(
        manifest,
        {"schema_version", "kind", "policy", "counts", "record_components", "objects"},
        "quarantine manifest",
    )
    if manifest["schema_version"] != FORMAT_VERSION or manifest["kind"] != QUARANTINE_KIND:
        raise LegacyExportError("The private quarantine format is unsupported.")
    policy = _object(manifest["policy"], "quarantine policy")
    if policy.get("data_role") != DATA_ROLE or policy.get("eligible_for_locked_test") is not False:
        raise LegacyExportError("The private quarantine has an unsafe data policy.")
    if policy.get("publishable") is not False:
        raise LegacyExportError("The private quarantine is incorrectly marked publishable.")
    if receipt_by_path.get("manifest.json") is None:
        raise LegacyExportError("The quarantine receipt does not anchor its manifest.")

    component_records: dict[str, list[JsonObject]] = {}
    declared_record_files: set[str] = set()
    for raw_component in _array(manifest["record_components"], "record components"):
        relative, declared_count = _validate_component(quarantine_root, raw_component)
        declared_record_files.add(relative)
        path = quarantine_root / relative
        _scan_portability(path, reject_timestamps=True)
        records = _read_jsonl(path)
        if declared_count != len(records):
            raise LegacyExportError("A quarantine record count does not match its manifest.")
        component_records[relative] = records

    expected_record_files = {
        "records/attempts.jsonl",
        "records/annotations.jsonl",
        "records/sessions.jsonl",
        "records/detections.jsonl",
    }
    if declared_record_files != expected_record_files:
        raise LegacyExportError("The quarantine record set is incomplete.")
    if set(receipt_by_path) != expected_record_files | {"manifest.json"}:
        raise LegacyExportError("The quarantine receipt component set is incomplete.")
    for raw_component in _array(manifest["record_components"], "record components"):
        component = _object(raw_component, "record component")
        record_path_value = component.get("path")
        if (
            not isinstance(record_path_value, str)
            or receipt_by_path.get(record_path_value) != component
        ):
            raise LegacyExportError("The quarantine receipt and manifest disagree.")
    attempts = component_records["records/attempts.jsonl"]
    annotations = component_records["records/annotations.jsonl"]
    sessions = component_records["records/sessions.jsonl"]
    detections = component_records["records/detections.jsonl"]
    for record in attempts:
        _validate_schema_instance(record, "live-attempt.schema.json", "live attempt")
        _validate_record_keys(record, "attempt")
    for record in annotations:
        _validate_schema_instance(record, "annotation.schema.json", "annotation")
        _validate_record_keys(record, "annotation")
    for record in sessions:
        _validate_schema_instance(record, "session.schema.json", "session")
        _validate_record_keys(record, "session")
    for record in detections:
        _validate_schema_instance(record, "detection.schema.json", "detection")
        _validate_record_keys(record, "detection")
    if any(record.get("model_run_id") not in source_runs for record in [*attempts, *sessions]):
        raise LegacyExportError("A live record references an unknown model run.")
    if any(
        _has_forbidden_record_key(record)
        for record in [*attempts, *annotations, *sessions, *detections]
    ):
        raise LegacyExportError("A quarantine record exposes a forbidden private field.")
    record_sequences = (
        (attempts, "attempt", 6),
        (annotations, "annotation", 6),
        (sessions, "session", 4),
        (detections, "detection", 6),
    )
    for records, prefix, width in record_sequences:
        expected_ids = [f"{prefix}-{index:0{width}d}" for index in range(1, len(records) + 1)]
        if [record.get("record_id") for record in records] != expected_ids:
            raise LegacyExportError("A quarantine record sequence is incomplete or unordered.")
    attempt_ids = {record.get("record_id") for record in attempts}
    if len(attempt_ids) != len(attempts) or any(
        record.get("attempt_ref") not in attempt_ids for record in annotations
    ):
        raise LegacyExportError("Quarantine annotations do not reconcile to attempts.")
    session_ids = {record.get("record_id") for record in sessions}
    if len(session_ids) != len(sessions) or any(
        record.get("session_ref") not in session_ids for record in detections
    ):
        raise LegacyExportError("Quarantine detections do not reconcile to sessions.")

    object_paths: set[str] = set()
    object_uris: set[str] = set()
    objects_by_uri: dict[str, JsonObject] = {}
    for raw_object in _array(manifest["objects"], "quarantine objects"):
        item = _object(raw_object, "quarantine object")
        _exact_keys(item, {"uri", "storage_key", "bytes", "sha256", "roles"}, "object")
        uri = item["uri"]
        digest = item["sha256"]
        if not isinstance(uri, str) or not isinstance(digest, str):
            raise LegacyExportError("A quarantine object identity is invalid.")
        if re.fullmatch(r"quarantine://sha256/[0-9a-f]{64}\.(json|keras|npz)", uri) is None:
            raise LegacyExportError("A quarantine object URI is invalid.")
        path = _safe_relative(quarantine_root, item["storage_key"])
        expected_bytes = item["bytes"]
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise LegacyExportError("A quarantine object byte count is invalid.")
        if (
            not isinstance(digest, str)
            or uri != (f"quarantine://sha256/{digest}{path.suffix}")
            or path.name != f"{digest}{path.suffix}"
        ):
            raise LegacyExportError("A quarantine object identity is inconsistent.")
        if not path.is_file() or path.stat().st_size != expected_bytes or _sha256(path) != digest:
            raise LegacyExportError("A quarantine object failed its integrity check.")
        object_paths.add(path.relative_to(quarantine_root).as_posix())
        object_uris.add(uri)
        objects_by_uri[uri] = item
    if len(objects_by_uri) != len(_array(manifest["objects"], "quarantine objects")):
        raise LegacyExportError("The quarantine manifest declares a duplicate object.")
    quarantined_json: dict[str, JsonObject] = {}
    for uri, item in objects_by_uri.items():
        if not uri.endswith(".json"):
            continue
        path = _safe_relative(quarantine_root, item["storage_key"])
        _scan_portability(path, reject_timestamps=False)
        document = _read_object(path)
        if _has_forbidden_record_key(document):
            raise LegacyExportError("A quarantined JSON object exposes a private field.")
        quarantined_json[uri] = document
    referenced_uris = {
        cast(str, record["segment_uri"])
        for record in attempts
        if record.get("segment_uri") is not None
    }
    if not referenced_uris.issubset(object_uris):
        raise LegacyExportError("A live attempt references a missing quarantine object.")
    for uri in referenced_uris:
        roles = _array(objects_by_uri[uri].get("roles"), "quarantine roles")
        if "live-attempt-segment" not in roles:
            raise LegacyExportError("A live segment object has an inconsistent role.")

    expected_public_objects: list[tuple[str, str, int | None, str]] = []
    for raw_artifact in _array(promoted["artifacts"], "promoted artifacts"):
        artifact = _object(raw_artifact, "promoted artifact")
        model_key = artifact["model_key"]
        if not isinstance(model_key, str):
            raise LegacyExportError("A promoted artifact model key is invalid.")
        for role in ("model", "label_map"):
            identity = _object(artifact[role], f"promoted {role}")
            uri = identity["quarantine_uri"]
            digest = identity["sha256"]
            byte_count = identity["bytes"]
            if not isinstance(uri, str) or not isinstance(digest, str):
                raise LegacyExportError("A promoted artifact identity is invalid.")
            if not isinstance(byte_count, int):
                raise LegacyExportError("A promoted artifact byte count is invalid.")
            quarantine_role = "promoted-model" if role == "model" else "promoted-label-map"
            expected_public_objects.append(
                (uri, digest, byte_count, f"{quarantine_role}:{model_key}")
            )
    plan_uri = plans["quarantine_uri"]
    plan_digest = plans["source_sha256"]
    if not isinstance(plan_uri, str) or not isinstance(plan_digest, str):
        raise LegacyExportError("The preprocessing-plan identity is invalid.")
    expected_public_objects.append((plan_uri, plan_digest, None, "preprocessing-plan-registry"))
    for uri, digest, byte_count, role in expected_public_objects:
        stored_object = objects_by_uri.get(uri)
        if stored_object is None or stored_object.get("sha256") != digest:
            raise LegacyExportError("A public artifact is absent from the private quarantine.")
        if byte_count is not None and stored_object.get("bytes") != byte_count:
            raise LegacyExportError("A public artifact byte count does not match quarantine.")
        if role not in _array(stored_object.get("roles"), "quarantine roles"):
            raise LegacyExportError("A public artifact has an inconsistent quarantine role.")
    public_plans = _object(plans["plans"], "preprocessing plans")
    if quarantined_json.get(plan_uri) != public_plans:
        raise LegacyExportError("The public preprocessing plans do not match quarantine.")
    for raw_artifact in _array(promoted["artifacts"], "promoted artifacts"):
        artifact = _object(raw_artifact, "promoted artifact")
        label_map = _object(artifact["label_map"], "promoted label map")
        label_uri = label_map["quarantine_uri"]
        if not isinstance(label_uri, str) or quarantined_json.get(label_uri) != label_map.get(
            "labels"
        ):
            raise LegacyExportError("A public label map does not match quarantine.")

    actual_files = {
        path.relative_to(quarantine_root).as_posix()
        for path in quarantine_root.rglob("*")
        if path.is_file()
    }
    expected_files = {"manifest.json"} | declared_record_files | object_paths
    if actual_files != expected_files:
        raise LegacyExportError("The quarantine has undeclared or missing files.")
    _scan_portability(quarantine_root / "manifest.json", reject_timestamps=True)

    counts = _object(manifest["counts"], "quarantine counts")
    object_items = list(objects_by_uri.values())
    segment_objects = [item for item in object_items if cast(str, item["uri"]).endswith(".npz")]
    referenced_segment_objects = [
        item
        for item in segment_objects
        if "live-attempt-segment" in _array(item.get("roles"), "quarantine roles")
    ]
    orphan_segment_objects = [
        item
        for item in segment_objects
        if "orphan-live-segment" in _array(item.get("roles"), "quarantine roles")
    ]
    if len(segment_objects) != len(referenced_segment_objects) + len(orphan_segment_objects):
        raise LegacyExportError("The quarantined segment roles do not reconcile.")
    expected_counts = {
        "attempts": len(attempts),
        "annotations": len(annotations),
        "sessions": len(sessions),
        "detections": len(detections),
        "annotated_attempts": len({record["attempt_ref"] for record in annotations}),
        "unannotated_attempts": len(attempts)
        - len({record["attempt_ref"] for record in annotations}),
        "quarantine_objects": len(object_uris),
        "segments": len(segment_objects),
        "referenced_segments": len(referenced_segment_objects),
        "orphan_segments": len(orphan_segment_objects),
        "promoted_models": len(_array(promoted["artifacts"], "promoted artifacts")),
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise LegacyExportError("The quarantine aggregate counts do not reconcile.")
    receipt_counts = _object(receipt.get("counts"), "receipt counts")
    if receipt_counts != counts:
        raise LegacyExportError("The public receipt does not match the private quarantine.")
    return len(attempts), len(annotations), len(detections), len(sessions)


def validate_legacy_export(
    *,
    public_root: Path,
    quarantine_root: Path | None = None,
) -> ValidationSummary:
    """Validate a committed public export and, optionally, its private quarantine."""
    if not public_root.is_dir():
        raise LegacyExportError("The public legacy export is unavailable.")
    manifest, receipt, promoted, plans, runs, source_runs = _validate_public(public_root.resolve())
    counts = _object(manifest["counts"], "public counts")
    attempts = cast(int, counts.get("attempts", 0))
    annotations = cast(int, counts.get("annotations", 0))
    detections = cast(int, counts.get("detections", 0))
    sessions = cast(int, counts.get("sessions", 0))
    quarantine_verified = False
    if quarantine_root is not None:
        if not quarantine_root.is_dir():
            raise LegacyExportError("The private legacy quarantine is unavailable.")
        private_counts = _validate_quarantine(
            quarantine_root.resolve(),
            receipt,
            promoted,
            plans,
            source_runs,
        )
        if private_counts != (attempts, annotations, detections, sessions):
            raise LegacyExportError("Public and private legacy counts do not reconcile.")
        quarantine_verified = True
    return ValidationSummary(
        runs=runs,
        attempts=attempts,
        annotations=annotations,
        detections=detections,
        sessions=sessions,
        quarantine_verified=quarantine_verified,
    )
