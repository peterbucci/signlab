"""Deterministic, read-only export of legacy development evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast
from urllib.parse import quote

from signlab.legacy.schemas import (
    DATA_ROLE,
    FORMAT_VERSION,
    PUBLIC_KIND,
    QUARANTINE_KIND,
    SCHEMAS,
)

type JsonObject = dict[str, object]

RUN_SHARD_SIZE = 100
KNOWN_LOGICAL_ROOTS = (
    ("data", "live_eval", "segments"),
    ("data", "landmarks"),
    ("data", "results_old"),
    ("data", "results"),
    ("runs", "sweeps"),
    ("data", "models"),
    ("data", "plans"),
    ("data", "live_eval"),
    ("data", "live"),
)
ARTIFACT_COLUMNS = {
    "model": "model_path",
    "label_map": "label_map_path",
    "configuration": "config_path",
    "history": "history_path",
    "predictions": "predictions_path",
    "metrics": "metrics_path",
}
MACHINE_PATH_PATTERNS = (
    re.compile(r"(?i)(?<![a-z0-9])[a-z]:[\\/]"),
    re.compile(r"(?i)(?:^|[\s\"'=(])/(?:users|home)/[^/\s]+(?:/|\\)"),
    re.compile(r"(?i)file:///(?:[a-z]:|users/|home/)"),
    re.compile(r"^\\\\[^\\]+\\[^\\]+"),
)
SENSITIVE_KEYS = {
    "email",
    "participant",
    "participant_id",
    "subject",
    "subject_id",
    "user_id",
    "username",
}
FREEFORM_KEYS = {"error", "freeform_note", "notes"}
SAFE_OBJECT_SUFFIXES = {".json", ".keras", ".npz"}
LEGACY_LABELS = frozenset({"NONE", "hello", "no", "nothing", "please", "thank you", "yes"})
LOCATOR_KEY_SUFFIXES = (
    "_dir",
    "_directory",
    "_file",
    "_files",
    "_locator",
    "_locators",
    "_path",
    "_paths",
    "_root",
    "_roots",
    "_uri",
    "_uris",
    "_url",
    "_urls",
)


class LegacyExportError(RuntimeError):
    """A sanitized legacy-export failure safe to display in a public CLI."""


@dataclass(frozen=True)
class ExportSummary:
    """Aggregate evidence from one successful export."""

    runs: int
    attempts: int
    annotations: int
    detections: int
    sessions: int
    promoted_models: int
    quarantined_segments: int
    quarantine_objects: int


def _read_json_object(path: Path) -> JsonObject:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LegacyExportError("A required JSON source is not an object.")
    return cast(JsonObject, value)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"".join(
        (
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode()
        for record in records
    )
    path.write_bytes(content)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component(path: Path, root: Path, *, records: int | None = None) -> JsonObject:
    relative = path.relative_to(root).as_posix()
    result: JsonObject = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _safe_source(legacy_root: Path, logical_path: str) -> Path:
    portable = PurePosixPath(logical_path)
    if portable.is_absolute() or not portable.parts or ".." in portable.parts:
        raise LegacyExportError("A legacy logical path is not portable.")
    candidate = legacy_root
    for part in portable.parts:
        candidate /= part
        if candidate.is_symlink():
            raise LegacyExportError("A legacy source path crosses a symbolic link.")
    resolved_root = legacy_root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise LegacyExportError("A legacy logical path escapes the supplied root.") from error
    return resolved


def _open_immutable_database(path: Path) -> sqlite3.Connection:
    if path.with_name(f"{path.name}-wal").exists():
        raise LegacyExportError("A legacy database has an active WAL and cannot be exported.")
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _run_git(legacy_root: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(legacy_root), *arguments],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env=environment,
        text=True,
    )
    return completed.stdout.strip()


def _expected_mapping(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LegacyExportError(f"The audit snapshot has an invalid {label} section.")
    return cast(JsonObject, value)


def _expected_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise LegacyExportError(f"The audit snapshot has an invalid {label} section.")
    return cast(list[object], value)


def _verify_file(legacy_root: Path, evidence: JsonObject) -> None:
    logical_path = evidence.get("logical_path")
    expected_bytes = evidence.get("bytes")
    expected_sha256 = evidence.get("sha256")
    if not isinstance(logical_path, str) or not isinstance(expected_bytes, int):
        raise LegacyExportError("The audit contains invalid file evidence.")
    if not isinstance(expected_sha256, str):
        raise LegacyExportError("The audit contains invalid file evidence.")
    source = _safe_source(legacy_root, logical_path)
    if not source.is_file():
        raise LegacyExportError("An audited legacy source artifact is missing.")
    if source.stat().st_size != expected_bytes or _sha256(source) != expected_sha256:
        raise LegacyExportError("An audited legacy source artifact changed.")


def _verify_source(legacy_root: Path, audit: JsonObject) -> None:
    if audit.get("schema_version") != 2:
        raise LegacyExportError("The legacy audit version is unsupported.")
    repository = _expected_mapping(audit.get("repository"), "repository")
    expected_head = repository.get("head_commit")
    expected_tree = repository.get("tree_object")
    if not isinstance(expected_head, str) or not isinstance(expected_tree, str):
        raise LegacyExportError("The legacy Git anchors are invalid.")
    try:
        actual_head = _run_git(legacy_root, "rev-parse", "HEAD")
        actual_tree = _run_git(legacy_root, "rev-parse", "HEAD^{tree}")
    except (OSError, subprocess.SubprocessError) as error:
        raise LegacyExportError("The legacy Git anchors could not be read.") from error
    if (actual_head, actual_tree) != (expected_head, expected_tree):
        raise LegacyExportError("The legacy Git anchors do not match the audit.")

    for raw_evidence in _expected_sequence(audit.get("integrity"), "integrity"):
        _verify_file(legacy_root, _expected_mapping(raw_evidence, "integrity item"))
    for raw_run in _expected_sequence(audit.get("reported_runs"), "reported runs"):
        run = _expected_mapping(raw_run, "reported run")
        model_evidence: JsonObject = {
            "logical_path": run.get("model_logical_path"),
            "bytes": run.get("model_bytes"),
            "sha256": run.get("model_sha256"),
        }
        _verify_file(legacy_root, model_evidence)
        for raw_evidence in _expected_sequence(run.get("source_evidence"), "source evidence"):
            _verify_file(legacy_root, _expected_mapping(raw_evidence, "source evidence item"))

    live_state = _expected_mapping(audit.get("live_state"), "live state")
    database_checks = (
        ("data/models/runs.db", "runs", "run_database", "runs"),
        ("data/models/live_feedback.db", "feedback", "feedback_database", "feedback_records"),
        ("data/live_eval/live_eval.db", "attempts", "live_evaluation_database", "attempts"),
    )
    for logical_path, table, section_name, count_name in database_checks:
        section = _expected_mapping(live_state.get(section_name), section_name)
        expected_count = section.get(count_name)
        if not isinstance(expected_count, int):
            raise LegacyExportError("The audit contains an invalid database count.")
        connection = _open_immutable_database(_safe_source(legacy_root, logical_path))
        try:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            actual_count = cast(int, row[0])
        finally:
            connection.close()
        if actual_count != expected_count:
            raise LegacyExportError("A legacy database count does not match the audit.")


def _path_parts(value: str) -> tuple[str, ...]:
    if "\\" in value or re.match(r"(?i)^[a-z]:", value):
        return PureWindowsPath(value).parts
    return PurePosixPath(value).parts


def _logical_path_from_value(value: str) -> PurePosixPath | None:
    parts = _path_parts(value)
    lowered = tuple(part.lower() for part in parts)
    for root in KNOWN_LOGICAL_ROOTS:
        for index in range(len(parts) - len(root) + 1):
            if lowered[index : index + len(root)] == root:
                selected = parts[index:]
                if ".." in selected:
                    return None
                return PurePosixPath(*selected)
    return None


def _portable_locator(value: str | None) -> str | None:
    if not value:
        return None
    logical_path = _logical_path_from_value(value)
    if logical_path is not None:
        encoded = "/".join(quote(part, safe="-._~") for part in logical_path.parts)
        return f"legacy://{encoded}"
    if any(pattern.search(value) for pattern in MACHINE_PATH_PATTERNS):
        return "redacted://machine-path"
    portable = PurePosixPath(value.replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        return "redacted://nonportable-path"
    if "/" in value or "\\" in value:
        # A slash in prose (for example, "orientation/scale") is not a path.
        # Known logical roots and absolute machine paths were handled above.
        if any(character.isspace() for character in value):
            return value
        encoded = "/".join(quote(part, safe="-._~") for part in portable.parts)
        return f"legacy-relative://{encoded}"
    return value


def _is_locator_field(key: str | None) -> bool:
    if key is None:
        return False
    lowered = key.casefold()
    return lowered in {"dir", "directory", "file", "locator", "path", "root", "uri", "url"} or (
        lowered.endswith(LOCATOR_KEY_SUFFIXES)
    )


def _sanitize(value: object, *, field: str | None = None) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        portable = PurePosixPath(value.replace("\\", "/"))
        if (
            _logical_path_from_value(value) is not None
            or any(pattern.search(value) for pattern in MACHINE_PATH_PATTERNS)
            or portable.is_absolute()
            or ".." in portable.parts
            or _is_locator_field(field)
        ):
            return _portable_locator(value)
        return value
    if isinstance(value, list):
        return [_sanitize(item, field=field) for item in value]
    if isinstance(value, dict):
        sanitized: JsonObject = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered_key = key.lower()
            if lowered_key in SENSITIVE_KEYS:
                sanitized[f"{lowered_key}_redacted"] = True
            elif lowered_key in FREEFORM_KEYS:
                sanitized[f"{lowered_key}_present"] = bool(child)
            else:
                sanitized[key] = _sanitize(child, field=key)
        return sanitized
    return str(value)


def _parse_json_value(value: object) -> object:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise LegacyExportError("A legacy JSON database field has an invalid type.")
    try:
        return _sanitize(json.loads(value))
    except json.JSONDecodeError as error:
        raise LegacyExportError("A legacy JSON database field is invalid.") from error


def _safe_token(value: object, *, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", text):
        return text
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"redacted-{digest}"


def _legacy_label(value: object, *, fallback: str) -> str:
    """Preserve the bounded historical vocabulary, including its spaced label."""
    if value is None:
        return fallback
    if not isinstance(value, str) or value not in LEGACY_LABELS:
        raise LegacyExportError("A legacy label is outside the audited vocabulary.")
    return value


def _normalized_run_timestamp(value: object, *, fallback: str) -> str:
    """Keep useful training-run chronology while rejecting arbitrary text."""
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise LegacyExportError("A legacy run timestamp has an invalid type.")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError as error:
        raise LegacyExportError("A legacy run timestamp is invalid.") from error


def _build_result_directory_index(legacy_root: Path) -> dict[str, tuple[PurePosixPath, ...]]:
    indexed: defaultdict[str, list[PurePosixPath]] = defaultdict(list)
    for logical_root in ("data/results", "data/results_old"):
        root = _safe_source(legacy_root, logical_root)
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise LegacyExportError("A legacy results tree contains a symbolic link.")
            if candidate.is_dir():
                indexed[candidate.name].append(
                    PurePosixPath(candidate.relative_to(legacy_root).as_posix())
                )
    return {name: tuple(sorted(paths, key=str)) for name, paths in indexed.items()}


def _locator_for_logical(path: PurePosixPath) -> str:
    return "legacy://" + "/".join(quote(part, safe="-._~") for part in path.parts)


def _artifact_entry(
    legacy_root: Path,
    raw_value: object,
    *,
    discovered_directories: Sequence[PurePosixPath],
    directory_role: bool = False,
) -> JsonObject:
    if raw_value in (None, ""):
        return {
            "registered_locator": None,
            "resolved_locator": None,
            "availability": "not-recorded",
        }
    if not isinstance(raw_value, str):
        raise LegacyExportError("A legacy artifact locator has an invalid type.")
    registered_path = _logical_path_from_value(raw_value)
    registered_locator = _portable_locator(raw_value)
    candidates: list[PurePosixPath] = []
    if (
        registered_path is not None
        and _safe_source(legacy_root, registered_path.as_posix()).exists()
    ):
        candidates.append(registered_path)
    basename = (
        PureWindowsPath(raw_value).name if "\\" in raw_value else PurePosixPath(raw_value).name
    )
    for directory in discovered_directories:
        candidate = directory if directory_role else directory / basename
        if _safe_source(legacy_root, candidate.as_posix()).exists():
            candidates.append(candidate)
    unique_candidates = tuple(sorted(set(candidates), key=str))
    if len(unique_candidates) == 1:
        return {
            "registered_locator": registered_locator,
            "resolved_locator": _locator_for_logical(unique_candidates[0]),
            "availability": "available",
        }
    if len(unique_candidates) > 1:
        return {
            "registered_locator": registered_locator,
            "resolved_locator": None,
            "availability": "ambiguous",
        }
    return {
        "registered_locator": registered_locator,
        "resolved_locator": None,
        "availability": "missing",
    }


def _run_validity(
    status: str,
    model_key: str,
    *,
    notes_present: bool,
    error_present: bool,
) -> JsonObject:
    notes = [
        "development-only",
        "sample-level-split-without-signer-or-session-grouping",
        "test-split-repeatedly-inspected",
        "no-learned-negative-class",
    ]
    if model_key == "mamba":
        notes.append("misleading-legacy-mamba-name")
    if status == "failed":
        notes.append("failed-run")
    elif status == "running":
        notes.append("stale-running-run")
    return {
        "data_role": DATA_ROLE,
        "eligible_for_locked_test": False,
        "notes": notes,
        "legacy_notes_present": notes_present,
        "legacy_error_present": error_present,
    }


def _source_run_id(run_name: str, global_settings: object) -> str:
    """Recover the cross-store run identity used by sweeps and live evaluation."""
    if isinstance(global_settings, dict):
        sweep_id = global_settings.get("sweep_id")
        if sweep_id is not None:
            return f"{_safe_token(sweep_id, fallback='unknown-sweep')}_{run_name}"
    return run_name


def _build_run_records(legacy_root: Path) -> list[JsonObject]:
    database = _safe_source(legacy_root, "data/models/runs.db")
    result_index = _build_result_directory_index(legacy_root)
    connection = _open_immutable_database(database)
    try:
        rows = connection.execute("SELECT * FROM runs ORDER BY id").fetchall()
    finally:
        connection.close()
    records: list[JsonObject] = []
    for ordinal, row in enumerate(rows, start=1):
        run_name = _safe_token(row["run_name"], fallback=f"unnamed-{ordinal:06d}")
        status = _safe_token(row["status"], fallback="unknown")
        model_key = _safe_token(row["model_key"], fallback="unknown")
        discovered_directories = result_index.get(run_name, ())
        model_settings = _parse_json_value(row["model_settings_json"])
        global_settings = _parse_json_value(row["global_settings_json"])
        artifacts: JsonObject = {
            "directory": _artifact_entry(
                legacy_root,
                row["artifacts_dir"],
                discovered_directories=discovered_directories,
                directory_role=True,
            )
        }
        for role, column in ARTIFACT_COLUMNS.items():
            artifacts[role] = _artifact_entry(
                legacy_root,
                row[column],
                discovered_directories=discovered_directories,
            )
        configuration: JsonObject = {
            "model_settings": model_settings,
            "global_settings": global_settings,
            "data_locator": _portable_locator(cast(str | None, row["data_root"])),
            "sequence_length": row["seq_len"],
            "test_fraction": row["test_frac"],
            "validation_fraction": row["val_frac"],
            "batch_size": row["batch_size"],
            "epochs": row["epochs"],
        }
        metrics: JsonObject = {
            "quick": _parse_json_value(row["quick_metrics_json"]),
            "test_loss": row["test_loss"],
            "test_accuracy": row["test_accuracy"],
            "samples_train": row["samples_train"],
            "samples_validation": row["samples_val"],
            "samples_test": row["samples_test"],
        }
        records.append(
            {
                "record_id": f"run-{ordinal:06d}",
                "run_name": run_name,
                "source_run_id": _source_run_id(run_name, global_settings),
                "started_at": _normalized_run_timestamp(row["started_at"], fallback="unknown"),
                "finished_at": (
                    _normalized_run_timestamp(row["finished_at"], fallback="unknown")
                    if row["finished_at"] is not None
                    else None
                ),
                "status": status,
                "model_key": model_key,
                "configuration": configuration,
                "metrics": metrics,
                "artifacts": artifacts,
                "validity": _run_validity(
                    status,
                    model_key,
                    notes_present=bool(row["notes"]),
                    error_present=bool(row["error"]),
                ),
            }
        )
    return records


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LegacyExportError("A legacy timestamp is invalid.") from error


def _offset_ms(value: object, origin: datetime) -> int:
    timestamp = _parse_time(value)
    if timestamp is None:
        return 0
    return max(0, int((timestamp - origin).total_seconds() * 1000))


def _duration_ms(start: object, end: object) -> int | None:
    start_time = _parse_time(start)
    end_time = _parse_time(end)
    if start_time is None or end_time is None:
        return None
    return max(0, int((end_time - start_time).total_seconds() * 1000))


class _ObjectStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._entries: dict[str, JsonObject] = {}

    def add(self, source: Path, *, role: str) -> str:
        if source.is_symlink() or not source.is_file():
            raise LegacyExportError("A quarantined source object is unavailable.")
        suffix = source.suffix.lower()
        if suffix not in SAFE_OBJECT_SUFFIXES:
            raise LegacyExportError("A quarantined source object has an unsupported type.")
        before = source.stat()
        digest = _sha256(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise LegacyExportError("A legacy source object changed during export.")
        relative = PurePosixPath("objects", "sha256", f"{digest}{suffix}")
        destination = self._root / Path(relative.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        if destination.stat().st_size != before.st_size or _sha256(destination) != digest:
            raise LegacyExportError("A quarantined object failed its integrity check.")
        uri = f"quarantine://sha256/{digest}{suffix}"
        entry = self._entries.setdefault(
            uri,
            {
                "uri": uri,
                "storage_key": relative.as_posix(),
                "bytes": before.st_size,
                "sha256": digest,
                "roles": [],
            },
        )
        roles = cast(list[str], entry["roles"])
        if role not in roles:
            roles.append(role)
            roles.sort()
        return uri

    def entries(self) -> list[JsonObject]:
        return [self._entries[key] for key in sorted(self._entries)]


def _all_segment_files(legacy_root: Path) -> list[Path]:
    root = _safe_source(legacy_root, "data/live_eval/segments")
    if not root.is_dir():
        raise LegacyExportError("The legacy live-segment directory is missing.")
    segments: list[Path] = []
    for candidate in sorted(root.rglob("*.npz"), key=lambda item: item.as_posix()):
        if candidate.is_symlink() or not candidate.is_file():
            raise LegacyExportError("The legacy segment tree is not a regular-file tree.")
        segments.append(candidate)
    return segments


def _resolve_segment_path(legacy_root: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise LegacyExportError("A legacy segment locator has an invalid type.")
    logical = _logical_path_from_value(value)
    if logical is None:
        raise LegacyExportError("A legacy segment locator cannot be sanitized.")
    source = _safe_source(legacy_root, logical.as_posix())
    if not source.is_file():
        raise LegacyExportError("A referenced legacy segment is missing.")
    return source


def _build_live_records(
    legacy_root: Path,
    store: _ObjectStore,
) -> tuple[list[JsonObject], list[JsonObject], list[JsonObject], list[JsonObject], dict[str, int]]:
    evaluation = _open_immutable_database(_safe_source(legacy_root, "data/live_eval/live_eval.db"))
    feedback = _open_immutable_database(_safe_source(legacy_root, "data/models/live_feedback.db"))
    try:
        attempt_rows = evaluation.execute(
            "SELECT * FROM attempts ORDER BY timestamp_start, attempt_id"
        ).fetchall()
        annotation_rows = feedback.execute(
            "SELECT * FROM feedback ORDER BY timestamp, feedback_id"
        ).fetchall()
        session_rows = feedback.execute("SELECT * FROM sessions ORDER BY started_at, id").fetchall()
        detection_rows = feedback.execute("SELECT * FROM detections ORDER BY ts, id").fetchall()
    finally:
        evaluation.close()
        feedback.close()

    attempt_times = [_parse_time(row["timestamp_start"]) for row in attempt_rows]
    attempt_origin = min(time for time in attempt_times if time is not None)
    attempt_aliases = {
        cast(str, row["attempt_id"]): f"attempt-{index:06d}"
        for index, row in enumerate(attempt_rows, start=1)
    }
    referenced_segments: set[Path] = set()
    attempts: list[JsonObject] = []
    for row in attempt_rows:
        raw_id = cast(str, row["attempt_id"])
        segment_source = _resolve_segment_path(legacy_root, row["segment_path"])
        segment_uri = None
        if segment_source is not None:
            referenced_segments.add(segment_source)
            segment_uri = store.add(segment_source, role="live-attempt-segment")
        predicted_label = _legacy_label(row["predicted_label"], fallback="NONE")
        model_run_id = _safe_token(row["model_run_id"], fallback="unknown")
        attempts.append(
            {
                "record_id": attempt_aliases[raw_id],
                "data_role": DATA_ROLE,
                "start_offset_ms": _offset_ms(row["timestamp_start"], attempt_origin),
                "duration_ms": _duration_ms(row["timestamp_start"], row["timestamp_end"]),
                "detected": bool(row["detected"]),
                "intended_label": (
                    _legacy_label(row["intended_label"], fallback="unknown")
                    if row["intended_label"] is not None
                    else None
                ),
                "predicted_label": predicted_label,
                "correct": bool(row["correct"]) if row["correct"] is not None else None,
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
                "latency_ms": int(row["latency_ms"]) if row["latency_ms"] is not None else None,
                "segment_frames": (
                    int(row["segment_frames"]) if row["segment_frames"] is not None else None
                ),
                "model_run_id": model_run_id,
                "segmentation": _parse_json_value(row["seg_params_json"]) or {},
                "segment_uri": segment_uri,
            }
        )

    annotations: list[JsonObject] = []
    for index, row in enumerate(annotation_rows, start=1):
        raw_attempt_id = cast(str, row["attempt_id"])
        if raw_attempt_id not in attempt_aliases:
            raise LegacyExportError("A feedback annotation references an unknown attempt.")
        annotations.append(
            {
                "record_id": f"annotation-{index:06d}",
                "attempt_ref": attempt_aliases[raw_attempt_id],
                "data_role": DATA_ROLE,
                # Use the attempt origin so feedback latency remains recoverable.
                "offset_ms": _offset_ms(row["timestamp"], attempt_origin),
                "feedback_type": _safe_token(row["feedback_type"], fallback="unknown"),
                "corrected_label": (
                    _legacy_label(row["corrected_label"], fallback="unknown")
                    if row["corrected_label"] is not None
                    else None
                ),
                "freeform_note_present": bool(row["freeform_note"]),
            }
        )

    session_times = [_parse_time(row["started_at"]) for row in session_rows]
    session_origin = min(time for time in session_times if time is not None)
    session_aliases = {
        int(row["id"]): f"session-{index:04d}" for index, row in enumerate(session_rows, start=1)
    }
    sessions: list[JsonObject] = []
    for row in session_rows:
        sessions.append(
            {
                "record_id": session_aliases[int(row["id"])],
                "data_role": DATA_ROLE,
                "model_run_id": _safe_token(row["model_name"], fallback="unknown"),
                "start_offset_ms": _offset_ms(row["started_at"], session_origin),
                "duration_ms": _duration_ms(row["started_at"], row["ended_at"]),
            }
        )

    detections: list[JsonObject] = []
    for index, row in enumerate(detection_rows, start=1):
        session_id = int(row["session_id"])
        if session_id not in session_aliases:
            raise LegacyExportError("A legacy detection references an unknown session.")
        detections.append(
            {
                "record_id": f"detection-{index:06d}",
                "session_ref": session_aliases[session_id],
                "data_role": DATA_ROLE,
                "offset_ms": _offset_ms(row["ts"], session_origin),
                "predicted_label": (
                    _legacy_label(row["predicted_label"], fallback="unknown")
                    if row["predicted_label"] is not None
                    else None
                ),
                "actual_label": (
                    _legacy_label(row["actual_label"], fallback="unknown")
                    if row["actual_label"] is not None
                    else None
                ),
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
                "correct": bool(row["is_correct"]) if row["is_correct"] is not None else None,
            }
        )

    all_segments = _all_segment_files(legacy_root)
    for source in all_segments:
        role = "live-attempt-segment" if source in referenced_segments else "orphan-live-segment"
        store.add(source, role=role)
    counts = {
        "attempts": len(attempts),
        "annotations": len(annotations),
        "annotated_attempts": len({record["attempt_ref"] for record in annotations}),
        "unannotated_attempts": len(attempts)
        - len({record["attempt_ref"] for record in annotations}),
        "sessions": len(sessions),
        "detections": len(detections),
        "segments": len(all_segments),
        "referenced_segments": len(referenced_segments),
        "orphan_segments": len(all_segments) - len(referenced_segments),
    }
    return attempts, annotations, sessions, detections, counts


def _build_promoted_artifacts(
    legacy_root: Path,
    audit: JsonObject,
    store: _ObjectStore,
) -> list[JsonObject]:
    dataset = _expected_mapping(audit.get("dataset_snapshot"), "dataset snapshot")
    selected_plan = _safe_token(dataset.get("selected_representation"), fallback="unknown")
    promoted: list[JsonObject] = []
    for raw_run in _expected_sequence(audit.get("reported_runs"), "reported runs"):
        run = _expected_mapping(raw_run, "reported run")
        run_id = _safe_token(run.get("run_id"), fallback="unknown")
        model_key = _safe_token(run.get("model_key"), fallback="unknown")
        model_path = run.get("model_logical_path")
        if not isinstance(model_path, str):
            raise LegacyExportError("A promoted model locator is invalid.")
        model_source = _safe_source(legacy_root, model_path)
        model_uri = store.add(model_source, role=f"promoted-model:{model_key}")
        label_evidence = next(
            (
                _expected_mapping(item, "label-map evidence")
                for item in _expected_sequence(run.get("source_evidence"), "source evidence")
                if isinstance(item, dict)
                and str(cast(dict[str, object], item).get("logical_path", "")).endswith(
                    "/label_map.json"
                )
            ),
            None,
        )
        if label_evidence is None or not isinstance(label_evidence.get("logical_path"), str):
            raise LegacyExportError("A promoted run has no label-map evidence.")
        label_path = cast(str, label_evidence["logical_path"])
        label_source = _safe_source(legacy_root, label_path)
        label_uri = store.add(label_source, role=f"promoted-label-map:{model_key}")
        label_map = _read_json_object(label_source)
        promoted.append(
            {
                "run_id": run_id,
                "model_key": model_key,
                "preprocessing_plan": selected_plan,
                "model": {
                    "bytes": run["model_bytes"],
                    "sha256": run["model_sha256"],
                    "quarantine_uri": model_uri,
                    "storage_status": "local-quarantine",
                },
                "label_map": {
                    "bytes": label_evidence["bytes"],
                    "sha256": label_evidence["sha256"],
                    "quarantine_uri": label_uri,
                    "labels": _sanitize(label_map),
                },
                "validity": {
                    "data_role": DATA_ROLE,
                    "eligible_for_locked_test": False,
                    "notes": [
                        "sample-level-split-without-signer-or-session-grouping",
                        "test-split-repeatedly-inspected",
                        "parity-candidate-not-champion",
                    ]
                    + (["misleading-legacy-mamba-name"] if model_key == "mamba" else []),
                },
            }
        )
    return promoted


def _write_schemas(public_root: Path) -> list[JsonObject]:
    components: list[JsonObject] = []
    for name, schema in sorted(SCHEMAS.items()):
        path = public_root / "schemas" / name
        _write_json(path, schema)
        components.append(_component(path, public_root))
    return components


def _build_export(
    legacy_root: Path,
    audit_path: Path,
    public_root: Path,
    quarantine_root: Path,
) -> ExportSummary:
    audit = _read_json_object(audit_path)
    _verify_source(legacy_root, audit)
    store = _ObjectStore(quarantine_root)

    runs = _build_run_records(legacy_root)
    attempts, annotations, sessions, detections, live_counts = _build_live_records(
        legacy_root, store
    )
    promoted = _build_promoted_artifacts(legacy_root, audit, store)
    plans_source = _safe_source(legacy_root, "data/plans/plans.json")
    plans_uri = store.add(plans_source, role="preprocessing-plan-registry")
    plans = _read_json_object(plans_source)

    record_sets: tuple[tuple[str, list[JsonObject]], ...] = (
        ("records/attempts.jsonl", attempts),
        ("records/annotations.jsonl", annotations),
        ("records/sessions.jsonl", sessions),
        ("records/detections.jsonl", detections),
    )
    private_components: list[JsonObject] = []
    for relative, records in record_sets:
        path = quarantine_root / relative
        _write_jsonl(path, records)
        private_components.append(_component(path, quarantine_root, records=len(records)))

    object_entries = store.entries()
    private_counts = {
        **live_counts,
        "promoted_models": len(promoted),
        "quarantine_objects": len(object_entries),
    }
    private_manifest: JsonObject = {
        "schema_version": FORMAT_VERSION,
        "kind": QUARANTINE_KIND,
        "policy": {
            "data_role": DATA_ROLE,
            "eligible_for_locked_test": False,
            "contains_private_artifacts": True,
            "publishable": False,
            "identifiers": "pseudonymized-ordinal",
            "timestamps": "live-records-relative-offsets-only",
        },
        "counts": private_counts,
        "record_components": private_components,
        "objects": object_entries,
    }
    private_manifest_path = quarantine_root / "manifest.json"
    _write_json(private_manifest_path, private_manifest)

    public_components = _write_schemas(public_root)
    for shard_index, start in enumerate(range(0, len(runs), RUN_SHARD_SIZE)):
        shard_records = runs[start : start + RUN_SHARD_SIZE]
        shard_path = public_root / "runs" / f"runs-{shard_index:03d}.json"
        _write_json(
            shard_path,
            {
                "schema_version": FORMAT_VERSION,
                "kind": "signlab.legacy-run-index",
                "shard": shard_index,
                "records": shard_records,
            },
        )
        public_components.append(_component(shard_path, public_root, records=len(shard_records)))

    plans_path = public_root / "preprocessing-plans.json"
    _write_json(
        plans_path,
        {
            "schema_version": FORMAT_VERSION,
            "kind": "signlab.legacy-preprocessing-plans",
            "source_sha256": _sha256(plans_source),
            "quarantine_uri": plans_uri,
            "plans": _sanitize(plans),
        },
    )
    public_components.append(_component(plans_path, public_root, records=len(plans)))

    promoted_path = public_root / "promoted-artifacts.json"
    _write_json(
        promoted_path,
        {
            "schema_version": FORMAT_VERSION,
            "kind": "signlab.legacy-promoted-artifacts",
            "artifacts": promoted,
        },
    )
    public_components.append(_component(promoted_path, public_root, records=len(promoted)))

    receipt_components = [
        _component(private_manifest_path, quarantine_root),
        *private_components,
    ]
    receipt_path = public_root / "quarantine-receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": FORMAT_VERSION,
            "kind": "signlab.legacy-quarantine-receipt",
            "policy": {
                "data_role": DATA_ROLE,
                "eligible_for_locked_test": False,
                "publishable": False,
                "durability": "local-only-pending-private-remote",
                "contains_individual_object_hashes": False,
            },
            "counts": private_counts,
            "components": receipt_components,
        },
    )
    public_components.append(_component(receipt_path, public_root))

    status_counts: defaultdict[str, int] = defaultdict(int)
    for record in runs:
        status_counts[cast(str, record["status"])] += 1
    source = _expected_mapping(audit.get("repository"), "repository")
    public_counts = {
        "runs": len(runs),
        "runs_succeeded": status_counts["succeeded"],
        "runs_failed": status_counts["failed"],
        "runs_running": status_counts["running"],
        "preprocessing_plans": len(plans),
        "promoted_models": len(promoted),
        **live_counts,
    }
    public_manifest: JsonObject = {
        "schema_version": FORMAT_VERSION,
        "kind": PUBLIC_KIND,
        "source": {
            "head_commit": source["head_commit"],
            "tree_object": source["tree_object"],
            "audit_sha256": _sha256(audit_path),
        },
        "policy": {
            "data_role": DATA_ROLE,
            "eligible_for_locked_test": False,
            "contains_private_artifacts": False,
            "durability": "local-only-pending-private-remote",
        },
        "counts": public_counts,
        "components": sorted(public_components, key=lambda item: cast(str, item["path"])),
    }
    _write_json(public_root / "manifest.json", public_manifest)

    _verify_source(legacy_root, audit)
    return ExportSummary(
        runs=len(runs),
        attempts=len(attempts),
        annotations=len(annotations),
        detections=len(detections),
        sessions=len(sessions),
        promoted_models=len(promoted),
        quarantined_segments=live_counts["segments"],
        quarantine_objects=len(object_entries),
    )


def _target_is_nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or next(path.iterdir(), None) is not None)


def _validate_output_boundaries(
    legacy_root: Path,
    public_output: Path,
    quarantine_output: Path,
) -> None:
    resolved_legacy = legacy_root.resolve()
    resolved_public = public_output.resolve()
    resolved_quarantine = quarantine_output.resolve()
    if resolved_public == resolved_quarantine:
        raise LegacyExportError("Public and quarantine outputs must be separate.")
    if resolved_public.is_relative_to(resolved_quarantine) or resolved_quarantine.is_relative_to(
        resolved_public
    ):
        raise LegacyExportError("Public and quarantine outputs must not be nested.")
    if resolved_public.is_relative_to(resolved_legacy) or resolved_quarantine.is_relative_to(
        resolved_legacy
    ):
        raise LegacyExportError("Export outputs must not be inside the legacy source.")
    if _target_is_nonempty(public_output) or _target_is_nonempty(quarantine_output):
        raise LegacyExportError("Export outputs must be absent or empty.")


def export_legacy_evidence(
    *,
    legacy_root: Path,
    audit_snapshot: Path,
    public_output: Path,
    quarantine_output: Path,
) -> ExportSummary:
    """Export public evidence and a private quarantine without mutating the source."""
    if not legacy_root.is_dir() or not audit_snapshot.is_file():
        raise LegacyExportError("The legacy root or audit snapshot is unavailable.")
    _validate_output_boundaries(legacy_root, public_output, quarantine_output)
    public_output.parent.mkdir(parents=True, exist_ok=True)
    quarantine_output.parent.mkdir(parents=True, exist_ok=True)
    public_staging = Path(tempfile.mkdtemp(prefix=".signlab-public-", dir=public_output.parent))
    quarantine_staging = Path(
        tempfile.mkdtemp(prefix=".signlab-private-", dir=quarantine_output.parent)
    )
    moved_public = False
    moved_quarantine = False
    try:
        summary = _build_export(
            legacy_root.resolve(),
            audit_snapshot.resolve(),
            public_staging,
            quarantine_staging,
        )
        from signlab.legacy.validator import validate_legacy_export

        validate_legacy_export(public_root=public_staging, quarantine_root=quarantine_staging)
        if public_output.exists():
            public_output.rmdir()
        if quarantine_output.exists():
            quarantine_output.rmdir()
        quarantine_staging.replace(quarantine_output)
        moved_quarantine = True
        public_staging.replace(public_output)
        moved_public = True
        return summary
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
        raise LegacyExportError("The legacy export could not complete safely.") from error
    finally:
        if public_staging.exists():
            shutil.rmtree(public_staging)
        if quarantine_staging.exists():
            shutil.rmtree(quarantine_staging)
        if moved_quarantine and not moved_public and quarantine_output.exists():
            shutil.rmtree(quarantine_output)
