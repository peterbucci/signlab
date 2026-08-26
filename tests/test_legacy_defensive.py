"""Defensive-path tests for legacy export trust boundaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from signlab.legacy import exporter, validator
from signlab.legacy.exporter import LegacyExportError


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("data/results/run/model.keras", "legacy://data/results/run/model.keras"),
        ("relative/folder", "legacy-relative://relative/folder"),
        (
            "orientation/scale is intentionally plain prose",
            "orientation/scale is intentionally plain prose",
        ),
        ("plain-token", "plain-token"),
        ("/opt/private/model.keras", "redacted://nonportable-path"),
        ("../private/model.keras", "redacted://nonportable-path"),
    ],
)
def test_portable_locator_classifies_historical_values(
    value: str | None,
    expected: str | None,
) -> None:
    assert exporter._portable_locator(value) == expected


def test_portable_locator_recognizes_windows_paths_without_leaking_them() -> None:
    known = "D" + ":" + r"\archive\data\results\run\model.keras"
    private = "D" + ":" + r"\private\model.keras"

    assert exporter._portable_locator(known) == "legacy://data/results/run/model.keras"
    assert exporter._portable_locator(private) == "redacted://machine-path"


def test_sanitizer_redacts_sensitive_and_freeform_values_recursively() -> None:
    class Marker:
        def __str__(self) -> str:
            return "marker"

    source = {
        "participant_id": "private-person",
        "notes": "operator prose",
        "nested": ["data/results/run", {"email": "private@example.invalid"}],
        "description": "input/output",
        "data_path": "relative/folder",
        "marker": Marker(),
        "enabled": True,
    }

    assert exporter._sanitize(source) == {
        "participant_id_redacted": True,
        "notes_present": True,
        "nested": ["legacy://data/results/run", {"email_redacted": True}],
        "description": "input/output",
        "data_path": "legacy-relative://relative/folder",
        "marker": "marker",
        "enabled": True,
    }


def test_json_field_parser_handles_empty_and_sanitized_values() -> None:
    assert exporter._parse_json_value(None) is None
    assert exporter._parse_json_value("") is None
    assert exporter._parse_json_value('{"error": "details"}') == {"error_present": True}


@pytest.mark.parametrize("value", [3, [], object()])
def test_json_field_parser_rejects_non_strings(value: object) -> None:
    with pytest.raises(LegacyExportError, match="invalid type"):
        exporter._parse_json_value(value)


def test_json_field_parser_rejects_malformed_json() -> None:
    with pytest.raises(LegacyExportError, match="field is invalid"):
        exporter._parse_json_value("{")


def test_safe_tokens_and_historical_labels_have_bounded_behavior() -> None:
    assert exporter._safe_token(None, fallback="fallback") == "fallback"
    assert exporter._safe_token("safe-token", fallback="fallback") == "safe-token"
    assert exporter._safe_token("unsafe token", fallback="fallback").startswith("redacted-")
    assert exporter._legacy_label(None, fallback="NONE") == "NONE"
    assert exporter._legacy_label("thank you", fallback="NONE") == "thank you"


@pytest.mark.parametrize("value", [7, "not-a-legacy-label"])
def test_historical_label_rejects_values_outside_the_audited_vocabulary(value: object) -> None:
    with pytest.raises(LegacyExportError, match="outside the audited vocabulary"):
        exporter._legacy_label(value, fallback="NONE")


def test_timestamp_helpers_normalize_and_bound_relative_time() -> None:
    origin = datetime.fromisoformat("2025-01-01T10:00:00")

    assert exporter._normalized_run_timestamp(None, fallback="unknown") == "unknown"
    assert exporter._normalized_run_timestamp("2025-01-01T10:00:00Z", fallback="unknown") == (
        "2025-01-01T10:00:00+00:00"
    )
    assert exporter._parse_time(None) is None
    assert exporter._parse_time("") is None
    assert exporter._offset_ms(None, origin) == 0
    assert exporter._offset_ms("2025-01-01T09:59:00", origin) == 0
    assert exporter._offset_ms("2025-01-01T10:00:01", origin) == 1000
    assert exporter._duration_ms(None, None) is None
    assert exporter._duration_ms("2025-01-01T10:00:02", "2025-01-01T10:00:01") == 0
    assert exporter._duration_ms("2025-01-01T10:00:00", "2025-01-01T10:00:01") == 1000


@pytest.mark.parametrize("value", [7, "not-a-timestamp"])
def test_run_timestamp_normalizer_rejects_invalid_values(value: object) -> None:
    with pytest.raises(LegacyExportError, match="run timestamp"):
        exporter._normalized_run_timestamp(value, fallback="unknown")


def test_relative_timestamp_parser_rejects_invalid_values() -> None:
    with pytest.raises(LegacyExportError, match="timestamp is invalid"):
        exporter._parse_time("not-a-timestamp")


def test_safe_source_accepts_only_descendants(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    source = root / "data" / "file.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")

    assert exporter._safe_source(root, "data/file.json") == source.resolve()
    with pytest.raises(LegacyExportError, match="not portable"):
        exporter._safe_source(root, "../outside.json")
    with pytest.raises(LegacyExportError, match="not portable"):
        exporter._safe_source(root, "/outside.json")


def test_immutable_database_rejects_an_active_wal(tmp_path: Path) -> None:
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence (id INTEGER)")
    database.with_name("source.db-wal").write_bytes(b"active")

    with pytest.raises(LegacyExportError, match="active WAL"):
        exporter._open_immutable_database(database)


def test_expected_container_helpers_reject_wrong_shapes() -> None:
    assert exporter._expected_mapping({"key": "value"}, "fixture") == {"key": "value"}
    assert exporter._expected_sequence([1], "fixture") == [1]
    with pytest.raises(LegacyExportError, match="invalid fixture section"):
        exporter._expected_mapping([], "fixture")
    with pytest.raises(LegacyExportError, match="invalid fixture section"):
        exporter._expected_sequence({}, "fixture")


def test_artifact_entry_distinguishes_available_ambiguous_missing_and_directory(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    first = legacy / "data" / "results" / "run"
    second = legacy / "data" / "results_old" / "run"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "model.keras").write_bytes(b"first")
    (second / "model.keras").write_bytes(b"second")
    discovered = (PurePosixPath("data/results/run"), PurePosixPath("data/results_old/run"))

    assert exporter._artifact_entry(legacy, None, discovered_directories=discovered) == {
        "registered_locator": None,
        "resolved_locator": None,
        "availability": "not-recorded",
    }
    with pytest.raises(LegacyExportError, match="invalid type"):
        exporter._artifact_entry(legacy, 3, discovered_directories=discovered)
    assert (
        exporter._artifact_entry(legacy, "model.keras", discovered_directories=discovered)[
            "availability"
        ]
        == "ambiguous"
    )
    assert (
        exporter._artifact_entry(
            legacy,
            "missing.keras",
            discovered_directories=discovered,
        )["availability"]
        == "missing"
    )
    directory = exporter._artifact_entry(
        legacy,
        "retired-location/run",
        discovered_directories=(PurePosixPath("data/results/run"),),
        directory_role=True,
    )
    assert directory["availability"] == "available"
    assert directory["resolved_locator"] == "legacy://data/results/run"


def test_run_validity_records_known_historical_limitations() -> None:
    succeeded = exporter._run_validity(
        "succeeded",
        "causal_gru",
        notes_present=False,
        error_present=False,
    )
    failed = exporter._run_validity(
        "failed",
        "mamba",
        notes_present=True,
        error_present=True,
    )
    running = exporter._run_validity(
        "running",
        "causal_tcn",
        notes_present=False,
        error_present=False,
    )

    assert "failed-run" in cast(list[str], failed["notes"])
    assert "misleading-legacy-mamba-name" in cast(list[str], failed["notes"])
    assert "stale-running-run" in cast(list[str], running["notes"])
    assert "failed-run" not in cast(list[str], succeeded["notes"])


def test_content_addressed_store_deduplicates_objects_and_roles(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    store = exporter._ObjectStore(tmp_path / "store")

    first_uri = store.add(source, role="z-role")
    second_uri = store.add(source, role="a-role")

    assert first_uri == second_uri
    assert store.entries()[0]["roles"] == ["a-role", "z-role"]
    with pytest.raises(LegacyExportError, match="unavailable"):
        store.add(tmp_path / "missing.json", role="missing")
    unsupported = tmp_path / "source.txt"
    unsupported.write_text("private", encoding="utf-8")
    with pytest.raises(LegacyExportError, match="unsupported type"):
        store.add(unsupported, role="unsupported")


def test_segment_resolution_rejects_unportable_or_missing_inputs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    segments = legacy / "data" / "live_eval" / "segments"
    segments.mkdir(parents=True)
    existing = segments / "event.npz"
    existing.write_bytes(b"event")

    assert exporter._resolve_segment_path(legacy, None) is None
    assert exporter._resolve_segment_path(legacy, "") is None
    assert (
        exporter._resolve_segment_path(
            legacy,
            "data/live_eval/segments/event.npz",
        )
        == existing.resolve()
    )
    with pytest.raises(LegacyExportError, match="invalid type"):
        exporter._resolve_segment_path(legacy, 7)
    with pytest.raises(LegacyExportError, match="cannot be sanitized"):
        exporter._resolve_segment_path(legacy, "unknown/location.npz")
    with pytest.raises(LegacyExportError, match="referenced legacy segment is missing"):
        exporter._resolve_segment_path(legacy, "data/live_eval/segments/missing.npz")


def test_segment_inventory_requires_a_directory(tmp_path: Path) -> None:
    with pytest.raises(LegacyExportError, match="segment directory is missing"):
        exporter._all_segment_files(tmp_path)


def test_target_nonempty_classification(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()
    file_path = tmp_path / "file"
    file_path.write_text("content", encoding="utf-8")
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "entry").write_text("content", encoding="utf-8")

    assert exporter._target_is_nonempty(missing) is False
    assert exporter._target_is_nonempty(empty) is False
    assert exporter._target_is_nonempty(file_path) is True
    assert exporter._target_is_nonempty(populated) is True


def test_json_readers_reject_malformed_or_non_object_values(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        exporter._read_json_object(malformed)
    with pytest.raises(LegacyExportError, match="not an object"):
        exporter._read_json_object(array)
    with pytest.raises(LegacyExportError, match="component is unreadable"):
        validator._read_object(malformed)
    with pytest.raises(LegacyExportError, match="not an object"):
        validator._read_object(array)


def test_validator_shape_helpers_reject_wrong_types() -> None:
    assert validator._object({"key": 1}, "fixture") == {"key": 1}
    assert validator._array([1], "fixture") == [1]
    validator._exact_keys({"key": 1}, {"key"}, "fixture")
    with pytest.raises(LegacyExportError, match="invalid fixture object"):
        validator._object([], "fixture")
    with pytest.raises(LegacyExportError, match="invalid fixture array"):
        validator._array({}, "fixture")
    with pytest.raises(LegacyExportError, match="unsupported fixture fields"):
        validator._exact_keys({"extra": 1}, {"key"}, "fixture")


def test_validator_rejects_unsafe_component_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(LegacyExportError, match="not a string"):
        validator._safe_relative(root, 7)
    with pytest.raises(LegacyExportError, match="not portable"):
        validator._safe_relative(root, "../escape.json")
    with pytest.raises(LegacyExportError, match="not portable"):
        validator._safe_relative(root, "/escape.json")


def test_component_validator_checks_shape_identity_and_record_count(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    component = root / "component.json"
    component.write_text("{}\n", encoding="utf-8")
    valid: dict[str, object] = {
        "path": "component.json",
        "bytes": component.stat().st_size,
        "sha256": exporter._sha256(component),
        "records": 1,
    }
    assert validator._validate_component(root, valid) == ("component.json", 1)

    cases = [
        ({**valid, "extra": True}, "unsupported fields"),
        ({**valid, "bytes": -1}, "byte count"),
        ({**valid, "sha256": "invalid"}, "digest"),
        ({**valid, "path": "missing.json"}, "component is missing"),
        ({**valid, "bytes": 999}, "integrity check"),
        ({**valid, "records": -1}, "record count"),
    ]
    for raw, message in cases:
        with pytest.raises(LegacyExportError, match=message):
            validator._validate_component(root, raw)


@pytest.mark.parametrize(
    ("value", "reject_timestamps", "message"),
    [
        ("D" + ":" + r"\private\file.json", False, "absolute machine path"),
        ("/" + "home/private/file.json", False, "absolute machine path"),
        (r"\\server\share\file.json", False, "absolute machine path"),
        ("123e4567e89b12d3" + "a456426614174000", False, "raw UUID-like"),
        ("2025-01-01T10:00:00", True, "absolute timestamp"),
    ],
)
def test_portability_scanner_rejects_private_values(
    tmp_path: Path,
    value: str,
    reject_timestamps: bool,
    message: str,
) -> None:
    path = tmp_path / "component.json"
    path.write_text(value, encoding="utf-8")

    with pytest.raises(LegacyExportError, match=message):
        validator._scan_portability(path, reject_timestamps=reject_timestamps)


def test_schema_closure_and_instance_validation_reject_open_or_invalid_contracts() -> None:
    assert validator._schema_is_closed({"type": "array", "items": {"type": "string"}})
    assert not validator._schema_is_closed(
        {"type": "object", "properties": {"key": {"type": "string"}}}
    )
    assert validator._schema_is_closed(
        {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "additionalProperties": False,
        }
    )
    with pytest.raises(LegacyExportError, match="invalid live attempt contract"):
        validator._validate_schema_instance({}, "live-attempt.schema.json", "live attempt")


def test_json_lines_reader_rejects_malformed_and_non_object_records(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{\n", encoding="utf-8")
    non_object = tmp_path / "array.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")

    with pytest.raises(LegacyExportError, match="component is unreadable"):
        validator._read_jsonl(malformed)
    with pytest.raises(LegacyExportError, match="record is not an object"):
        validator._read_jsonl(non_object)
