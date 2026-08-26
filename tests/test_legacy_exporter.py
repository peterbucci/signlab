"""End-to-end tests for the read-only legacy exporter."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from signlab.legacy.exporter import (
    SENSITIVE_KEYS,
    ExportSummary,
    LegacyExportError,
    export_legacy_evidence,
)
from signlab.legacy.validator import ValidationSummary, validate_legacy_export

type JsonObject = dict[str, object]

SENSITIVE_KEY_FORMS = tuple(
    sorted(
        SENSITIVE_KEYS
        | {variant for key in SENSITIVE_KEYS for variant in (key.upper(), key.title())}
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonObject:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(JsonObject, value)


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[JsonObject]:
    values: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: Any = json.loads(line)
        assert isinstance(value, dict)
        values.append(cast(JsonObject, value))
    return values


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    return result.stdout.strip()


def _create_run_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = """
        CREATE TABLE runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_name TEXT,
          started_at TEXT NOT NULL,
          finished_at TEXT,
          status TEXT NOT NULL,
          model_key TEXT NOT NULL,
          model_settings_json TEXT NOT NULL,
          global_settings_json TEXT NOT NULL,
          data_root TEXT NOT NULL,
          seq_len INTEGER NOT NULL,
          test_frac REAL NOT NULL,
          val_frac REAL NOT NULL,
          batch_size INTEGER NOT NULL,
          epochs INTEGER NOT NULL,
          artifacts_dir TEXT NOT NULL,
          model_path TEXT,
          label_map_path TEXT,
          config_path TEXT,
          history_path TEXT,
          predictions_path TEXT,
          metrics_path TEXT,
          quick_metrics_json TEXT,
          test_loss REAL,
          test_accuracy REAL,
          samples_train INTEGER,
          samples_val INTEGER,
          samples_test INTEGER,
          error TEXT,
          notes TEXT
        )
    """
    columns = (
        "run_name, started_at, finished_at, status, model_key, "
        "model_settings_json, global_settings_json, data_root, seq_len, "
        "test_frac, val_frac, batch_size, epochs, artifacts_dir, model_path, "
        "label_map_path, config_path, history_path, predictions_path, metrics_path, "
        "quick_metrics_json, test_loss, test_accuracy, samples_train, samples_val, "
        "samples_test, error, notes"
    )
    placeholder = ", ".join("?" for _ in range(28))
    stale_root = "Z" + ":" + r"\retired-machine\archive\run-relocated"
    rows: list[tuple[Any, ...]] = [
        (
            "run-good",
            "2025-01-01T10:00:00",
            "2025-01-01T10:01:00",
            "succeeded",
            "causal_gru",
            json.dumps({"hidden": 32, "participant_id": "private"}),
            json.dumps({"notes": "reviewed", "data_root": "data/landmarks/v1"}),
            "data/landmarks/v1",
            30,
            0.2,
            0.2,
            8,
            2,
            "data/results/run-good",
            "data/results/run-good/model.keras",
            "data/results/run-good/label_map.json",
            "data/results/run-good/config.json",
            None,
            None,
            None,
            json.dumps({"accuracy": 0.8}),
            0.4,
            0.8,
            8,
            2,
            2,
            None,
            "reviewed",
        ),
        (
            "run-relocated",
            "2025-01-02T10:00:00",
            "2025-01-02T10:01:00",
            "failed",
            "mamba",
            "{}",
            "{}",
            "data/landmarks/v1",
            30,
            0.2,
            0.2,
            8,
            2,
            stale_root,
            stale_root + r"\sign_model_best.keras",
            stale_root + r"\label_map.json",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "training stopped",
            None,
        ),
        (
            "run-shared",
            "2025-01-03T10:00:00",
            None,
            "running",
            "causal_tcn",
            "{}",
            "{}",
            "data/landmarks/v1",
            30,
            0.2,
            0.2,
            8,
            2,
            "run-shared",
            "model.keras",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ]
    with sqlite3.connect(path) as connection:
        connection.execute(schema)
        connection.executemany(f"INSERT INTO runs ({columns}) VALUES ({placeholder})", rows)


def _create_live_evaluation_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE attempts (
              attempt_id TEXT PRIMARY KEY,
              timestamp_start TEXT NOT NULL,
              timestamp_end TEXT NOT NULL,
              intended_label TEXT,
              detected INTEGER NOT NULL,
              predicted_label TEXT NOT NULL,
              correct INTEGER,
              confidence REAL,
              latency_ms INTEGER,
              segment_frames INTEGER,
              model_run_id TEXT NOT NULL,
              seg_params_json TEXT NOT NULL,
              segment_path TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "raw-attempt-one",
                    "2025-01-04T10:00:00",
                    "2025-01-04T10:00:01",
                    "hello",
                    1,
                    "hello",
                    1,
                    0.91,
                    12,
                    30,
                    "run-good",
                    json.dumps({"threshold": 0.5}),
                    "data/live_eval/segments/referenced.npz",
                ),
                (
                    "raw-attempt-two",
                    "2025-01-04T10:00:02",
                    "2025-01-04T10:00:03",
                    None,
                    0,
                    "NONE",
                    None,
                    None,
                    None,
                    None,
                    "run-good",
                    "{}",
                    None,
                ),
            ],
        )


def _create_feedback_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
              id INTEGER PRIMARY KEY,
              model_name TEXT NOT NULL,
              started_at TEXT NOT NULL,
              ended_at TEXT
            );
            CREATE TABLE detections (
              id INTEGER PRIMARY KEY,
              session_id INTEGER NOT NULL,
              ts TEXT NOT NULL,
              sign_label TEXT NOT NULL,
              confidence REAL,
              predicted_label TEXT,
              actual_label TEXT,
              is_correct INTEGER
            );
            CREATE TABLE feedback (
              feedback_id INTEGER PRIMARY KEY,
              attempt_id TEXT NOT NULL,
              feedback_type TEXT NOT NULL,
              corrected_label TEXT,
              freeform_note TEXT,
              timestamp TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (7, "run-good", "2025-01-04T09:59:00", "2025-01-04T10:05:00"),
        )
        connection.execute(
            "INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, 7, "2025-01-04T10:00:00", "hello", 0.91, "hello", "hello", 1),
        )
        connection.executemany(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "raw-attempt-one", "confirm", None, None, "2025-01-04T10:01:00"),
                (
                    2,
                    "raw-attempt-one",
                    "wrong",
                    "no",
                    "operator correction",
                    "2025-01-04T10:02:00",
                ),
            ],
        )


def _file_evidence(repository: Path, relative: str) -> dict[str, object]:
    path = repository / relative
    return {
        "logical_path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _create_synthetic_legacy_repository(root: Path, audit_path: Path) -> None:
    label_map = {"0": "hello", "1": "no"}
    plans = {
        "synthetic-plan": {
            "name": "synthetic-plan",
            "description": "Synthetic contract fixture.",
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T10:05:00",
            "steps": [{"key": "center_wrist", "params": {}}],
        }
    }
    for relative in (
        "data/results/run-good/model.keras",
        "data/results/run-relocated/sign_model_best.keras",
        "data/results/run-shared/model.keras",
        "data/results_old/run-shared/model.keras",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"model:{relative}".encode())
    for relative in (
        "data/results/run-good/label_map.json",
        "data/results/run-relocated/label_map.json",
    ):
        _write_json(root / relative, label_map)
    _write_json(root / "data/results/run-good/config.json", {"batch_size": 8})
    _write_json(root / "data/plans/plans.json", plans)
    segments = root / "data/live_eval/segments"
    segments.mkdir(parents=True)
    (segments / "referenced.npz").write_bytes(b"referenced-segment")
    (segments / "orphan.npz").write_bytes(b"orphan-segment")
    _create_run_database(root / "data/models/runs.db")
    _create_feedback_database(root / "data/models/live_feedback.db")
    _create_live_evaluation_database(root / "data/live_eval/live_eval.db")

    subprocess.run(["git", "init", "-b", "main", str(root)], check=True, capture_output=True)
    _git(root, "config", "user.name", "SignLab Tests")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Synthetic immutable legacy source")

    promoted_model = "data/results/run-relocated/sign_model_best.keras"
    promoted_label_map = "data/results/run-relocated/label_map.json"
    audit = {
        "schema_version": 2,
        "repository": {
            "head_commit": _git(root, "rev-parse", "HEAD"),
            "tree_object": _git(root, "rev-parse", "HEAD^{tree}"),
        },
        "dataset_snapshot": {"selected_representation": "synthetic-plan"},
        "live_state": {
            "run_database": {"runs": 3},
            "feedback_database": {"feedback_records": 2},
            "live_evaluation_database": {"attempts": 2},
        },
        "integrity": [
            _file_evidence(root, "data/models/runs.db"),
            _file_evidence(root, "data/models/live_feedback.db"),
            _file_evidence(root, "data/live_eval/live_eval.db"),
            _file_evidence(root, "data/plans/plans.json"),
        ],
        "reported_runs": [
            {
                "run_id": "run-relocated",
                "model_key": "mamba",
                "model_logical_path": promoted_model,
                "model_bytes": (root / promoted_model).stat().st_size,
                "model_sha256": _sha256(root / promoted_model),
                "source_evidence": [_file_evidence(root, promoted_label_map)],
            }
        ],
    }
    _write_json(audit_path, audit)


@pytest.fixture(scope="module")
def pristine_export(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("validated-legacy-export")
    legacy_root = root / "legacy"
    audit_snapshot = root / "legacy-state.json"
    public_root = root / "public"
    quarantine_root = root / "quarantine"
    _create_synthetic_legacy_repository(legacy_root, audit_snapshot)
    export_legacy_evidence(
        legacy_root=legacy_root,
        audit_snapshot=audit_snapshot,
        public_output=public_root,
        quarantine_output=quarantine_root,
    )
    return public_root, quarantine_root


@pytest.fixture
def export_copy(
    tmp_path: Path,
    pristine_export: tuple[Path, Path],
) -> tuple[Path, Path]:
    source_public, source_quarantine = pristine_export
    public_root = tmp_path / "public"
    quarantine_root = tmp_path / "quarantine"
    shutil.copytree(source_public, public_root)
    shutil.copytree(source_quarantine, quarantine_root)
    return public_root, quarantine_root


def _refresh_public_component(public_root: Path, relative: str) -> None:
    manifest_path = public_root / "manifest.json"
    manifest = _read_json(manifest_path)
    components = cast(list[JsonObject], manifest["components"])
    component = next(item for item in components if item["path"] == relative)
    path = public_root / relative
    component["bytes"] = path.stat().st_size
    component["sha256"] = _sha256(path)
    _write_json(manifest_path, manifest)


def _refresh_private_chain(
    public_root: Path,
    quarantine_root: Path,
    *,
    record_path: str | None = None,
) -> None:
    manifest_path = quarantine_root / "manifest.json"
    if record_path is not None:
        manifest = _read_json(manifest_path)
        components = cast(list[JsonObject], manifest["record_components"])
        component = next(item for item in components if item["path"] == record_path)
        path = quarantine_root / record_path
        component["bytes"] = path.stat().st_size
        component["sha256"] = _sha256(path)
        component["records"] = len(path.read_text(encoding="utf-8").splitlines())
        _write_json(manifest_path, manifest)

    receipt_path = public_root / "quarantine-receipt.json"
    receipt = _read_json(receipt_path)
    receipt_components = cast(list[JsonObject], receipt["components"])
    for component in receipt_components:
        relative = cast(str, component["path"])
        path = quarantine_root / relative
        component["bytes"] = path.stat().st_size
        component["sha256"] = _sha256(path)
        if relative.startswith("records/"):
            component["records"] = len(path.read_text(encoding="utf-8").splitlines())
    _write_json(receipt_path, receipt)
    _refresh_public_component(public_root, "quarantine-receipt.json")


def test_exporter_builds_and_validates_a_portable_bundle_without_mutating_source(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    audit_snapshot = tmp_path / "legacy-state.json"
    public_root = tmp_path / "public"
    quarantine_root = tmp_path / "quarantine"
    _create_synthetic_legacy_repository(legacy_root, audit_snapshot)
    head_before = _git(legacy_root, "rev-parse", "HEAD")
    tree_before = _git(legacy_root, "rev-parse", "HEAD^{tree}")

    summary = export_legacy_evidence(
        legacy_root=legacy_root,
        audit_snapshot=audit_snapshot,
        public_output=public_root,
        quarantine_output=quarantine_root,
    )

    assert summary == ExportSummary(
        runs=3,
        attempts=2,
        annotations=2,
        detections=1,
        sessions=1,
        promoted_models=1,
        quarantined_segments=2,
        quarantine_objects=5,
    )
    assert validate_legacy_export(
        public_root=public_root,
        quarantine_root=quarantine_root,
    ) == ValidationSummary(
        runs=3,
        attempts=2,
        annotations=2,
        detections=1,
        sessions=1,
        quarantine_verified=True,
    )
    assert _git(legacy_root, "rev-parse", "HEAD") == head_before
    assert _git(legacy_root, "rev-parse", "HEAD^{tree}") == tree_before
    assert _git(legacy_root, "status", "--short") == ""

    run_shard = _read_json(public_root / "runs" / "runs-000.json")
    run_records = cast(list[JsonObject], run_shard["records"])
    relocated_artifacts = cast(JsonObject, run_records[1]["artifacts"])
    relocated_directory = cast(JsonObject, relocated_artifacts["directory"])
    relocated_model = cast(JsonObject, relocated_artifacts["model"])
    ambiguous_artifacts = cast(JsonObject, run_records[2]["artifacts"])
    ambiguous_model = cast(JsonObject, ambiguous_artifacts["model"])
    assert relocated_directory["availability"] == "available"
    assert relocated_directory["resolved_locator"] == "legacy://data/results/run-relocated"
    assert relocated_model["availability"] == "available"
    assert ambiguous_model["availability"] == "ambiguous"

    source_run_ids = {cast(str, record["source_run_id"]) for record in run_records}
    promoted = _read_json(public_root / "promoted-artifacts.json")
    promoted_artifacts = cast(list[JsonObject], promoted["artifacts"])
    assert {cast(str, artifact["run_id"]) for artifact in promoted_artifacts}.issubset(
        source_run_ids
    )
    attempt_records = _read_jsonl(quarantine_root / "records" / "attempts.jsonl")
    session_records = _read_jsonl(quarantine_root / "records" / "sessions.jsonl")
    assert {cast(str, record["model_run_id"]) for record in attempt_records}.issubset(
        source_run_ids
    )
    assert {cast(str, record["model_run_id"]) for record in session_records}.issubset(
        source_run_ids
    )

    private_manifest = _read_json(quarantine_root / "manifest.json")
    private_policy = cast(JsonObject, private_manifest["policy"])
    assert private_policy["timestamps"] == "live-records-relative-offsets-only"
    private_objects = cast(list[JsonObject], private_manifest["objects"])
    public_label_map = cast(JsonObject, promoted_artifacts[0]["label_map"])
    label_uri = public_label_map["quarantine_uri"]
    label_object = next(item for item in private_objects if item["uri"] == label_uri)
    quarantined_labels = _read_json(quarantine_root / cast(str, label_object["storage_key"]))
    assert public_label_map["labels"] == quarantined_labels

    public_plans = _read_json(public_root / "preprocessing-plans.json")
    plan_object = next(
        item for item in private_objects if item["uri"] == public_plans["quarantine_uri"]
    )
    quarantined_plans = _read_json(quarantine_root / cast(str, plan_object["storage_key"]))
    historical_plan = cast(JsonObject, quarantined_plans["synthetic-plan"])
    assert historical_plan["created_at"] == "2025-01-01T10:00:00"
    assert historical_plan["updated_at"] == "2025-01-01T10:05:00"

    attempts = (quarantine_root / "records/attempts.jsonl").read_text(encoding="utf-8")
    annotations = (quarantine_root / "records/annotations.jsonl").read_text(encoding="utf-8")
    assert "raw-attempt" not in attempts
    assert "operator correction" not in annotations
    assert "2025-01" not in attempts + annotations

    second_public = tmp_path / "public-second"
    second_quarantine = tmp_path / "quarantine-second"
    assert (
        export_legacy_evidence(
            legacy_root=legacy_root,
            audit_snapshot=audit_snapshot,
            public_output=second_public,
            quarantine_output=second_quarantine,
        )
        == summary
    )
    assert _tree_bytes(second_public) == _tree_bytes(public_root)
    assert _tree_bytes(second_quarantine) == _tree_bytes(quarantine_root)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-receipt-component",
        "receipt-manifest-disagreement",
        "attempt-sequence",
        "annotation-reference",
        "detection-reference",
        "segment-reference",
        "segment-role",
        "missing-public-artifact",
        "public-artifact-role",
        "undeclared-file",
        "unclassified-segment",
        "aggregate-count",
        "receipt-count",
        "unknown-promoted-run",
        "unknown-attempt-model",
        "unknown-session-model",
        "label-map-content",
        "duplicate-promoted-run",
        "duplicate-promoted-model",
        "promoted-model-key-mismatch",
    ],
)
def test_private_validator_rejects_integrity_and_relationship_tampering(
    export_copy: tuple[Path, Path],
    case: str,
) -> None:
    public_root, quarantine_root = export_copy
    receipt_path = public_root / "quarantine-receipt.json"
    private_manifest_path = quarantine_root / "manifest.json"
    receipt = _read_json(receipt_path)
    private_manifest = _read_json(private_manifest_path)

    if case == "duplicate-receipt-component":
        components = cast(list[JsonObject], receipt["components"])
        components.append(dict(components[0]))
        _write_json(receipt_path, receipt)
        _refresh_public_component(public_root, "quarantine-receipt.json")
    elif case == "receipt-manifest-disagreement":
        components = cast(list[JsonObject], receipt["components"])
        record_component = next(
            item for item in components if item["path"] == "records/attempts.jsonl"
        )
        del record_component["records"]
        _write_json(receipt_path, receipt)
        _refresh_public_component(public_root, "quarantine-receipt.json")
    elif case in {"attempt-sequence", "annotation-reference", "detection-reference"}:
        relative = {
            "attempt-sequence": "records/attempts.jsonl",
            "annotation-reference": "records/annotations.jsonl",
            "detection-reference": "records/detections.jsonl",
        }[case]
        path = quarantine_root / relative
        records = _read_jsonl(path)
        if case == "attempt-sequence":
            records[0]["record_id"] = "attempt-000003"
        elif case == "annotation-reference":
            records[0]["attempt_ref"] = "attempt-999999"
        else:
            records[0]["session_ref"] = "session-9999"
        _write_jsonl(path, records)
        _refresh_private_chain(public_root, quarantine_root, record_path=relative)
    elif case in {"segment-reference", "segment-role"}:
        attempts_path = quarantine_root / "records" / "attempts.jsonl"
        attempts = _read_jsonl(attempts_path)
        segment_uri = cast(str, attempts[0]["segment_uri"])
        if case == "segment-reference":
            attempts[0]["segment_uri"] = f"quarantine://sha256/{'0' * 64}.npz"
            _write_jsonl(attempts_path, attempts)
            _refresh_private_chain(
                public_root,
                quarantine_root,
                record_path="records/attempts.jsonl",
            )
        else:
            objects = cast(list[JsonObject], private_manifest["objects"])
            segment = next(item for item in objects if item["uri"] == segment_uri)
            segment["roles"] = ["orphan-live-segment"]
            _write_json(private_manifest_path, private_manifest)
            _refresh_private_chain(public_root, quarantine_root)
    elif case in {"missing-public-artifact", "public-artifact-role"}:
        promoted = _read_json(public_root / "promoted-artifacts.json")
        artifact = cast(list[JsonObject], promoted["artifacts"])[0]
        model = cast(JsonObject, artifact["model"])
        model_uri = cast(str, model["quarantine_uri"])
        objects = cast(list[JsonObject], private_manifest["objects"])
        model_object = next(item for item in objects if item["uri"] == model_uri)
        if case == "missing-public-artifact":
            objects.remove(model_object)
        else:
            model_object["roles"] = ["unrelated-role"]
        _write_json(private_manifest_path, private_manifest)
        _refresh_private_chain(public_root, quarantine_root)
    elif case == "undeclared-file":
        (quarantine_root / "undeclared.json").write_text("{}\n", encoding="utf-8")
    elif case == "unclassified-segment":
        objects = cast(list[JsonObject], private_manifest["objects"])
        orphan = next(
            item for item in objects if "orphan-live-segment" in cast(list[str], item["roles"])
        )
        orphan["roles"] = ["unclassified-segment"]
        _write_json(private_manifest_path, private_manifest)
        _refresh_private_chain(public_root, quarantine_root)
    elif case == "aggregate-count":
        counts = cast(JsonObject, private_manifest["counts"])
        counts["sessions"] = cast(int, counts["sessions"]) + 1
        _write_json(private_manifest_path, private_manifest)
        _refresh_private_chain(public_root, quarantine_root)
    elif case == "receipt-count":
        public_manifest = _read_json(public_root / "manifest.json")
        public_counts = cast(JsonObject, public_manifest["counts"])
        receipt_counts = cast(JsonObject, receipt["counts"])
        public_counts["attempts"] = cast(int, public_counts["attempts"]) + 1
        receipt_counts["attempts"] = cast(int, receipt_counts["attempts"]) + 1
        _write_json(receipt_path, receipt)
        _write_json(public_root / "manifest.json", public_manifest)
        _refresh_public_component(public_root, "quarantine-receipt.json")
    elif case in {
        "unknown-promoted-run",
        "label-map-content",
        "duplicate-promoted-run",
        "duplicate-promoted-model",
        "promoted-model-key-mismatch",
    }:
        promoted_path = public_root / "promoted-artifacts.json"
        promoted = _read_json(promoted_path)
        artifacts = cast(list[JsonObject], promoted["artifacts"])
        artifact = artifacts[0]
        if case == "unknown-promoted-run":
            artifact["run_id"] = "missing-run"
        elif case == "label-map-content":
            label_map = cast(JsonObject, artifact["label_map"])
            label_map["labels"] = {"0": "tampered-label"}
        elif case == "promoted-model-key-mismatch":
            artifact["model_key"] = "causal_gru"
        else:
            duplicate = cast(JsonObject, json.loads(json.dumps(artifact)))
            if case == "duplicate-promoted-run":
                duplicate["model_key"] = "causal_gru"
            else:
                duplicate["run_id"] = "run-good"
            artifacts.append(duplicate)
        _write_json(promoted_path, promoted)
        _refresh_public_component(public_root, "promoted-artifacts.json")
    else:
        relative = (
            "records/attempts.jsonl"
            if case == "unknown-attempt-model"
            else "records/sessions.jsonl"
        )
        path = quarantine_root / relative
        records = _read_jsonl(path)
        records[0]["model_run_id"] = "missing-run"
        _write_jsonl(path, records)
        _refresh_private_chain(public_root, quarantine_root, record_path=relative)

    with pytest.raises(LegacyExportError):
        validate_legacy_export(
            public_root=public_root,
            quarantine_root=quarantine_root,
        )


@pytest.mark.parametrize("sensitive_key", SENSITIVE_KEY_FORMS)
def test_private_live_records_reject_every_nested_sensitive_key(
    export_copy: tuple[Path, Path],
    sensitive_key: str,
) -> None:
    public_root, quarantine_root = export_copy
    attempts_path = quarantine_root / "records" / "attempts.jsonl"
    attempts = _read_jsonl(attempts_path)
    attempts[0]["segmentation"] = {"nested": {sensitive_key: "opaque-identifier"}}
    _write_jsonl(attempts_path, attempts)
    _refresh_private_chain(
        public_root,
        quarantine_root,
        record_path="records/attempts.jsonl",
    )

    with pytest.raises(LegacyExportError):
        validate_legacy_export(
            public_root=public_root,
            quarantine_root=quarantine_root,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "trace_token",
            "123e4567-e89b" + "-12d3-a456-426614174000",
            "raw UUID-like",
        ),
        ("observed_at", "2025-01-01T10:00:00", "absolute timestamp"),
    ],
)
def test_private_live_records_reject_raw_identifiers_and_absolute_timestamps(
    export_copy: tuple[Path, Path],
    field: str,
    value: str,
    message: str,
) -> None:
    public_root, quarantine_root = export_copy
    attempts_path = quarantine_root / "records" / "attempts.jsonl"
    attempts = _read_jsonl(attempts_path)
    attempts[0]["segmentation"] = {field: value}
    _write_jsonl(attempts_path, attempts)
    _refresh_private_chain(
        public_root,
        quarantine_root,
        record_path="records/attempts.jsonl",
    )

    with pytest.raises(LegacyExportError, match=message):
        validate_legacy_export(
            public_root=public_root,
            quarantine_root=quarantine_root,
        )


def test_validator_rejects_missing_public_or_private_roots(
    tmp_path: Path,
    pristine_export: tuple[Path, Path],
) -> None:
    public_root, _ = pristine_export
    with pytest.raises(LegacyExportError, match="public legacy export is unavailable"):
        validate_legacy_export(public_root=tmp_path / "missing-public")
    with pytest.raises(LegacyExportError, match="private legacy quarantine is unavailable"):
        validate_legacy_export(
            public_root=public_root,
            quarantine_root=tmp_path / "missing-private",
        )


@pytest.mark.parametrize("boundary", ["same", "nested", "inside-source", "nonempty"])
def test_exporter_rejects_unsafe_output_boundaries(tmp_path: Path, boundary: str) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    audit_snapshot = tmp_path / "audit.json"
    audit_snapshot.write_text("{}\n", encoding="utf-8")
    public_root = tmp_path / "public"
    quarantine_root = tmp_path / "private"
    if boundary == "same":
        quarantine_root = public_root
    elif boundary == "nested":
        quarantine_root = public_root / "private"
    elif boundary == "inside-source":
        public_root = legacy_root / "public"
    else:
        public_root.mkdir()
        (public_root / "existing.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(LegacyExportError):
        export_legacy_evidence(
            legacy_root=legacy_root,
            audit_snapshot=audit_snapshot,
            public_output=public_root,
            quarantine_output=quarantine_root,
        )


def test_exporter_rolls_back_staging_when_the_audit_is_invalid(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    audit_snapshot = tmp_path / "audit.json"
    audit_snapshot.write_text("not-json\n", encoding="utf-8")
    public_root = tmp_path / "public"
    quarantine_root = tmp_path / "private"

    with pytest.raises(LegacyExportError, match="could not complete safely"):
        export_legacy_evidence(
            legacy_root=legacy_root,
            audit_snapshot=audit_snapshot,
            public_output=public_root,
            quarantine_output=quarantine_root,
        )

    assert not public_root.exists()
    assert not quarantine_root.exists()
    assert list(tmp_path.glob(".signlab-*-*")) == []
