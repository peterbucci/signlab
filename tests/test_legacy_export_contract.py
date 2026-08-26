"""Regression tests for the portable legacy-evidence export."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from signlab.legacy.exporter import SENSITIVE_KEYS, LegacyExportError
from signlab.legacy.schemas import DATA_ROLE, FORMAT_VERSION, PUBLIC_KIND, SCHEMAS
from signlab.legacy.validator import ValidationSummary, validate_legacy_export

type JsonObject = dict[str, object]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EXPORT_ROOT = REPOSITORY_ROOT / "docs" / "legacy" / "export" / "v1"
EXPECTED_COUNTS = {
    "runs": 464,
    "runs_succeeded": 457,
    "runs_failed": 6,
    "runs_running": 1,
    "attempts": 532,
    "annotations": 408,
    "annotated_attempts": 401,
    "unannotated_attempts": 131,
    "detections": 45,
    "sessions": 5,
    "segments": 529,
    "referenced_segments": 525,
    "orphan_segments": 4,
    "promoted_models": 4,
    "preprocessing_plans": 3,
}
EXPECTED_MODELS = {
    "causal_gru": (2428032, "f69c07838a477df0853a6cdb71b1acb9a933e0de1491359da0cae5462584e46c"),
    "causal_lstm": (3210098, "b175174c2daa4ccd33bb9fb49120d7399a61aab95c3bdfa378ba40aa57ace7ee"),
    "causal_tcn": (2458533, "d5e3d15cead7436342c15229e94a6becdfd41b1254f3caa5332cf64ad9cf3f0c"),
    "mamba": (467230, "89647b516068950818bffabb9c269f31bef924bdcce0230ea509af9fea17559e"),
}
MACHINE_PATH_PATTERNS = (
    re.compile(r"(?i)(?<![a-z0-9])[a-z]:[\\/]"),
    re.compile(r"(?i)(?:^|[\s\"'=(])/(?:users|home)/[^/\s]+(?:/|\\)"),
    re.compile(r"(?i)file:///(?:[a-z]:|users/|home/)"),
    re.compile(r"\\\\[^\\\s]+\\[^\\\s]+"),
)
RAW_UUID_PATTERN = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
FORBIDDEN_PRIVATE_KEYS = {
    "absolute_path",
    "attempt_id",
    "email",
    "error",
    "freeform_note",
    "participant",
    "participant_id",
    "subject",
    "subject_id",
    "user_id",
    "username",
}
SENSITIVE_KEY_FORMS = tuple(
    sorted(
        SENSITIVE_KEYS
        | {variant for key in SENSITIVE_KEYS for variant in (key.upper(), key.title())}
    )
)


def _read_object(path: Path) -> JsonObject:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(JsonObject, value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_object(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _component(path: Path, root: Path, *, records: int | None = None) -> JsonObject:
    result: JsonObject = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _walk(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _run_records(root: Path = PUBLIC_EXPORT_ROOT) -> list[JsonObject]:
    records: list[JsonObject] = []
    for path in sorted((root / "runs").glob("runs-*.json")):
        shard = _read_object(path)
        raw_records = shard["records"]
        assert isinstance(raw_records, list)
        records.extend(cast(list[JsonObject], raw_records))
    return records


def _synthetic_run(*, configuration: JsonObject | None = None) -> JsonObject:
    artifact = {
        "availability": "not-recorded",
        "registered_locator": None,
        "resolved_locator": None,
    }
    run_configuration: JsonObject = {
        "model_settings": {},
        "global_settings": {},
        "data_locator": "legacy-relative://data/landmarks/synthetic",
        "sequence_length": 30,
        "test_fraction": 0.2,
        "validation_fraction": 0.2,
        "batch_size": 8,
        "epochs": 2,
    }
    if configuration is not None:
        run_configuration.update(configuration)
    return {
        "record_id": "run-000001",
        "run_name": "synthetic-run",
        "source_run_id": "synthetic-run",
        "started_at": "redacted-0000000000000000",
        "finished_at": "redacted-0000000000000000",
        "status": "succeeded",
        "model_key": "synthetic",
        "configuration": run_configuration,
        "metrics": {
            "quick": {},
            "test_loss": 0.2,
            "test_accuracy": 0.8,
            "samples_train": 8,
            "samples_validation": 2,
            "samples_test": 2,
        },
        "artifacts": {
            name: dict(artifact)
            for name in (
                "configuration",
                "directory",
                "history",
                "label_map",
                "metrics",
                "model",
                "predictions",
            )
        },
        "validity": {
            "data_role": DATA_ROLE,
            "eligible_for_locked_test": False,
            "legacy_error_present": False,
            "legacy_notes_present": False,
            "notes": [DATA_ROLE],
        },
    }


def _build_synthetic_public_export(
    root: Path,
    *,
    configuration: JsonObject | None = None,
) -> Path:
    run_path = root / "runs" / "runs-000.json"
    _write_object(
        run_path,
        {
            "schema_version": FORMAT_VERSION,
            "kind": "signlab.legacy-run-index",
            "shard": 0,
            "records": [_synthetic_run(configuration=configuration)],
        },
    )
    _write_object(
        root / "promoted-artifacts.json",
        {
            "schema_version": FORMAT_VERSION,
            "kind": "signlab.legacy-promoted-artifacts",
            "artifacts": [],
        },
    )
    zero_digest = "0" * 64
    _write_object(
        root / "preprocessing-plans.json",
        {
            "schema_version": FORMAT_VERSION,
            "kind": "signlab.legacy-preprocessing-plans",
            "source_sha256": zero_digest,
            "quarantine_uri": f"quarantine://sha256/{zero_digest}.json",
            "plans": {
                "synthetic-plan": {
                    "name": "synthetic-plan",
                    "description": "Synthetic contract fixture.",
                    "created_at": "2025-01-01T10:00:00",
                    "updated_at": "2025-01-01T10:05:00",
                    "steps": [{"key": "center_wrist", "params": {}}],
                }
            },
        },
    )
    counts = {
        "runs": 1,
        "runs_succeeded": 1,
        "runs_failed": 0,
        "runs_running": 0,
        "attempts": 0,
        "annotations": 0,
        "annotated_attempts": 0,
        "unannotated_attempts": 0,
        "detections": 0,
        "sessions": 0,
        "segments": 0,
        "referenced_segments": 0,
        "orphan_segments": 0,
        "promoted_models": 0,
        "preprocessing_plans": 1,
    }
    private_counts = {
        "attempts": 0,
        "annotations": 0,
        "annotated_attempts": 0,
        "unannotated_attempts": 0,
        "sessions": 0,
        "detections": 0,
        "segments": 0,
        "referenced_segments": 0,
        "orphan_segments": 0,
        "promoted_models": 0,
        "quarantine_objects": 0,
    }
    private_receipt_components = [
        {
            "path": path,
            "bytes": 0,
            "sha256": zero_digest,
            **({"records": 0} if path.startswith("records/") else {}),
        }
        for path in (
            "manifest.json",
            "records/attempts.jsonl",
            "records/annotations.jsonl",
            "records/sessions.jsonl",
            "records/detections.jsonl",
        )
    ]
    _write_object(
        root / "quarantine-receipt.json",
        {
            "schema_version": FORMAT_VERSION,
            "kind": "signlab.legacy-quarantine-receipt",
            "policy": {
                "contains_individual_object_hashes": False,
                "data_role": DATA_ROLE,
                "durability": "local-only-pending-private-remote",
                "eligible_for_locked_test": False,
                "publishable": False,
            },
            "counts": private_counts,
            "components": private_receipt_components,
        },
    )
    for name, schema in SCHEMAS.items():
        _write_object(root / "schemas" / name, schema)

    components = []
    record_counts = {
        "runs/runs-000.json": 1,
        "promoted-artifacts.json": 0,
        "preprocessing-plans.json": 1,
    }
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        records = record_counts.get(relative)
        components.append(_component(path, root, records=records))
    _write_object(
        root / "manifest.json",
        {
            "schema_version": FORMAT_VERSION,
            "kind": PUBLIC_KIND,
            "source": {
                "head_commit": "0" * 40,
                "tree_object": "1" * 40,
                "audit_sha256": "2" * 64,
            },
            "policy": {
                "data_role": DATA_ROLE,
                "eligible_for_locked_test": False,
                "contains_private_artifacts": False,
                "durability": "local-only-pending-private-remote",
            },
            "counts": counts,
            "components": components,
        },
    )
    return root


def _refresh_public_component(root: Path, relative: str) -> None:
    manifest = _read_object(root / "manifest.json")
    components = cast(list[JsonObject], manifest["components"])
    path = root / relative
    component = next(item for item in components if item["path"] == relative)
    component["bytes"] = path.stat().st_size
    component["sha256"] = _sha256(path)
    _write_object(root / "manifest.json", manifest)


def test_committed_public_export_validates_without_legacy_dependencies() -> None:
    summary = validate_legacy_export(public_root=PUBLIC_EXPORT_ROOT)

    assert summary == ValidationSummary(
        runs=464,
        attempts=532,
        annotations=408,
        detections=45,
        sessions=5,
        quarantine_verified=False,
    )


def test_public_manifest_and_run_index_reconcile() -> None:
    manifest = _read_object(PUBLIC_EXPORT_ROOT / "manifest.json")
    assert manifest["counts"] == EXPECTED_COUNTS
    assert manifest["policy"] == {
        "contains_private_artifacts": False,
        "data_role": DATA_ROLE,
        "durability": "local-only-pending-private-remote",
        "eligible_for_locked_test": False,
    }

    components = cast(list[JsonObject], manifest["components"])
    declared_paths = [cast(str, item["path"]) for item in components]
    actual_paths = sorted(
        path.relative_to(PUBLIC_EXPORT_ROOT).as_posix()
        for path in PUBLIC_EXPORT_ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    assert sorted(declared_paths) == actual_paths
    assert len(declared_paths) == len(set(declared_paths))
    for item in components:
        path = PUBLIC_EXPORT_ROOT / cast(str, item["path"])
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)

    records = _run_records()
    assert [record["record_id"] for record in records] == [
        f"run-{index:06d}" for index in range(1, 465)
    ]
    assert Counter(record["status"] for record in records) == {
        "succeeded": 457,
        "failed": 6,
        "running": 1,
    }
    for record in records:
        assert isinstance(record["configuration"], dict)
        assert isinstance(record["metrics"], dict)
        configuration = cast(JsonObject, record["configuration"])
        data_locator = configuration["data_locator"]
        if isinstance(data_locator, str) and "data/landmarks" in data_locator:
            assert data_locator.startswith(("legacy://", "legacy-relative://"))
        validity = cast(JsonObject, record["validity"])
        assert validity["data_role"] == DATA_ROLE
        assert validity["eligible_for_locked_test"] is False

    relocated_directories: list[JsonObject] = []
    for record in records:
        artifacts = cast(JsonObject, record["artifacts"])
        directory = cast(JsonObject, artifacts["directory"])
        if (
            directory["registered_locator"] != directory["resolved_locator"]
            and directory["availability"] == "available"
        ):
            relocated_directories.append(directory)
    assert relocated_directories
    assert all(item["resolved_locator"] is not None for item in relocated_directories)


def test_promoted_models_label_maps_and_plans_are_hash_anchored() -> None:
    promoted = _read_object(PUBLIC_EXPORT_ROOT / "promoted-artifacts.json")
    artifacts = cast(list[JsonObject], promoted["artifacts"])
    assert {cast(str, item["model_key"]) for item in artifacts} == set(EXPECTED_MODELS)

    for artifact in artifacts:
        key = cast(str, artifact["model_key"])
        model = cast(JsonObject, artifact["model"])
        label_map = cast(JsonObject, artifact["label_map"])
        validity = cast(JsonObject, artifact["validity"])
        expected_bytes, expected_digest = EXPECTED_MODELS[key]
        assert (model["bytes"], model["sha256"]) == (expected_bytes, expected_digest)
        assert model["quarantine_uri"] == f"quarantine://sha256/{expected_digest}.keras"
        assert model["storage_status"] == "local-quarantine"
        assert label_map["sha256"] == (
            "c0a1e624ee1d6353d6377c2391030df15106ec2bea193323c07f0da1a4c9499d"
        )
        assert label_map["labels"] == {
            "0": "hello",
            "1": "no",
            "2": "please",
            "3": "thank you",
            "4": "yes",
        }
        assert validity["data_role"] == DATA_ROLE
        assert validity["eligible_for_locked_test"] is False

    plans = _read_object(PUBLIC_EXPORT_ROOT / "preprocessing-plans.json")
    assert set(cast(JsonObject, plans["plans"])) == {
        "baseline_pos_handscale_30f",
        "pos_vel_handscale_smooth_30f",
        "shape_motion_angles_robust_30f",
    }
    assert plans["source_sha256"] == (
        "5a727cc6218dcc2f82cb72cdec00bd7348f1f9d924677011548e2200d32cea0d"
    )
    assert plans["quarantine_uri"] == (
        "quarantine://sha256/5a727cc6218dcc2f82cb72cdec00bd7348f1f9d924677011548e2200d32cea0d.json"
    )
    robust_plan = cast(
        JsonObject,
        cast(JsonObject, plans["plans"])["shape_motion_angles_robust_30f"],
    )
    description = cast(str, robust_plan["description"])
    assert "orientation/scale" in description
    assert not description.startswith(("legacy://", "legacy-relative://"))


def test_public_export_is_portable_and_contains_no_private_payloads() -> None:
    files = [path for path in PUBLIC_EXPORT_ROOT.rglob("*") if path.is_file()]
    assert files
    assert all(path.suffix == ".json" for path in files)
    assert max(path.stat().st_size for path in files) < 1024 * 1024

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not any(pattern.search(text) for pattern in MACHINE_PATH_PATTERNS), path
        assert RAW_UUID_PATTERN.search(text) is None, path
        assert "sb128" not in text.casefold(), path
        assert "peterbucci" not in text.casefold(), path
        value: Any = json.loads(text)
        keys = {cast(str, key) for node in _walk(value) if isinstance(node, dict) for key in node}
        assert keys.isdisjoint(FORBIDDEN_PRIVATE_KEYS), path


def test_committed_schemas_match_code_and_close_required_objects() -> None:
    for name, expected in SCHEMAS.items():
        schema = _read_object(PUBLIC_EXPORT_ROOT / "schemas" / name)
        assert schema == expected
        for node in _walk(schema):
            if not isinstance(node, dict):
                continue
            if "required" in node:
                assert "properties" in node
                required = set(cast(list[str], node["required"]))
                properties = cast(JsonObject, node["properties"])
                assert required.issubset(properties)
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False


def test_synthetic_public_export_validates_without_legacy_sources(tmp_path: Path) -> None:
    root = _build_synthetic_public_export(tmp_path / "portable-export")

    assert validate_legacy_export(public_root=root) == ValidationSummary(
        runs=1,
        attempts=0,
        annotations=0,
        detections=0,
        sessions=0,
        quarantine_verified=False,
    )


@pytest.mark.parametrize("sensitive_key", SENSITIVE_KEY_FORMS)
def test_public_export_rejects_every_nested_sensitive_key(
    tmp_path: Path,
    sensitive_key: str,
) -> None:
    root = _build_synthetic_public_export(tmp_path / "sensitive-public")
    run_path = root / "runs" / "runs-000.json"
    shard = _read_object(run_path)
    record = cast(list[JsonObject], shard["records"])[0]
    configuration = cast(JsonObject, record["configuration"])
    model_settings = cast(JsonObject, configuration["model_settings"])
    model_settings["nested"] = {sensitive_key: "opaque-identifier"}
    _write_object(run_path, shard)
    _refresh_public_component(root, "runs/runs-000.json")

    with pytest.raises(LegacyExportError):
        validate_legacy_export(public_root=root)


def test_public_export_rejects_a_canonical_hyphenated_uuid(tmp_path: Path) -> None:
    root = _build_synthetic_public_export(tmp_path / "uuid-public")
    run_path = root / "runs" / "runs-000.json"
    shard = _read_object(run_path)
    record = cast(list[JsonObject], shard["records"])[0]
    configuration = cast(JsonObject, record["configuration"])
    model_settings = cast(JsonObject, configuration["model_settings"])
    model_settings["trace_token"] = "123e4567-e89b" + "-12d3-a456-426614174000"
    _write_object(run_path, shard)
    _refresh_public_component(root, "runs/runs-000.json")

    with pytest.raises(LegacyExportError, match="raw UUID-like"):
        validate_legacy_export(public_root=root)


def test_all_five_public_run_shards_are_visible_to_git() -> None:
    shards = sorted((PUBLIC_EXPORT_ROOT / "runs").glob("runs-*.json"))
    assert [path.name for path in shards] == [f"runs-{index:03d}.json" for index in range(5)]

    for path in shards:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 1, f"Git ignores required run shard {path.name}"


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ({"data_locator": "C" + ":" + r"\Users\person\private"}, "absolute machine path"),
        (
            {"data_locator": "123e4567e89b12d3" + "a456426614174000"},
            "raw UUID-like",
        ),
    ],
)
def test_synthetic_export_rejects_private_machine_data(
    tmp_path: Path,
    configuration: JsonObject,
    message: str,
) -> None:
    root = _build_synthetic_public_export(
        tmp_path / "unsafe-export",
        configuration=configuration,
    )

    with pytest.raises(LegacyExportError, match=message):
        validate_legacy_export(public_root=root)


@pytest.mark.parametrize(
    "case",
    [
        "run-id-sequence",
        "shard-identity",
        "shard-record-count",
        "undeclared-file",
        "plan-uri",
        "promoted-plan",
        "promoted-uri",
        "promoted-count",
        "receipt-count",
        "receipt-components",
        "forbidden-key",
        "schema-drift",
    ],
)
def test_synthetic_public_validator_rejects_contract_tampering(
    tmp_path: Path,
    case: str,
) -> None:
    root = _build_synthetic_public_export(tmp_path / "tampered-export")
    manifest = _read_object(root / "manifest.json")
    run_path = root / "runs" / "runs-000.json"
    run_shard = _read_object(run_path)
    records = cast(list[JsonObject], run_shard["records"])
    promoted_path = root / "promoted-artifacts.json"
    promoted = _read_object(promoted_path)
    plans_path = root / "preprocessing-plans.json"
    plans = _read_object(plans_path)
    receipt_path = root / "quarantine-receipt.json"
    receipt = _read_object(receipt_path)

    if case == "run-id-sequence":
        records[0]["record_id"] = "run-000002"
        _write_object(run_path, run_shard)
        _refresh_public_component(root, "runs/runs-000.json")
    elif case == "shard-identity":
        run_shard["shard"] = 1
        _write_object(run_path, run_shard)
        _refresh_public_component(root, "runs/runs-000.json")
    elif case == "shard-record-count":
        components = cast(list[JsonObject], manifest["components"])
        run_component = next(item for item in components if item["path"] == "runs/runs-000.json")
        run_component["records"] = 2
        _write_object(root / "manifest.json", manifest)
    elif case == "undeclared-file":
        _write_object(root / "undeclared.json", {})
    elif case == "plan-uri":
        plans["quarantine_uri"] = f"quarantine://sha256/{'1' * 64}.json"
        _write_object(plans_path, plans)
        _refresh_public_component(root, "preprocessing-plans.json")
    elif case == "promoted-plan":
        promoted["artifacts"] = [
            {
                "run_id": "synthetic-run",
                "model_key": "synthetic",
                "preprocessing_plan": "unknown-plan",
                "model": {
                    "bytes": 1,
                    "sha256": "3" * 64,
                    "quarantine_uri": f"quarantine://sha256/{'3' * 64}.keras",
                    "storage_status": "local-quarantine",
                },
                "label_map": {
                    "bytes": 1,
                    "sha256": "4" * 64,
                    "quarantine_uri": f"quarantine://sha256/{'4' * 64}.json",
                    "labels": {"0": "hello"},
                },
                "validity": {
                    "data_role": DATA_ROLE,
                    "eligible_for_locked_test": False,
                    "notes": [DATA_ROLE],
                },
            }
        ]
        _write_object(promoted_path, promoted)
        _refresh_public_component(root, "promoted-artifacts.json")
    elif case == "promoted-uri":
        artifact = {
            "run_id": "synthetic-run",
            "model_key": "synthetic",
            "preprocessing_plan": "synthetic-plan",
            "model": {
                "bytes": 1,
                "sha256": "3" * 64,
                "quarantine_uri": f"quarantine://sha256/{'4' * 64}.keras",
                "storage_status": "local-quarantine",
            },
            "label_map": {
                "bytes": 1,
                "sha256": "4" * 64,
                "quarantine_uri": f"quarantine://sha256/{'4' * 64}.json",
                "labels": {"0": "hello"},
            },
            "validity": {
                "data_role": DATA_ROLE,
                "eligible_for_locked_test": False,
                "notes": [DATA_ROLE],
            },
        }
        promoted["artifacts"] = [artifact]
        _write_object(promoted_path, promoted)
        _refresh_public_component(root, "promoted-artifacts.json")
    elif case == "promoted-count":
        components = cast(list[JsonObject], manifest["components"])
        component = next(item for item in components if item["path"] == "promoted-artifacts.json")
        component["records"] = 1
        _write_object(root / "manifest.json", manifest)
    elif case == "receipt-count":
        counts = cast(JsonObject, receipt["counts"])
        counts["attempts"] = 1
        _write_object(receipt_path, receipt)
        _refresh_public_component(root, "quarantine-receipt.json")
    elif case == "receipt-components":
        receipt["components"] = cast(list[JsonObject], receipt["components"])[1:]
        _write_object(receipt_path, receipt)
        _refresh_public_component(root, "quarantine-receipt.json")
    elif case == "forbidden-key":
        configuration = cast(JsonObject, records[0]["configuration"])
        settings = cast(JsonObject, configuration["model_settings"])
        settings["participant_id"] = "redacted-value"
        _write_object(run_path, run_shard)
        _refresh_public_component(root, "runs/runs-000.json")
    else:
        schema_path = root / "schemas" / "session.schema.json"
        schema = _read_object(schema_path)
        schema["title"] = "Drifted schema"
        _write_object(schema_path, schema)
        _refresh_public_component(root, "schemas/session.schema.json")

    with pytest.raises(LegacyExportError):
        validate_legacy_export(public_root=root)
