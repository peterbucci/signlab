from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_AUDIT_PATH = REPOSITORY_ROOT / "docs" / "legacy-audit.md"
LEGACY_SCHEMA_PATH = REPOSITORY_ROOT / "docs" / "legacy" / "legacy-state.schema.json"
LEGACY_STATE_PATH = REPOSITORY_ROOT / "docs" / "legacy" / "legacy-state.json"

EXPECTED_HEAD = "0e630168727e0d526a1f040ce912db26c1cefc72"
EXPECTED_TREE = "828edda9a38ef5bdd0f0e501265482b49de000be"
EXPECTED_INVENTORY = {
    "data/landmarks": (3712, 54_683_024),
    "data/live_eval": (530, 6_059_649),
    "data/models": (8, 2_050_135),
    "data/nothing_label": (4954, 2_043_571_289),
    "data/plans": (1, 3_410),
    "data/raw_videos": (310, 341_300_064),
    "data/results": (220, 210_305_035),
    "data/results_old": (1102, 429_307_581),
    "runs/sweeps": (17_461, 16_508_757_238),
}
EXPECTED_LABELS = {"hello": 60, "no": 71, "please": 58, "thank you": 53, "yes": 68}
EXPECTED_INTEGRITY_HASHES = {
    "data/live/live_imports.json": (
        "fa20489bb2181462f80940ec3bcfc34ad546273e85ee1dd385e4838af9552598"
    ),
    "data/live_eval/live_eval.db": (
        "0f606831befc27cd488004e62bf7f96ef87582d02fbefe795f4cd512c6ad952b"
    ),
    "data/models/live_feedback.db": (
        "1cfaf5185278a5722b869f66c4e4e00d80c4540e9147b931b781524b9e6a75b5"
    ),
    "data/models/runs.db": "b23da61e7600ebaf9eb03336af50716a0170a4264b1db518b241777af5815921",
    "data/plans/plans.json": "5a727cc6218dcc2f82cb72cdec00bd7348f1f9d924677011548e2200d32cea0d",
}
EXPECTED_RUN_EVIDENCE = {
    "20251222_154233_gru_phase_3_run_001": {
        "model_key": "causal_gru",
        "model_sha256": "f69c07838a477df0853a6cdb71b1acb9a933e0de1491359da0cae5462584e46c",
        "source_sha256": {
            "48714ee48008b9c43edc70c6f4347d7f864f08058dd880bb72fbc6a14d9f83d3",
            "9f1c88e7523a5191b0c7901dcbf73f3146ed7df68415c3b21b574aa48284f3b6",
            "c0a1e624ee1d6353d6377c2391030df15106ec2bea193323c07f0da1a4c9499d",
            "8b3ea92cec2c1d002e8e2242e8a7a26a7c89333a3f9e4865dd91ab20de8ff794",
        },
    },
    "20251222_161016_lstm_phase_c_run_003": {
        "model_key": "causal_lstm",
        "model_sha256": "b175174c2daa4ccd33bb9fb49120d7399a61aab95c3bdfa378ba40aa57ace7ee",
        "source_sha256": {
            "d619db0b9cb2de83f2f1347cb2edb3f3c41edd831de35ed29c737b0e30ea5ade",
            "05b4c624fe8a8d2e8a2ed322fc97ada43057219f78575ae8b1330513ebe3cf0f",
            "c0a1e624ee1d6353d6377c2391030df15106ec2bea193323c07f0da1a4c9499d",
            "6e96d05cbdef41a28bc90ef989564b4e2ba29304b14dbd502aedaf3c56d2cd20",
        },
    },
    "20251222_162425_tcn_phase_c_run_003": {
        "model_key": "causal_tcn",
        "model_sha256": "d5e3d15cead7436342c15229e94a6becdfd41b1254f3caa5332cf64ad9cf3f0c",
        "source_sha256": {
            "55a7f62b003d1403b42701cda7ce3d7257d6dfa968cbffd310dff9e8132ffcc7",
            "d5635cdf363f88e7945f826f968c8885089cc8ab2bdc4e08e0c769477cec4889",
            "c0a1e624ee1d6353d6377c2391030df15106ec2bea193323c07f0da1a4c9499d",
            "4bae18905924f1f9c089a15c199cd9bbe7d09e4346e4ff4e9ea6b141f1342131",
        },
    },
    "20251222_163528_mamba_phase_c_run_001": {
        "model_key": "mamba",
        "model_sha256": "89647b516068950818bffabb9c269f31bef924bdcce0230ea509af9fea17559e",
        "source_sha256": {
            "5483a37500fd5b406545d4a845725df7a01da7b19e8b0826742e303f78ae94d9",
            "20ec133ec420a9ab9009b5100b5ac03dbfce584eb1ee3a411e2d362d7b5e5bfc",
            "c0a1e624ee1d6353d6377c2391030df15106ec2bea193323c07f0da1a4c9499d",
            "4575ed4ec7e4cfaa6bbfac53d5ab85c98c6cb03866852077a3b1a2706f692db3",
        },
    },
}


def _load_state() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(LEGACY_STATE_PATH.read_text(encoding="utf-8")),
    )


def _macro_f1(matrix: Sequence[Sequence[int]]) -> float:
    scores: list[float] = []
    for index in range(len(matrix)):
        true_positive = matrix[index][index]
        false_positive = sum(row[index] for row in matrix) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _sha_values(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "sha256":
                assert isinstance(child, str)
                yield child
            else:
                yield from _sha_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _sha_values(child)


def _assert_schema(value: object, schema: dict[str, Any]) -> None:
    expected_type = schema.get("type")
    type_checks = {
        "array": lambda candidate: isinstance(candidate, list),
        "boolean": lambda candidate: isinstance(candidate, bool),
        "integer": lambda candidate: isinstance(candidate, int) and not isinstance(candidate, bool),
        "number": lambda candidate: (
            isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
        ),
        "object": lambda candidate: isinstance(candidate, dict),
        "string": lambda candidate: isinstance(candidate, str),
    }
    if expected_type is not None:
        assert type_checks[expected_type](value), expected_type
    if "const" in schema:
        assert value == schema["const"]
    if "pattern" in schema:
        assert isinstance(value, str)
        assert re.fullmatch(schema["pattern"], value)
    if "minimum" in schema:
        assert isinstance(value, (int, float))
        assert value >= schema["minimum"]

    if isinstance(value, dict):
        assert schema.get("additionalProperties") is False
        properties = schema["properties"]
        assert set(properties) == set(schema["required"]) == set(value)
        for key, child in value.items():
            _assert_schema(child, properties[key])
    elif isinstance(value, list):
        assert len(value) >= schema.get("minItems", 0)
        for child in value:
            _assert_schema(child, schema["items"])


def _assert_every_object_schema_is_closed(schema: object) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            assert set(schema["properties"]) == set(schema["required"])
        for child in schema.values():
            _assert_every_object_schema_is_closed(child)
    elif isinstance(schema, list):
        for child in schema:
            _assert_every_object_schema_is_closed(child)


def test_legacy_snapshot_has_a_golden_identity_and_honest_recovery_status() -> None:
    state = _load_state()
    repository = state["repository"]

    assert state["schema_version"] == 2
    assert repository["head_commit"] == EXPECTED_HEAD
    assert repository["tree_object"] == EXPECTED_TREE
    assert repository["identification_anchor"] == f"git-object:{EXPECTED_HEAD}"
    assert repository["branch"] == "master"
    assert repository["remotes"] == []
    assert repository["tags"] == []
    assert repository["recovery"]["status"] == "incomplete-local-only"
    assert "durable artifact backup" in repository["recovery"]["limitation"]
    assert repository["tracked_files"] == {
        "bytecode_files": 183,
        "file_count": 303,
        "python_source_files": 114,
    }


def test_legacy_snapshot_matches_the_golden_aggregate_contract() -> None:
    state = _load_state()
    inventory = {
        item["logical_path"]: (item["file_count"], item["bytes"])
        for item in state["artifact_inventory"]
    }
    integrity = {item["logical_path"]: item["sha256"] for item in state["integrity"]}

    assert inventory == EXPECTED_INVENTORY
    assert integrity == EXPECTED_INTEGRITY_HASHES
    assert state["dataset_snapshot"]["labels"] == EXPECTED_LABELS
    assert state["dataset_snapshot"]["total_samples"] == 310
    assert state["dataset_snapshot"]["reported_split"] == {
        "method": "two stratified sample-level train_test_split calls",
        "random_state": 42,
        "test_fraction": 0.2,
        "test_samples": 62,
        "train_fraction": 0.6,
        "train_samples": 186,
        "validation_fraction": 0.2,
        "validation_samples": 62,
    }
    assert state["live_state"]["run_database"] == {
        "by_model": {"causal_gru": 130, "causal_lstm": 130, "causal_tcn": 77, "mamba": 127},
        "by_status": {"failed": 6, "running": 1, "succeeded": 457},
        "runs": 464,
    }


def test_reported_metrics_recompute_from_sanitized_confusion_matrices() -> None:
    state = _load_state()
    labels = state["reported_label_order"]
    runs = {run["run_id"]: run for run in state["reported_runs"]}

    assert labels == list(EXPECTED_LABELS)
    assert set(runs) == set(EXPECTED_RUN_EVIDENCE)
    for run_id, expected in EXPECTED_RUN_EVIDENCE.items():
        run = runs[run_id]
        matrix = run["confusion_matrix"]

        assert run["model_key"] == expected["model_key"]
        assert run["model_sha256"] == expected["model_sha256"]
        assert {item["sha256"] for item in run["source_evidence"]} == expected["source_sha256"]
        assert len(matrix) == len(labels)
        assert all(len(row) == len(labels) for row in matrix)
        assert all(isinstance(count, int) and count >= 0 for row in matrix for count in row)

        sample_count = sum(sum(row) for row in matrix)
        correct_count = sum(matrix[index][index] for index in range(len(labels)))
        assert sample_count == run["test_samples"] == 62
        assert correct_count == run["correct_test_predictions"]
        assert math.isclose(correct_count / sample_count, run["test_accuracy"], abs_tol=1e-7)
        assert math.isclose(_macro_f1(matrix), run["macro_f1"], abs_tol=1e-12)


def test_public_snapshot_satisfies_a_recursive_closed_schema() -> None:
    state = _load_state()
    schema = json.loads(LEGACY_SCHEMA_PATH.read_text(encoding="utf-8"))

    _assert_every_object_schema_is_closed(schema)
    _assert_schema(state, schema)

    logical_paths = [item["logical_path"] for item in state["artifact_inventory"]]
    logical_paths.extend(item["logical_path"] for item in state["integrity"])
    logical_paths.extend(run["model_logical_path"] for run in state["reported_runs"])
    logical_paths.extend(
        item["logical_path"] for run in state["reported_runs"] for item in run["source_evidence"]
    )
    for logical_path in logical_paths:
        parsed = PurePosixPath(logical_path)
        assert not parsed.is_absolute()
        assert ".." not in parsed.parts
        assert "\\" not in logical_path

    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in _sha_values(state))


def test_public_audit_files_do_not_leak_local_paths_or_contact_details() -> None:
    public_text = "\n".join(
        (
            LEGACY_AUDIT_PATH.read_text(encoding="utf-8"),
            LEGACY_SCHEMA_PATH.read_text(encoding="utf-8"),
            LEGACY_STATE_PATH.read_text(encoding="utf-8"),
        )
    )
    prohibited_patterns = {
        "email address": r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
        "Unix user directory": r"(?i)/(?:users|home)/[^/\s`]+(?:/|\\)",
        "UNC path": r"\\\\[a-z0-9._$-]+[\\/]",
        "Windows absolute path": r"(?i)(?<![a-z])[a-z]:[\\/]",
    }
    for description, pattern in prohibited_patterns.items():
        assert re.search(pattern, public_text) is None, description


def test_markdown_and_machine_readable_evidence_agree() -> None:
    state = _load_state()
    markdown = LEGACY_AUDIT_PATH.read_text(encoding="utf-8")

    assert EXPECTED_HEAD in markdown
    assert "incomplete-local-only" in markdown
    for item in state["artifact_inventory"]:
        assert f"`{item['logical_path']}`" in markdown
        assert f"{item['file_count']:,}" in markdown
        assert f"{item['bytes']:,}" in markdown
    for run in state["reported_runs"]:
        assert f"{run['correct_test_predictions']} / {run['test_samples']}" in markdown
        assert f"{run['test_accuracy']:.4f}" in markdown
        assert f"{run['macro_f1']:.4f}" in markdown
