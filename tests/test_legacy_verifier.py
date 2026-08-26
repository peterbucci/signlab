from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from scripts import verify_legacy_audit as verifier


@dataclass(frozen=True)
class SyntheticLegacy:
    root: Path
    snapshot: dict[str, Any]
    snapshot_path: Path


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(root: Path, path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "logical_path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _create_database(path: Path, statements: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(statements)
        connection.commit()
    finally:
        connection.close()


def _filesystem_fingerprint(root: Path) -> tuple[tuple[str, ...], dict[str, tuple[int, int, str]]]:
    directories = tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir())
    )
    files = {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha256(path),
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    return directories, files


@pytest.fixture
def synthetic_legacy(tmp_path: Path) -> SyntheticLegacy:
    root = tmp_path / "operator-private-root"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(root, "config", "user.name", "Fixture Author")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _write_text(root / "tracked.py", "VALUE = 1\n")
    _git(root, "add", "--", "tracked.py")
    _git(root, "commit", "-m", "fixture")

    _write_text(root / "data/raw_videos/alpha/a.txt", "alpha\n")
    _write_text(root / "data/raw_videos/beta/b.txt", "beta\n")
    _write_text(root / "data/nothing_label/raw_videos/negative.txt", "negative\n")
    _write_text(root / "data/nothing_label/landmarks/negative.txt", "landmark\n")
    run_root = root / "data/results/run-001"
    _write_text(run_root / "sign_model_best.keras", "model fixture\n")
    _write_text(run_root / "label_map.json", '{"0": "alpha", "1": "beta"}\n')
    _write_text(
        run_root / "confusion_matrix.json",
        json.dumps(
            {"alpha": {"alpha": 1, "beta": 1}, "beta": {"alpha": 0, "beta": 1}},
            indent=2,
        )
        + "\n",
    )
    _write_text(
        run_root / "test_predictions.csv",
        "file,true_idx,true_label,pred_idx,pred_label,correct\n"
        "one,0,alpha,0,alpha,1\n"
        "two,0,alpha,1,beta,0\n"
        "three,1,beta,1,beta,1\n",
    )
    _write_text(
        run_root / "metrics.json",
        json.dumps(
            {
                "train_accuracy": 0.75,
                "val_accuracy": 0.5,
                "test_accuracy": 2 / 3,
                "n_train_samples": 4,
                "n_val_samples": 2,
                "n_test_samples": 3,
            },
            indent=2,
        )
        + "\n",
    )
    _write_text(root / "data/live/live_imports.json", '{"runs": ["run-001"]}\n')
    _write_text(root / "data/plans/plans.json", "{}\n")

    runs_database = root / "data/models/runs.db"
    _create_database(
        runs_database,
        """
        CREATE TABLE runs (status TEXT NOT NULL, model_key TEXT NOT NULL);
        INSERT INTO runs VALUES ('succeeded', 'fixture_model');
        """,
    )
    feedback_database = root / "data/models/live_feedback.db"
    _create_database(
        feedback_database,
        """
        CREATE TABLE sessions (id INTEGER);
        CREATE TABLE detections (id INTEGER);
        CREATE TABLE feedback (id INTEGER);
        INSERT INTO sessions VALUES (1);
        INSERT INTO detections VALUES (1);
        INSERT INTO feedback VALUES (1);
        """,
    )
    evaluation_database = root / "data/live_eval/live_eval.db"
    _create_database(
        evaluation_database,
        """
        CREATE TABLE attempts (id INTEGER);
        CREATE TABLE replay_results (id INTEGER);
        INSERT INTO attempts VALUES (1);
        """,
    )

    inventory_paths = (
        "data/raw_videos",
        "data/nothing_label",
        "data/results",
        "data/models",
        "data/live_eval",
        "data/live",
        "data/plans",
    )
    inventory = []
    for logical_path in inventory_paths:
        file_count, byte_count = verifier._file_totals(root / logical_path)
        inventory.append(
            {
                "bytes": byte_count,
                "file_count": file_count,
                "logical_path": logical_path,
                "policy": "synthetic fixture",
            }
        )

    tracked = _git(root, "ls-files").splitlines()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    matrix = [[1, 1], [0, 1]]
    source_files = (
        run_root / "test_predictions.csv",
        run_root / "confusion_matrix.json",
        run_root / "label_map.json",
        run_root / "metrics.json",
    )
    snapshot: dict[str, Any] = {
        "schema_version": 2,
        "repository": {
            "branch": _git(root, "branch", "--show-current"),
            "commit_timestamp": "fixture",
            "head_commit": _git(root, "rev-parse", "HEAD"),
            "identification_anchor": "fixture",
            "remotes": [],
            "recovery": {
                "artifact_locator": "fixture",
                "code_locator": "fixture",
                "limitation": "fixture",
                "status": "incomplete-local-only",
            },
            "tags": [],
            "tree_object": _git(root, "rev-parse", "HEAD^{tree}"),
            "tracked_files": {
                "bytecode_files": sum(path.endswith(".pyc") for path in tracked),
                "file_count": len(tracked),
                "python_source_files": sum(path.endswith(".py") for path in tracked),
            },
            "worktree": {
                "modified_files": sum(not line.startswith("??") for line in status),
                "untracked_files": sum(line.startswith("??") for line in status),
                "untracked_note": "fixture",
            },
        },
        "artifact_inventory": inventory,
        "dataset_snapshot": {
            "labels": {"alpha": 1, "beta": 1},
            "negative_material": {
                "derived_landmark_files": 1,
                "included_in_reported_five_class_runs": False,
                "raw_video_files": 1,
            },
            "reported_split": {
                "method": "fixture",
                "random_state": 1,
                "test_fraction": 1 / 3,
                "test_samples": 3,
                "train_fraction": 0.5,
                "train_samples": 4,
                "validation_fraction": 1 / 6,
                "validation_samples": 2,
            },
            "selected_representation": "fixture",
            "sequence_length": 1,
            "total_samples": 2,
        },
        "live_state": {
            "feedback_database": {"detections": 1, "feedback_records": 1, "sessions": 1},
            "live_evaluation_database": {"attempts": 1, "replay_results": 0},
            "run_database": {
                "by_model": {"fixture_model": 1},
                "by_status": {"succeeded": 1},
                "runs": 1,
            },
        },
        "reported_label_order": ["alpha", "beta"],
        "reported_runs": [
            {
                "confusion_matrix": matrix,
                "correct_test_predictions": 2,
                "live_imported": True,
                "macro_f1": verifier._macro_f1(matrix),
                "model_bytes": (run_root / "sign_model_best.keras").stat().st_size,
                "model_key": "fixture_model",
                "model_logical_path": "data/results/run-001/sign_model_best.keras",
                "model_sha256": _sha256(run_root / "sign_model_best.keras"),
                "run_id": "run-001",
                "source_evidence": [_evidence(root, path) for path in source_files],
                "test_accuracy": 2 / 3,
                "test_samples": 3,
                "train_accuracy": 0.75,
                "validation_accuracy": 0.5,
            }
        ],
        "integrity": [
            _evidence(root, root / "data/plans/plans.json"),
            _evidence(root, root / "data/live/live_imports.json"),
            _evidence(root, runs_database),
            _evidence(root, feedback_database),
            _evidence(root, evaluation_database),
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    _write_text(snapshot_path, json.dumps(snapshot, indent=2) + "\n")
    return SyntheticLegacy(root=root, snapshot=snapshot, snapshot_path=snapshot_path)


def test_full_verifier_succeeds_without_changing_source(
    synthetic_legacy: SyntheticLegacy,
) -> None:
    before = _filesystem_fingerprint(synthetic_legacy.root)

    errors = verifier.verify_legacy_root(synthetic_legacy.root, synthetic_legacy.snapshot)

    assert errors == []
    assert _filesystem_fingerprint(synthetic_legacy.root) == before


def test_verifier_detects_snapshot_and_prediction_mismatches(
    synthetic_legacy: SyntheticLegacy,
) -> None:
    wrong_snapshot = copy.deepcopy(synthetic_legacy.snapshot)
    wrong_snapshot["dataset_snapshot"]["labels"]["alpha"] = 2
    assert any(
        "raw label 'alpha'" in mismatch
        for mismatch in verifier.verify_legacy_root(synthetic_legacy.root, wrong_snapshot)
    )

    prediction_path = synthetic_legacy.root / "data/results/run-001/test_predictions.csv"
    _write_text(
        prediction_path,
        "file,true_idx,true_label,pred_idx,pred_label,correct\n"
        "one,0,alpha,0,alpha,1\n"
        "two,0,alpha,0,alpha,1\n"
        "three,1,beta,1,beta,1\n",
    )
    prediction_snapshot = copy.deepcopy(synthetic_legacy.snapshot)
    prediction_evidence = prediction_snapshot["reported_runs"][0]["source_evidence"][0]
    prediction_evidence.update(_evidence(synthetic_legacy.root, prediction_path))
    errors = verifier.verify_legacy_root(synthetic_legacy.root, prediction_snapshot)
    assert any("prediction/confusion agreement" in mismatch for mismatch in errors)


def test_verifier_rejects_traversal(synthetic_legacy: SyntheticLegacy) -> None:
    traversal_snapshot = copy.deepcopy(synthetic_legacy.snapshot)
    traversal_snapshot["integrity"][0]["logical_path"] = "../outside"

    with pytest.raises(ValueError, match="escapes the legacy root"):
        verifier.verify_legacy_root(synthetic_legacy.root, traversal_snapshot)


def test_cli_exit_codes_and_errors_redact_operator_root(
    synthetic_legacy: SyntheticLegacy,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        verifier.main(
            [
                "--legacy-root",
                str(synthetic_legacy.root),
                "--snapshot",
                str(synthetic_legacy.snapshot_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    mismatch = copy.deepcopy(synthetic_legacy.snapshot)
    mismatch["repository"]["head_commit"] = "f" * 40
    mismatch_path = tmp_path / "mismatch.json"
    _write_text(mismatch_path, json.dumps(mismatch))
    assert (
        verifier.main(
            [
                "--legacy-root",
                str(synthetic_legacy.root),
                "--snapshot",
                str(mismatch_path),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert str(synthetic_legacy.root) not in output.err

    traversal = copy.deepcopy(synthetic_legacy.snapshot)
    traversal["integrity"][0]["logical_path"] = "../outside"
    traversal_path = tmp_path / "traversal.json"
    _write_text(traversal_path, json.dumps(traversal))
    assert (
        verifier.main(
            [
                "--legacy-root",
                str(synthetic_legacy.root),
                "--snapshot",
                str(traversal_path),
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert str(synthetic_legacy.root) not in output.err
    assert "unavailable or malformed" in output.err

    missing_root = tmp_path / "private-missing-root"
    assert verifier.main(["--legacy-root", str(missing_root)]) == 2
    output = capsys.readouterr()
    assert str(missing_root) not in output.err
