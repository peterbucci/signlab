"""Read-only verification of the public legacy-audit snapshot.

The legacy root is supplied by the operator and is never persisted. The verifier
only reads Git metadata, file metadata/content hashes, aggregate JSON, and SQLite
counts. It does not copy artifacts or print participant-level records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = REPOSITORY_ROOT / "docs" / "legacy" / "legacy-state.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _run_git(root: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_logical_path(root: Path, logical_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(logical_path)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Logical path escapes the legacy root: {logical_path}") from error
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_totals(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _open_immutable_database(database: Path) -> sqlite3.Connection:
    if database.with_name(f"{database.name}-wal").exists():
        raise ValueError("A database has an active WAL and cannot be inspected immutably.")
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.execute("PRAGMA query_only = ON")
    return connection


def _database_rows(database: Path, table: str) -> int:
    connection = _open_immutable_database(database)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def _database_group_counts(database: Path, column: str) -> dict[str, int]:
    connection = _open_immutable_database(database)
    try:
        rows = connection.execute(
            f"SELECT {column}, COUNT(*) FROM runs GROUP BY {column}"
        ).fetchall()
        return {str(key): int(count) for key, count in rows}
    finally:
        connection.close()


def _macro_f1(matrix: Sequence[Sequence[int]]) -> float:
    class_count = len(matrix)
    scores: list[float] = []
    for index in range(class_count):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(class_count)) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return sum(scores) / class_count


def _matrix_from_mapping(mapping: dict[str, Any], labels: Sequence[str]) -> list[list[int]]:
    return [[int(mapping[actual][predicted]) for predicted in labels] for actual in labels]


def _matrix_from_predictions(path: Path, labels: Sequence[str]) -> list[list[int]]:
    label_indexes = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"true_label", "pred_label"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Prediction evidence is missing required label columns.")
        for row in reader:
            try:
                true_index = label_indexes[row["true_label"]]
                predicted_index = label_indexes[row["pred_label"]]
            except KeyError as error:
                raise ValueError("Prediction evidence contains an unknown label.") from error
            if row.get("true_idx") not in (None, "", str(true_index)):
                raise ValueError("Prediction evidence contains an inconsistent true-label index.")
            if row.get("pred_idx") not in (None, "", str(predicted_index)):
                raise ValueError(
                    "Prediction evidence contains an inconsistent predicted-label index."
                )
            expected_correct = str(int(true_index == predicted_index))
            if row.get("correct") not in (None, "", expected_correct):
                raise ValueError("Prediction evidence contains an inconsistent correctness flag.")
            matrix[true_index][predicted_index] += 1
    return matrix


def _compare(errors: list[str], label: str, actual: object, expected: object) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _compare_float(
    errors: list[str], label: str, actual: float, expected: float, *, tolerance: float = 1e-12
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _verify_file(
    errors: list[str], legacy_root: Path, evidence: dict[str, Any], *, label: str
) -> None:
    path = _resolve_logical_path(legacy_root, str(evidence["logical_path"]))
    if not path.is_file():
        errors.append(f"{label}: source artifact is missing")
        return
    _compare(errors, f"{label} bytes", path.stat().st_size, evidence["bytes"])
    _compare(errors, f"{label} SHA-256", _sha256(path), evidence["sha256"])


def verify_legacy_root(legacy_root: Path, snapshot: dict[str, Any]) -> list[str]:
    """Return sanitized mismatch messages without mutating the legacy project."""
    errors: list[str] = []
    repository = snapshot["repository"]

    _compare(
        errors, "Git HEAD", _run_git(legacy_root, "rev-parse", "HEAD"), repository["head_commit"]
    )
    _compare(
        errors,
        "Git tree",
        _run_git(legacy_root, "rev-parse", "HEAD^{tree}"),
        repository["tree_object"],
    )
    _compare(
        errors,
        "Git branch",
        _run_git(legacy_root, "branch", "--show-current"),
        repository["branch"],
    )

    tracked = _run_git(legacy_root, "ls-files").splitlines()
    tracked_snapshot = repository["tracked_files"]
    _compare(errors, "tracked file count", len(tracked), tracked_snapshot["file_count"])
    _compare(
        errors,
        "tracked Python source count",
        sum(path.endswith(".py") for path in tracked),
        tracked_snapshot["python_source_files"],
    )
    _compare(
        errors,
        "tracked bytecode count",
        sum(path.endswith(".pyc") for path in tracked),
        tracked_snapshot["bytecode_files"],
    )

    status = _run_git(legacy_root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    _compare(
        errors,
        "modified worktree entries",
        sum(not line.startswith("??") for line in status),
        repository["worktree"]["modified_files"],
    )
    _compare(
        errors,
        "untracked worktree entries",
        sum(line.startswith("??") for line in status),
        repository["worktree"]["untracked_files"],
    )

    for item in snapshot["artifact_inventory"]:
        logical_path = str(item["logical_path"])
        source = _resolve_logical_path(legacy_root, logical_path)
        if not source.is_dir():
            errors.append(f"inventory {logical_path}: directory is missing")
            continue
        file_count, byte_count = _file_totals(source)
        _compare(errors, f"inventory {logical_path} files", file_count, item["file_count"])
        _compare(errors, f"inventory {logical_path} bytes", byte_count, item["bytes"])

    dataset = snapshot["dataset_snapshot"]
    raw_root = _resolve_logical_path(legacy_root, "data/raw_videos")
    for label, expected_count in dataset["labels"].items():
        label_root = raw_root / label
        actual_count = sum(path.is_file() for path in label_root.rglob("*"))
        _compare(errors, f"raw label {label!r}", actual_count, expected_count)

    negative = dataset["negative_material"]
    negative_root = _resolve_logical_path(legacy_root, "data/nothing_label")
    _compare(
        errors,
        "negative raw video files",
        _file_totals(negative_root / "raw_videos")[0],
        negative["raw_video_files"],
    )
    _compare(
        errors,
        "negative landmark files",
        _file_totals(negative_root / "landmarks")[0],
        negative["derived_landmark_files"],
    )

    for item in snapshot["integrity"]:
        _verify_file(errors, legacy_root, item, label=f"integrity {item['logical_path']}")

    live_state = snapshot["live_state"]
    runs_database = _resolve_logical_path(legacy_root, "data/models/runs.db")
    _compare(
        errors,
        "run database rows",
        _database_rows(runs_database, "runs"),
        live_state["run_database"]["runs"],
    )
    _compare(
        errors,
        "run database status counts",
        _database_group_counts(runs_database, "status"),
        live_state["run_database"]["by_status"],
    )
    _compare(
        errors,
        "run database model counts",
        _database_group_counts(runs_database, "model_key"),
        live_state["run_database"]["by_model"],
    )

    feedback_database = _resolve_logical_path(legacy_root, "data/models/live_feedback.db")
    feedback_snapshot = live_state["feedback_database"]
    for table, snapshot_key in (
        ("sessions", "sessions"),
        ("detections", "detections"),
        ("feedback", "feedback_records"),
    ):
        _compare(
            errors,
            f"feedback database {table} rows",
            _database_rows(feedback_database, table),
            feedback_snapshot[snapshot_key],
        )

    evaluation_database = _resolve_logical_path(legacy_root, "data/live_eval/live_eval.db")
    evaluation_snapshot = live_state["live_evaluation_database"]
    for table in ("attempts", "replay_results"):
        _compare(
            errors,
            f"live-evaluation database {table} rows",
            _database_rows(evaluation_database, table),
            evaluation_snapshot[table],
        )

    labels = snapshot["reported_label_order"]
    verified_run_ids: list[str] = []
    for run in snapshot["reported_runs"]:
        run_id = str(run["run_id"])
        verified_run_ids.append(run_id)
        run_prefix = f"data/results/{run_id}/"
        if not str(run["model_logical_path"]).startswith(run_prefix):
            errors.append(f"run {run_id}: model locator does not stay inside its run directory")
        _verify_file(
            errors,
            legacy_root,
            {
                "bytes": run["model_bytes"],
                "logical_path": run["model_logical_path"],
                "sha256": run["model_sha256"],
            },
            label=f"run {run_id} model",
        )
        for evidence in run["source_evidence"]:
            if not str(evidence["logical_path"]).startswith(run_prefix):
                errors.append(f"run {run_id}: source locator escapes its run directory")
            _verify_file(errors, legacy_root, evidence, label=f"run {run_id} source")

        run_root = _resolve_logical_path(legacy_root, run_prefix)
        label_map = _read_json(run_root / "label_map.json")
        actual_labels = [label_map[str(index)] for index in range(len(label_map))]
        _compare(errors, f"run {run_id} label order", actual_labels, labels)

        confusion_mapping = _read_json(run_root / "confusion_matrix.json")
        confusion_matrix = _matrix_from_mapping(confusion_mapping, labels)
        prediction_matrix = _matrix_from_predictions(run_root / "test_predictions.csv", labels)
        _compare(
            errors,
            f"run {run_id} prediction/confusion agreement",
            prediction_matrix,
            confusion_matrix,
        )
        _compare(
            errors,
            f"run {run_id} confusion matrix",
            confusion_matrix,
            run["confusion_matrix"],
        )

        matrix = run["confusion_matrix"]
        sample_count = sum(sum(row) for row in matrix)
        correct_count = sum(matrix[index][index] for index in range(len(matrix)))
        _compare(errors, f"run {run_id} sample count", sample_count, run["test_samples"])
        _compare(
            errors,
            f"run {run_id} correct predictions",
            correct_count,
            run["correct_test_predictions"],
        )
        _compare_float(
            errors,
            f"run {run_id} accuracy",
            correct_count / sample_count,
            run["test_accuracy"],
            tolerance=1e-7,
        )
        _compare_float(
            errors,
            f"run {run_id} macro-F1",
            _macro_f1(matrix),
            run["macro_f1"],
        )

        metrics = _read_json(run_root / "metrics.json")
        for source_key, snapshot_key in (
            ("train_accuracy", "train_accuracy"),
            ("val_accuracy", "validation_accuracy"),
            ("test_accuracy", "test_accuracy"),
        ):
            _compare_float(
                errors,
                f"run {run_id} {source_key}",
                float(metrics[source_key]),
                float(run[snapshot_key]),
                tolerance=1e-7,
            )
        for source_key, expected in (
            ("n_train_samples", dataset["reported_split"]["train_samples"]),
            ("n_val_samples", dataset["reported_split"]["validation_samples"]),
            ("n_test_samples", dataset["reported_split"]["test_samples"]),
        ):
            _compare(errors, f"run {run_id} {source_key}", metrics[source_key], expected)

    imports = _read_json(_resolve_logical_path(legacy_root, "data/live/live_imports.json"))
    _compare(errors, "live-imported run IDs", sorted(imports["runs"]), sorted(verified_run_ids))
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the SignLab legacy audit against an operator-supplied legacy root."
    )
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    legacy_root = arguments.legacy_root.resolve()
    if not legacy_root.is_dir():
        print("Legacy root is not a directory.", file=sys.stderr)
        return 2

    try:
        errors = verify_legacy_root(legacy_root, _read_json(arguments.snapshot))
    except (OSError, ValueError, KeyError, sqlite3.Error, subprocess.SubprocessError):
        print(
            "Verification could not complete; source evidence was unavailable or malformed.",
            file=sys.stderr,
        )
        return 2

    if errors:
        print("Legacy audit verification failed:", file=sys.stderr)
        for mismatch in errors:
            print(f"- {mismatch}", file=sys.stderr)
        return 1

    print("Legacy audit verified: Git state, aggregate counts, metrics, and hashes match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
