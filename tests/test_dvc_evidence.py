from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from signlab.reproducibility import DVC_VERSION, evidence
from signlab.reproducibility.provenance import (
    PUBLIC_REPOSITORY,
    DvcMetadataRepositoryRole,
    build_dvc_snapshot,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_COMMIT = "1" * 40


def _copy_controls(target: Path) -> Path:
    for name in ("uv.lock", "dvc.yaml", "dvc.lock"):
        shutil.copy2(REPOSITORY_ROOT / name, target / name)
    return target.resolve()


def _install_successful_commands(
    monkeypatch: pytest.MonkeyPatch,
    *,
    origin: str = f"{PUBLIC_REPOSITORY}.git",
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def run(_repository: Path, command: Sequence[str]) -> str:
        call = tuple(command)
        calls.append(call)
        if call == ("git", "status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if call == ("git", "rev-parse", "--verify", "HEAD^{commit}"):
            return _COMMIT
        if call == ("git", "remote", "get-url", "origin"):
            return origin
        if call == (sys.executable, "-I", "-m", "dvc", "--version"):
            return DVC_VERSION
        if call == (sys.executable, "-I", "-m", "dvc", "status", "--json"):
            return "{}"
        raise AssertionError(call)

    monkeypatch.setattr(evidence, "_run", run)
    return calls


def test_capture_public_snapshot_requires_clean_git_dvc_and_public_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _copy_controls(tmp_path)
    calls = _install_successful_commands(monkeypatch)

    snapshot = evidence.capture_dvc_snapshot(
        repository,
        metadata_repository_role="public-fixture",
    )

    assert snapshot.metadata_repository_role == "public-fixture"
    assert snapshot.metadata_repository == PUBLIC_REPOSITORY
    assert snapshot.metadata_git_commit == _COMMIT
    assert snapshot.git_working_tree_clean is True
    assert snapshot.dvc_workspace_clean is True
    assert ("git", "remote", "get-url", "origin") in calls


def test_capture_protected_snapshot_checks_private_origin_but_does_not_record_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _copy_controls(tmp_path)
    private_origin = "ssh://git@private.example.test/signlab-metadata.git"
    calls = _install_successful_commands(monkeypatch, origin=private_origin)

    snapshot = evidence.capture_dvc_snapshot(
        repository,
        metadata_repository_role="protected-metadata",
    )

    assert snapshot.metadata_repository_role == "protected-metadata"
    assert snapshot.metadata_repository is None
    assert private_origin not in snapshot.model_dump_json()
    assert ("git", "remote", "get-url", "origin") in calls


@pytest.mark.parametrize(
    ("role", "origin"),
    [
        ("public-fixture", "ssh://git@private.example.test/signlab-metadata.git"),
        ("protected-metadata", f"{PUBLIC_REPOSITORY}.git"),
    ],
)
def test_capture_rejects_repository_role_origin_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: DvcMetadataRepositoryRole,
    origin: str,
) -> None:
    repository = _copy_controls(tmp_path)
    _install_successful_commands(monkeypatch, origin=origin)

    with pytest.raises(evidence.DvcEvidenceError):
        evidence.capture_dvc_snapshot(repository, metadata_repository_role=role)


@pytest.mark.parametrize(
    ("failing_command", "response"),
    [
        (
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            " M private-sentinel",
        ),
        ((sys.executable, "-I", "-m", "dvc", "--version"), "3.67.2"),
        ((sys.executable, "-I", "-m", "dvc", "status", "--json"), '{"changed": true}'),
    ],
)
def test_capture_rejects_dirty_or_unlocked_state_without_echoing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_command: tuple[str, ...],
    response: str,
) -> None:
    repository = _copy_controls(tmp_path)
    _install_successful_commands(monkeypatch)
    successful = evidence._run

    def fail_selected(root: Path, command: Sequence[str]) -> str:
        if tuple(command) == failing_command:
            return response
        return successful(root, command)

    monkeypatch.setattr(evidence, "_run", fail_selected)

    with pytest.raises(evidence.DvcEvidenceError) as captured:
        evidence.capture_dvc_snapshot(
            repository,
            metadata_repository_role="public-fixture",
        )

    assert "private-sentinel" not in str(captured.value)
    assert str(tmp_path) not in str(captured.value)


def test_subprocess_failure_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-command-output"

    def fail(
        command: Sequence[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout=secret, stderr=secret)

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(
        evidence.DvcEvidenceError,
        match=r"^provenance command failed$",
    ) as captured:
        evidence._run(tmp_path, ("git", "status"))

    assert secret not in str(captured.value)


def test_snapshot_writer_publishes_canonical_json_only_under_reproduction_reports(
    tmp_path: Path,
) -> None:
    repository = _copy_controls(tmp_path)
    snapshot = build_dvc_snapshot(
        repository,
        _COMMIT,
        metadata_repository_role="public-fixture",
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    )

    output = evidence.write_dvc_snapshot(
        snapshot,
        repository,
        "reports/reproduction/snapshot.json",
    )

    assert json.loads(output.read_bytes()) == snapshot.model_dump(mode="json", round_trip=True)
    assert output.read_bytes().endswith(b"\n")
    with pytest.raises(evidence.DvcEvidenceError, match="already exists"):
        evidence.write_dvc_snapshot(
            snapshot,
            repository,
            "reports/reproduction/snapshot.json",
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "snapshot.json",
        "reports/private/snapshot.json",
        "reports/reproduction/snapshot.txt",
        "reports/reproduction/../snapshot.json",
        "/reports/reproduction/snapshot.json",
    ],
)
def test_snapshot_writer_rejects_paths_outside_the_ignored_report_root(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository = _copy_controls(tmp_path)
    snapshot = build_dvc_snapshot(
        repository,
        _COMMIT,
        metadata_repository_role="protected-metadata",
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    )

    with pytest.raises(evidence.DvcEvidenceError, match="output path is invalid"):
        evidence.write_dvc_snapshot(snapshot, repository, relative_path)
