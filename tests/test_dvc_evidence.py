from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from signlab.reproducibility import evidence
from signlab.reproducibility.provenance import (
    DVC_VERSION,
    DvcMetadataRepositoryRole,
    DvcSnapshotV1,
    build_dvc_snapshot,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_COMMIT = "1" * 40


def _copy_controls(target: Path) -> Path:
    for name in ("uv.lock", "dvc.yaml", "dvc.lock"):
        shutil.copy2(REPOSITORY_ROOT / name, target / name)
    return target.resolve()


def _successful_run(
    arguments: list[str],
    **_kwargs: object,
) -> subprocess.CompletedProcess[str]:
    cwd = _kwargs["cwd"]
    assert isinstance(cwd, Path)
    if arguments == ["git", "rev-parse", "--show-toplevel"]:
        stdout = str(cwd.resolve()) + "\n"
    elif arguments == ["git", "remote", "get-url", "origin"]:
        stdout = "https://github.com/peterbucci/signlab.git\n"
    elif arguments[:2] == ["git", "rev-parse"]:
        stdout = _COMMIT + "\n"
    elif arguments[:2] == ["git", "show"]:
        name = arguments[-1].split(":", maxsplit=1)[1]
        stdout = (cwd / name).read_text(encoding="utf-8")
    elif arguments[:2] == ["git", "status"]:
        stdout = ""
    elif arguments[-1] == "--version":
        stdout = DVC_VERSION + "\n"
    elif arguments[-2:] == ["config", "cache.type"]:
        stdout = "reflink,copy\n"
    elif arguments[-2:] == ["status", "--json"]:
        stdout = "{}\n"
    else:  # pragma: no cover - makes unexpected subprocess expansion obvious.
        raise AssertionError(arguments)
    return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")


def test_capture_binds_two_matching_clean_git_and_dvc_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_controls(tmp_path)
    calls: list[list[str]] = []

    def record_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        assert kwargs["shell"] is False
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["DVC_NO_ANALYTICS"] == "true"
        return _successful_run(arguments, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_run)

    snapshot = evidence.capture_dvc_snapshot(
        root,
        metadata_repository_role="public-fixture",
    )

    assert snapshot.metadata_git_commit == _COMMIT
    assert snapshot.git_working_tree_clean is True
    assert snapshot.dvc_workspace_clean is True
    assert len(calls) == 17
    assert calls.count(["git", "rev-parse", "--verify", "HEAD"]) == 2
    assert all(call[1:4] == ["-I", "-m", "dvc"] for call in calls if call[0] == sys.executable)


def test_capture_records_a_protected_metadata_commit_without_its_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_controls(tmp_path)

    def protected_run(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result = _successful_run(arguments, **kwargs)
        if arguments == ["git", "remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout="ssh://git@private.example.test/signlab-metadata.git\n",
                stderr="",
            )
        return result

    monkeypatch.setattr(subprocess, "run", protected_run)

    snapshot = evidence.capture_dvc_snapshot(
        root,
        metadata_repository_role="protected-metadata",
    )

    assert snapshot.metadata_repository_role == "protected-metadata"
    assert snapshot.metadata_repository is None
    assert snapshot.metadata_git_commit == _COMMIT


@pytest.mark.parametrize(
    ("role", "origin"),
    [
        ("public-fixture", "ssh://git@private.example.test/signlab-metadata.git"),
        ("protected-metadata", "https://github.com/peterbucci/signlab.git"),
    ],
)
def test_capture_rejects_a_repository_role_that_does_not_match_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: DvcMetadataRepositoryRole,
    origin: str,
) -> None:
    root = _copy_controls(tmp_path)

    def mismatched_origin(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result = _successful_run(arguments, **kwargs)
        if arguments == ["git", "remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=f"{origin}\n",
                stderr="",
            )
        return result

    monkeypatch.setattr(subprocess, "run", mismatched_origin)

    with pytest.raises(evidence.DvcEvidenceError):
        evidence.capture_dvc_snapshot(
            root,
            metadata_repository_role=role,
        )


@pytest.mark.parametrize(
    ("command_suffix", "stdout"),
    [
        (("git", "status"), " M private-sentinel\n"),
        (("--version",), "3.67.2\n"),
        (("config", "cache.type"), "hardlink\n"),
        (("status", "--json"), '{"changed": ["private-sentinel"]}\n'),
    ],
)
def test_capture_rejects_dirty_or_incompatible_state_without_echoing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_suffix: tuple[str, ...],
    stdout: str,
) -> None:
    root = _copy_controls(tmp_path)

    def seeded_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = _successful_run(arguments, **kwargs)
        matches = (
            arguments[:2] == list(command_suffix)
            if command_suffix == ("git", "status")
            else tuple(arguments[-len(command_suffix) :]) == command_suffix
        )
        if matches:
            return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")
        return result

    monkeypatch.setattr(subprocess, "run", seeded_run)

    with pytest.raises(evidence.DvcEvidenceError) as raised:
        evidence.capture_dvc_snapshot(
            root,
            metadata_repository_role="public-fixture",
        )

    assert raised.value.code == "dvc.evidence.capture.invalid"
    assert "private" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_capture_rejects_non_json_dvc_status_and_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_controls(tmp_path)

    def invalid_status(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = _successful_run(arguments, **kwargs)
        if arguments[-2:] == ["status", "--json"]:
            return subprocess.CompletedProcess(arguments, 0, stdout="private", stderr="")
        return result

    monkeypatch.setattr(subprocess, "run", invalid_status)
    with pytest.raises(evidence.DvcEvidenceError):
        evidence.capture_dvc_snapshot(
            root,
            metadata_repository_role="public-fixture",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("private", 1)),
    )
    with pytest.raises(evidence.DvcEvidenceError, match="could not be captured safely"):
        evidence.capture_dvc_snapshot(
            root,
            metadata_repository_role="public-fixture",
        )


def test_capture_detects_state_change_between_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_controls(tmp_path)
    monkeypatch.setattr(subprocess, "run", _successful_run)
    first = build_dvc_snapshot(
        root,
        _COMMIT,
        metadata_repository_role="public-fixture",
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    )
    second = first.model_copy(update={"metadata_git_commit": "2" * 40})
    snapshots: list[DvcSnapshotV1] = [first, first, second]
    monkeypatch.setattr(evidence, "build_dvc_snapshot", lambda *_args, **_kwargs: snapshots.pop(0))

    with pytest.raises(evidence.DvcEvidenceError):
        evidence.capture_dvc_snapshot(
            root,
            metadata_repository_role="public-fixture",
        )


def test_evidence_environment_removes_cloud_and_private_remote_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "private-sentinel")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "private-sentinel")
    monkeypatch.setenv("DVC_CACHE_TYPE", "hardlink")
    monkeypatch.setenv("GIT_DIR", "private-sentinel")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "private-sentinel")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "private-sentinel")
    monkeypatch.setenv("PYTHONPATH", "private-sentinel")
    monkeypatch.setenv("SIGNLAB_DVC_REMOTE_URL", "private-sentinel")
    monkeypatch.setenv("UNRELATED_SAFE_SETTING", "retained")

    environment = evidence._safe_environment(tmp_path / "isolated-home")

    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "AZURE_CLIENT_SECRET" not in environment
    assert "DVC_CACHE_TYPE" not in environment
    assert "GIT_DIR" not in environment
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment
    assert "MLFLOW_TRACKING_TOKEN" not in environment
    assert "PYTHONPATH" not in environment
    assert "SIGNLAB_DVC_REMOTE_URL" not in environment
    assert environment["DVC_EXP_AUTO_PUSH"] == "false"
    assert environment["DVC_STUDIO_OFFLINE"] == "true"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "UNRELATED_SAFE_SETTING" not in environment
    assert environment["HOME"] == str(tmp_path / "isolated-home")


def test_capture_rejects_a_nested_or_mismatched_git_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_controls(tmp_path)

    def mismatched_root(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result = _successful_run(arguments, **kwargs)
        if arguments == ["git", "rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=str(root.parent.resolve()) + "\n",
                stderr="",
            )
        return result

    monkeypatch.setattr(subprocess, "run", mismatched_root)

    with pytest.raises(evidence.DvcEvidenceError):
        evidence.capture_dvc_snapshot(
            root,
            metadata_repository_role="public-fixture",
        )


def test_capture_rejects_controls_that_do_not_match_head_even_when_git_is_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_controls(tmp_path)

    def mismatched_blob(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result = _successful_run(arguments, **kwargs)
        if arguments[:2] == ["git", "show"] and arguments[-1].endswith(":uv.lock"):
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=result.stdout + "# committed-drift\n",
                stderr="",
            )
        return result

    monkeypatch.setattr(subprocess, "run", mismatched_blob)

    with pytest.raises(evidence.DvcEvidenceError):
        evidence.capture_dvc_snapshot(
            root,
            metadata_repository_role="public-fixture",
        )


def test_snapshot_writer_publishes_one_canonical_ignored_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_controls(tmp_path)
    monkeypatch.setattr(subprocess, "run", _successful_run)
    snapshot = evidence.capture_dvc_snapshot(
        root,
        metadata_repository_role="public-fixture",
    )

    output = evidence.write_dvc_snapshot(
        snapshot,
        root,
        "reports/reproduction/public-snapshot.json",
    )

    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed == snapshot.model_dump(mode="json", round_trip=True)
    assert output.read_bytes().endswith(b"\n")
    with pytest.raises(evidence.DvcEvidenceError):
        evidence.write_dvc_snapshot(
            snapshot,
            root,
            "reports/reproduction/public-snapshot.json",
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "snapshot.json",
        "reports/private/snapshot.json",
        "reports/reproduction/snapshot.txt",
        "reports/reproduction/../snapshot.json",
    ],
)
def test_snapshot_writer_rejects_destinations_outside_the_ignored_report_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    root = _copy_controls(tmp_path)
    monkeypatch.setattr(subprocess, "run", _successful_run)
    snapshot = evidence.capture_dvc_snapshot(
        root,
        metadata_repository_role="public-fixture",
    )

    with pytest.raises(evidence.DvcEvidenceError):
        evidence.write_dvc_snapshot(snapshot, root, relative_path)
