from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from scripts import verify_dvc_clean_room as clean_room

from signlab.reproducibility import DVC_VERSION
from signlab.reproducibility.stages import STAGE_NAMES

_COMMIT = "1" * 40
_DIGEST = "sha256:" + "a" * 64


def _digest_map() -> dict[str, str]:
    return {stage: f"sha256:{index:064x}" for index, stage in enumerate(STAGE_NAMES, 1)}


def _valid_report() -> dict[str, object]:
    output_hashes = _digest_map()
    return {
        "consent": clean_room.CONSENT_STATUS,
        "dvc_lock_sha256": _DIGEST,
        "dvc_snapshot_sha256": _DIGEST,
        "dvc_version": DVC_VERSION,
        "failed_phase": None,
        "fixture_only": True,
        "git_commit": _COMMIT,
        "phases": list(clean_room._PHASES),
        "producer_output_sha256": output_hashes,
        "pulled_output_sha256": dict(output_hashes),
        "schema_version": clean_room.REPORT_SCHEMA,
        "stage_lock_sha256": _digest_map(),
        "stage_names": list(STAGE_NAMES),
        "status": "passed",
    }


def test_environment_preserves_virtualenv_bin_without_resolving_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_bin = tmp_path / "managed-python" / "bin"
    virtualenv_bin = tmp_path / "project" / ".venv" / "bin"
    base_bin.mkdir(parents=True)
    virtualenv_bin.mkdir(parents=True)
    base_python = base_bin / "python"
    base_python.write_bytes(b"python")
    virtualenv_python = virtualenv_bin / "python"
    try:
        virtualenv_python.symlink_to(base_python)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    monkeypatch.setattr(sys, "executable", str(virtualenv_python))
    monkeypatch.setenv("PATH", "inherited-path")

    environment = clean_room._environment(tmp_path / "clone")

    first_path = environment["PATH"].split(os.pathsep)[0]
    assert first_path == str(virtualenv_bin)
    assert first_path != str(base_python.resolve().parent)


def test_environment_excludes_private_remote_and_provider_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_names = {
        "SIGNLAB_DVC_REMOTE_URL": "private-sentinel",
        "SIGNLAB_DVC_ENDPOINT_URL": "private-sentinel",
        "AWS_SECRET_ACCESS_KEY": "private-sentinel",
        "AZURE_CLIENT_SECRET": "private-sentinel",
        "GOOGLE_APPLICATION_CREDENTIALS": "private-sentinel",
        "DVC_STUDIO_TOKEN": "private-sentinel",
        "MLFLOW_TRACKING_TOKEN": "private-sentinel",
    }
    for name, value in private_names.items():
        monkeypatch.setenv(name, value)

    environment = clean_room._environment(tmp_path / "clone")

    assert not private_names.keys() & environment.keys()
    assert environment["DVC_NO_ANALYTICS"] == "true"
    assert environment["DVC_STUDIO_OFFLINE"] == "true"
    assert environment["PYTHONPATH"] == str(tmp_path / "clone" / "src")


def test_phase_failure_exposes_only_allowlisted_phase_and_completed_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "s3://private-bucket/private-path"
    completed: list[clean_room.Phase] = ["preflight"]

    def fail() -> None:
        raise RuntimeError(secret)

    with pytest.raises(clean_room.CleanRoomVerificationError) as captured:
        clean_room._phase("pull", completed, fail)

    assert captured.value.phase == "pull"
    assert captured.value.completed == ("preflight",)
    assert secret not in str(captured.value)
    assert capsys.readouterr().out == "Clean-room phase started: pull.\n"


def test_main_writes_a_sanitized_failure_report_and_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "private-person-raw-mov-sentinel"
    report_path = (tmp_path / "proof.json").resolve()

    def fail(_source: Path, _report: Path) -> dict[str, object]:
        try:
            raise RuntimeError(secret)
        except RuntimeError as error:
            raise clean_room.CleanRoomVerificationError(
                "pull",
                ("preflight", "clone-producer", "reproduce", "push", "clone-consumer"),
            ) from error

    monkeypatch.setattr(clean_room, "run_clean_room", fail)

    assert clean_room.main(["--report", str(report_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Clean-room verification failed during phase: pull.\n"
    report = json.loads(report_path.read_bytes())
    assert report == {
        "consent": "not_checked",
        "failed_phase": "pull",
        "fixture_only": True,
        "phases": ["preflight", "clone-producer", "reproduce", "push", "clone-consumer"],
        "schema_version": clean_room.REPORT_SCHEMA,
        "status": "failed",
    }
    serialized = report_path.read_text(encoding="utf-8")
    assert secret not in serialized
    assert str(tmp_path) not in serialized


def test_main_redacts_unexpected_exception_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "private-unexpected-sentinel"

    def fail(_source: Path, _report: Path) -> dict[str, object]:
        raise RuntimeError(secret)

    monkeypatch.setattr(clean_room, "run_clean_room", fail)
    report_path = (tmp_path / "proof.json").resolve()

    assert clean_room.main(["--report", str(report_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Clean-room verification failed during phase: preflight.\n"
    assert secret not in captured.err
    assert not report_path.exists()


def test_subprocess_failure_does_not_echo_command_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-subprocess-output"

    def fail(
        command: Sequence[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout=secret, stderr=secret)

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(RuntimeError, match=r"^command failed$") as captured:
        clean_room._run(("private-command",), cwd=tmp_path, environment={})

    assert secret not in str(captured.value)


def test_windows_dvc_lock_newlines_are_normalized_without_changing_content(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "dvc.lock"
    lock.write_bytes(b"schema: '2.0'\r\nstages: {}\r\n")

    clean_room._normalize_dvc_lock_newlines(tmp_path)

    assert lock.read_bytes() == b"schema: '2.0'\nstages: {}\n"


def test_success_report_is_canonical_and_requires_matching_outputs() -> None:
    report = _valid_report()

    payload = clean_room.canonical_report_bytes(report)

    assert payload.endswith(b"\n")
    assert json.loads(payload) == report
    assert clean_room.validate_report(json.loads(payload)) == report
    changed = dict(report)
    changed["pulled_output_sha256"] = {
        **_digest_map(),
        STAGE_NAMES[0]: "sha256:" + "f" * 64,
    }
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room.validate_report(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("git_commit", "not-a-commit"),
        ("dvc_version", "latest"),
        ("dvc_lock_sha256", "not-a-digest"),
        ("producer_output_sha256", []),
    ],
)
def test_success_report_rejects_invalid_identity_fields(
    field: str,
    value: object,
) -> None:
    report = _valid_report()
    report[field] = value
    if field == "producer_output_sha256":
        report["pulled_output_sha256"] = value

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room.validate_report(report)


def test_help_and_invalid_arguments_are_stable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert clean_room.main(["--help"]) == 0
    help_output = capsys.readouterr()
    assert help_output.out == clean_room._usage()
    assert help_output.err == ""

    assert clean_room.main([]) == 2
    invalid_output = capsys.readouterr()
    assert invalid_output.out == ""
    assert invalid_output.err == clean_room._usage()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SIGNLAB_RUN_DVC_CLEAN_ROOM") != "1",
    reason="set SIGNLAB_RUN_DVC_CLEAN_ROOM=1 on a clean committed checkout",
)
def test_clean_committed_checkout_reproduces_from_an_empty_consumer(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "clean-room-report.json"

    report = clean_room.run_clean_room(repository, report_path)

    assert report_path.read_bytes() == clean_room.canonical_report_bytes(report)
    assert clean_room.validate_report(report)["status"] == "passed"
