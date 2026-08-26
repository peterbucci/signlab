"""Adversarial tests for the standalone synthetic DVC clean-room proof."""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from scripts import verify_dvc_clean_room as clean_room

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.reproducibility.provenance import DVC_VERSION, EXPECTED_DVC_STAGES
from signlab.reproducibility.stages import STAGE_REGISTRY

type JsonObject = dict[str, object]
type ReportMutation = Callable[[JsonObject], object]

_COMMIT = "a" * 40
_DIGEST = f"sha256:{'b' * 64}"


def _digest_map(offset: int = 0) -> dict[str, str]:
    return {
        stage: f"sha256:{index + offset:064x}"
        for index, stage in enumerate(EXPECTED_DVC_STAGES, start=1)
    }


def _valid_report() -> JsonObject:
    return {
        "schema_version": clean_room.REPORT_SCHEMA,
        "fixture_only": True,
        "git_commit": _COMMIT,
        "dvc_version": DVC_VERSION,
        "uv_lock_sha256": _DIGEST,
        "dvc_yaml_sha256": _DIGEST,
        "dvc_lock_sha256": _DIGEST,
        "dvc_snapshot_sha256": _DIGEST,
        "stage_names": list(EXPECTED_DVC_STAGES),
        "stage_lock_sha256": _digest_map(),
        "producer_output_sha256": _digest_map(10),
        "pulled_output_sha256": _digest_map(20),
        "offline_output_sha256": _digest_map(30),
        **{field: True for field in clean_room._REPORT_BOOLEAN_FIELDS},
        "consent": clean_room.CONSENT_STATUS,
    }


def _write_fixture_outputs(repository: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    upstream = f"sha256:{'0' * 64}"
    for spec in STAGE_REGISTRY:
        path = repository.joinpath(*spec.output_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "fixture_only": True,
            "implementation": "fixture-only/1",
            "payload": {"synthetic": spec.name},
            "profile": "public-synthetic-reproducibility",
            "schema_version": "synthetic-dvc-stage/1",
            "stage": spec.name,
            "upstream_sha256": upstream,
        }
        payload = canonical_json_bytes(document) + b"\n"
        path.write_bytes(payload)
        digest = clean_room._sha256(payload)
        hashes[spec.name] = digest
        upstream = digest
    return hashes


def test_sanitized_environment_excludes_credentials_and_private_dvc_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leaked_names = {
        "AWS_ACCESS_KEY_ID": "private-access",
        "AWS_SECRET_ACCESS_KEY": "private-secret",
        "AWS_SESSION_TOKEN": "private-session",
        "SIGNLAB_DVC_REMOTE_URL": "private-remote",
        "SIGNLAB_DVC_TOKEN": "private-token",
        "DVC_STUDIO_TOKEN": "private-studio-token",
        "DVC_EXP_GIT_REMOTE": "private-exp-remote",
        "MLFLOW_TRACKING_TOKEN": "private-mlflow-token",
    }
    for name, value in leaked_names.items():
        monkeypatch.setenv(name, value)
    clone = tmp_path / "clone"
    environment_home = tmp_path / "home"

    environment = clean_room._sanitized_environment(clone, environment_home)

    assert not leaked_names.keys() & environment.keys()
    assert environment["DVC_NO_ANALYTICS"] == "true"
    assert environment["DVC_GLOBAL_CONFIG_DIR"].startswith(str(environment_home))
    assert environment["DVC_SYSTEM_CONFIG_DIR"].startswith(str(environment_home))
    assert environment["DVC_SITE_CACHE_DIR"].startswith(str(environment_home))
    assert environment["DVC_STUDIO_OFFLINE"] == "true"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["PYTHONPATH"] == str(clone / "src")
    assert environment["HOME"] == str(environment_home)
    assert environment["LOGNAME"] == "signlab-clean-room"
    assert environment["USER"] == "signlab-clean-room"
    assert environment["USERNAME"] == "signlab-clean-room"
    assert environment["USERPROFILE"] == str(environment_home)
    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).resolve().parent)


def test_offline_environment_blocks_network_without_adding_remote_state(tmp_path: Path) -> None:
    base = clean_room._sanitized_environment(tmp_path / "clone", tmp_path / "home")

    environment = clean_room._offline_environment(base)

    assert environment["DVC_NO_ANALYTICS"] == "true"
    assert environment["HTTP_PROXY"] == "http://127.0.0.1:9"
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert environment["NO_PROXY"] == ""
    assert not any(name.startswith(("AWS_", "SIGNLAB_DVC_")) for name in environment)


def test_run_dvc_uses_current_python_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int = clean_room.COMMAND_TIMEOUT_SECONDS,
    ) -> str:
        del cwd, environment, timeout_seconds
        observed.append(tuple(command))
        return ""

    monkeypatch.setattr(clean_room, "_run", fake_run)

    clean_room._run_dvc(tmp_path, {}, "status", "--json")

    assert observed == [(sys.executable, "-I", "-m", "dvc", "status", "--json")]


def test_exact_stage_markers_are_required_in_canonical_order() -> None:
    output = "\n".join(
        f"Synthetic reproduction stage completed: {stage}." for stage in EXPECTED_DVC_STAGES
    )

    clean_room._require_all_stages_executed(output)

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._require_all_stages_executed(output.rsplit("\n", maxsplit=1)[0])
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._require_all_stages_executed(f"{output}\n{output.splitlines()[-1]}")
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._require_all_stages_executed("\n".join(reversed(output.splitlines())))


def test_stage_marker_failure_does_not_echo_subprocess_output() -> None:
    secret = "/".join(("C:", "Users", "private", "person", "raw.mov"))
    with pytest.raises(clean_room.CleanRoomVerificationError) as captured:
        clean_room._require_all_stages_executed(secret)

    assert secret not in str(captured.value)


def test_dvc_status_requires_a_json_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def changed_status(
        repository: Path,
        environment: Mapping[str, str],
        *arguments: str,
    ) -> str:
        del repository, environment, arguments
        return '{"changed": [{"path": "synthetic"}]}'

    monkeypatch.setattr(clean_room, "_run_dvc", changed_status)
    assert not clean_room._dvc_is_clean(tmp_path, {})

    def malformed_status(
        repository: Path,
        environment: Mapping[str, str],
        *arguments: str,
    ) -> str:
        del repository, environment, arguments
        return "[]"

    monkeypatch.setattr(clean_room, "_run_dvc", malformed_status)
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._dvc_status_document(tmp_path, {})


def test_control_file_normalization_accepts_only_uniform_utf8_line_endings() -> None:
    expected = b"schema: '2.0'\nstages: {}\n"

    assert clean_room._normalized_control_bytes(expected) == expected
    assert clean_room._normalized_control_bytes(expected.replace(b"\n", b"\r\n")) == expected

    for invalid in (
        b"schema: 2.0\r\nstages: {}\n",
        b"schema: 2.0\r",
        b"\xef\xbb\xbfschema: 2.0\n",
        b"schema: \0\n",
        b"schema: \xff\n",
    ):
        with pytest.raises(clean_room.CleanRoomVerificationError):
            clean_room._normalized_control_bytes(invalid)


def test_fixture_outputs_are_canonical_synthetic_and_sha256_identified(tmp_path: Path) -> None:
    expected = _write_fixture_outputs(tmp_path)

    actual = clean_room._fixture_output_hashes(tmp_path)

    assert actual == expected
    assert tuple(actual) == EXPECTED_DVC_STAGES


def test_fixture_output_rejects_noncanonical_json(tmp_path: Path) -> None:
    _write_fixture_outputs(tmp_path)
    first = STAGE_REGISTRY[0]
    path = tmp_path.joinpath(*first.output_path.split("/"))
    document = parse_json_object(path.read_bytes())
    path.write_text(
        "{\n" + ",\n".join(f'  "{key}": {value!r}' for key, value in document.items()) + "\n}",
        encoding="utf-8",
    )

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._fixture_output_hashes(tmp_path)


def test_fixture_output_rejects_a_symlink(tmp_path: Path) -> None:
    _write_fixture_outputs(tmp_path)
    first = STAGE_REGISTRY[0]
    path = tmp_path.joinpath(*first.output_path.split("/"))
    target = tmp_path / "outside.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._fixture_output_hashes(tmp_path)


def test_fixture_output_rejects_a_hardlink(tmp_path: Path) -> None:
    _write_fixture_outputs(tmp_path)
    first = STAGE_REGISTRY[0]
    path = tmp_path.joinpath(*first.output_path.split("/"))
    target = tmp_path / "outside.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        os.link(target, path)
    except OSError:
        pytest.skip("hardlinks are unavailable on this platform")

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._fixture_output_hashes(tmp_path)


def test_consumer_empty_state_detects_cache_and_outputs(tmp_path: Path) -> None:
    (tmp_path / ".dvc").mkdir()

    assert clean_room._consumer_state_is_empty(tmp_path) == (True, True)

    cache = tmp_path / ".dvc" / "cache"
    cache.mkdir()
    (cache / "object").write_text("cached", encoding="utf-8")
    first_output = tmp_path.joinpath(*STAGE_REGISTRY[0].output_path.split("/"))
    first_output.parent.mkdir(parents=True)
    first_output.write_text("generated", encoding="utf-8")

    assert clean_room._consumer_state_is_empty(tmp_path) == (False, False)


def test_validated_consumer_deletion_is_confined_to_temporary_workspace(tmp_path: Path) -> None:
    repository = tmp_path / "consumer"
    repository.mkdir()
    cache = repository / ".dvc" / "cache"
    cache.mkdir(parents=True)
    (cache / "object").write_text("cached", encoding="utf-8")
    _write_fixture_outputs(repository)
    outside = tmp_path.parent / f"{tmp_path.name}-sentinel"
    outside.write_text("keep", encoding="utf-8")
    try:
        clean_room._delete_validated_consumer_state(repository, tmp_path)

        assert not cache.exists()
        assert all(
            not repository.joinpath(*spec.output_path.split("/")).exists()
            for spec in STAGE_REGISTRY
        )
        assert outside.read_text(encoding="utf-8") == "keep"
    finally:
        outside.unlink(missing_ok=True)


def test_consumer_deletion_rejects_repository_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._delete_validated_consumer_state(outside, workspace)


def test_consumer_deletion_rejects_cache_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "consumer"
    dvc_directory = repository / ".dvc"
    dvc_directory.mkdir(parents=True)
    outside_cache = tmp_path / "outside-cache"
    outside_cache.mkdir()
    cache = dvc_directory / "cache"
    try:
        cache.symlink_to(outside_cache, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._delete_validated_consumer_state(repository, tmp_path)


def test_consumer_deletion_rejects_nested_cache_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "consumer"
    cache = repository / ".dvc" / "cache"
    cache.mkdir(parents=True)
    outside_cache = tmp_path / "outside-cache-object"
    outside_cache.write_text("keep", encoding="utf-8")
    try:
        (cache / "object").symlink_to(outside_cache)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._delete_validated_consumer_state(repository, tmp_path)
    assert outside_cache.read_text(encoding="utf-8") == "keep"


def test_report_is_canonical_allowlisted_and_path_free() -> None:
    report = _valid_report()

    first = clean_room.canonical_report_bytes(report)
    reparsed = parse_json_object(first)
    second = clean_room.canonical_report_bytes(reparsed)

    assert first == second
    assert first == canonical_json_bytes(reparsed)
    assert not first.endswith(b"\n")
    assert b"not_checked" in first
    assert b"://" not in first
    assert b"md5" not in first.lower()
    assert b"\\\\" not in first
    assert b"/".join((b"C:", b"Users", b"")) not in first


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update({"unexpected": True}),
        lambda report: report.update({"fixture_only": False}),
        lambda report: report.update({"git_commit": "/".join(("C:", "Users", "private"))}),
        lambda report: report.update({"dvc_version": "latest"}),
        lambda report: report.update({"consent": "verified"}),
        lambda report: report.update({"stage_names": list(reversed(EXPECTED_DVC_STAGES))}),
        lambda report: report.update({"dvc_lock_sha256": "0" * 32}),
        lambda report: cast(dict[str, str], report["stage_lock_sha256"]).pop("feature"),
        lambda report: cast(dict[str, str], report["producer_output_sha256"]).__setitem__(
            "ingest", "md5:private"
        ),
        lambda report: report.update({clean_room._REPORT_BOOLEAN_FIELDS[0]: False}),
    ],
)
def test_report_rejects_untruthful_or_nonportable_fields(
    mutation: ReportMutation,
) -> None:
    report = copy.deepcopy(_valid_report())
    mutation(report)

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room.validate_report(report)


def test_report_digest_maps_do_not_depend_on_json_key_order() -> None:
    report = _valid_report()
    digest_map = cast(dict[str, str], report["stage_lock_sha256"])
    report["stage_lock_sha256"] = dict(reversed(tuple(digest_map.items())))

    validated = clean_room.validate_report(report)

    assert tuple(cast(dict[str, str], validated["stage_lock_sha256"])) == EXPECTED_DVC_STAGES


def test_report_target_must_be_new_absolute_json_outside_source_and_sandbox(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    sandbox = tmp_path / "sandbox"
    reports = tmp_path / "reports"
    source.mkdir()
    sandbox.mkdir()
    reports.mkdir()
    target = reports / "proof.json"

    assert clean_room._prepare_report_path(target, source, sandbox) == target

    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._prepare_report_path(Path("relative.json"), source, sandbox)
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._prepare_report_path(source / "proof.json", source, sandbox)
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._prepare_report_path(sandbox / "proof.json", source, sandbox)
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._prepare_report_path(reports / "proof.txt", source, sandbox)
    target.write_text("existing", encoding="utf-8")
    with pytest.raises(clean_room.CleanRoomVerificationError):
        clean_room._prepare_report_path(target, source, sandbox)


def test_main_sanitizes_internal_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    secret = "/".join(("C:", "Users", "private", "person", "private.mov"))

    def fail(repository: Path, report: Path) -> JsonObject:
        del repository, report
        raise RuntimeError(secret)

    monkeypatch.setattr(clean_room, "run_clean_room", fail)

    assert clean_room.main(["--report", str(tmp_path / "proof.json")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Clean-room verification failed.\n"
    assert secret not in captured.err


def test_main_help_and_invalid_arguments_are_path_free(
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
def test_clean_committed_checkout_reproduces_without_the_remote(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "clean-room-report.json"

    report = clean_room.run_clean_room(repository, report_path)

    assert report_path.read_bytes() == clean_room.canonical_report_bytes(report)
    assert clean_room.validate_report(report)["consent"] == "not_checked"
