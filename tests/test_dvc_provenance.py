"""Adversarial tests for the standalone DVC provenance boundary."""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256
from signlab.reproducibility import provenance as provenance_module
from signlab.reproducibility.provenance import (
    DVC_VERSION,
    EXPECTED_DVC_STAGES,
    MAX_CONTROL_FILE_BYTES,
    DvcProvenanceError,
    DvcSnapshotV1,
    build_dvc_snapshot,
    dvc_snapshot_digest,
    dvc_snapshot_mlflow_projection,
    parse_dvc_lock,
    validate_dvc_snapshot,
)

_COMMIT = "1" * 40
_SLASH = chr(47)
_WINDOWS_PRIVATE_PERSON = f"{chr(67)}:{_SLASH}Users{_SLASH}private{_SLASH}person"
_POSIX_PRIVATE_PERSON = f"{_SLASH}home{_SLASH}private{_SLASH}person"
_WINDOWS_PRIVATE_FILE = f"{chr(67)}:{_SLASH}Users{_SLASH}private{_SLASH}file"
_POSIX_PRIVATE_FILE = f"{_SLASH}home{_SLASH}private{_SLASH}file"
_PRIVATE_FILE_URI = f"file:{_SLASH}{_SLASH}{_SLASH}private{_SLASH}file"
type JsonMapping = dict[str, object]
type PayloadMutation = Callable[[JsonMapping], None]


def _md5(number: int, *, directory: bool = False) -> str:
    suffix = ".dir" if directory else ""
    return f"{number:032x}{suffix}"


def _artifact(
    path: str,
    number: int,
    *,
    size: int,
    file_count: int | None = None,
) -> JsonMapping:
    value: JsonMapping = {
        "path": path,
        "hash": "md5",
        "md5": _md5(number, directory=file_count is not None),
        "size": size,
    }
    if file_count is not None:
        value["nfiles"] = file_count
    return value


def _lock_payload() -> JsonMapping:
    stages: JsonMapping = {}
    previous = _artifact("configs/pipeline/synthetic-dvc.yaml", 1, size=128)
    output_paths = {
        "ingest": "data/interim/ingest",
        "validate": "data/interim/validated",
        "extract": "data/interim/extracted",
        "quality": "data/interim/quality",
        "split": "data/splits/synthetic",
        "feature": "data/processed/features",
    }
    for index, stage_name in enumerate(EXPECTED_DVC_STAGES, start=1):
        output = _artifact(
            output_paths[stage_name],
            100 + index,
            size=1_000 + index,
            file_count=index,
        )
        stages[stage_name] = {
            "cmd": f"uv run --locked signlab data run-stage {stage_name}",
            "deps": [copy.deepcopy(previous)],
            "params": {
                "configs/pipeline/synthetic-dvc.yaml": {
                    "fixture.profile": "synthetic",
                    "fixture.seed": 17,
                }
            },
            "outs": [copy.deepcopy(output)],
        }
        previous = output
    return {"schema": "2.0", "stages": stages}


def _render_lock(payload: JsonMapping | None = None) -> str:
    return yaml.safe_dump(
        payload or _lock_payload(),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def _stages(payload: JsonMapping) -> JsonMapping:
    return cast(JsonMapping, payload["stages"])


def _stage(payload: JsonMapping, stage_name: str = "ingest") -> JsonMapping:
    return cast(JsonMapping, _stages(payload)[stage_name])


def _artifacts(
    payload: JsonMapping,
    field_name: str,
    stage_name: str = "ingest",
) -> list[JsonMapping]:
    return cast(list[JsonMapping], _stage(payload, stage_name)[field_name])


def _write_repository(root: Path, lock_text: str | None = None) -> dict[str, bytes]:
    values = {
        "uv.lock": b"version = 1\n",
        "dvc.yaml": b"stages: {}\n",
        "dvc.lock": (lock_text or _render_lock()).encode("utf-8"),
    }
    for name, value in values.items():
        (root / name).write_bytes(value)
    return values


def _build(root: Path) -> DvcSnapshotV1:
    return build_dvc_snapshot(
        root,
        _COMMIT,
        metadata_repository_role="public-fixture",
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    )


def _assert_lock_invalid(document: str | bytes | bytearray) -> None:
    with pytest.raises(DvcProvenanceError) as captured:
        parse_dvc_lock(document)
    assert captured.value.category == "lock.invalid"
    assert str(captured.value) == "DVC lock metadata is invalid or unsupported"


def test_build_snapshot_records_exact_control_and_stage_identities(tmp_path: Path) -> None:
    values = _write_repository(tmp_path)

    snapshot = _build(tmp_path)

    assert snapshot.schema_version == "dvc-snapshot/1"
    assert snapshot.metadata_repository_role == "public-fixture"
    assert snapshot.metadata_repository == "https://github.com/peterbucci/signlab"
    assert snapshot.metadata_git_commit == _COMMIT
    assert snapshot.git_working_tree_clean is True
    assert snapshot.dvc_workspace_clean is True
    assert snapshot.dvc_version == DVC_VERSION == "3.67.1"
    assert snapshot.uv_lock_sha256 == f"sha256:{hashlib.sha256(values['uv.lock']).hexdigest()}"
    assert snapshot.dvc_yaml_sha256 == (f"sha256:{hashlib.sha256(values['dvc.yaml']).hexdigest()}")
    assert snapshot.dvc_lock_sha256 == (f"sha256:{hashlib.sha256(values['dvc.lock']).hexdigest()}")
    assert tuple(stage.stage_name for stage in snapshot.stages) == EXPECTED_DVC_STAGES

    ingest = snapshot.stages[0]
    assert ingest.dependencies[0].path == "configs/pipeline/synthetic-dvc.yaml"
    assert ingest.dependencies[0].hash_type == "md5"
    assert ingest.dependencies[0].hash_value == _md5(1)
    assert ingest.dependencies[0].size_bytes == 128
    assert ingest.dependencies[0].file_count is None
    assert ingest.outputs[0].path == "data/interim/ingest"
    assert ingest.outputs[0].hash_value == _md5(101, directory=True)
    assert ingest.outputs[0].file_count == 1
    assert dvc_snapshot_digest(snapshot) == dvc_snapshot_digest(canonical_json_bytes(snapshot))


def test_snapshot_model_does_not_advertise_an_unpublished_schema_resource() -> None:
    schema = DvcSnapshotV1.model_json_schema(mode="validation")

    assert "$id" not in schema
    assert "dvc-snapshot-1.schema.json" not in repr(schema)


def test_protected_metadata_snapshot_is_opaque_and_distinct(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    public = _build(tmp_path)

    protected = build_dvc_snapshot(
        tmp_path,
        _COMMIT,
        metadata_repository_role="protected-metadata",
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    )

    assert protected.metadata_repository_role == "protected-metadata"
    assert protected.metadata_repository is None
    assert protected.metadata_git_commit == _COMMIT
    assert dvc_snapshot_digest(protected) != dvc_snapshot_digest(public)
    projection = dvc_snapshot_mlflow_projection(protected)
    assert projection.tags["signlab.dvc.metadata_repository_role"] == "protected-metadata"
    assert "signlab.dvc.metadata_repository" not in projection.tags
    assert "github.com" not in repr((dict(projection.params), dict(projection.tags)))


def test_stage_identity_hashes_normalized_entry_with_stage_domain() -> None:
    parsed = parse_dvc_lock(_render_lock())
    expected_entry = {
        "cmd": "uv run --locked signlab data run-stage ingest",
        "deps": [
            {
                "path": "configs/pipeline/synthetic-dvc.yaml",
                "hash": "md5",
                "md5": _md5(1),
                "size": 128,
            }
        ],
        "params": {
            "configs/pipeline/synthetic-dvc.yaml": {
                "fixture.profile": "synthetic",
                "fixture.seed": 17,
            }
        },
        "outs": [
            {
                "path": "data/interim/ingest",
                "hash": "md5",
                "md5": _md5(101, directory=True),
                "size": 1_001,
                "nfiles": 1,
            }
        ],
    }

    assert parsed[0].lock_entry_sha256 == canonical_sha256(
        expected_entry,
        domain="dvc-stage-lock/1/ingest",
    )
    assert parsed[0].lock_entry_sha256 != canonical_sha256(
        expected_entry,
        domain="dvc-stage-lock/1/validate",
    )


def test_stage_digests_ignore_yaml_layout_but_change_only_for_changed_entry() -> None:
    original = _render_lock()
    layout_changed = f"\n{original.replace('stages:', 'stages:  ', 1)}"
    crlf = original.replace("\n", "\r\n")
    changed = original.replace(
        "run-stage validate",
        "run-stage validate --strict",
        1,
    )

    original_stages = parse_dvc_lock(original)
    assert parse_dvc_lock(layout_changed) == original_stages
    assert parse_dvc_lock(crlf) == original_stages
    changed_stages = parse_dvc_lock(changed)
    for before, after in zip(original_stages, changed_stages, strict=True):
        if before.stage_name == "validate":
            assert before.lock_entry_sha256 != after.lock_entry_sha256
        else:
            assert before.lock_entry_sha256 == after.lock_entry_sha256


def test_snapshot_control_hashes_are_portable_across_native_newlines(tmp_path: Path) -> None:
    values = _write_repository(tmp_path)
    expected = _build(tmp_path)
    for name, value in values.items():
        (tmp_path / name).write_bytes(value.replace(b"\n", b"\r\n"))

    actual = _build(tmp_path)

    assert actual == expected


def test_snapshot_reader_round_trips_revalidates_and_is_frozen(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    snapshot = _build(tmp_path)

    assert validate_dvc_snapshot(snapshot) == snapshot
    assert validate_dvc_snapshot(canonical_json_bytes(snapshot)) == snapshot
    assert validate_dvc_snapshot(snapshot.model_dump(mode="json")) == snapshot
    with pytest.raises(ValidationError):
        snapshot.dvc_version = "3.67.2"  # type: ignore[assignment]

    forged = snapshot.model_copy(update={"dvc_version": "3.67.2"})
    with pytest.raises(DvcProvenanceError, match="snapshot is invalid"):
        validate_dvc_snapshot(forged)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("schema_version", "dvc-snapshot/2"),
        ("metadata_repository_role", "private"),
        ("metadata_repository", "https://example.invalid/private"),
        ("metadata_git_commit", "f" * 39),
        ("git_working_tree_clean", False),
        ("dvc_workspace_clean", False),
        ("dvc_version", "3.67.2"),
        ("dvc_lock_sha256", "sha256:" + "A" * 64),
    ],
)
def test_snapshot_reader_rejects_identity_drift(
    tmp_path: Path,
    field_name: str,
    replacement: object,
) -> None:
    _write_repository(tmp_path)
    document = _build(tmp_path).model_dump(mode="json")
    document[field_name] = replacement

    with pytest.raises(DvcProvenanceError, match="snapshot is invalid"):
        validate_dvc_snapshot(document)


def test_snapshot_reader_rejects_repository_role_locator_mismatches(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    public = _build(tmp_path).model_dump(mode="json")
    public["metadata_repository_role"] = "protected-metadata"

    protected = build_dvc_snapshot(
        tmp_path,
        _COMMIT,
        metadata_repository_role="protected-metadata",
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    ).model_dump(mode="json")
    protected["metadata_repository"] = "https://github.com/peterbucci/signlab"

    for document in (public, protected):
        with pytest.raises(DvcProvenanceError, match="snapshot is invalid"):
            validate_dvc_snapshot(document)


def test_snapshot_reader_requires_exact_ordered_six_stage_inventory(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    document = _build(tmp_path).model_dump(mode="json")

    for stages in (
        list(reversed(cast(list[object], document["stages"]))),
        cast(list[object], document["stages"])[:-1],
        [*cast(list[object], document["stages"]), cast(list[object], document["stages"])[0]],
    ):
        changed = copy.deepcopy(document)
        changed["stages"] = stages
        with pytest.raises(DvcProvenanceError, match="snapshot is invalid"):
            validate_dvc_snapshot(changed)


def test_snapshot_reader_rejects_extra_and_duplicate_json_members(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    snapshot = _build(tmp_path)
    extra = snapshot.model_dump(mode="json")
    extra["remote_url"] = "private-sentinel"

    with pytest.raises(DvcProvenanceError) as extra_error:
        validate_dvc_snapshot(extra)
    assert "private-sentinel" not in str(extra_error.value)

    encoded = canonical_json_bytes(snapshot).decode("utf-8")
    duplicate = '{"schema_version":"dvc-snapshot/1",' + encoded[1:]
    with pytest.raises(DvcProvenanceError):
        validate_dvc_snapshot(duplicate)


def test_mlflow_projection_is_immutable_searchable_and_path_free(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    snapshot = _build(tmp_path)

    projection = dvc_snapshot_mlflow_projection(snapshot)

    assert projection.params["signlab.dvc.metadata_git_commit"] == _COMMIT
    assert projection.params["signlab.dvc.lock_sha256"] == snapshot.dvc_lock_sha256
    assert projection.tags == {
        "signlab.provenance.schema": "dvc-snapshot/1",
        "signlab.dvc.metadata_repository_role": "public-fixture",
        "signlab.dvc.metadata_git_commit": _COMMIT,
        "signlab.dvc.metadata_repository": "peterbucci/signlab",
        "signlab.dvc.version": "3.67.1",
        "signlab.dvc.reproducible": "true",
    }
    for stage in snapshot.stages:
        assert (
            projection.params[f"signlab.dvc.stage.{stage.stage_name}.sha256"]
            == stage.lock_entry_sha256
        )
    rendered = repr((dict(projection.params), dict(projection.tags))).casefold()
    for forbidden in ("data/", "configs/", "run-stage", "remote", "secret", "password"):
        assert forbidden not in rendered
    with pytest.raises(TypeError):
        cast(dict[str, str], projection.params)["new"] = "value"
    with pytest.raises(TypeError):
        cast(dict[str, str], projection.tags)["new"] = "value"


@pytest.mark.parametrize(
    "document",
    [
        "schema: '2.0'\nschema: '2.0'\nstages: {}\n",
        "schema: '2.0'\nstages: &all {}\ncopy: *all\n",
        "schema: '2.0'\nstages:\n  ingest:\n    <<: {cmd: safe}\n",
        "schema: '2.0'\nstages: !!set {ingest: null}\n",
    ],
    ids=["duplicate-key", "anchor-alias", "merge", "non-json-tag"],
)
def test_lock_parser_rejects_yaml_ambiguity(document: str) -> None:
    _assert_lock_invalid(document)


def _unknown_stage(payload: JsonMapping) -> None:
    stages = _stages(payload)
    stages["publish"] = stages.pop("feature")


def _missing_stage(payload: JsonMapping) -> None:
    del _stages(payload)["feature"]


def _unknown_root_field(payload: JsonMapping) -> None:
    payload["remote"] = "private-sentinel"


def _unknown_stage_field(payload: JsonMapping) -> None:
    _stage(payload)["wdir"] = "private-sentinel"


def _unknown_artifact_field(payload: JsonMapping) -> None:
    _artifacts(payload, "outs")[0]["etag"] = "private-sentinel"


def _numeric_schema(payload: JsonMapping) -> None:
    payload["schema"] = 2.0


@pytest.mark.parametrize(
    "mutation",
    [
        _unknown_stage,
        _missing_stage,
        _unknown_root_field,
        _unknown_stage_field,
        _unknown_artifact_field,
        _numeric_schema,
    ],
    ids=[
        "unknown-stage",
        "missing-stage",
        "unknown-root-field",
        "unknown-stage-field",
        "unknown-artifact-field",
        "numeric-schema",
    ],
)
def test_lock_parser_fails_closed_on_inventory_or_shape_drift(
    mutation: PayloadMutation,
) -> None:
    payload = _lock_payload()
    mutation(payload)

    with pytest.raises(DvcProvenanceError) as captured:
        parse_dvc_lock(_render_lock(payload))
    assert "private-sentinel" not in str(captured.value)


def test_lock_parser_allows_exact_interpreter_control_only_as_a_dependency() -> None:
    payload = _lock_payload()
    _artifacts(payload, "deps")[0]["path"] = ".python-version"

    parsed = parse_dvc_lock(_render_lock(payload))

    assert parsed[0].dependencies[0].path == ".python-version"

    output_control = _lock_payload()
    _artifacts(output_control, "outs")[0]["path"] = ".python-version"
    _assert_lock_invalid(_render_lock(output_control))

    parameter_control = _lock_payload()
    params = cast(dict[str, object], _stage(parameter_control)["params"])
    params[".python-version"] = params.pop("configs/pipeline/synthetic-dvc.yaml")
    _assert_lock_invalid(_render_lock(parameter_control))


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".python-version/child",
        "configs/.python-version",
        "..python-version",
    ],
)
def test_lock_parser_rejects_every_other_hidden_dependency_path(path: str) -> None:
    payload = _lock_payload()
    _artifacts(payload, "deps")[0]["path"] = path
    _assert_lock_invalid(_render_lock(payload))


@pytest.mark.parametrize(
    "path",
    [
        "../private",
        _POSIX_PRIVATE_PERSON,
        _WINDOWS_PRIVATE_PERSON,
        "data\\private\\person",
        "data/../private",
        "con/fixture",
        "data/private/",
    ],
)
def test_lock_parser_rejects_unsafe_dependency_and_output_paths(path: str) -> None:
    for field_name in ("deps", "outs"):
        payload = _lock_payload()
        _artifacts(payload, field_name)[0]["path"] = path
        _assert_lock_invalid(_render_lock(payload))


@pytest.mark.parametrize(
    "command",
    [
        f"uv run command {_WINDOWS_PRIVATE_FILE}",
        f"uv run command {_POSIX_PRIVATE_FILE}",
        f"uv run command {_PRIVATE_FILE_URI}",
        "uv run command data\\private\\file",
        "uv run command\nsecond-command",
    ],
)
def test_lock_parser_rejects_commands_with_machine_paths_or_controls(command: str) -> None:
    payload = _lock_payload()
    _stage(payload)["cmd"] = command
    _assert_lock_invalid(_render_lock(payload))


def test_lock_parser_rejects_case_collisions_and_cross_role_paths() -> None:
    case_collision = _lock_payload()
    duplicated = copy.deepcopy(_artifacts(case_collision, "deps")[0])
    duplicated["path"] = "CONFIGS/PIPELINE/SYNTHETIC-DVC.YAML"
    _artifacts(case_collision, "deps").append(duplicated)
    _assert_lock_invalid(_render_lock(case_collision))

    cross_role = _lock_payload()
    _artifacts(cross_role, "outs")[0]["path"] = _artifacts(cross_role, "deps")[0]["path"]
    _assert_lock_invalid(_render_lock(cross_role))

    global_collision = _lock_payload()
    _artifacts(global_collision, "outs", "validate")[0]["path"] = "DATA/INTERIM/INGEST"
    _assert_lock_invalid(_render_lock(global_collision))


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("hash", "etag"),
        ("md5", "A" * 32),
        ("md5", "0" * 31),
        ("size", -1),
        ("size", True),
        ("size", (2**53)),
        ("nfiles", -1),
        ("nfiles", True),
    ],
)
def test_lock_parser_rejects_invalid_native_hash_and_size_metadata(
    field_name: str,
    replacement: object,
) -> None:
    payload = _lock_payload()
    _artifacts(payload, "outs")[0][field_name] = replacement
    _assert_lock_invalid(_render_lock(payload))


def test_lock_parser_requires_directory_hash_and_file_count_together() -> None:
    missing_count = _lock_payload()
    del _artifacts(missing_count, "outs")[0]["nfiles"]
    _assert_lock_invalid(_render_lock(missing_count))

    count_on_file = _lock_payload()
    output = _artifacts(count_on_file, "outs")[0]
    output["md5"] = _md5(101)
    _assert_lock_invalid(_render_lock(count_on_file))


@pytest.mark.parametrize(
    "document",
    [
        b"",
        b"\xff",
        b"\xef\xbb\xbfschema: '2.0'\n",
        b"schema: '2.0'\r\nstages: {}\n",
        b"schema: '2.0'\rstages: {}\r",
        b"schema: '2.0'\n\0",
        b"x" * (MAX_CONTROL_FILE_BYTES + 1),
    ],
    ids=["empty", "invalid-utf8", "bom", "mixed-newlines", "bare-cr", "nul", "oversized"],
)
def test_lock_parser_rejects_unsafe_bytes(document: bytes) -> None:
    _assert_lock_invalid(document)


def test_build_uses_explicit_commit_and_clean_state_without_a_git_repository(
    tmp_path: Path,
) -> None:
    _write_repository(tmp_path)

    assert _build(tmp_path).metadata_git_commit == _COMMIT
    for git_clean, dvc_clean in ((False, True), (True, False), (False, False)):
        with pytest.raises(DvcProvenanceError, match="snapshot is invalid"):
            build_dvc_snapshot(
                tmp_path,
                _COMMIT,
                metadata_repository_role="public-fixture",
                git_working_tree_clean=git_clean,
                dvc_workspace_clean=dvc_clean,
            )


def test_build_rejects_missing_non_regular_and_unsafe_control_files(tmp_path: Path) -> None:
    private_root = tmp_path / "participant-private-sentinel"
    private_root.mkdir()
    _write_repository(private_root)

    (private_root / "dvc.lock").unlink()
    (private_root / "dvc.lock").mkdir()
    with pytest.raises(DvcProvenanceError) as directory_error:
        _build(private_root)
    assert directory_error.value.category == "control_file.invalid"
    assert "participant-private-sentinel" not in str(directory_error.value)

    (private_root / "dvc.lock").rmdir()
    with pytest.raises(DvcProvenanceError) as missing_error:
        _build(private_root)
    assert missing_error.value.category == "control_file.invalid"
    assert "participant-private-sentinel" not in str(missing_error.value)


def test_build_rejects_oversized_and_non_utf8_control_files(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    (tmp_path / "uv.lock").write_bytes(b"x" * (MAX_CONTROL_FILE_BYTES + 1))
    with pytest.raises(DvcProvenanceError, match="control files are unavailable"):
        _build(tmp_path)

    (tmp_path / "uv.lock").write_bytes(b"\xff")
    with pytest.raises(DvcProvenanceError, match="control files are unavailable"):
        _build(tmp_path)


def test_build_rejects_symlinked_control_file(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    target = tmp_path / "real-lock"
    target.write_text(_render_lock(), encoding="utf-8", newline="\n")
    (tmp_path / "dvc.lock").unlink()
    try:
        (tmp_path / "dvc.lock").symlink_to(target)
    except OSError:
        pytest.skip("this account cannot create file symlinks")

    with pytest.raises(DvcProvenanceError, match="control files are unavailable"):
        _build(tmp_path)


def test_build_rejects_hardlinked_control_file(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    alias = tmp_path / "lock-hardlink-alias"
    try:
        os.link(tmp_path / "dvc.lock", alias)
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")

    with pytest.raises(DvcProvenanceError, match="control files are unavailable"):
        _build(tmp_path)


@pytest.mark.parametrize("boundary", ["root", "control-file"])
def test_build_rejects_windows_reparse_points_at_control_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    _write_repository(tmp_path)
    target = tmp_path if boundary == "root" else tmp_path / "uv.lock"
    target_status = os.lstat(target)
    original = provenance_module._is_reparse

    def classify(details: os.stat_result) -> bool:
        return details.st_dev == target_status.st_dev and details.st_ino == target_status.st_ino

    monkeypatch.setattr(provenance_module, "_is_reparse", classify)
    try:
        with pytest.raises(DvcProvenanceError, match="control files are unavailable"):
            _build(tmp_path)
    finally:
        monkeypatch.setattr(provenance_module, "_is_reparse", original)


def test_build_rejects_final_control_path_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_repository(tmp_path)
    target_status = os.lstat(tmp_path / "uv.lock")
    original = provenance_module._path_file_identity
    calls = 0

    def drifting_identity(details: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        identity = original(details)
        if details.st_dev == target_status.st_dev and details.st_ino == target_status.st_ino:
            calls += 1
            if calls == 2:
                return (*identity, 1)
        return identity

    monkeypatch.setattr(provenance_module, "_path_file_identity", drifting_identity)

    with pytest.raises(DvcProvenanceError, match="control files are unavailable"):
        _build(tmp_path)


def test_build_rejects_control_parent_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_repository(tmp_path)
    target_status = os.lstat(tmp_path)
    original = provenance_module._directory_identity
    calls = 0

    def drifting_identity(details: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        identity = original(details)
        if details.st_dev == target_status.st_dev and details.st_ino == target_status.st_ino:
            calls += 1
            if calls == 6:
                return (*identity, 1)
        return identity

    monkeypatch.setattr(provenance_module, "_directory_identity", drifting_identity)

    with pytest.raises(DvcProvenanceError, match="control files are unavailable"):
        _build(tmp_path)


def test_control_file_os_failure_suppresses_the_path_bearing_context(tmp_path: Path) -> None:
    private_root = tmp_path / "participant-private-sentinel"
    private_root.mkdir()

    with pytest.raises(DvcProvenanceError) as captured:
        _build(private_root)

    assert captured.value.category == "control_file.invalid"
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True
    assert "participant-private-sentinel" not in str(captured.value)


def test_invalid_lock_failure_does_not_echo_sensitive_content() -> None:
    document = (
        "schema: '2.0'\n"
        "stages:\n"
        "  ingest:\n"
        "    cmd: participant-private-secret\n"
        "    cmd: duplicate-private-secret\n"
    )

    with pytest.raises(DvcProvenanceError) as captured:
        parse_dvc_lock(document)
    assert "private" not in str(captured.value).casefold()
    assert "secret" not in str(captured.value).casefold()
