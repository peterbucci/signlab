from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from signlab.contracts.canonical import canonical_json_bytes
from signlab.reproducibility.provenance import (
    PUBLIC_REPOSITORY,
    DvcMetadataRepositoryRole,
    DvcProvenanceError,
    DvcSnapshotV1,
    build_dvc_snapshot,
    dvc_experiment_metadata,
    dvc_snapshot_digest,
    validate_dvc_snapshot,
)
from signlab.reproducibility.stages import STAGE_NAMES

COMMIT = "a" * 40


def _lock_document() -> dict[str, object]:
    return {
        "schema": "2.0",
        "stages": {
            stage: {
                "cmd": f"python -m signlab.cli data run-reproduction-stage {stage}",
                "deps": [{"path": f"fixture/{stage}.in", "md5": "1" * 32}],
                "outs": [{"path": f"fixture/{stage}.out", "md5": "2" * 32}],
            }
            for stage in STAGE_NAMES
        },
    }


def _encode(text: str, newline: str) -> bytes:
    return text.replace("\n", newline).encode("utf-8")


def _write_repository(
    root: Path,
    *,
    lock: dict[str, object] | None = None,
    newline: str = "\n",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("uv.lock").write_bytes(_encode("version = 1\n", newline))
    root.joinpath("dvc.yaml").write_bytes(
        _encode("stages:\n  fixture:\n    cmd: fixture\n", newline)
    )
    rendered_lock = yaml.safe_dump(lock or _lock_document(), sort_keys=False)
    root.joinpath("dvc.lock").write_bytes(_encode(rendered_lock, newline))


def _build(
    root: Path,
    *,
    role: DvcMetadataRepositoryRole = "public-fixture",
) -> DvcSnapshotV1:
    return build_dvc_snapshot(
        root,
        COMMIT,
        metadata_repository_role=role,
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    )


def test_snapshot_records_git_dvc_controls_stage_hashes_and_experiment_metadata(
    tmp_path: Path,
) -> None:
    _write_repository(tmp_path)

    snapshot = _build(tmp_path)
    protected = _build(tmp_path, role="protected-metadata")

    assert snapshot.metadata_repository == PUBLIC_REPOSITORY
    assert protected.metadata_repository is None
    assert snapshot.metadata_git_commit == COMMIT
    assert snapshot.git_working_tree_clean is True
    assert snapshot.dvc_workspace_clean is True
    assert tuple(stage.stage_name for stage in snapshot.stages) == STAGE_NAMES
    for digest in (
        snapshot.uv_lock_sha256,
        snapshot.dvc_yaml_sha256,
        snapshot.dvc_lock_sha256,
        *(stage.lock_entry_sha256 for stage in snapshot.stages),
        dvc_snapshot_digest(snapshot),
    ):
        assert digest.startswith("sha256:")
        assert len(digest) == 71

    serialized = canonical_json_bytes(snapshot.model_dump(mode="json", round_trip=True))
    assert validate_dvc_snapshot(serialized) == snapshot

    metadata = dvc_experiment_metadata(snapshot)
    assert metadata == {
        "git.commit": COMMIT,
        "dvc.version": snapshot.dvc_version,
        "dvc.lock.sha256": snapshot.dvc_lock_sha256,
        "dvc.snapshot.sha256": dvc_snapshot_digest(snapshot),
        **{
            f"dvc.stage.{stage.stage_name}.sha256": stage.lock_entry_sha256
            for stage in snapshot.stages
        },
    }
    assert PUBLIC_REPOSITORY not in metadata.values()
    assert not any("/" in value or "\\" in value for value in metadata.values())


def test_changing_one_lock_entry_changes_only_that_stage_and_snapshot_identity(
    tmp_path: Path,
) -> None:
    lock = _lock_document()
    _write_repository(tmp_path, lock=lock)
    before = _build(tmp_path)

    stages = cast(dict[str, object], lock["stages"])
    feature = cast(dict[str, object], stages["feature"])
    feature["params"] = {"fixture_seed": 2}
    _write_repository(tmp_path, lock=lock)
    after = _build(tmp_path)

    before_hashes = {stage.stage_name: stage.lock_entry_sha256 for stage in before.stages}
    after_hashes = {stage.stage_name: stage.lock_entry_sha256 for stage in after.stages}
    assert [name for name in STAGE_NAMES if before_hashes[name] != after_hashes[name]] == [
        "feature"
    ]
    assert before.dvc_lock_sha256 != after.dvc_lock_sha256
    assert dvc_snapshot_digest(before) != dvc_snapshot_digest(after)


def test_snapshot_hashes_are_portable_across_lf_and_crlf_control_files(tmp_path: Path) -> None:
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    _write_repository(lf_root, newline="\n")
    _write_repository(crlf_root, newline="\r\n")

    assert _build(lf_root) == _build(crlf_root)


def test_snapshot_rejects_dirty_state_lock_inventory_drift_and_private_locator(
    tmp_path: Path,
) -> None:
    _write_repository(tmp_path)
    for git_clean, dvc_clean in ((False, True), (True, False)):
        with pytest.raises(DvcProvenanceError, match="must be clean"):
            build_dvc_snapshot(
                tmp_path,
                COMMIT,
                metadata_repository_role="public-fixture",
                git_working_tree_clean=git_clean,
                dvc_workspace_clean=dvc_clean,
            )

    lock = _lock_document()
    stages = cast(dict[str, object], lock["stages"])
    del stages["feature"]
    _write_repository(tmp_path, lock=lock)
    with pytest.raises(DvcProvenanceError, match="snapshot is invalid"):
        _build(tmp_path)

    _write_repository(tmp_path)
    protected = _build(tmp_path, role="protected-metadata").model_dump(mode="json", round_trip=True)
    protected["metadata_repository"] = "https://private.example.test/metadata"
    with pytest.raises(DvcProvenanceError, match="snapshot is invalid"):
        validate_dvc_snapshot(protected)
