"""Small, portable Git/DVC snapshot recorded beside experiment metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import Field, model_validator

from signlab.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import GitCommit, StrictContractModel
from signlab.contracts.taxonomy import Sha256Digest
from signlab.reproducibility import DVC_VERSION
from signlab.reproducibility.stages import STAGE_NAMES

PUBLIC_REPOSITORY = "https://github.com/peterbucci/signlab"
PUBLIC_REPOSITORY_ORIGINS = frozenset(
    {
        PUBLIC_REPOSITORY,
        f"{PUBLIC_REPOSITORY}.git",
        "git@github.com:peterbucci/signlab.git",
        "ssh://git@github.com/peterbucci/signlab.git",
    }
)
MAX_CONTROL_FILE_BYTES = 4 * 1024 * 1024

type DvcMetadataRepositoryRole = Literal["public-fixture", "protected-metadata"]


class DvcProvenanceError(ValueError):
    """Raised when committed DVC metadata cannot produce a trustworthy snapshot."""


class DvcStageIdentityV1(StrictContractModel):
    """Hash of one complete stage entry from ``dvc.lock``."""

    stage_name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    lock_entry_sha256: Sha256Digest


class DvcSnapshotV1(StrictContractModel):
    """Minimal lineage record suitable for later experiment tracking."""

    schema_version: Literal["dvc-snapshot/1"]
    metadata_repository_role: DvcMetadataRepositoryRole
    metadata_repository: str | None
    metadata_git_commit: GitCommit
    git_working_tree_clean: Literal[True]
    dvc_workspace_clean: Literal[True]
    dvc_version: Literal["3.67.1"]
    uv_lock_sha256: Sha256Digest
    dvc_yaml_sha256: Sha256Digest
    dvc_lock_sha256: Sha256Digest
    stages: tuple[DvcStageIdentityV1, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _validate_repository_and_stages(self) -> DvcSnapshotV1:
        if self.metadata_repository_role == "public-fixture":
            if self.metadata_repository != PUBLIC_REPOSITORY:
                raise ValueError("public fixture snapshot must identify the public repository")
        elif self.metadata_repository is not None:
            raise ValueError("protected repository locator must remain private")
        if tuple(stage.stage_name for stage in self.stages) != STAGE_NAMES:
            raise ValueError("snapshot stages must match the registered DVC graph")
        return self


type DvcSnapshotInput = DvcSnapshotV1 | str | bytes | bytearray | Mapping[str, object]


def _normalized_bytes(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DvcProvenanceError("DVC control file is unavailable") from error
    if not payload or len(payload) > MAX_CONTROL_FILE_BYTES or b"\0" in payload:
        raise DvcProvenanceError("DVC control file is invalid")
    return payload.replace(b"\r\n", b"\n")


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _stage_identities(lock_payload: bytes) -> tuple[DvcStageIdentityV1, ...]:
    try:
        document = yaml.safe_load(lock_payload)
        stages = document["stages"]
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise DvcProvenanceError("dvc.lock is invalid") from error
    if type(stages) is not dict or tuple(stages) != STAGE_NAMES:
        raise DvcProvenanceError("dvc.lock stages do not match the registered graph")
    identities: list[DvcStageIdentityV1] = []
    for stage_name in STAGE_NAMES:
        entry = stages.get(stage_name)
        if type(entry) is not dict:
            raise DvcProvenanceError("dvc.lock stage entry is invalid")
        try:
            digest = canonical_sha256(
                cast(Mapping[str, object], entry),
                domain="dvc-lock-stage/1",
            )
        except (TypeError, ValueError) as error:
            raise DvcProvenanceError("dvc.lock stage entry is invalid") from error
        identities.append(DvcStageIdentityV1(stage_name=stage_name, lock_entry_sha256=digest))
    return tuple(identities)


def build_dvc_snapshot(
    repository: Path,
    commit: str,
    *,
    metadata_repository_role: DvcMetadataRepositoryRole,
    git_working_tree_clean: bool,
    dvc_workspace_clean: bool,
) -> DvcSnapshotV1:
    """Build a snapshot from the three committed control files."""

    if not git_working_tree_clean or not dvc_workspace_clean:
        raise DvcProvenanceError("Git and DVC state must be clean")
    uv_lock = _normalized_bytes(repository / "uv.lock")
    dvc_yaml = _normalized_bytes(repository / "dvc.yaml")
    dvc_lock = _normalized_bytes(repository / "dvc.lock")
    try:
        return DvcSnapshotV1(
            schema_version="dvc-snapshot/1",
            metadata_repository_role=metadata_repository_role,
            metadata_repository=(
                PUBLIC_REPOSITORY if metadata_repository_role == "public-fixture" else None
            ),
            metadata_git_commit=commit,
            git_working_tree_clean=True,
            dvc_workspace_clean=True,
            dvc_version=DVC_VERSION,
            uv_lock_sha256=_sha256(uv_lock),
            dvc_yaml_sha256=_sha256(dvc_yaml),
            dvc_lock_sha256=_sha256(dvc_lock),
            stages=_stage_identities(dvc_lock),
        )
    except (TypeError, ValueError) as error:
        raise DvcProvenanceError("DVC snapshot is invalid") from error


def validate_dvc_snapshot(document: DvcSnapshotInput) -> DvcSnapshotV1:
    """Validate a snapshot model, JSON value, or JSON document."""

    try:
        if isinstance(document, DvcSnapshotV1):
            return document
        if isinstance(document, (str, bytes, bytearray)):
            document = parse_json_object(document)
        return DvcSnapshotV1.model_validate_json(canonical_json_bytes(document))
    except (TypeError, ValueError) as error:
        raise DvcProvenanceError("DVC snapshot is invalid") from error


def dvc_snapshot_digest(document: DvcSnapshotInput) -> str:
    """Return the canonical SHA-256 identity of a validated snapshot."""

    snapshot = validate_dvc_snapshot(document)
    try:
        return canonical_sha256(
            snapshot.model_dump(mode="json", round_trip=True),
            domain="dvc-snapshot/1",
        )
    except (TypeError, ValueError) as error:
        raise DvcProvenanceError("DVC snapshot is invalid") from error


def dvc_experiment_metadata(document: DvcSnapshotInput) -> dict[str, str]:
    """Project a snapshot into tracker-neutral string metadata for Story #27."""

    snapshot = validate_dvc_snapshot(document)
    metadata = {
        "git.commit": snapshot.metadata_git_commit,
        "dvc.version": snapshot.dvc_version,
        "dvc.lock.sha256": snapshot.dvc_lock_sha256,
        "dvc.snapshot.sha256": dvc_snapshot_digest(snapshot),
    }
    metadata.update(
        {
            f"dvc.stage.{stage.stage_name}.sha256": stage.lock_entry_sha256
            for stage in snapshot.stages
        }
    )
    return metadata


__all__ = [
    "DVC_VERSION",
    "MAX_CONTROL_FILE_BYTES",
    "PUBLIC_REPOSITORY",
    "PUBLIC_REPOSITORY_ORIGINS",
    "DvcMetadataRepositoryRole",
    "DvcProvenanceError",
    "DvcSnapshotInput",
    "DvcSnapshotV1",
    "DvcStageIdentityV1",
    "build_dvc_snapshot",
    "dvc_experiment_metadata",
    "dvc_snapshot_digest",
    "validate_dvc_snapshot",
]
