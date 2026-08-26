"""Capture and write clean Git/DVC provenance snapshots."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.reproducibility import DVC_VERSION
from signlab.reproducibility.provenance import (
    PUBLIC_REPOSITORY_ORIGINS,
    DvcMetadataRepositoryRole,
    DvcSnapshotInput,
    DvcSnapshotV1,
    build_dvc_snapshot,
    validate_dvc_snapshot,
)


class DvcEvidenceError(ValueError):
    """Raised with a stable message that contains no command output or paths."""


def _run(repository: Path, command: Sequence[str]) -> str:
    environment = dict(os.environ)
    environment["DVC_NO_ANALYTICS"] = "true"
    try:
        result = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DvcEvidenceError("provenance command failed") from error
    if result.returncode != 0:
        raise DvcEvidenceError("provenance command failed")
    return result.stdout.strip()


def capture_dvc_snapshot(
    repository: Path,
    *,
    metadata_repository_role: DvcMetadataRepositoryRole,
) -> DvcSnapshotV1:
    """Capture a snapshot only when the selected Git and DVC state is clean."""

    try:
        root = repository.resolve(strict=True)
    except OSError as error:
        raise DvcEvidenceError("repository is unavailable") from error
    git_status = _run(root, ("git", "status", "--porcelain=v1", "--untracked-files=all"))
    if git_status:
        raise DvcEvidenceError("Git working tree is not clean")
    commit = _run(root, ("git", "rev-parse", "--verify", "HEAD^{commit}"))
    origin = _run(root, ("git", "remote", "get-url", "origin"))
    if (metadata_repository_role == "public-fixture") != (origin in PUBLIC_REPOSITORY_ORIGINS):
        raise DvcEvidenceError("repository identity does not match its declared role")
    dvc_version = _run(root, (sys.executable, "-I", "-m", "dvc", "--version"))
    if dvc_version != DVC_VERSION:
        raise DvcEvidenceError("locked DVC version is unavailable")
    dvc_status = _run(root, (sys.executable, "-I", "-m", "dvc", "status", "--json"))
    try:
        dvc_clean = parse_json_object(dvc_status) == {}
    except (TypeError, ValueError) as error:
        raise DvcEvidenceError("DVC status is invalid") from error
    if not dvc_clean:
        raise DvcEvidenceError("DVC workspace is not clean")
    try:
        return build_dvc_snapshot(
            root,
            commit,
            metadata_repository_role=metadata_repository_role,
            git_working_tree_clean=True,
            dvc_workspace_clean=True,
        )
    except (OSError, TypeError, ValueError) as error:
        raise DvcEvidenceError("DVC snapshot could not be captured") from error


def _validated_output(repository: Path, relative_path: str) -> Path:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or path.suffix.casefold() != ".json"
        or tuple(path.parts[:2]) != ("reports", "reproduction")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DvcEvidenceError("snapshot output path is invalid")
    target = repository.joinpath(*path.parts)
    if target.exists() or target.is_symlink():
        raise DvcEvidenceError("snapshot output already exists")
    return target


def write_dvc_snapshot(
    snapshot: DvcSnapshotInput,
    repository: Path,
    relative_path: str,
) -> Path:
    """Atomically write a validated snapshot beneath ignored reproduction reports."""

    checked = validate_dvc_snapshot(snapshot)
    target = _validated_output(repository.resolve(), relative_path)
    payload = canonical_json_bytes(checked.model_dump(mode="json", round_trip=True)) + b"\n"
    temporary_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
        temporary_path.replace(target)
    except OSError as error:
        raise DvcEvidenceError("DVC snapshot could not be written") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


__all__ = ["DvcEvidenceError", "capture_dvc_snapshot", "write_dvc_snapshot"]
