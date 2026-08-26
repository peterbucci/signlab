"""Streaming verification for dataset row artifacts materialized by DVC."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Never, cast

from signlab.contracts.core import ArtifactRefV1, WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import (
    ClipsTableV1,
    DatasetTable,
    DerivedArtifactsTableV1,
    RecordingsTableV1,
    TableName,
)

_REPARSE_POINT: Final = 0x400


class DatasetStorageError(ValueError):
    """Sanitized failure for missing, unsafe, or mismatched row-artifact bytes."""

    def __init__(self) -> None:
        self.code = "dataset.storage.artifact_bytes.invalid"
        super().__init__("dataset row-artifact bytes are unavailable or invalid")


@dataclass(frozen=True, slots=True)
class RowArtifactVerificationResult:
    """Positive storage evidence; counts are for internal reconciliation only."""

    artifact_byte_integrity: Literal["verified"]
    artifacts_verified: int
    total_bytes_verified: int


@dataclass(frozen=True, slots=True)
class _ResolvedArtifact:
    path: Path
    resolved_path: Path
    before: os.stat_result
    directories: tuple[tuple[Path, tuple[int, ...]], ...]


def _fail() -> Never:
    raise DatasetStorageError from None


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_POINT)


def _lstat_regular(path: Path, *, directory: bool) -> os.stat_result:
    try:
        details = os.lstat(path)
    except (OSError, ValueError):
        _fail()
    valid_type = stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode)
    if stat.S_ISLNK(details.st_mode) or _is_reparse(details) or not valid_type:
        _fail()
    if not directory and details.st_nlink != 1:
        _fail()
    return details


def _resolve_existing(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        _fail()


def _directory_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        stat.S_IFMT(details.st_mode),
        details.st_ctime_ns,
        getattr(details, "st_file_attributes", 0),
    )


def _file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        stat.S_IFMT(details.st_mode),
        details.st_nlink,
        details.st_size,
        details.st_mtime_ns,
        getattr(details, "st_file_attributes", 0),
    )


def _path_file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (*_file_identity(details), details.st_ctime_ns)


def _resolve_regular_artifact(
    root: Path,
    locator: WorkspaceRelativeLocatorV1,
) -> _ResolvedArtifact:
    root_status = _lstat_regular(root, directory=True)
    checked_root = _resolve_existing(root)
    resolved_root_status = _lstat_regular(checked_root, directory=True)
    if _directory_identity(root_status) != _directory_identity(resolved_root_status):
        _fail()
    directories: list[tuple[Path, tuple[int, ...]]] = [
        (checked_root, _directory_identity(resolved_root_status))
    ]
    candidate = root
    segments = locator.path.split("/")
    for index, segment in enumerate(segments):
        candidate /= segment
        details = _lstat_regular(candidate, directory=index < len(segments) - 1)
        if index < len(segments) - 1:
            directories.append((candidate, _directory_identity(details)))
    resolved_candidate = _resolve_existing(candidate)
    if not resolved_candidate.is_relative_to(checked_root):
        _fail()
    return _ResolvedArtifact(
        path=candidate,
        resolved_path=resolved_candidate,
        before=details,
        directories=tuple(directories),
    )


def _validate_resolved_artifact_after_read(resolved: _ResolvedArtifact) -> None:
    path_after = _lstat_regular(resolved.path, directory=False)
    if (
        _path_file_identity(path_after) != _path_file_identity(resolved.before)
        or _resolve_existing(resolved.path) != resolved.resolved_path
    ):
        _fail()
    for directory, expected_identity in resolved.directories:
        if _directory_identity(_lstat_regular(directory, directory=True)) != expected_identity:
            _fail()


def _verify_file(reference: ArtifactRefV1, root: Path) -> int:
    locator = reference.locator
    if not isinstance(locator, WorkspaceRelativeLocatorV1):
        _fail()
    resolved = _resolve_regular_artifact(root, locator)
    if resolved.before.st_size != reference.size_bytes:
        _fail()

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved.path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_ISLNK(opened.st_mode)
                or _is_reparse(opened)
                or opened.st_nlink != 1
            ):
                _fail()
            digest = hashlib.sha256()
            bytes_read = 0
            while True:
                chunk = os.read(descriptor, 1_048_576)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)
                if bytes_read > reference.size_bytes:
                    _fail()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except DatasetStorageError:
        raise
    except (OSError, ValueError):
        _fail()

    actual_sha256 = f"sha256:{digest.hexdigest()}"
    if (
        _file_identity(resolved.before) != _file_identity(opened)
        or _file_identity(opened) != _file_identity(after)
        or bytes_read != reference.size_bytes
        or actual_sha256 != reference.sha256
    ):
        _fail()
    _validate_resolved_artifact_after_read(resolved)
    return bytes_read


def collect_row_artifact_references(
    tables: Mapping[TableName, DatasetTable],
) -> tuple[ArtifactRefV1, ...]:
    """Collect every materialized recording, clip, and derived artifact exactly once."""

    try:
        recordings = cast(RecordingsTableV1, tables["recordings"])
        clips = cast(ClipsTableV1, tables["clips"])
        derived = cast(DerivedArtifactsTableV1, tables["derived_artifacts"])
        candidates = (
            *(row.media for row in recordings.rows),
            *(row.artifact for row in clips.rows if row.artifact is not None),
            *(row.artifact for row in derived.rows),
        )
    except (AttributeError, KeyError, TypeError):
        _fail()
    by_id: dict[str, ArtifactRefV1] = {}
    by_path: dict[str, ArtifactRefV1] = {}
    for artifact in candidates:
        existing_id = by_id.get(artifact.artifact_id)
        if existing_id is not None and existing_id != artifact:
            _fail()
        locator = artifact.locator
        if not isinstance(locator, WorkspaceRelativeLocatorV1):
            _fail()
        path_key = locator.path.casefold()
        existing_path = by_path.get(path_key)
        if existing_path is not None and existing_path != artifact:
            _fail()
        by_id[artifact.artifact_id] = artifact
        by_path[path_key] = artifact
    return tuple(sorted(by_id.values(), key=lambda item: item.artifact_id))


def verify_artifact_references(
    references: tuple[ArtifactRefV1, ...],
    workspace_root: str | Path,
) -> RowArtifactVerificationResult:
    """Stream-check a canonical, unique reference set beneath one explicit root."""

    try:
        if not references:
            _fail()
        identities = tuple(reference.artifact_id for reference in references)
        if identities != tuple(sorted(set(identities))):
            _fail()
        locators = tuple(reference.locator for reference in references)
        if not all(isinstance(locator, WorkspaceRelativeLocatorV1) for locator in locators):
            _fail()
        path_keys = tuple(
            cast(WorkspaceRelativeLocatorV1, locator).path.casefold() for locator in locators
        )
        if len(path_keys) != len(set(path_keys)):
            _fail()
        root = Path(workspace_root)
        _lstat_regular(root, directory=True)
        total_bytes = sum(_verify_file(reference, root) for reference in references)
        return RowArtifactVerificationResult(
            artifact_byte_integrity="verified",
            artifacts_verified=len(references),
            total_bytes_verified=total_bytes,
        )
    except DatasetStorageError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail()


def verify_dataset_row_artifacts(
    tables: Mapping[TableName, DatasetTable],
    workspace_root: str | Path,
) -> RowArtifactVerificationResult:
    """Verify all row-level bytes after semantic table validation succeeds."""

    return verify_artifact_references(
        collect_row_artifact_references(tables),
        workspace_root,
    )


__all__ = [
    "DatasetStorageError",
    "RowArtifactVerificationResult",
    "collect_row_artifact_references",
    "verify_artifact_references",
    "verify_dataset_row_artifacts",
]
