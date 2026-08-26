"""Verify manifest-referenced row artifacts after an authorized DVC pull."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from signlab.contracts.core import ArtifactRefV1, WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import (
    ClipsTableV1,
    DatasetTable,
    DerivedArtifactsTableV1,
    RecordingsTableV1,
    TableName,
)


class DatasetStorageError(ValueError):
    """Raised without disclosing a private artifact locator."""

    def __init__(self) -> None:
        self.code = "dataset.storage.artifact_bytes.invalid"
        super().__init__("dataset row-artifact bytes are unavailable or invalid")


@dataclass(frozen=True, slots=True)
class RowArtifactVerificationResult:
    """Positive byte-integrity evidence for a manifest."""

    artifact_byte_integrity: Literal["verified"]
    artifacts_verified: int
    total_bytes_verified: int


def collect_row_artifact_references(
    tables: Mapping[TableName, DatasetTable],
) -> tuple[ArtifactRefV1, ...]:
    """Collect unique recording, clip, and derived-artifact references."""

    try:
        recordings = cast(RecordingsTableV1, tables["recordings"])
        clips = cast(ClipsTableV1, tables["clips"])
        derived = cast(DerivedArtifactsTableV1, tables["derived_artifacts"])
        candidates = (
            *(row.media for row in recordings.rows),
            *(row.artifact for row in clips.rows if row.artifact is not None),
            *(row.artifact for row in derived.rows),
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise DatasetStorageError from error

    by_id: dict[str, ArtifactRefV1] = {}
    by_path: dict[str, ArtifactRefV1] = {}
    for artifact in candidates:
        locator = artifact.locator
        if not isinstance(locator, WorkspaceRelativeLocatorV1):
            raise DatasetStorageError
        prior_id = by_id.get(artifact.artifact_id)
        prior_path = by_path.get(locator.path.casefold())
        if (prior_id is not None and prior_id != artifact) or (
            prior_path is not None and prior_path != artifact
        ):
            raise DatasetStorageError
        by_id[artifact.artifact_id] = artifact
        by_path[locator.path.casefold()] = artifact
    return tuple(sorted(by_id.values(), key=lambda reference: reference.artifact_id))


def _verify_file(reference: ArtifactRefV1, root: Path) -> int:
    locator = reference.locator
    if not isinstance(locator, WorkspaceRelativeLocatorV1):
        raise DatasetStorageError
    try:
        candidate = root.joinpath(*locator.path.split("/"))
        if candidate.is_symlink():
            raise DatasetStorageError
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise DatasetStorageError
        size = resolved.stat().st_size
        if size != reference.size_bytes:
            raise DatasetStorageError
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except DatasetStorageError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetStorageError from error
    if f"sha256:{digest.hexdigest()}" != reference.sha256:
        raise DatasetStorageError
    return size


def verify_artifact_references(
    references: tuple[ArtifactRefV1, ...],
    workspace_root: str | Path,
) -> RowArtifactVerificationResult:
    """Stream-check an ordered unique set beneath an explicit workspace root."""

    if not references:
        raise DatasetStorageError
    identities = tuple(reference.artifact_id for reference in references)
    if identities != tuple(sorted(set(identities))):
        raise DatasetStorageError
    locators = tuple(reference.locator for reference in references)
    if not all(isinstance(locator, WorkspaceRelativeLocatorV1) for locator in locators):
        raise DatasetStorageError
    paths = tuple(cast(WorkspaceRelativeLocatorV1, locator).path.casefold() for locator in locators)
    if len(paths) != len(set(paths)):
        raise DatasetStorageError
    try:
        root = Path(workspace_root).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DatasetStorageError from error
    if not root.is_dir():
        raise DatasetStorageError
    total_bytes = sum(_verify_file(reference, root) for reference in references)
    return RowArtifactVerificationResult(
        artifact_byte_integrity="verified",
        artifacts_verified=len(references),
        total_bytes_verified=total_bytes,
    )


def verify_dataset_row_artifacts(
    tables: Mapping[TableName, DatasetTable],
    workspace_root: str | Path,
) -> RowArtifactVerificationResult:
    """Verify every row-level reference after semantic validation succeeds."""

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
