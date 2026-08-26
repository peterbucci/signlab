"""Deterministic, privacy-safe import of a finalized collection sidecar.

Source locations exist only at this application boundary.  Published contracts
contain opaque source keys, content identities, and canonical object locators;
they never contain the caller's filenames or machine paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final, Literal, cast

from pydantic import ValidationError

from signlab.contracts.core import ArtifactRefV1, WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    AnnotationsTableV1,
    ClipsTableV1,
    DatasetTable,
    DatasetTableRefV1,
    DatasetTableSetV1,
    DerivedArtifactsTableV1,
    ParticipantsTableV1,
    RecordingRowV1,
    RecordingsTableV1,
    SessionsTableV1,
    TableName,
    content_addressed_row_locations,
)
from signlab.contracts.governance import LineageInventoryV1
from signlab.contracts.ingest import (
    CaptureAttemptV1,
    CollectionSidecarV1,
    PromptOccurrenceV1,
    RawDatasetContentV1,
    RawDatasetManifestV1,
    project_annotation_rows,
    raw_dataset_content_digest,
    require_importable_sidecar,
)
from signlab.contracts.quarantine import (
    QuarantineInventoryV1,
    quarantine_attempt_workspace_path,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference
from signlab.datasets.parquet import write_dataset_table
from signlab.datasets.raw_bundle import (
    COLLECTION_SIDECAR_FILENAME,
    LINEAGE_INVENTORY_FILENAME,
    QUARANTINE_INVENTORY_FILENAME,
    RAW_MANIFEST_FILENAME,
    RawDatasetBundleError,
    RawDatasetBundleValidationResult,
    ValidatedRawDatasetBundle,
    build_quarantine_inventory,
    build_raw_lineage_inventory,
    validate_raw_dataset_bundle,
)

type CollectionSidecarInput = CollectionSidecarV1 | str | bytes | bytearray | Mapping[str, object]
type SourceLocation = str | os.PathLike[str]
type SourceMap = Mapping[str, SourceLocation]
type DatasetImportErrorCategory = Literal[
    "sidecar.invalid",
    "source.invalid",
    "source.conflict",
    "destination.invalid",
    "destination.conflict",
    "publication.failed",
]

_CHUNK_SIZE: Final = 1024 * 1024
_ERROR_MESSAGES: Final[dict[DatasetImportErrorCategory, str]] = {
    "sidecar.invalid": "collection sidecar is not importable",
    "source.invalid": "collection source bytes are unavailable or invalid",
    "source.conflict": "collection source identities conflict",
    "destination.invalid": "raw dataset destination is invalid",
    "destination.conflict": "raw dataset destination contains different content",
    "publication.failed": "raw dataset bundle could not be published",
}


class DatasetImportError(ValueError):
    """A stable importer error that never discloses participant source paths."""

    def __init__(self, category: DatasetImportErrorCategory) -> None:
        self.category = category
        self.code = f"dataset.import.{category}"
        super().__init__(_ERROR_MESSAGES[category])


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """A private, process-local source identity used while copying bytes."""

    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class ImportedRawDatasetBundle:
    """One newly published or already-identical raw bundle."""

    status: Literal["published", "unchanged"]
    manifest: RawDatasetManifestV1
    manifest_path: Path
    validation: RawDatasetBundleValidationResult
    accepted_recordings: int
    retry_attempts: int
    quarantined_attempts: int
    skipped_occurrences: int


def _checkpoint(_phase: str) -> None:
    """No-op test seam for proving that every pre-publication failure is atomic."""


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _require_source_root(source_root: str | Path) -> Path:
    try:
        candidate = Path(source_root)
        if _is_linklike(candidate):
            raise DatasetImportError("source.invalid")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise DatasetImportError("source.invalid")
        return root
    except DatasetImportError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DatasetImportError("source.invalid") from error


def _resolve_source(root: Path, location: SourceLocation) -> Path:
    try:
        raw = location.as_posix() if isinstance(location, Path) else os.fspath(location)
        if not isinstance(raw, str) or not raw or "\\" in raw:
            raise DatasetImportError("source.invalid")
        windows = PureWindowsPath(raw)
        if Path(raw).is_absolute() or windows.is_absolute() or windows.drive:
            raise DatasetImportError("source.invalid")
        locator = WorkspaceRelativeLocatorV1.model_validate(
            {"kind": "workspace_relative", "path": raw},
            strict=True,
        )
        candidate = root
        for segment in locator.path.split("/"):
            candidate = candidate / segment
            if _is_linklike(candidate):
                raise DatasetImportError("source.invalid")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise DatasetImportError("source.invalid")
        return resolved
    except DatasetImportError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, ValidationError) as error:
        raise DatasetImportError("source.invalid") from error


def _fingerprint(path: Path) -> SourceFingerprint:
    try:
        if _is_linklike(path) or path.resolve(strict=True) != path:
            raise DatasetImportError("source.invalid")
        before = path.stat()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                size += len(chunk)
                digest.update(chunk)
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if (
            _is_linklike(path)
            or path.resolve(strict=True) != path
            or before_identity != after_identity
            or size != after.st_size
        ):
            raise DatasetImportError("source.invalid")
        return SourceFingerprint(
            path=path,
            sha256=f"sha256:{digest.hexdigest()}",
            size_bytes=size,
            device=after.st_dev,
            inode=after.st_ino,
            modified_ns=after.st_mtime_ns,
        )
    except DatasetImportError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetImportError("source.invalid") from error


def _all_attempts(sidecar: CollectionSidecarV1) -> tuple[CaptureAttemptV1, ...]:
    return tuple(attempt for occurrence in sidecar.occurrences for attempt in occurrence.attempts)


def _preflight_sources(
    sidecar: CollectionSidecarV1,
    source_root: Path,
    source_map: SourceMap,
) -> dict[str, SourceFingerprint]:
    attempts = _all_attempts(sidecar)
    expected_keys = {attempt.source_key for attempt in attempts}
    try:
        locations: dict[str, SourceLocation] = dict(source_map)
        if set(locations) != expected_keys:
            raise DatasetImportError("source.invalid")
    except DatasetImportError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise DatasetImportError("source.invalid") from error

    fingerprints: dict[str, SourceFingerprint] = {}
    for attempt in attempts:
        fingerprint = _fingerprint(_resolve_source(source_root, locations[attempt.source_key]))
        if (
            fingerprint.sha256 != attempt.expected_sha256
            or fingerprint.size_bytes != attempt.expected_size_bytes
        ):
            raise DatasetImportError("source.invalid")
        fingerprints[attempt.source_key] = fingerprint

    accepted = tuple(attempt for attempt in attempts if attempt.outcome == "accepted")
    accepted_hashes = tuple(attempt.expected_sha256 for attempt in accepted)
    if len(accepted_hashes) != len(set(accepted_hashes)):
        raise DatasetImportError("source.conflict")
    grant_ids: list[str] = []
    for attempt in accepted:
        if attempt.consent_grant is None:
            raise DatasetImportError("sidecar.invalid")
        grant_ids.append(attempt.consent_grant.grant_id)
    if len(grant_ids) != len(set(grant_ids)):
        raise DatasetImportError("source.conflict")
    return fingerprints


def _accepted_attempt(occurrence: PromptOccurrenceV1) -> CaptureAttemptV1:
    accepted = tuple(attempt for attempt in occurrence.attempts if attempt.outcome == "accepted")
    if len(accepted) != 1:
        raise DatasetImportError("sidecar.invalid")
    return accepted[0]


def _build_tables(sidecar: CollectionSidecarV1) -> dict[TableName, DatasetTable]:
    participant_by_id = {row.participant_id: row for row in sidecar.participants}
    session_by_id = {row.session_id: row for row in sidecar.sessions}
    recordings: list[RecordingRowV1] = []
    for occurrence in sidecar.occurrences:
        if occurrence.state != "accepted":
            continue
        attempt = _accepted_attempt(occurrence)
        if attempt.consent_grant is None:
            raise DatasetImportError("sidecar.invalid")
        session = session_by_id[occurrence.session_id]
        digest = attempt.expected_sha256.removeprefix("sha256:")
        provisional = ArtifactRefV1(
            schema_version="artifact-reference/1",
            artifact_id=attempt.recording_id,
            role="raw_recording",
            media_type=attempt.media_type,
            sha256=attempt.expected_sha256,
            size_bytes=attempt.expected_size_bytes,
            locator=WorkspaceRelativeLocatorV1(
                kind="workspace_relative",
                path=(f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{attempt.recording_id}"),
            ),
        )
        recordings.append(
            RecordingRowV1(
                recording_id=attempt.recording_id,
                participant_id=occurrence.participant_id,
                session_id=occurrence.session_id,
                device_id=session.device_id,
                captured_at=attempt.recorded_at,
                duration_us=attempt.duration_us,
                handedness=attempt.handedness,
                mirror_state=attempt.mirror_state,
                rotation_degrees=attempt.rotation_degrees,
                audio_present=False,
                media=provisional,
                consent_grant=attempt.consent_grant,
            )
        )
    checked_recordings = tuple(sorted(recordings, key=lambda row: row.recording_id))
    participant_ids = {row.participant_id for row in checked_recordings}
    session_ids = {row.session_id for row in checked_recordings}
    participants = tuple(
        sorted(
            (participant_by_id[participant_id] for participant_id in participant_ids),
            key=lambda row: row.participant_id,
        )
    )
    sessions = tuple(
        sorted(
            (session_by_id[session_id] for session_id in session_ids),
            key=lambda row: row.session_id,
        )
    )
    return {
        "participants": ParticipantsTableV1(
            schema_version="participants-table/1",
            rows=participants,
        ),
        "sessions": SessionsTableV1(
            schema_version="sessions-table/1",
            rows=sessions,
        ),
        "recordings": RecordingsTableV1(
            schema_version="recordings-table/1",
            rows=checked_recordings,
        ),
        "clips": ClipsTableV1(schema_version="clips-table/1", rows=()),
        "annotations": AnnotationsTableV1(
            schema_version="annotations-table/1",
            rows=project_annotation_rows(sidecar),
        ),
        "derived_artifacts": DerivedArtifactsTableV1(
            schema_version="derived-artifacts-table/1",
            rows=(),
        ),
    }


def _copy_verified(fingerprint: SourceFingerprint, destination: Path) -> None:
    temporary: Path | None = None
    try:
        current = fingerprint.path.stat()
        if (
            _is_linklike(fingerprint.path)
            or fingerprint.path.resolve(strict=True) != fingerprint.path
            or current.st_dev != fingerprint.device
            or current.st_ino != fingerprint.inode
            or current.st_size != fingerprint.size_bytes
            or current.st_mtime_ns != fingerprint.modified_ns
        ):
            raise DatasetImportError("source.invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with fingerprint.path.open("rb") as source:
                for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = fingerprint.path.stat()
        if (
            _is_linklike(fingerprint.path)
            or fingerprint.path.resolve(strict=True) != fingerprint.path
            or after.st_dev != fingerprint.device
            or after.st_ino != fingerprint.inode
            or after.st_size != fingerprint.size_bytes
            or after.st_mtime_ns != fingerprint.modified_ns
            or size != fingerprint.size_bytes
            or f"sha256:{digest.hexdigest()}" != fingerprint.sha256
        ):
            raise DatasetImportError("source.invalid")
        os.replace(temporary, destination)
        temporary = None
    except DatasetImportError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetImportError("source.invalid") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _attempt_path(attempt: CaptureAttemptV1) -> str:
    if attempt.outcome == "accepted":
        reference = ArtifactRefV1(
            schema_version="artifact-reference/1",
            artifact_id=attempt.recording_id,
            role="raw_recording",
            media_type=attempt.media_type,
            sha256=attempt.expected_sha256,
            size_bytes=attempt.expected_size_bytes,
            locator=WorkspaceRelativeLocatorV1(
                kind="workspace_relative",
                path="placeholder",
            ),
        )
        return content_addressed_row_locations(reference)[0]
    return quarantine_attempt_workspace_path(attempt.attempt_id, attempt.expected_sha256)


def _stage_attempt_bytes(
    sidecar: CollectionSidecarV1,
    fingerprints: Mapping[str, SourceFingerprint],
    staging: Path,
) -> None:
    for occurrence in sidecar.occurrences:
        for attempt in occurrence.attempts:
            relative = _attempt_path(attempt)
            _copy_verified(
                fingerprints[attempt.source_key],
                staging.joinpath(*relative.split("/")),
            )


def _write_json_durably(path: Path, document: object) -> None:
    if hasattr(document, "model_dump"):
        payload = document.model_dump(mode="json", round_trip=True)
    else:
        payload = document
    captured = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(captured)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError) as error:
        raise DatasetImportError("publication.failed") from error


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _table_artifact_id(sidecar: CollectionSidecarV1, table_name: TableName) -> str:
    value = f"{sidecar.dataset_id}:{sidecar.dataset_version}:{table_name}"
    return f"table_{hashlib.sha256(value.encode('ascii')).hexdigest()[:32]}"


def _write_tables(
    sidecar: CollectionSidecarV1,
    tables: Mapping[TableName, DatasetTable],
    staging: Path,
) -> DatasetTableSetV1:
    references: dict[TableName, DatasetTableRefV1] = {}
    table_root = staging / "tables"
    table_root.mkdir()
    for table_name in DATASET_TABLE_SCHEMA_VERSIONS:
        relative = f"tables/{table_name}.parquet"
        target = staging.joinpath(*relative.split("/"))
        written = write_dataset_table(tables[table_name], target)
        _fsync_file(target)
        references[table_name] = DatasetTableRefV1(
            schema_version="dataset-table-reference/1",
            table_name=table_name,
            table_schema_version=written.table_schema_version,
            row_count=written.row_count,
            content_sha256=written.content_sha256,
            artifact=ArtifactRefV1(
                schema_version="artifact-reference/1",
                artifact_id=_table_artifact_id(sidecar, table_name),
                role="dataset_table",
                media_type="application/vnd.apache.parquet",
                sha256=written.sha256,
                size_bytes=written.size_bytes,
                locator=WorkspaceRelativeLocatorV1(
                    kind="workspace_relative",
                    path=relative,
                ),
            ),
        )
    return DatasetTableSetV1(schema_version="dataset-table-set/1", **references)


def _build_manifest(
    sidecar: CollectionSidecarV1,
    tables: Mapping[TableName, DatasetTable],
    staging: Path,
) -> tuple[RawDatasetManifestV1, LineageInventoryV1, QuarantineInventoryV1]:
    table_set = _write_tables(sidecar, tables, staging)
    recordings = cast(RecordingsTableV1, tables["recordings"]).rows
    inventory = build_raw_lineage_inventory(sidecar, recordings)
    quarantine_inventory = build_quarantine_inventory(sidecar)
    _write_json_durably(staging / COLLECTION_SIDECAR_FILENAME, sidecar)
    _write_json_durably(staging / LINEAGE_INVENTORY_FILENAME, inventory)
    _write_json_durably(staging / QUARANTINE_INVENTORY_FILENAME, quarantine_inventory)
    content = RawDatasetContentV1(
        schema_version="raw-dataset-content/1",
        taxonomy=sidecar.taxonomy,
        governance_policy=sidecar.governance_policy.policy_document,
        lineage_inventory_sha256=inventory.inventory_sha256,
        collection_sidecar_sha256=sidecar.collection_sidecar_sha256,
        tables=table_set,
    )
    manifest = RawDatasetManifestV1(
        schema_version="raw-dataset-manifest/1",
        dataset_id=sidecar.dataset_id,
        version=sidecar.dataset_version,
        content=content,
        raw_data_sha256=raw_dataset_content_digest(content),
    )
    return manifest, inventory, quarantine_inventory


def _load_existing(destination: Path) -> ValidatedRawDatasetBundle:
    try:
        if destination.is_symlink() or not destination.is_dir():
            raise DatasetImportError("destination.invalid")
        manifest_path = destination / RAW_MANIFEST_FILENAME
        return validate_raw_dataset_bundle(manifest_path.read_bytes(), destination)
    except DatasetImportError:
        raise
    except (OSError, RawDatasetBundleError, RuntimeError, ValueError) as error:
        raise DatasetImportError("destination.invalid") from error


def _existing_is_identical(
    destination: Path,
    manifest: RawDatasetManifestV1,
    sidecar: CollectionSidecarV1,
    inventory: LineageInventoryV1,
    quarantine_inventory: QuarantineInventoryV1,
) -> ValidatedRawDatasetBundle:
    existing = _load_existing(destination)
    if (
        existing.manifest != manifest
        or existing.sidecar != sidecar
        or existing.inventory != inventory
        or existing.quarantine_inventory != quarantine_inventory
    ):
        raise DatasetImportError("destination.conflict")
    return existing


def _publish_or_reconcile(
    staging: Path,
    destination: Path,
    manifest: RawDatasetManifestV1,
    sidecar: CollectionSidecarV1,
    inventory: LineageInventoryV1,
    quarantine_inventory: QuarantineInventoryV1,
) -> tuple[Literal["published", "unchanged"], ValidatedRawDatasetBundle]:
    if destination.exists() and any(destination.iterdir()):
        return "unchanged", _existing_is_identical(
            destination,
            manifest,
            sidecar,
            inventory,
            quarantine_inventory,
        )

    removed_empty = False
    try:
        if destination.is_symlink():
            raise DatasetImportError("destination.invalid")
        if destination.exists():
            if not destination.is_dir():
                raise DatasetImportError("destination.invalid")
            destination.rmdir()
            removed_empty = True
        staging.rename(destination)
        with suppress(OSError):
            _fsync_directory(destination.parent)
        return "published", validate_raw_dataset_bundle(manifest, destination)
    except DatasetImportError:
        raise
    except (OSError, RawDatasetBundleError, RuntimeError, ValueError) as error:
        if removed_empty and not destination.exists():
            with suppress(OSError):
                destination.mkdir()
        # A concurrent identical importer may have won the atomic rename.
        if destination.exists():
            try:
                return "unchanged", _existing_is_identical(
                    destination,
                    manifest,
                    sidecar,
                    inventory,
                    quarantine_inventory,
                )
            except DatasetImportError:
                pass
        raise DatasetImportError("publication.failed") from error


def import_collection_sidecar(
    sidecar: CollectionSidecarInput,
    *,
    source_root: str | Path,
    source_map: SourceMap,
    destination: str | Path,
) -> ImportedRawDatasetBundle:
    """Validate, normalize, and atomically publish one complete collection.

    Replaying byte-identical inputs verifies and returns the existing bundle as
    ``unchanged``.  Any difference at an occupied destination fails closed.
    """

    try:
        checked_sidecar = require_importable_sidecar(sidecar)
    except (TypeError, ValueError) as error:
        raise DatasetImportError("sidecar.invalid") from error
    # Story #17 has no authenticated collection-readiness or private-storage
    # verifier.  Never turn a caller assertion into authorization for real media.
    if checked_sidecar.fixture_only is not True:
        raise DatasetImportError("sidecar.invalid")
    if checked_sidecar.taxonomy != taxonomy_reference(load_builtin_taxonomy()):
        raise DatasetImportError("sidecar.invalid")
    root = _require_source_root(source_root)
    fingerprints = _preflight_sources(checked_sidecar, root, source_map)
    try:
        tables = _build_tables(checked_sidecar)
    except DatasetImportError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise DatasetImportError("sidecar.invalid") from error

    output = Path(destination)
    try:
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise DatasetImportError("destination.invalid")
        output.parent.mkdir(parents=True, exist_ok=True)
    except DatasetImportError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetImportError("destination.invalid") from error

    try:
        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.staging-",
        ) as temporary_directory:
            staging = Path(temporary_directory)
            _stage_attempt_bytes(checked_sidecar, fingerprints, staging)
            _checkpoint("media")
            manifest, inventory, quarantine_inventory = _build_manifest(
                checked_sidecar,
                tables,
                staging,
            )
            _checkpoint("metadata")
            # The manifest is the completion marker and is always written last.
            _write_json_durably(staging / RAW_MANIFEST_FILENAME, manifest)
            _checkpoint("manifest")
            validate_raw_dataset_bundle(manifest, staging)
            for directory in (staging / "tables", staging):
                _fsync_directory(directory)
            status, published = _publish_or_reconcile(
                staging,
                output,
                manifest,
                checked_sidecar,
                inventory,
                quarantine_inventory,
            )
    except DatasetImportError:
        raise
    except RawDatasetBundleError as error:
        raise DatasetImportError("publication.failed") from error
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DatasetImportError("publication.failed") from error

    attempts = _all_attempts(checked_sidecar)
    return ImportedRawDatasetBundle(
        status=status,
        manifest=published.manifest,
        manifest_path=output / RAW_MANIFEST_FILENAME,
        validation=published.validation,
        accepted_recordings=sum(attempt.outcome == "accepted" for attempt in attempts),
        retry_attempts=sum(attempt.outcome == "retry" for attempt in attempts),
        quarantined_attempts=sum(attempt.outcome == "quarantined" for attempt in attempts),
        skipped_occurrences=sum(
            occurrence.state == "skipped" for occurrence in checked_sidecar.occurrences
        ),
    )


__all__ = [
    "CollectionSidecarInput",
    "DatasetImportError",
    "DatasetImportErrorCategory",
    "ImportedRawDatasetBundle",
    "SourceLocation",
    "SourceMap",
    "import_collection_sidecar",
]
