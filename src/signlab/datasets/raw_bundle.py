"""Validation helpers for an atomically published raw ingest bundle.

The raw bundle is a hand-off between collection and sample extraction.  It is
deliberately separate from ``dataset-manifest/2``: raw recordings are not
trainable samples and therefore must not be made to satisfy a sample contract
by inventing derived data.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, cast

from signlab.contracts.core import ArtifactRefV1, WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    AnnotationsTableV1,
    ClipsTableV1,
    DatasetTable,
    DerivedArtifactsTableV1,
    ParticipantRowV1,
    ParticipantsTableV1,
    RecordingRowV1,
    RecordingsTableV1,
    SessionRowV1,
    SessionsTableV1,
    TableName,
    content_addressed_row_locations,
)
from signlab.contracts.governance import (
    GovernanceAssetV1,
    LineageInventoryV1,
    lineage_inventory_digest,
    validate_lineage_inventory,
)
from signlab.contracts.ingest import (
    AttemptReasonCode,
    CaptureAttemptV1,
    CollectionSidecarV1,
    RawDatasetManifestV1,
    project_annotation_rows,
    validate_collection_sidecar,
    validate_raw_dataset_manifest,
)
from signlab.contracts.quarantine import (
    QuarantinedAttemptAssetV1,
    QuarantineInventoryV1,
    quarantine_attempt_workspace_path,
    quarantine_inventory_digest,
    validate_quarantine_inventory,
)
from signlab.datasets.parquet import read_dataset_table
from signlab.datasets.storage import DatasetStorageError, verify_dataset_row_artifacts

type RawDatasetManifestInput = RawDatasetManifestV1 | str | bytes | bytearray | Mapping[str, object]
type RawDatasetBundleErrorCategory = Literal[
    "manifest.invalid",
    "metadata.invalid",
    "table_bytes.invalid",
    "semantics.invalid",
    "row_artifact_bytes.invalid",
]

RAW_MANIFEST_FILENAME: Final = "raw-dataset-manifest.json"
COLLECTION_SIDECAR_FILENAME: Final = "collection-sidecar.json"
LINEAGE_INVENTORY_FILENAME: Final = "lineage-inventory.json"
QUARANTINE_INVENTORY_FILENAME: Final = "quarantine-inventory.json"
_MAX_METADATA_BYTES: Final = 16 * 1024 * 1024
_ERROR_MESSAGES: Final[dict[RawDatasetBundleErrorCategory, str]] = {
    "manifest.invalid": "raw dataset bundle manifest is invalid",
    "metadata.invalid": "raw dataset bundle metadata could not be verified",
    "table_bytes.invalid": "raw dataset bundle table bytes could not be verified",
    "semantics.invalid": "raw dataset bundle semantic relationships are invalid",
    "row_artifact_bytes.invalid": "raw dataset bundle artifact bytes could not be verified",
}


class RawDatasetBundleError(ValueError):
    """A stable raw-bundle failure that does not reveal source locations."""

    def __init__(self, category: RawDatasetBundleErrorCategory) -> None:
        self.category = category
        self.code = f"dataset.raw_bundle.{category}"
        super().__init__(_ERROR_MESSAGES[category])


@dataclass(frozen=True, slots=True)
class RawDatasetBundleValidationResult:
    """Positive evidence returned only after the entire raw bundle is checked."""

    raw_data_sha256: str
    parquet_table_bytes: Literal["verified"]
    semantic_integrity: Literal["verified"]
    artifact_byte_integrity: Literal["verified"]
    collection_sidecar_integrity: Literal["verified"]
    lineage_inventory_integrity: Literal["verified"]
    quarantine_inventory_integrity: Literal["verified"]
    consent_authorization: Literal["not_checked"]


@dataclass(frozen=True, slots=True)
class ValidatedRawDatasetBundle:
    """Validated contracts and normalized tables loaded from a published bundle."""

    manifest: RawDatasetManifestV1
    sidecar: CollectionSidecarV1
    inventory: LineageInventoryV1
    quarantine_inventory: QuarantineInventoryV1
    tables: Mapping[TableName, DatasetTable]
    validation: RawDatasetBundleValidationResult


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _read_fixed_metadata(root: Path, filename: str) -> bytes:
    path = root / filename
    try:
        if _is_linklike(path):
            raise RawDatasetBundleError("metadata.invalid")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise RawDatasetBundleError("metadata.invalid")
        if resolved.stat().st_size > _MAX_METADATA_BYTES:
            raise RawDatasetBundleError("metadata.invalid")
        return resolved.read_bytes()
    except RawDatasetBundleError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise RawDatasetBundleError("metadata.invalid") from error


def _load_tables(
    manifest: RawDatasetManifestV1,
    root: Path,
) -> dict[TableName, DatasetTable]:
    tables: dict[TableName, DatasetTable] = {}
    try:
        for table_name in DATASET_TABLE_SCHEMA_VERSIONS:
            reference = getattr(manifest.content.tables, table_name)
            tables[table_name] = read_dataset_table(reference, root)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        raise RawDatasetBundleError("table_bytes.invalid") from error
    return tables


def _accepted_attempt(sidecar: CollectionSidecarV1, recording_id: str) -> CaptureAttemptV1:
    for occurrence in sidecar.occurrences:
        if occurrence.state == "accepted":
            accepted = tuple(
                attempt for attempt in occurrence.attempts if attempt.outcome == "accepted"
            )
            if len(accepted) == 1 and accepted[0].recording_id == recording_id:
                return accepted[0]
    raise RawDatasetBundleError("semantics.invalid")


def _expected_participants(
    sidecar: CollectionSidecarV1,
    recordings: tuple[RecordingRowV1, ...],
) -> tuple[ParticipantRowV1, ...]:
    used = {recording.participant_id for recording in recordings}
    return tuple(row for row in sidecar.participants if row.participant_id in used)


def _expected_sessions(
    sidecar: CollectionSidecarV1,
    recordings: tuple[RecordingRowV1, ...],
) -> tuple[SessionRowV1, ...]:
    used = {recording.session_id for recording in recordings}
    return tuple(row for row in sidecar.sessions if row.session_id in used)


def _canonical_recording_path(recording: RecordingRowV1) -> str:
    return content_addressed_row_locations(recording.media)[0]


def _validate_normalized_semantics(
    manifest: RawDatasetManifestV1,
    sidecar: CollectionSidecarV1,
    tables: Mapping[TableName, DatasetTable],
) -> tuple[RecordingRowV1, ...]:
    try:
        participants = cast(ParticipantsTableV1, tables["participants"])
        sessions = cast(SessionsTableV1, tables["sessions"])
        recordings_table = cast(RecordingsTableV1, tables["recordings"])
        clips = cast(ClipsTableV1, tables["clips"])
        annotations = cast(AnnotationsTableV1, tables["annotations"])
        derived = cast(DerivedArtifactsTableV1, tables["derived_artifacts"])
        recordings = recordings_table.rows

        if clips.rows or derived.rows:
            raise RawDatasetBundleError("semantics.invalid")
        if manifest.dataset_id != sidecar.dataset_id or manifest.version != sidecar.dataset_version:
            raise RawDatasetBundleError("semantics.invalid")
        if manifest.content.taxonomy != sidecar.taxonomy:
            raise RawDatasetBundleError("semantics.invalid")
        if manifest.content.governance_policy != sidecar.governance_policy.policy_document:
            raise RawDatasetBundleError("semantics.invalid")
        if manifest.content.collection_sidecar_sha256 != sidecar.collection_sidecar_sha256:
            raise RawDatasetBundleError("semantics.invalid")
        if participants.rows != _expected_participants(sidecar, recordings):
            raise RawDatasetBundleError("semantics.invalid")
        if sessions.rows != _expected_sessions(sidecar, recordings):
            raise RawDatasetBundleError("semantics.invalid")
        if annotations.rows != project_annotation_rows(sidecar):
            raise RawDatasetBundleError("semantics.invalid")

        participant_by_id = {row.participant_id: row for row in participants.rows}
        session_by_id = {row.session_id: row for row in sessions.rows}
        accepted_ids = tuple(
            cast(str, occurrence.accepted_recording_id)
            for occurrence in sidecar.occurrences
            if occurrence.state == "accepted"
        )
        if tuple(recording.recording_id for recording in recordings) != tuple(sorted(accepted_ids)):
            raise RawDatasetBundleError("semantics.invalid")

        media_digests: set[str] = set()
        grant_ids: set[str] = set()
        for recording in recordings:
            attempt = _accepted_attempt(sidecar, recording.recording_id)
            session = session_by_id.get(recording.session_id)
            if session is None or recording.participant_id not in participant_by_id:
                raise RawDatasetBundleError("semantics.invalid")
            locator = recording.media.locator
            if not isinstance(locator, WorkspaceRelativeLocatorV1):
                raise RawDatasetBundleError("semantics.invalid")
            if (
                recording.participant_id
                != next(
                    occurrence.participant_id
                    for occurrence in sidecar.occurrences
                    if occurrence.accepted_recording_id == recording.recording_id
                )
                or recording.session_id
                != next(
                    occurrence.session_id
                    for occurrence in sidecar.occurrences
                    if occurrence.accepted_recording_id == recording.recording_id
                )
                or recording.device_id != session.device_id
                or recording.captured_at != attempt.recorded_at
                or recording.duration_us != attempt.duration_us
                or recording.handedness != attempt.handedness
                or recording.mirror_state != attempt.mirror_state
                or recording.rotation_degrees != attempt.rotation_degrees
                or recording.audio_present is not False
                or recording.media.artifact_id != recording.recording_id
                or recording.media.role != "raw_recording"
                or recording.media.media_type != attempt.media_type
                or recording.media.sha256 != attempt.expected_sha256
                or recording.media.size_bytes != attempt.expected_size_bytes
                or locator.path != _canonical_recording_path(recording)
                or recording.consent_grant != attempt.consent_grant
                or recording.consent_grant.taxonomy != sidecar.taxonomy
                or not recording.consent_grant.scope.raw_media_retention
            ):
                raise RawDatasetBundleError("semantics.invalid")
            captured_at = datetime.strptime(recording.captured_at, "%Y-%m-%dT%H:%M:%SZ")
            session_start = datetime.strptime(session.started_at, "%Y-%m-%dT%H:%M:%SZ")
            session_finish = datetime.strptime(session.finished_at, "%Y-%m-%dT%H:%M:%SZ")
            remaining = session_finish - captured_at
            remaining_us = (
                remaining.days * 86_400 + remaining.seconds
            ) * 1_000_000 + remaining.microseconds
            if (
                not session_start <= captured_at <= session_finish
                or recording.duration_us > remaining_us
            ):
                raise RawDatasetBundleError("semantics.invalid")
            if recording.media.sha256 in media_digests:
                raise RawDatasetBundleError("semantics.invalid")
            if recording.consent_grant.grant_id in grant_ids:
                raise RawDatasetBundleError("semantics.invalid")
            media_digests.add(recording.media.sha256)
            grant_ids.add(recording.consent_grant.grant_id)

        annotation_ids: set[str] = set()
        recording_by_id = {row.recording_id: row for row in recordings}
        for annotation in annotations.rows:
            source_recording = recording_by_id.get(annotation.source_recording_id)
            if (
                annotation.annotation_id in annotation_ids
                or source_recording is None
                or annotation.participant_id != source_recording.participant_id
                or annotation.session_id != source_recording.session_id
                or annotation.interval.end_us > source_recording.duration_us
            ):
                raise RawDatasetBundleError("semantics.invalid")
            annotation_ids.add(annotation.annotation_id)
    except RawDatasetBundleError:
        raise
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as error:
        raise RawDatasetBundleError("semantics.invalid") from error
    return recordings


def _lineage_asset_id(recording_id: str) -> str:
    identity = hashlib.sha256(recording_id.encode("ascii")).hexdigest()[:32]
    return f"asset_{identity}"


def build_raw_lineage_inventory(
    sidecar: CollectionSidecarV1,
    recordings: tuple[RecordingRowV1, ...],
) -> LineageInventoryV1:
    """Build the deterministic raw-recording inventory committed by the sidecar."""

    authority = sidecar.store_id
    assets = tuple(
        GovernanceAssetV1(
            schema_version="governance-asset/1",
            asset_id=(asset_id := _lineage_asset_id(recording.recording_id)),
            asset_kind="raw_recording",
            logical_uri=f"signlab://{authority}/raw_recording/{asset_id}",
            sha256=recording.media.sha256,
            taxonomy=sidecar.taxonomy,
            created_at=recording.captured_at,
            participant_ids=(recording.participant_id,),
            recording_ids=(recording.recording_id,),
            receipt_ids=(recording.consent_grant.receipt_id,),
            grant_ids=(recording.consent_grant.grant_id,),
            parent_asset_ids=(),
            lifecycle_state="active",
            invalidated_at=None,
        )
        for recording in recordings
    )
    payload: dict[str, object] = {
        "schema_version": "lineage-inventory/1",
        "inventory_id": sidecar.inventory_id,
        "taxonomy": sidecar.taxonomy.model_dump(mode="json", round_trip=True),
        "generated_at": cast(str, sidecar.finalized_at),
        "assets": [asset.model_dump(mode="json", round_trip=True) for asset in assets],
    }
    return LineageInventoryV1(
        schema_version="lineage-inventory/1",
        inventory_id=sidecar.inventory_id,
        taxonomy=sidecar.taxonomy,
        generated_at=cast(str, sidecar.finalized_at),
        assets=assets,
        inventory_sha256=lineage_inventory_digest(payload),
    )


def build_quarantine_inventory(sidecar: CollectionSidecarV1) -> QuarantineInventoryV1:
    """Build the exact unconsented inventory for every retained nonaccepted attempt."""

    assets: list[QuarantinedAttemptAssetV1] = []
    for occurrence in sidecar.occurrences:
        for attempt in occurrence.attempts:
            if attempt.outcome == "accepted":
                continue
            relative = quarantine_attempt_workspace_path(
                attempt.attempt_id,
                attempt.expected_sha256,
            )
            assets.append(
                QuarantinedAttemptAssetV1(
                    schema_version="quarantined-attempt-asset/1",
                    attempt_id=attempt.attempt_id,
                    recording_id=attempt.recording_id,
                    source_key=attempt.source_key,
                    participant_id=occurrence.participant_id,
                    session_id=occurrence.session_id,
                    outcome=attempt.outcome,
                    reason_code=cast(AttemptReasonCode, attempt.reason_code),
                    recorded_at=attempt.recorded_at,
                    artifact=ArtifactRefV1(
                        schema_version="artifact-reference/1",
                        artifact_id=attempt.attempt_id,
                        role="quarantined_capture_attempt",
                        media_type=attempt.media_type,
                        sha256=attempt.expected_sha256,
                        size_bytes=attempt.expected_size_bytes,
                        locator=WorkspaceRelativeLocatorV1(
                            kind="workspace_relative",
                            path=relative,
                        ),
                    ),
                    lifecycle_state="quarantined",
                    consent_evidence_status="absent",
                )
            )
    ordered = tuple(
        sorted(
            assets,
            key=lambda asset: (asset.participant_id, asset.recording_id, asset.attempt_id),
        )
    )
    payload: dict[str, object] = {
        "schema_version": "quarantine-inventory/1",
        "collection_id": sidecar.collection_id,
        "store_id": sidecar.store_id,
        "taxonomy": sidecar.taxonomy.model_dump(mode="json", round_trip=True),
        "collection_sidecar_sha256": sidecar.collection_sidecar_sha256,
        "generated_at": cast(str, sidecar.finalized_at),
        "assets": [asset.model_dump(mode="json", round_trip=True) for asset in ordered],
    }
    return QuarantineInventoryV1(
        schema_version="quarantine-inventory/1",
        collection_id=sidecar.collection_id,
        store_id=sidecar.store_id,
        taxonomy=sidecar.taxonomy,
        collection_sidecar_sha256=sidecar.collection_sidecar_sha256,
        generated_at=cast(str, sidecar.finalized_at),
        assets=ordered,
        quarantine_inventory_sha256=quarantine_inventory_digest(payload),
    )


def _verify_nonaccepted_attempts(
    inventory: QuarantineInventoryV1,
    root: Path,
) -> None:
    try:
        for asset in inventory.assets:
            locator = asset.artifact.locator
            if not isinstance(locator, WorkspaceRelativeLocatorV1):
                raise RawDatasetBundleError("row_artifact_bytes.invalid")
            path = root.joinpath(*locator.path.split("/"))
            if _is_linklike(path):
                raise RawDatasetBundleError("row_artifact_bytes.invalid")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise RawDatasetBundleError("row_artifact_bytes.invalid")
            if resolved.stat().st_size != asset.artifact.size_bytes:
                raise RawDatasetBundleError("row_artifact_bytes.invalid")
            actual = hashlib.sha256()
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(chunk)
            if f"sha256:{actual.hexdigest()}" != asset.artifact.sha256:
                raise RawDatasetBundleError("row_artifact_bytes.invalid")
    except RawDatasetBundleError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise RawDatasetBundleError("row_artifact_bytes.invalid") from error


def _expected_bundle_files(
    sidecar: CollectionSidecarV1,
    recordings: tuple[RecordingRowV1, ...],
    quarantine_inventory: QuarantineInventoryV1,
) -> set[str]:
    recording_by_id = {recording.recording_id: recording for recording in recordings}
    expected = {
        RAW_MANIFEST_FILENAME,
        COLLECTION_SIDECAR_FILENAME,
        LINEAGE_INVENTORY_FILENAME,
        QUARANTINE_INVENTORY_FILENAME,
        *(f"tables/{table_name}.parquet" for table_name in DATASET_TABLE_SCHEMA_VERSIONS),
    }
    for occurrence in sidecar.occurrences:
        for attempt in occurrence.attempts:
            if attempt.outcome == "accepted":
                recording = recording_by_id.get(attempt.recording_id)
                if recording is None:
                    raise RawDatasetBundleError("semantics.invalid")
                expected.add(_canonical_recording_path(recording))
    for asset in quarantine_inventory.assets:
        locator = asset.artifact.locator
        if not isinstance(locator, WorkspaceRelativeLocatorV1):
            raise RawDatasetBundleError("semantics.invalid")
        expected.add(locator.path)
    return expected


def _verify_exact_file_inventory(
    sidecar: CollectionSidecarV1,
    recordings: tuple[RecordingRowV1, ...],
    quarantine_inventory: QuarantineInventoryV1,
    root: Path,
) -> None:
    expected_files = _expected_bundle_files(sidecar, recordings, quarantine_inventory)
    expected_directories: set[str] = set()
    for filename in expected_files:
        parent = Path(filename).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    try:
        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in directories:
                candidate = current_path / name
                if _is_linklike(candidate):
                    raise RawDatasetBundleError("metadata.invalid")
                actual_directories.add(candidate.relative_to(root).as_posix())
            for name in filenames:
                candidate = current_path / name
                if _is_linklike(candidate) or not candidate.is_file():
                    raise RawDatasetBundleError("metadata.invalid")
                actual_files.add(candidate.relative_to(root).as_posix())
    except RawDatasetBundleError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise RawDatasetBundleError("metadata.invalid") from error
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RawDatasetBundleError("metadata.invalid")


def validate_raw_dataset_bundle(
    manifest: RawDatasetManifestInput,
    workspace_root: str | Path,
) -> ValidatedRawDatasetBundle:
    """Load and verify every contract, table, and media byte in a raw bundle."""

    try:
        checked_manifest = validate_raw_dataset_manifest(manifest)
    except (TypeError, ValueError) as error:
        raise RawDatasetBundleError("manifest.invalid") from error
    try:
        root_input = Path(workspace_root)
        if _is_linklike(root_input):
            raise RawDatasetBundleError("metadata.invalid")
        root = root_input.resolve(strict=True)
        if not root.is_dir():
            raise RawDatasetBundleError("metadata.invalid")
    except RawDatasetBundleError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RawDatasetBundleError("metadata.invalid") from error

    try:
        disk_manifest = validate_raw_dataset_manifest(
            _read_fixed_metadata(root, RAW_MANIFEST_FILENAME)
        )
    except (RawDatasetBundleError, TypeError, ValueError) as error:
        raise RawDatasetBundleError("manifest.invalid") from error
    if disk_manifest != checked_manifest:
        raise RawDatasetBundleError("manifest.invalid")

    try:
        sidecar = validate_collection_sidecar(
            _read_fixed_metadata(root, COLLECTION_SIDECAR_FILENAME)
        )
        inventory = validate_lineage_inventory(
            _read_fixed_metadata(root, LINEAGE_INVENTORY_FILENAME)
        )
        quarantine_inventory = validate_quarantine_inventory(
            _read_fixed_metadata(root, QUARANTINE_INVENTORY_FILENAME)
        )
    except RawDatasetBundleError:
        raise
    except (TypeError, ValueError) as error:
        raise RawDatasetBundleError("metadata.invalid") from error
    tables = _load_tables(checked_manifest, root)
    recordings = _validate_normalized_semantics(checked_manifest, sidecar, tables)
    expected_inventory = build_raw_lineage_inventory(sidecar, recordings)
    if (
        inventory != expected_inventory
        or inventory.inventory_sha256 != checked_manifest.content.lineage_inventory_sha256
    ):
        raise RawDatasetBundleError("semantics.invalid")
    expected_quarantine_inventory = build_quarantine_inventory(sidecar)
    if quarantine_inventory != expected_quarantine_inventory:
        raise RawDatasetBundleError("semantics.invalid")
    try:
        verify_dataset_row_artifacts(tables, root)
        _verify_nonaccepted_attempts(quarantine_inventory, root)
        _verify_exact_file_inventory(sidecar, recordings, quarantine_inventory, root)
    except DatasetStorageError as error:
        raise RawDatasetBundleError("row_artifact_bytes.invalid") from error

    validation = RawDatasetBundleValidationResult(
        raw_data_sha256=checked_manifest.raw_data_sha256,
        parquet_table_bytes="verified",
        semantic_integrity="verified",
        artifact_byte_integrity="verified",
        collection_sidecar_integrity="verified",
        lineage_inventory_integrity="verified",
        quarantine_inventory_integrity="verified",
        consent_authorization="not_checked",
    )
    return ValidatedRawDatasetBundle(
        manifest=checked_manifest,
        sidecar=sidecar,
        inventory=inventory,
        quarantine_inventory=quarantine_inventory,
        tables=tables,
        validation=validation,
    )


__all__ = [
    "COLLECTION_SIDECAR_FILENAME",
    "LINEAGE_INVENTORY_FILENAME",
    "QUARANTINE_INVENTORY_FILENAME",
    "RAW_MANIFEST_FILENAME",
    "RawDatasetBundleError",
    "RawDatasetBundleErrorCategory",
    "RawDatasetBundleValidationResult",
    "RawDatasetManifestInput",
    "ValidatedRawDatasetBundle",
    "build_quarantine_inventory",
    "build_raw_lineage_inventory",
    "validate_raw_dataset_bundle",
]
