"""Fail-closed semantic validation for table-backed dataset manifests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, Never, cast

from pydantic import BaseModel

from signlab.contracts.core import same_artifact_reference
from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    AnnotationRowV1,
    AnnotationsTableV1,
    ClipRowV1,
    ClipsTableV1,
    DatasetManifestV2,
    DatasetTable,
    DatasetTableInput,
    DerivedArtifactRowV1,
    DerivedArtifactsTableV1,
    ParticipantsTableV1,
    RecordingRowV1,
    RecordingsTableV1,
    SessionsTableV1,
    TableName,
    dataset_table_digest,
    validate_dataset_table,
)
from signlab.contracts.governance import (
    ConsentAuthorizationVerifier,
    ConsentEventLogV1,
    ConsentReceiptV1,
    DocumentRef,
    RecordingConsentGrantV1,
    ScopePermission,
    grant_authorizes_at,
)
from signlab.contracts.pipeline import (
    PipelineContractError,
    SplitManifestV1,
    assert_split_compatible,
    validate_dataset_manifest_v2,
    validate_split_manifest,
)

type DatasetValidationCategory = Literal[
    "authorization.dependencies",
    "authorization.denied",
    "consent.binding",
    "contract.invalid",
    "foreign_key.invalid",
    "group.invalid",
    "identity.invalid",
    "interval.invalid",
    "label.invalid",
    "lineage.invalid",
    "sample_projection.invalid",
    "split.invalid",
    "table.inventory",
    "table.reference",
]
type VerificationState = Literal["verified", "not_checked"]
type DatasetManifestV2Input = DatasetManifestV2 | str | bytes | bytearray | Mapping[str, object]
type ConsentEvidenceLookup = Callable[
    [RecordingConsentGrantV1], tuple[ConsentReceiptV1, ConsentEventLogV1]
]

_ERROR_MESSAGES: Final[dict[DatasetValidationCategory, str]] = {
    "authorization.dependencies": (
        "current consent authorization requires every authenticated dependency"
    ),
    "authorization.denied": "current consent authorization was not verified",
    "consent.binding": "dataset consent evidence is incompatible",
    "contract.invalid": "dataset manifest is invalid",
    "foreign_key.invalid": "dataset table relationships are incomplete",
    "group.invalid": "dataset grouping metadata is inconsistent",
    "identity.invalid": "dataset identities are not canonical and unique",
    "interval.invalid": "dataset media intervals are inconsistent",
    "label.invalid": "dataset annotation labels are inconsistent",
    "lineage.invalid": "dataset artifact lineage is inconsistent",
    "sample_projection.invalid": "dataset sample projection is inconsistent",
    "split.invalid": "dataset split inheritance was not verified",
    "table.inventory": "dataset requires exactly the six registered tables",
    "table.reference": "dataset table identity does not match its manifest reference",
}

_EXPECTED_TABLE_TYPES: Final[dict[TableName, type[DatasetTable]]] = {
    "participants": ParticipantsTableV1,
    "sessions": SessionsTableV1,
    "recordings": RecordingsTableV1,
    "clips": ClipsTableV1,
    "annotations": AnnotationsTableV1,
    "derived_artifacts": DerivedArtifactsTableV1,
}
_TABLE_PRIMARY_KEYS: Final[dict[TableName, str]] = {
    "participants": "participant_id",
    "sessions": "session_id",
    "recordings": "recording_id",
    "clips": "clip_id",
    "annotations": "annotation_id",
    "derived_artifacts": "derived_artifact_id",
}


class DatasetValidationError(ValueError):
    """A sanitized dataset failure with a stable machine-readable category."""

    def __init__(self, category: DatasetValidationCategory) -> None:
        self.category = category
        self.code = f"dataset.{category}"
        super().__init__(_ERROR_MESSAGES[category])


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    """Semantic checks only; Parquet/artifact bytes require the storage verifier."""

    semantic_integrity: Literal["verified"]
    artifact_byte_integrity: Literal["not_checked"]
    split_compatibility: VerificationState
    consent_authorization: VerificationState


@dataclass(frozen=True, slots=True)
class _CheckedTables:
    participants: ParticipantsTableV1
    sessions: SessionsTableV1
    recordings: RecordingsTableV1
    clips: ClipsTableV1
    annotations: AnnotationsTableV1
    derived_artifacts: DerivedArtifactsTableV1


@dataclass(frozen=True, slots=True)
class _LineageNode:
    parent_artifact_ids: tuple[str, ...]
    participant_id: str
    session_id: str
    source_recording_id: str
    root_recording_id: str | None
    split_id: str | None
    partition: str | None
    is_derived: bool


def _fail(category: DatasetValidationCategory) -> Never:
    raise DatasetValidationError(category) from None


def _validate_manifest(document: DatasetManifestV2Input) -> DatasetManifestV2:
    try:
        return validate_dataset_manifest_v2(document)
    except (TypeError, ValueError):
        _fail("contract.invalid")


def _validate_tables(
    manifest: DatasetManifestV2,
    tables: Mapping[str, DatasetTableInput],
) -> _CheckedTables:
    if set(tables) != set(DATASET_TABLE_SCHEMA_VERSIONS):
        _fail("table.inventory")

    checked: dict[TableName, DatasetTable] = {}
    for table_name in DATASET_TABLE_SCHEMA_VERSIONS:
        untrusted_table = tables[table_name]
        if isinstance(untrusted_table, _EXPECTED_TABLE_TYPES[table_name]):
            primary_key = _TABLE_PRIMARY_KEYS[table_name]
            identities = tuple(cast(str, getattr(row, primary_key)) for row in untrusted_table.rows)
            if identities != tuple(sorted(set(identities))):
                _fail("identity.invalid")
        try:
            table = validate_dataset_table(untrusted_table)
        except (KeyError, TypeError, ValueError):
            _fail("table.reference")
        if not isinstance(table, _EXPECTED_TABLE_TYPES[table_name]):
            _fail("table.reference")

        reference = getattr(manifest.content.tables, table_name)
        try:
            content_sha256 = dataset_table_digest(table)
        except (TypeError, ValueError):
            _fail("table.reference")
        if (
            reference.table_name != table_name
            or reference.table_schema_version != table.schema_version
            or reference.row_count != len(table.rows)
            or reference.content_sha256 != content_sha256
        ):
            _fail("table.reference")
        checked[table_name] = table

    return _CheckedTables(
        participants=cast(ParticipantsTableV1, checked["participants"]),
        sessions=cast(SessionsTableV1, checked["sessions"]),
        recordings=cast(RecordingsTableV1, checked["recordings"]),
        clips=cast(ClipsTableV1, checked["clips"]),
        annotations=cast(AnnotationsTableV1, checked["annotations"]),
        derived_artifacts=cast(DerivedArtifactsTableV1, checked["derived_artifacts"]),
    )


def _unique_by[RowT: BaseModel](
    rows: tuple[RowT, ...],
    field_name: str,
) -> dict[str, RowT]:
    identities = tuple(cast(str, getattr(row, field_name)) for row in rows)
    if identities != tuple(sorted(set(identities))):
        _fail("identity.invalid")
    return dict(zip(identities, rows, strict=True))


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _reconcile_relations(
    manifest: DatasetManifestV2,
    tables: _CheckedTables,
) -> tuple[
    dict[str, RecordingRowV1],
    dict[str, ClipRowV1],
    dict[str, AnnotationRowV1],
    dict[str, DerivedArtifactRowV1],
]:
    participants = _unique_by(tables.participants.rows, "participant_id")
    sessions = _unique_by(tables.sessions.rows, "session_id")
    recordings = _unique_by(tables.recordings.rows, "recording_id")
    clips = _unique_by(tables.clips.rows, "clip_id")
    annotations = _unique_by(tables.annotations.rows, "annotation_id")
    derived = _unique_by(tables.derived_artifacts.rows, "derived_artifact_id")
    grant_ids = tuple(recording.consent_grant.grant_id for recording in recordings.values())
    if len(grant_ids) != len(set(grant_ids)):
        _fail("identity.invalid")
    table_artifact_ids = {
        getattr(manifest.content.tables, table_name).artifact.artifact_id
        for table_name in DATASET_TABLE_SCHEMA_VERSIONS
    }
    row_artifact_ids = {
        *(recording.media.artifact_id for recording in recordings.values()),
        *(clip.artifact.artifact_id for clip in clips.values() if clip.artifact is not None),
        *(row.artifact.artifact_id for row in derived.values()),
    }
    if table_artifact_ids.intersection(row_artifact_ids):
        _fail("identity.invalid")

    for session in sessions.values():
        if session.participant_id not in participants:
            _fail("foreign_key.invalid")

    for recording in recordings.values():
        related_session = sessions.get(recording.session_id)
        if related_session is None:
            _fail("foreign_key.invalid")
        if (
            recording.participant_id != related_session.participant_id
            or recording.device_id != related_session.device_id
            or recording.mirror_state != related_session.mirror_state
            or recording.rotation_degrees != related_session.rotation_degrees
        ):
            _fail("group.invalid")
        captured_at = _parse_utc(recording.captured_at)
        session_started_at = _parse_utc(related_session.started_at)
        session_finished_at = _parse_utc(related_session.finished_at)
        if not session_started_at <= captured_at <= session_finished_at:
            _fail("interval.invalid")
        remaining = session_finished_at - captured_at
        remaining_us = (
            remaining.days * 86_400 + remaining.seconds
        ) * 1_000_000 + remaining.microseconds
        if recording.duration_us > remaining_us:
            _fail("interval.invalid")
        if recording.consent_grant.taxonomy != manifest.content.taxonomy:
            _fail("consent.binding")
        if not recording.consent_grant.scope.raw_media_retention:
            _fail("consent.binding")

    for clip in clips.values():
        source_recording = recordings.get(clip.source_recording_id)
        if source_recording is None:
            _fail("foreign_key.invalid")
        if (
            clip.participant_id != source_recording.participant_id
            or clip.session_id != source_recording.session_id
            or clip.handedness != source_recording.handedness
        ):
            _fail("group.invalid")
        if clip.mirror_state != source_recording.mirror_state:
            _fail("group.invalid")
        if clip.interval.end_us > source_recording.duration_us:
            _fail("interval.invalid")

    for annotation in annotations.values():
        source_recording = recordings.get(annotation.source_recording_id)
        if source_recording is None:
            _fail("foreign_key.invalid")
        if (
            annotation.participant_id != source_recording.participant_id
            or annotation.session_id != source_recording.session_id
        ):
            _fail("group.invalid")
        if annotation.interval.end_us > source_recording.duration_us:
            _fail("interval.invalid")
        if annotation.clip_id is not None:
            source_clip = clips.get(annotation.clip_id)
            if source_clip is None:
                _fail("foreign_key.invalid")
            if (
                source_clip.source_recording_id != annotation.source_recording_id
                or source_clip.participant_id != annotation.participant_id
                or source_clip.session_id != annotation.session_id
            ):
                _fail("group.invalid")
            if (
                annotation.interval.start_us < source_clip.interval.start_us
                or annotation.interval.end_us > source_clip.interval.end_us
            ):
                _fail("interval.invalid")

    samples_by_recording: dict[str, bool] = {}
    for row in derived.values():
        source_recording = recordings.get(row.source_recording_id)
        if source_recording is None:
            _fail("foreign_key.invalid")
        if (
            row.participant_id != source_recording.participant_id
            or row.session_id != source_recording.session_id
            or row.handedness != source_recording.handedness
            or row.mirror_state != source_recording.mirror_state
        ):
            _fail("group.invalid")
        derived_clip: ClipRowV1 | None = None
        if row.clip_id is not None:
            derived_clip = clips.get(row.clip_id)
            if derived_clip is None:
                _fail("foreign_key.invalid")
            if (
                derived_clip.source_recording_id != row.source_recording_id
                or derived_clip.participant_id != row.participant_id
                or derived_clip.session_id != row.session_id
            ):
                _fail("group.invalid")
        if row.annotation_id is not None:
            source_annotation = annotations.get(row.annotation_id)
            if source_annotation is None:
                _fail("foreign_key.invalid")
            if (
                source_annotation.source_recording_id != row.source_recording_id
                or source_annotation.participant_id != row.participant_id
                or source_annotation.session_id != row.session_id
                or (
                    source_annotation.clip_id is not None
                    and source_annotation.clip_id != row.clip_id
                )
            ):
                _fail("group.invalid")
            if derived_clip is not None and (
                source_annotation.interval.start_us < derived_clip.interval.start_us
                or source_annotation.interval.end_us > derived_clip.interval.end_us
            ):
                _fail("interval.invalid")
        if row.sample_id is not None:
            if row.annotation_id is None:
                _fail("label.invalid")
            annotation = annotations[row.annotation_id]
            if (
                not annotation.eligible_for_training
                or annotation.disposition != "class_label"
                or annotation.label_id != row.label_id
            ):
                _fail("label.invalid")
            samples_by_recording[row.source_recording_id] = True

    derived_recording_ids = {row.source_recording_id for row in derived.values()}
    for recording_id in derived_recording_ids:
        if not recordings[recording_id].consent_grant.scope.derived_features:
            _fail("consent.binding")
    for recording_id in samples_by_recording:
        if not recordings[recording_id].consent_grant.scope.model_training:
            _fail("consent.binding")

    return recordings, clips, annotations, derived


def _validate_lineage(
    recordings: Mapping[str, RecordingRowV1],
    clips: Mapping[str, ClipRowV1],
    derived: Mapping[str, DerivedArtifactRowV1],
) -> None:
    nodes: dict[str, _LineageNode] = {}

    def add_node(artifact_id: str, node: _LineageNode) -> None:
        if artifact_id in nodes:
            _fail("identity.invalid")
        nodes[artifact_id] = node

    for recording in recordings.values():
        add_node(
            recording.media.artifact_id,
            _LineageNode(
                parent_artifact_ids=(),
                participant_id=recording.participant_id,
                session_id=recording.session_id,
                source_recording_id=recording.recording_id,
                root_recording_id=recording.recording_id,
                split_id=None,
                partition=None,
                is_derived=False,
            ),
        )
    for clip in clips.values():
        if clip.artifact is None:
            continue
        recording = recordings[clip.source_recording_id]
        add_node(
            clip.artifact.artifact_id,
            _LineageNode(
                parent_artifact_ids=(recording.media.artifact_id,),
                participant_id=clip.participant_id,
                session_id=clip.session_id,
                source_recording_id=clip.source_recording_id,
                root_recording_id=None,
                split_id=None,
                partition=None,
                is_derived=False,
            ),
        )
    for row in derived.values():
        add_node(
            row.artifact.artifact_id,
            _LineageNode(
                parent_artifact_ids=row.parent_artifact_ids,
                participant_id=row.participant_id,
                session_id=row.session_id,
                source_recording_id=row.source_recording_id,
                root_recording_id=None,
                split_id=row.split_id,
                partition=row.partition,
                is_derived=True,
            ),
        )

    states: dict[str, int] = {}
    roots_by_artifact: dict[str, frozenset[str]] = {}

    for starting_artifact_id in tuple(nodes):
        if states.get(starting_artifact_id) == 2:
            continue
        stack: list[tuple[str, bool]] = [(starting_artifact_id, False)]
        while stack:
            artifact_id, parents_visited = stack.pop()
            node = nodes.get(artifact_id)
            if node is None:
                _fail("lineage.invalid")

            state = states.get(artifact_id, 0)
            if parents_visited:
                if state == 2:
                    continue
                if state != 1:
                    _fail("lineage.invalid")
                if node.root_recording_id is not None:
                    result = frozenset({node.root_recording_id})
                else:
                    root_ids: set[str] = set()
                    for parent_id in node.parent_artifact_ids:
                        if states.get(parent_id) != 2:
                            _fail("lineage.invalid")
                        resolved_parent = nodes[parent_id]
                        if (
                            resolved_parent.is_derived
                            and resolved_parent.split_id is not None
                            and (
                                resolved_parent.split_id != node.split_id
                                or resolved_parent.partition != node.partition
                            )
                        ):
                            _fail("split.invalid")
                        root_ids.update(roots_by_artifact[parent_id])
                    result = frozenset(root_ids)
                if result != frozenset({node.source_recording_id}):
                    _fail("lineage.invalid")
                states[artifact_id] = 2
                roots_by_artifact[artifact_id] = result
                continue

            if state == 2:
                continue
            if state == 1:
                _fail("lineage.invalid")
            states[artifact_id] = 1
            stack.append((artifact_id, True))
            for parent_id in reversed(node.parent_artifact_ids):
                parent = nodes.get(parent_id)
                if parent is None:
                    _fail("lineage.invalid")
                if (
                    parent.participant_id != node.participant_id
                    or parent.session_id != node.session_id
                    or parent.source_recording_id != node.source_recording_id
                ):
                    _fail("lineage.invalid")
                if states.get(parent_id) == 1:
                    _fail("lineage.invalid")
                if states.get(parent_id) != 2:
                    stack.append((parent_id, False))


def _validate_sample_projection(
    manifest: DatasetManifestV2,
    derived: Mapping[str, DerivedArtifactRowV1],
) -> dict[str, DerivedArtifactRowV1]:
    sample_rows = [row for row in derived.values() if row.sample_id is not None]
    sample_ids = tuple(cast(str, row.sample_id) for row in sample_rows)
    if len(sample_ids) != len(set(sample_ids)):
        _fail("sample_projection.invalid")
    by_sample = dict(zip(sample_ids, sample_rows, strict=True))
    projected_ids = tuple(sample.sample_id for sample in manifest.content.samples)
    if projected_ids != tuple(sorted(by_sample)):
        _fail("sample_projection.invalid")
    for sample in manifest.content.samples:
        row = by_sample[sample.sample_id]
        if (
            sample.participant_id != row.participant_id
            or sample.session_id != row.session_id
            or sample.source_recording_id != row.source_recording_id
            or sample.label_id != row.label_id
            or not same_artifact_reference(sample.artifact, row.artifact)
        ):
            _fail("sample_projection.invalid")
    return by_sample


def _validate_split(
    manifest: DatasetManifestV2,
    split: SplitManifestV1 | None,
    sample_rows: Mapping[str, DerivedArtifactRowV1],
    annotations: Mapping[str, AnnotationRowV1],
) -> VerificationState:
    for row in sample_rows.values():
        if row.annotation_id is not None:
            annotation = annotations[row.annotation_id]
            if annotation.other_kind == "transition_fragment" and row.partition == "test":
                _fail("split.invalid")
    if split is None:
        return "not_checked"
    try:
        checked_split = validate_split_manifest(split)
        assert_split_compatible(manifest, checked_split)
    except (PipelineContractError, TypeError, ValueError):
        _fail("split.invalid")

    assigned: dict[str, str] = {}
    for partition in checked_split.partitions:
        for sample_id in partition.sample_ids:
            assigned[sample_id] = partition.name
    for sample_id, row in sample_rows.items():
        if row.split_id != checked_split.split_id or row.partition != assigned.get(sample_id):
            _fail("split.invalid")
    return "verified"


def _authorize_current_consent(
    recordings: Mapping[str, RecordingRowV1],
    *,
    governance_policy: DocumentRef,
    evidence_lookup: ConsentEvidenceLookup | None,
    authorization_verifier: ConsentAuthorizationVerifier | None,
    permission: ScopePermission | None,
    authorization_at: str | None,
) -> VerificationState:
    dependencies = (
        evidence_lookup,
        authorization_verifier,
        permission,
        authorization_at,
    )
    if all(dependency is None for dependency in dependencies):
        return "not_checked"
    if any(dependency is None for dependency in dependencies):
        _fail("authorization.dependencies")

    lookup = cast(ConsentEvidenceLookup, evidence_lookup)
    verifier = cast(ConsentAuthorizationVerifier, authorization_verifier)
    checked_permission = cast(ScopePermission, permission)
    checked_at = cast(str, authorization_at)
    for recording in recordings.values():
        try:
            receipt, event_log = lookup(recording.consent_grant)
            authorized = receipt.governance_policy == governance_policy and grant_authorizes_at(
                recording.consent_grant,
                receipt,
                event_log,
                checked_permission,
                checked_at,
                purpose_id=recording.consent_grant.purpose_id,
                study_id=recording.consent_grant.study_id,
                authorization_verifier=verifier,
            )
        except Exception:
            _fail("authorization.denied")
        if authorized is not True:
            _fail("authorization.denied")
    return "verified"


def validate_dataset_manifest_tables(
    manifest: DatasetManifestV2Input,
    tables: Mapping[str, DatasetTableInput],
    *,
    split: SplitManifestV1 | None = None,
    consent_evidence_lookup: ConsentEvidenceLookup | None = None,
    consent_authorization_verifier: ConsentAuthorizationVerifier | None = None,
    authorization_permission: ScopePermission | None = None,
    authorization_at: str | None = None,
) -> DatasetValidationResult:
    """Validate one complete table-backed dataset and optional external evidence.

    Structural consent snapshots are always reconciled. Current authorization and
    split inheritance are reported as verified only when their exact dependencies
    are supplied and pass.
    """

    checked_manifest = _validate_manifest(manifest)
    checked_tables = _validate_tables(checked_manifest, tables)
    recordings, clips, annotations, derived = _reconcile_relations(
        checked_manifest,
        checked_tables,
    )
    _validate_lineage(recordings, clips, derived)
    sample_rows = _validate_sample_projection(checked_manifest, derived)
    split_state = _validate_split(checked_manifest, split, sample_rows, annotations)
    consent_state = _authorize_current_consent(
        recordings,
        governance_policy=checked_manifest.content.governance_policy,
        evidence_lookup=consent_evidence_lookup,
        authorization_verifier=consent_authorization_verifier,
        permission=authorization_permission,
        authorization_at=authorization_at,
    )
    return DatasetValidationResult(
        semantic_integrity="verified",
        artifact_byte_integrity="not_checked",
        split_compatibility=split_state,
        consent_authorization=consent_state,
    )


__all__ = [
    "ConsentEvidenceLookup",
    "DatasetValidationCategory",
    "DatasetValidationError",
    "DatasetValidationResult",
    "validate_dataset_manifest_tables",
]
