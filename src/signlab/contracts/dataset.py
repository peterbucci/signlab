"""Versioned dataset manifests and normalized dataset-table contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Final, Literal, Self, cast, get_args

from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    ArtifactRefV1,
    NonNegativeSafeInteger,
    PositiveSafeInteger,
    SchemaName,
    SemanticVersion,
    StableId,
    StrictContractModel,
    UtcTimestamp,
    WorkspaceRelativeLocatorV1,
    contract_config,
)
from signlab.contracts.governance import (
    DocumentRef,
    ParticipantId,
    RecordingConsentGrantV1,
    RecordingId,
    validate_recording_consent_grant,
)
from signlab.contracts.taxonomy import (
    EXPECTED_CLASS_IDS,
    EXPECTED_OTHER_KINDS,
    Sha256Digest,
    TaxonomyRef,
)

SampleId = Annotated[str, StringConstraints(pattern=r"^sample_[0-9a-f]{32}$")]
SessionId = Annotated[str, StringConstraints(pattern=r"^session_[0-9a-f]{32}$")]
DeviceId = Annotated[str, StringConstraints(pattern=r"^device_[0-9a-f]{32}$")]
ClipId = Annotated[str, StringConstraints(pattern=r"^clip_[0-9a-f]{32}$")]
AnnotationId = Annotated[str, StringConstraints(pattern=r"^annotation_[0-9a-f]{32}$")]
DerivedArtifactId = Annotated[
    str,
    StringConstraints(pattern=r"^derived_artifact_[0-9a-f]{32}$"),
]

LabelId = Literal["hello", "no", "please", "thank_you", "yes", "other"]
PartitionName = Literal["train", "validation", "test"]
Handedness = Literal["left", "right", "both", "ambidextrous", "unknown"]
MirrorState = Literal["not_mirrored", "mirrored"]
TableName = Literal[
    "participants",
    "sessions",
    "recordings",
    "clips",
    "annotations",
    "derived_artifacts",
]
TableSchemaVersion = Literal[
    "participants-table/1",
    "sessions-table/1",
    "recordings-table/1",
    "clips-table/1",
    "annotations-table/1",
    "derived-artifacts-table/1",
]
OtherKind = Literal[
    "partial_target",
    "transition_fragment",
    "oov_gesture",
    "incidental_activity",
    "two_hand_non_target",
]
if get_args(LabelId) != EXPECTED_CLASS_IDS or get_args(OtherKind) != EXPECTED_OTHER_KINDS:
    raise RuntimeError("dataset label types must match the immutable taxonomy")

DATASET_TABLE_SCHEMA_VERSIONS: Final[dict[TableName, TableSchemaVersion]] = {
    "participants": "participants-table/1",
    "sessions": "sessions-table/1",
    "recordings": "recordings-table/1",
    "clips": "clips-table/1",
    "annotations": "annotations-table/1",
    "derived_artifacts": "derived-artifacts-table/1",
}
_SAMPLE_DERIVATION_KINDS: Final = frozenset({"crop", "augmentation", "window"})


class DatasetContractError(ValueError):
    """Raised when dataset-table content is invalid or incompatible."""


def content_addressed_row_locations(artifact: ArtifactRefV1) -> tuple[str, str]:
    """Return the only privacy-safe row locations for immutable artifact bytes."""

    digest = artifact.sha256.removeprefix("sha256:")
    suffix = f"sha256/p-{digest[:2]}/sha256-{digest}/{artifact.artifact_id}"
    return f"objects/{suffix}", f"signlab://objects/{suffix}"


def _require_content_addressed_row_locator(artifact: ArtifactRefV1) -> None:
    workspace_path, artifact_uri = content_addressed_row_locations(artifact)
    locator = artifact.locator
    if isinstance(locator, WorkspaceRelativeLocatorV1):
        valid = locator.path == workspace_path
    else:
        valid = locator.uri == artifact_uri
    if not valid:
        raise ValueError("dataset row artifact locator must use its canonical content address")


def _require_canonical_table_locator(reference: DatasetTableRefV1) -> None:
    locator = reference.artifact.locator
    if isinstance(locator, WorkspaceRelativeLocatorV1):
        valid = locator.path == f"tables/{reference.table_name}.parquet"
    else:
        valid = locator.uri == f"signlab://tables/{reference.table_name}"
    if not valid:
        raise ValueError("dataset table locator must use its registered canonical name")


class MediaIntervalV1(StrictContractModel):
    """A non-empty half-open interval in root-recording microseconds."""

    schema_version: Literal["media-interval/1"]
    start_us: NonNegativeSafeInteger
    end_us: PositiveSafeInteger

    @model_validator(mode="after")
    def _require_non_empty_interval(self) -> Self:
        if self.end_us <= self.start_us:
            raise ValueError("media interval must be non-empty and ordered")
        return self


class DatasetSampleIdentityV1(StrictContractModel):
    """Minimal membership and leakage-group identity for one immutable sample."""

    sample_id: SampleId
    participant_id: ParticipantId
    session_id: SessionId
    source_recording_id: RecordingId
    label_id: LabelId
    artifact: ArtifactRefV1

    @model_validator(mode="after")
    def _bind_artifact_to_sample(self) -> Self:
        if self.artifact.artifact_id != self.sample_id or self.artifact.role != "sample_data":
            raise ValueError("sample artifact identity and role must bind to the sample")
        return self


class ParticipantRowV1(StrictContractModel):
    """One pseudonymous participant without identity-vault attributes."""

    participant_id: ParticipantId
    handedness: Handedness

    @model_validator(mode="after")
    def _require_declared_handedness(self) -> Self:
        if self.handedness == "both":
            raise ValueError("participant handedness must use ambidextrous rather than both")
        return self


class SessionRowV1(StrictContractModel):
    """One bounded capture session with non-identifying device facts."""

    session_id: SessionId
    participant_id: ParticipantId
    device_id: DeviceId
    started_at: UtcTimestamp
    finished_at: UtcTimestamp
    capture_mode: Literal["isolated", "continuous"]
    capture_software_version: SemanticVersion
    camera_facing: Literal["front", "rear", "external", "unknown"]
    frame_width_px: PositiveSafeInteger
    frame_height_px: PositiveSafeInteger
    frame_rate_numerator: PositiveSafeInteger
    frame_rate_denominator: PositiveSafeInteger
    rotation_degrees: Literal[0, 90, 180, 270]
    mirror_state: MirrorState

    @model_validator(mode="after")
    def _require_ordered_session(self) -> Self:
        if _parse_utc(self.finished_at) <= _parse_utc(self.started_at):
            raise ValueError("session finish time must be later than start time")
        return self


class RecordingRowV1(StrictContractModel):
    """One raw recording bound to grouping, bytes, and its consent snapshot."""

    recording_id: RecordingId
    participant_id: ParticipantId
    session_id: SessionId
    device_id: DeviceId
    captured_at: UtcTimestamp
    duration_us: PositiveSafeInteger
    handedness: Handedness
    mirror_state: MirrorState
    rotation_degrees: Literal[0, 90, 180, 270]
    audio_present: Literal[False]
    media: ArtifactRefV1
    consent_grant: RecordingConsentGrantV1

    @model_validator(mode="after")
    def _bind_recording_evidence(self) -> Self:
        validate_recording_consent_grant(self.consent_grant)
        if self.handedness == "ambidextrous":
            raise ValueError("recording handedness must describe the observed performance")
        if (
            self.consent_grant.recording_id != self.recording_id
            or self.consent_grant.participant_id != self.participant_id
            or self.consent_grant.captured_at != self.captured_at
        ):
            raise ValueError("recording grouping and time must match its consent grant")
        if (
            self.media.artifact_id != self.recording_id
            or self.media.role != "raw_recording"
            or not self.media.media_type.startswith("video/")
        ):
            raise ValueError("recording media identity, role, and type are incompatible")
        _require_content_addressed_row_locator(self.media)
        return self


class ClipRowV1(StrictContractModel):
    """One temporal view whose coordinates remain rooted in the source recording."""

    clip_id: ClipId
    participant_id: ParticipantId
    session_id: SessionId
    source_recording_id: RecordingId
    interval: MediaIntervalV1
    handedness: Handedness
    mirror_state: MirrorState
    artifact: ArtifactRefV1 | None

    @model_validator(mode="after")
    def _bind_optional_clip_bytes(self) -> Self:
        if self.handedness == "ambidextrous":
            raise ValueError("clip handedness must describe the observed performance")
        if self.artifact is not None and (
            self.artifact.artifact_id != self.clip_id or self.artifact.role != "clip_media"
        ):
            raise ValueError("materialized clip identity and role must bind to the clip")
        if self.artifact is not None:
            _require_content_addressed_row_locator(self.artifact)
        return self


class AnnotationRowV1(StrictContractModel):
    """One reviewed label or coded exclusion in root-recording time."""

    annotation_id: AnnotationId
    participant_id: ParticipantId
    session_id: SessionId
    source_recording_id: RecordingId
    clip_id: ClipId | None
    interval: MediaIntervalV1
    disposition: Literal["class_label", "ambiguous", "ignore"]
    label_id: LabelId | None
    other_kind: OtherKind | None
    reason_code: StableId | None
    review_status: Literal["draft", "reviewed", "adjudicated"]
    eligible_for_training: bool

    @model_validator(mode="after")
    def _require_disposition_shape(self) -> Self:
        if self.disposition == "class_label":
            if self.label_id is None or self.reason_code is not None:
                raise ValueError("class annotations require a label and no exclusion reason")
            if (self.label_id == "other") != (self.other_kind is not None):
                raise ValueError("only the other label requires a registered other kind")
        elif self.label_id is not None or self.other_kind is not None or self.reason_code is None:
            raise ValueError("ambiguous and ignored annotations require only a coded reason")
        expected_eligibility = self.disposition == "class_label" and self.review_status in {
            "reviewed",
            "adjudicated",
        }
        if self.eligible_for_training != expected_eligibility:
            raise ValueError("training eligibility must follow disposition and review status")
        return self


class DerivedArtifactRowV1(StrictContractModel):
    """One derived artifact with immutable root grouping and optional sample identity."""

    derived_artifact_id: DerivedArtifactId
    derivation_kind: Literal[
        "crop",
        "augmentation",
        "window",
        "landmark_extraction",
        "feature_extraction",
    ]
    parent_artifact_ids: tuple[StableId, ...] = Field(min_length=1)
    participant_id: ParticipantId
    session_id: SessionId
    source_recording_id: RecordingId
    clip_id: ClipId | None
    annotation_id: AnnotationId | None
    sample_id: SampleId | None
    label_id: LabelId | None
    split_id: StableId | None
    partition: PartitionName | None
    handedness: Handedness
    mirror_state: MirrorState
    operation_id: StableId
    operation_version: SemanticVersion
    artifact: ArtifactRefV1

    @model_validator(mode="after")
    def _require_complete_lineage_shape(self) -> Self:
        if self.handedness == "ambidextrous":
            raise ValueError("derived handedness must describe the observed performance")
        if self.parent_artifact_ids != tuple(sorted(set(self.parent_artifact_ids))):
            raise ValueError("derived parent artifact IDs must be unique and sorted")
        if self.artifact.artifact_id in self.parent_artifact_ids:
            raise ValueError("a derived artifact cannot be its own parent")
        sample_fields = (self.sample_id, self.label_id, self.split_id, self.partition)
        is_sample = all(value is not None for value in sample_fields)
        if any(value is not None for value in sample_fields) and not is_sample:
            raise ValueError("sample identity, label, split, and partition are all-or-none")
        if self.derivation_kind in _SAMPLE_DERIVATION_KINDS and (
            not is_sample or self.clip_id is None or self.annotation_id is None
        ):
            raise ValueError("crop, augmentation, and window samples require source and split data")
        if is_sample:
            if self.artifact.artifact_id != self.sample_id or self.artifact.role != "sample_data":
                raise ValueError(
                    "derived sample artifact identity and role must bind to the sample"
                )
        elif (
            self.artifact.artifact_id != self.derived_artifact_id
            or self.artifact.role != "derived_data"
        ):
            raise ValueError("intermediate artifact identity and role must bind to its row")
        _require_content_addressed_row_locator(self.artifact)
        return self


def _require_sorted_unique_rows(rows: tuple[BaseModel, ...], field_name: str) -> None:
    identities = tuple(cast(str, getattr(row, field_name)) for row in rows)
    if identities != tuple(sorted(set(identities))):
        raise ValueError(f"{field_name} values must be unique and sorted")


class ParticipantsTableV1(StrictContractModel):
    schema_version: Literal["participants-table/1"]
    rows: tuple[ParticipantRowV1, ...]

    @model_validator(mode="after")
    def _canonical_rows(self) -> Self:
        _require_sorted_unique_rows(self.rows, "participant_id")
        return self


class SessionsTableV1(StrictContractModel):
    schema_version: Literal["sessions-table/1"]
    rows: tuple[SessionRowV1, ...]

    @model_validator(mode="after")
    def _canonical_rows(self) -> Self:
        _require_sorted_unique_rows(self.rows, "session_id")
        return self


class RecordingsTableV1(StrictContractModel):
    schema_version: Literal["recordings-table/1"]
    rows: tuple[RecordingRowV1, ...]

    @model_validator(mode="after")
    def _canonical_rows(self) -> Self:
        _require_sorted_unique_rows(self.rows, "recording_id")
        return self


class ClipsTableV1(StrictContractModel):
    schema_version: Literal["clips-table/1"]
    rows: tuple[ClipRowV1, ...]

    @model_validator(mode="after")
    def _canonical_rows(self) -> Self:
        _require_sorted_unique_rows(self.rows, "clip_id")
        return self


class AnnotationsTableV1(StrictContractModel):
    schema_version: Literal["annotations-table/1"]
    rows: tuple[AnnotationRowV1, ...]

    @model_validator(mode="after")
    def _canonical_rows(self) -> Self:
        _require_sorted_unique_rows(self.rows, "annotation_id")
        return self


class DerivedArtifactsTableV1(StrictContractModel):
    schema_version: Literal["derived-artifacts-table/1"]
    rows: tuple[DerivedArtifactRowV1, ...]

    @model_validator(mode="after")
    def _canonical_rows(self) -> Self:
        _require_sorted_unique_rows(self.rows, "derived_artifact_id")
        return self


type DatasetTable = (
    ParticipantsTableV1
    | SessionsTableV1
    | RecordingsTableV1
    | ClipsTableV1
    | AnnotationsTableV1
    | DerivedArtifactsTableV1
)
type DatasetTableInput = DatasetTable | str | bytes | bytearray | Mapping[str, object]

DATASET_TABLE_ROW_MODELS: Final[dict[TableName, type[BaseModel]]] = {
    "participants": ParticipantRowV1,
    "sessions": SessionRowV1,
    "recordings": RecordingRowV1,
    "clips": ClipRowV1,
    "annotations": AnnotationRowV1,
    "derived_artifacts": DerivedArtifactRowV1,
}
DATASET_TABLE_WRAPPER_MODELS: Final[dict[TableName, type[BaseModel]]] = {
    "participants": ParticipantsTableV1,
    "sessions": SessionsTableV1,
    "recordings": RecordingsTableV1,
    "clips": ClipsTableV1,
    "annotations": AnnotationsTableV1,
    "derived_artifacts": DerivedArtifactsTableV1,
}
DATASET_TABLE_PRIMARY_KEYS: Final[dict[TableName, str]] = {
    "participants": "participant_id",
    "sessions": "session_id",
    "recordings": "recording_id",
    "clips": "clip_id",
    "annotations": "annotation_id",
    "derived_artifacts": "derived_artifact_id",
}
DATASET_TABLE_MODELS: Final[dict[str, type[BaseModel]]] = {
    DATASET_TABLE_SCHEMA_VERSIONS[table_name]: model
    for table_name, model in DATASET_TABLE_WRAPPER_MODELS.items()
}


def validate_dataset_table(document: DatasetTableInput) -> DatasetTable:
    """Validate one exact table wrapper without coercion or migration."""

    if isinstance(document, BaseModel):
        document = cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
    try:
        payload = parse_json_object(document)
    except CanonicalizationError as error:
        raise DatasetContractError("dataset table is not an interoperable JSON object") from error
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or schema_version not in DATASET_TABLE_MODELS:
        supported = ", ".join(sorted(DATASET_TABLE_MODELS))
        raise DatasetContractError(f"unsupported dataset table schema; supported: {supported}")
    try:
        checked = DATASET_TABLE_MODELS[schema_version].model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except (CanonicalizationError, ValidationError) as error:
        raise DatasetContractError(f"invalid {schema_version} table") from error
    return cast(DatasetTable, checked)


def dataset_table_digest(document: DatasetTableInput) -> str:
    """Hash logical table rows independently of their Parquet representation."""

    checked = validate_dataset_table(document)
    try:
        payload = checked.model_dump(mode="json", round_trip=True)
        return canonical_sha256(
            cast(Mapping[str, object], _without_artifact_locators(payload)),
            domain=checked.schema_version,
        )
    except CanonicalizationError as error:
        raise DatasetContractError("validated dataset table cannot be canonicalized") from error


def _without_artifact_locators(value: object) -> object:
    if isinstance(value, dict):
        copied = {
            key: _without_artifact_locators(nested)
            for key, nested in value.items()
            if not (value.get("schema_version") == "artifact-reference/1" and key == "locator")
        }
        return copied
    if isinstance(value, list):
        return [_without_artifact_locators(item) for item in value]
    return value


class DatasetTableRefV1(StrictContractModel):
    """Logical table content plus an independently verified Parquet byte reference."""

    schema_version: Literal["dataset-table-reference/1"]
    table_name: TableName
    table_schema_version: TableSchemaVersion
    row_count: NonNegativeSafeInteger
    content_sha256: Sha256Digest
    artifact: ArtifactRefV1

    @model_validator(mode="after")
    def _bind_table_reference(self) -> Self:
        if self.table_schema_version != DATASET_TABLE_SCHEMA_VERSIONS[self.table_name]:
            raise ValueError("table name and schema version are incompatible")
        if (
            self.artifact.role != "dataset_table"
            or self.artifact.media_type != "application/vnd.apache.parquet"
        ):
            raise ValueError("dataset table artifact role and media type are incompatible")
        _require_canonical_table_locator(self)
        return self


class DatasetTableSetV1(StrictContractModel):
    """The exact six normalized table identities required by dataset-manifest/2."""

    schema_version: Literal["dataset-table-set/1"]
    participants: DatasetTableRefV1
    sessions: DatasetTableRefV1
    recordings: DatasetTableRefV1
    clips: DatasetTableRefV1
    annotations: DatasetTableRefV1
    derived_artifacts: DatasetTableRefV1

    @model_validator(mode="after")
    def _bind_named_tables(self) -> Self:
        for table_name in DATASET_TABLE_SCHEMA_VERSIONS:
            reference = cast(DatasetTableRefV1, getattr(self, table_name))
            if reference.table_name != table_name:
                raise ValueError("every named dataset table must bind its exact table name")
        if any(
            reference.row_count == 0
            for reference in (self.participants, self.sessions, self.recordings)
        ):
            raise ValueError("participant, session, and recording tables must be non-empty")
        artifact_ids = tuple(
            cast(DatasetTableRefV1, getattr(self, name)).artifact.artifact_id
            for name in DATASET_TABLE_SCHEMA_VERSIONS
        )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("dataset table artifacts must have unique IDs")
        return self


class DatasetContentV1(StrictContractModel):
    """Storage-independent semantic content used to calculate a stable data identity."""

    schema_version: Literal["dataset-content/1"]
    taxonomy: TaxonomyRef
    governance_policy: DocumentRef
    lineage_inventory_sha256: Sha256Digest
    sample_schema_version: SchemaName
    samples: tuple[DatasetSampleIdentityV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_canonical_unique_samples(self) -> Self:
        _validate_dataset_content(self.governance_policy, self.samples)
        return self


class DatasetContentV2(StrictContractModel):
    """Detailed table-backed content used by the current dataset writer."""

    schema_version: Literal["dataset-content/2"]
    taxonomy: TaxonomyRef
    governance_policy: DocumentRef
    lineage_inventory_sha256: Sha256Digest
    sample_schema_version: SchemaName
    tables: DatasetTableSetV1
    samples: tuple[DatasetSampleIdentityV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_canonical_unique_samples(self) -> Self:
        _validate_dataset_content(self.governance_policy, self.samples)
        for sample in self.samples:
            _require_content_addressed_row_locator(sample.artifact)
        if len(self.samples) > self.tables.derived_artifacts.row_count:
            raise ValueError("sample projection cannot exceed derived-artifact rows")
        return self


type DatasetContent = DatasetContentV1 | DatasetContentV2


def _validate_dataset_content(
    governance_policy: DocumentRef,
    samples: tuple[DatasetSampleIdentityV1, ...],
) -> None:
    if governance_policy.document_type != "governance_policy":
        raise ValueError("dataset content must bind the registered governance policy")
    sample_ids = tuple(sample.sample_id for sample in samples)
    if sample_ids != tuple(sorted(set(sample_ids))):
        raise ValueError("dataset samples must have unique IDs in sorted order")
    artifact_digests = tuple(sample.artifact.sha256 for sample in samples)
    if len(artifact_digests) != len(set(artifact_digests)):
        raise ValueError("dataset samples must not duplicate artifact content")


def _sample_semantic_payload(sample: DatasetSampleIdentityV1) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "participant_id": sample.participant_id,
        "session_id": sample.session_id,
        "source_recording_id": sample.source_recording_id,
        "label_id": sample.label_id,
        "artifact": {
            "artifact_id": sample.artifact.artifact_id,
            "role": sample.artifact.role,
            "media_type": sample.artifact.media_type,
            "sha256": sample.artifact.sha256,
            "size_bytes": sample.artifact.size_bytes,
        },
    }


def _dataset_semantic_payload(content: DatasetContent) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": content.schema_version,
        "taxonomy": content.taxonomy.model_dump(mode="json", round_trip=True),
        "governance_policy": content.governance_policy.model_dump(mode="json", round_trip=True),
        "lineage_inventory_sha256": content.lineage_inventory_sha256,
        "sample_schema_version": content.sample_schema_version,
        "samples": [_sample_semantic_payload(sample) for sample in content.samples],
    }
    if isinstance(content, DatasetContentV2):
        payload["tables"] = {
            table_name: {
                "table_name": reference.table_name,
                "table_schema_version": reference.table_schema_version,
                "row_count": reference.row_count,
                "content_sha256": reference.content_sha256,
            }
            for table_name in DATASET_TABLE_SCHEMA_VERSIONS
            for reference in (cast(DatasetTableRefV1, getattr(content.tables, table_name)),)
        }
    return payload


def dataset_content_digest(content: DatasetContent) -> str:
    """Hash logical sample/table content independently of storage representation."""

    try:
        checked = type(content).model_validate_json(
            content.model_dump_json(round_trip=True), strict=True
        )
        return canonical_sha256(
            _dataset_semantic_payload(checked),
            domain=checked.schema_version,
        )
    except (CanonicalizationError, ValidationError) as error:
        raise DatasetContractError(
            "dataset content is invalid or cannot be canonicalized"
        ) from error


class DatasetManifestV1(StrictContractModel):
    """Portable dataset envelope; detailed participant tables arrive in Story #15."""

    model_config = contract_config("dataset-manifest-1.schema.json")

    schema_version: Literal["dataset-manifest/1"]
    dataset_id: StableId
    version: SemanticVersion
    content: DatasetContentV1
    data_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_data_identity(self) -> Self:
        if self.data_sha256 != dataset_content_digest(self.content):
            raise ValueError("data_sha256 does not match canonical storage-independent content")
        return self


class DatasetManifestV2(StrictContractModel):
    """Current dataset envelope committing to six normalized table identities."""

    model_config = contract_config("dataset-manifest-2.schema.json")

    schema_version: Literal["dataset-manifest/2"]
    dataset_id: StableId
    version: SemanticVersion
    content: DatasetContentV2
    data_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_data_identity(self) -> Self:
        if self.data_sha256 != dataset_content_digest(self.content):
            raise ValueError("data_sha256 does not match canonical storage-independent content")
        return self


type DatasetManifest = DatasetManifestV1 | DatasetManifestV2


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "DATASET_TABLE_MODELS",
    "DATASET_TABLE_PRIMARY_KEYS",
    "DATASET_TABLE_ROW_MODELS",
    "DATASET_TABLE_SCHEMA_VERSIONS",
    "DATASET_TABLE_WRAPPER_MODELS",
    "AnnotationId",
    "AnnotationRowV1",
    "AnnotationsTableV1",
    "ClipId",
    "ClipRowV1",
    "ClipsTableV1",
    "DatasetContent",
    "DatasetContentV1",
    "DatasetContentV2",
    "DatasetContractError",
    "DatasetManifest",
    "DatasetManifestV1",
    "DatasetManifestV2",
    "DatasetSampleIdentityV1",
    "DatasetTable",
    "DatasetTableInput",
    "DatasetTableRefV1",
    "DatasetTableSetV1",
    "DerivedArtifactId",
    "DerivedArtifactRowV1",
    "DerivedArtifactsTableV1",
    "DeviceId",
    "Handedness",
    "LabelId",
    "MediaIntervalV1",
    "MirrorState",
    "OtherKind",
    "ParticipantRowV1",
    "ParticipantsTableV1",
    "PartitionName",
    "RecordingId",
    "RecordingRowV1",
    "RecordingsTableV1",
    "SampleId",
    "SessionId",
    "SessionRowV1",
    "SessionsTableV1",
    "TableName",
    "TableSchemaVersion",
    "content_addressed_row_locations",
    "dataset_content_digest",
    "dataset_table_digest",
    "validate_dataset_table",
]
