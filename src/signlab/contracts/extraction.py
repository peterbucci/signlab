"""Strict contracts for version-pinned MediaPipe Tasks landmark extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal, Self, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    FiniteFloat,
    NonNegativeSafeInteger,
    PositiveSafeInteger,
    SafeInteger,
    SemanticVersion,
    StableId,
    StrictContractModel,
    contract_config,
)
from signlab.contracts.dataset import DerivedArtifactRowV1, MirrorState
from signlab.contracts.governance import RecordingId
from signlab.contracts.ingest import DatasetId, RawDatasetManifestV1, validate_raw_dataset_manifest
from signlab.contracts.taxonomy import Sha256Digest

MEDIAPIPE_PACKAGE_VERSION: Final = "1.0.1"
HAND_LANDMARK_COUNT: Final = 21
HAND_SLOT_IDS: Final = ("hand_0", "hand_1")
BODY_ANCHOR_NAMES: Final = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
TIMESTAMP_RULE: Final = "source_pts_relative_us_then_strict_monotonic_floor_ms/1"
TRACKING_ALGORITHM: Final = "deterministic_wrist_mcp_centroid_minimum_cost"
TRACKING_ALGORITHM_VERSION: Final = "1.0.0"

HandSlotId = Literal["hand_0", "hand_1"]
BodyAnchorName = Literal[
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PinnedThreshold = Annotated[
    float,
    Field(strict=True, ge=0.5, le=0.5, allow_inf_nan=False),
]
PinnedMaxSpatialCost = Annotated[
    float,
    Field(strict=True, ge=0.25, le=0.25, allow_inf_nan=False),
]
PinnedHandednessPenalty = Annotated[
    float,
    Field(strict=True, ge=0.05, le=0.05, allow_inf_nan=False),
]
PinnedAmbiguityMargin = Annotated[
    float,
    Field(strict=True, ge=1e-9, le=1e-9, allow_inf_nan=False),
]


class ExtractionContractError(ValueError):
    """Raised when extraction content is invalid or incompatible."""


class MediaPipeTaskAssetV1(StrictContractModel):
    """Exact identity of one cross-runtime MediaPipe Tasks model bundle."""

    schema_version: Literal["mediapipe-task-asset/1"]
    task_kind: Literal["hand_landmarker", "pose_landmarker"]
    model_id: Literal[
        "mediapipe-hand-landmarker-full",
        "mediapipe-pose-landmarker-lite",
    ]
    model_revision: Literal["1.0.0"]
    filename: Literal["hand_landmarker.task", "pose_landmarker_lite.task"]
    sha256: Sha256Digest
    size_bytes: PositiveSafeInteger
    compatible_runtimes: tuple[Literal["browser", "python"], Literal["browser", "python"]]

    @model_validator(mode="after")
    def _require_registered_asset(self) -> Self:
        expected = {
            "hand_landmarker": (
                "mediapipe-hand-landmarker-full",
                "hand_landmarker.task",
                "sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
                7_819_105,
            ),
            "pose_landmarker": (
                "mediapipe-pose-landmarker-lite",
                "pose_landmarker_lite.task",
                "sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
                5_777_746,
            ),
        }[self.task_kind]
        actual = (self.model_id, self.filename, self.sha256, self.size_bytes)
        if actual != expected:
            raise ValueError("MediaPipe task asset does not match the registered model bytes")
        if self.compatible_runtimes != ("browser", "python"):
            raise ValueError("MediaPipe task runtimes must be unique and in canonical order")
        return self


class MediaPipeExtractionConfigV1(StrictContractModel):
    """Fully pinned detector, timestamp, and deterministic tracking configuration."""

    schema_version: Literal["mediapipe-extraction-config/1"]
    config_id: StableId
    version: SemanticVersion
    python_package: Literal["mediapipe"]
    python_package_version: Literal["1.0.1"]
    browser_package: Literal["@mediapipe/tasks-vision"]
    browser_package_version: Literal["1.0.1"]
    decoder_package: Literal["av"]
    decoder_package_version: Literal["18.1.0"]
    delegate: Literal["CPU"]
    running_mode: Literal["VIDEO"]
    num_hands: Literal[2]
    num_poses: Literal[1]
    min_hand_detection_confidence: PinnedThreshold
    min_hand_presence_confidence: PinnedThreshold
    min_hand_tracking_confidence: PinnedThreshold
    min_pose_detection_confidence: PinnedThreshold
    min_pose_presence_confidence: PinnedThreshold
    min_pose_tracking_confidence: PinnedThreshold
    body_anchors: tuple[
        BodyAnchorName,
        BodyAnchorName,
        BodyAnchorName,
        BodyAnchorName,
        BodyAnchorName,
        BodyAnchorName,
    ]
    timestamp_rule: Literal["source_pts_relative_us_then_strict_monotonic_floor_ms/1"]
    tracking_algorithm: Literal["deterministic_wrist_mcp_centroid_minimum_cost"]
    tracking_algorithm_version: Literal["1.0.0"]
    max_spatial_cost: PinnedMaxSpatialCost
    handedness_disagreement_penalty: PinnedHandednessPenalty
    ambiguity_margin: PinnedAmbiguityMargin
    hand_task_asset: MediaPipeTaskAssetV1
    pose_task_asset: MediaPipeTaskAssetV1

    @model_validator(mode="after")
    def _require_exact_task_roles_and_anchor_order(self) -> Self:
        if self.hand_task_asset.task_kind != "hand_landmarker":
            raise ValueError("hand_task_asset must contain the registered hand task")
        if self.pose_task_asset.task_kind != "pose_landmarker":
            raise ValueError("pose_task_asset must contain the registered pose task")
        if self.body_anchors != BODY_ANCHOR_NAMES:
            raise ValueError("body anchors must use the registered six-anchor order")
        return self


class Point3V1(StrictContractModel):
    """One finite 3-D point with optional task-provided confidence channels."""

    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
    visibility: UnitFloat | None
    presence: UnitFloat | None


class HandSlotV1(StrictContractModel):
    """One stable hand identity, explicitly masked when no hand occupies the slot."""

    slot_id: HandSlotId
    present: bool
    detector_index: NonNegativeSafeInteger | None
    tracking_id: HandSlotId | None
    handedness: Literal["left", "right"] | None
    handedness_confidence: UnitFloat | None
    image_landmarks: tuple[Point3V1, ...] | None = Field(min_length=21, max_length=21)
    world_landmarks: tuple[Point3V1, ...] | None = Field(min_length=21, max_length=21)

    @model_validator(mode="after")
    def _require_present_or_all_null(self) -> Self:
        observations = (
            self.detector_index,
            self.tracking_id,
            self.handedness,
            self.handedness_confidence,
            self.image_landmarks,
            self.world_landmarks,
        )
        if self.present:
            if any(value is None for value in observations):
                raise ValueError(
                    "a present hand requires identity, handedness, and both point sets"
                )
            if self.tracking_id != self.slot_id:
                raise ValueError("hand tracking identity must remain bound to its stable slot")
        elif any(value is not None for value in observations):
            raise ValueError("an absent hand must have an explicit all-null observation payload")
        return self


class BodyAnchorV1(StrictContractModel):
    """One selected pose anchor, explicitly masked when it is not observable."""

    name: BodyAnchorName
    present: bool
    image_point: Point3V1 | None
    world_point: Point3V1 | None

    @model_validator(mode="after")
    def _require_present_or_all_null(self) -> Self:
        if self.present and (self.image_point is None or self.world_point is None):
            raise ValueError("a present body anchor requires image and world points")
        if not self.present and (self.image_point is not None or self.world_point is not None):
            raise ValueError("an absent body anchor must have an all-null point payload")
        return self


class LandmarkFrameV1(StrictContractModel):
    """One decoded source frame with raw task observations and explicit masks."""

    schema_version: Literal["landmark-frame/1"]
    source_recording_id: RecordingId
    frame_index: NonNegativeSafeInteger
    source_pts: SafeInteger
    source_time_base_numerator: PositiveSafeInteger
    source_time_base_denominator: PositiveSafeInteger
    relative_timestamp_us: NonNegativeSafeInteger
    task_timestamp_ms: NonNegativeSafeInteger
    invalid: bool
    invalid_reason: Literal["source_frame_invalid", "task_inference_failed"] | None
    hands: tuple[HandSlotV1, HandSlotV1]
    body_anchors: tuple[
        BodyAnchorV1,
        BodyAnchorV1,
        BodyAnchorV1,
        BodyAnchorV1,
        BodyAnchorV1,
        BodyAnchorV1,
    ]
    observed_hand_count: Annotated[NonNegativeSafeInteger, Field(le=2)]
    observed_body_anchor_count: Annotated[NonNegativeSafeInteger, Field(le=6)]

    @model_validator(mode="after")
    def _require_masks_order_and_counts(self) -> Self:
        if tuple(hand.slot_id for hand in self.hands) != HAND_SLOT_IDS:
            raise ValueError("frame hand slots must be exactly hand_0 then hand_1")
        if tuple(anchor.name for anchor in self.body_anchors) != BODY_ANCHOR_NAMES:
            raise ValueError("frame body anchors must use the registered six-anchor order")
        hand_count = sum(hand.present for hand in self.hands)
        anchor_count = sum(anchor.present for anchor in self.body_anchors)
        if self.observed_hand_count != hand_count:
            raise ValueError("observed hand count does not match the hand masks")
        if self.observed_body_anchor_count != anchor_count:
            raise ValueError("observed body-anchor count does not match the anchor masks")
        if self.invalid != (self.invalid_reason is not None):
            raise ValueError("invalid frame mask and coded reason must be present together")
        if self.invalid and (hand_count or anchor_count):
            raise ValueError("an invalid frame cannot contain task observations")
        if self.task_timestamp_ms < self.relative_timestamp_us // 1_000:
            raise ValueError("task timestamp cannot precede floor(relative_timestamp_us / 1000)")
        return self


class LandmarkFramesTableV1(StrictContractModel):
    """Semantic rows for one source recording's Parquet landmark sequence."""

    schema_version: Literal["landmark-frames-table/1"]
    rows: tuple[LandmarkFrameV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_one_canonical_timeline(self) -> Self:
        first = self.rows[0]
        if tuple(frame.frame_index for frame in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("landmark frame indexes must be consecutive from zero")
        if any(frame.source_recording_id != first.source_recording_id for frame in self.rows):
            raise ValueError("a landmark frames table must contain exactly one source recording")
        time_base = (first.source_time_base_numerator, first.source_time_base_denominator)
        if any(
            (frame.source_time_base_numerator, frame.source_time_base_denominator) != time_base
            for frame in self.rows
        ):
            raise ValueError("source time base must remain constant within a sequence")
        if first.relative_timestamp_us != 0 or first.task_timestamp_ms != 0:
            raise ValueError("the first decoded frame must establish relative timestamp zero")
        previous_pts: int | None = None
        previous_task_ms: int | None = None
        for frame in self.rows:
            if previous_pts is not None and frame.source_pts <= previous_pts:
                raise ValueError("source PTS values must increase strictly")
            expected_us = (
                (frame.source_pts - first.source_pts)
                * first.source_time_base_numerator
                * 1_000_000
                // first.source_time_base_denominator
            )
            if frame.relative_timestamp_us != expected_us:
                raise ValueError("relative timestamp does not match source PTS and time base")
            floor_ms = frame.relative_timestamp_us // 1_000
            expected_task_ms = max(
                floor_ms, 0 if previous_task_ms is None else previous_task_ms + 1
            )
            if frame.task_timestamp_ms != expected_task_ms:
                raise ValueError(
                    "MediaPipe VIDEO timestamp does not follow the collision-free recurrence"
                )
            previous_pts = frame.source_pts
            previous_task_ms = frame.task_timestamp_ms
        return self


class LandmarkObservationCountsV1(StrictContractModel):
    """Auditable observation totals without introducing a quality policy."""

    frame_count: PositiveSafeInteger
    valid_frame_count: NonNegativeSafeInteger
    invalid_frame_count: NonNegativeSafeInteger
    zero_hand_frame_count: NonNegativeSafeInteger
    one_hand_frame_count: NonNegativeSafeInteger
    two_hand_frame_count: NonNegativeSafeInteger
    hand_observation_count: NonNegativeSafeInteger
    body_anchor_observation_count: NonNegativeSafeInteger
    body_anchor_presence_counts: tuple[
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
    ]

    @model_validator(mode="after")
    def _require_possible_totals(self) -> Self:
        if self.valid_frame_count + self.invalid_frame_count != self.frame_count:
            raise ValueError("valid and invalid frame counts must cover the sequence")
        if (
            self.zero_hand_frame_count + self.one_hand_frame_count + self.two_hand_frame_count
            != self.valid_frame_count
        ):
            raise ValueError("zero-, one-, and two-hand counts must cover every valid frame")
        if self.hand_observation_count != (
            self.one_hand_frame_count + (2 * self.two_hand_frame_count)
        ):
            raise ValueError("hand observation total must match per-frame hand counts")
        if self.hand_observation_count > self.valid_frame_count * len(HAND_SLOT_IDS):
            raise ValueError("hand observations exceed the possible valid-frame total")
        if self.body_anchor_observation_count > self.valid_frame_count * len(BODY_ANCHOR_NAMES):
            raise ValueError("body-anchor observations exceed the possible valid-frame total")
        if any(count > self.valid_frame_count for count in self.body_anchor_presence_counts):
            raise ValueError("per-anchor presence count exceeds the valid-frame total")
        if self.body_anchor_observation_count != sum(self.body_anchor_presence_counts):
            raise ValueError("body-anchor observation total must match per-anchor counts")
        return self


class LandmarkSequenceRefV1(StrictContractModel):
    """Lineage, exact input/output bytes, and semantic identity of one sequence."""

    schema_version: Literal["landmark-sequence-reference/1"]
    lineage: DerivedArtifactRowV1
    source_media_sha256: Sha256Digest
    source_media_size_bytes: PositiveSafeInteger
    source_rotation_degrees: Literal[0, 90, 180, 270]
    source_mirror_state: MirrorState
    frames_schema_version: Literal["landmark-frames-table/1"]
    content_sha256: Sha256Digest
    counts: LandmarkObservationCountsV1

    @model_validator(mode="after")
    def _require_landmark_lineage(self) -> Self:
        if self.lineage.derivation_kind != "landmark_extraction":
            raise ValueError("landmark sequence lineage must use landmark_extraction")
        if self.lineage.parent_artifact_ids != (self.lineage.source_recording_id,):
            raise ValueError("landmark sequence must have its source recording as its sole parent")
        if any(
            value is not None
            for value in (
                self.lineage.clip_id,
                self.lineage.annotation_id,
                self.lineage.sample_id,
                self.lineage.label_id,
                self.lineage.split_id,
                self.lineage.partition,
            )
        ):
            raise ValueError("raw landmark sequences cannot acquire sample or split identity")
        if self.lineage.mirror_state != self.source_mirror_state:
            raise ValueError("source mirror state must match inherited landmark lineage")
        artifact = self.lineage.artifact
        if artifact.media_type != "application/vnd.apache.parquet":
            raise ValueError("landmark sequence artifacts must use Parquet")
        return self


class LandmarkExtractionManifestV1(StrictContractModel):
    """Self-digested extraction handoff bound to exact raw data and task assets."""

    model_config = contract_config("landmark-extraction-manifest-1.schema.json")

    schema_version: Literal["landmark-extraction-manifest/1"]
    extraction_id: StableId
    version: SemanticVersion
    raw_dataset_id: DatasetId
    raw_dataset_version: SemanticVersion
    raw_data_sha256: Sha256Digest
    raw_dataset_manifest_sha256: Sha256Digest
    config: MediaPipeExtractionConfigV1
    config_sha256: Sha256Digest
    sequences: tuple[LandmarkSequenceRefV1, ...] = Field(min_length=1)
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_canonical_bound_sequences_and_identity(self) -> Self:
        recording_ids = tuple(sequence.lineage.source_recording_id for sequence in self.sequences)
        sequence_ids = tuple(sequence.lineage.derived_artifact_id for sequence in self.sequences)
        if recording_ids != tuple(sorted(set(recording_ids))):
            raise ValueError("extraction sequences must have unique source recordings in order")
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("extraction sequence artifact IDs must be unique")
        for sequence in self.sequences:
            if (
                sequence.lineage.operation_id != self.config.config_id
                or sequence.lineage.operation_version != self.config.version
            ):
                raise ValueError("sequence lineage must bind the exact extraction configuration")
        if self.config_sha256 != mediapipe_extraction_config_digest(self.config):
            raise ValueError("config_sha256 does not match the pinned extraction configuration")
        if self.manifest_sha256 != landmark_extraction_manifest_digest(self):
            raise ValueError("manifest_sha256 does not match canonical extraction content")
        return self


type ExtractionInput = BaseModel | str | bytes | bytearray | Mapping[str, object]


def _validate_model[ModelT: BaseModel](
    document: ExtractionInput,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        if isinstance(document, BaseModel):
            payload = cast(
                Mapping[str, object],
                document.model_dump(mode="json", round_trip=True),
            )
        else:
            payload = cast(Mapping[str, object], parse_json_object(document))
        return model.model_validate_json(canonical_json_bytes(payload), strict=True)
    except (CanonicalizationError, ValidationError) as error:
        raise ExtractionContractError(f"invalid {label}") from error


def validate_mediapipe_extraction_config(
    document: ExtractionInput,
) -> MediaPipeExtractionConfigV1:
    """Validate one exact extraction configuration without coercion."""

    return _validate_model(document, MediaPipeExtractionConfigV1, "MediaPipe extraction config")


def mediapipe_extraction_config_digest(document: ExtractionInput) -> str:
    """Return the stable identity of the fully pinned extraction configuration."""

    checked = validate_mediapipe_extraction_config(document)
    return canonical_sha256(checked, domain=checked.schema_version)


def validate_landmark_frames_table(document: ExtractionInput) -> LandmarkFramesTableV1:
    """Validate one semantic landmark table independently of Parquet encoding."""

    return _validate_model(document, LandmarkFramesTableV1, "landmark frames table")


def landmark_frames_table_digest(document: ExtractionInput) -> str:
    """Hash semantic frame rows rather than storage-specific Parquet bytes."""

    checked = validate_landmark_frames_table(document)
    return canonical_sha256(checked, domain=checked.schema_version)


def landmark_observation_counts(
    document: LandmarkFramesTableV1 | ExtractionInput,
) -> LandmarkObservationCountsV1:
    """Derive exact mask totals for one validated landmark sequence."""

    checked = (
        document
        if isinstance(document, LandmarkFramesTableV1)
        else validate_landmark_frames_table(document)
    )
    invalid = sum(frame.invalid for frame in checked.rows)
    valid_frames = tuple(frame for frame in checked.rows if not frame.invalid)
    hand_counts = tuple(frame.observed_hand_count for frame in valid_frames)
    return LandmarkObservationCountsV1(
        frame_count=len(checked.rows),
        valid_frame_count=len(checked.rows) - invalid,
        invalid_frame_count=invalid,
        zero_hand_frame_count=hand_counts.count(0),
        one_hand_frame_count=hand_counts.count(1),
        two_hand_frame_count=hand_counts.count(2),
        hand_observation_count=sum(frame.observed_hand_count for frame in checked.rows),
        body_anchor_observation_count=sum(
            frame.observed_body_anchor_count for frame in checked.rows
        ),
        body_anchor_presence_counts=cast(
            tuple[int, int, int, int, int, int],
            tuple(
                sum(frame.body_anchors[index].present for frame in valid_frames)
                for index in range(len(BODY_ANCHOR_NAMES))
            ),
        ),
    )


def assert_landmark_sequence_ref_matches_table(
    reference: LandmarkSequenceRefV1,
    table: LandmarkFramesTableV1 | ExtractionInput,
) -> None:
    """Bind a lightweight sequence reference to exact semantic Parquet rows."""

    checked = (
        table if isinstance(table, LandmarkFramesTableV1) else validate_landmark_frames_table(table)
    )
    if checked.rows[0].source_recording_id != reference.lineage.source_recording_id:
        raise ExtractionContractError("landmark table source recording does not match its lineage")
    if reference.content_sha256 != landmark_frames_table_digest(checked):
        raise ExtractionContractError("landmark table semantic digest does not match its reference")
    if reference.counts != landmark_observation_counts(checked):
        raise ExtractionContractError(
            "landmark table observation counts do not match its reference"
        )


def raw_dataset_manifest_digest(document: RawDatasetManifestV1 | ExtractionInput) -> str:
    """Return the exact canonical identity used by extraction manifests."""

    try:
        checked = (
            document
            if isinstance(document, RawDatasetManifestV1)
            else validate_raw_dataset_manifest(document)
        )
        return canonical_sha256(checked, domain=checked.schema_version)
    except (CanonicalizationError, ValueError) as error:
        raise ExtractionContractError("invalid raw dataset manifest identity") from error


def landmark_extraction_manifest_digest(
    document: LandmarkExtractionManifestV1 | Mapping[str, object],
) -> str:
    """Hash a complete extraction manifest while excluding its self-digest field."""

    if isinstance(document, BaseModel):
        payload = cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
    else:
        payload = dict(document)
    payload.pop("manifest_sha256", None)
    try:
        return canonical_sha256(payload, domain="landmark-extraction-manifest/1")
    except CanonicalizationError as error:
        raise ExtractionContractError("extraction manifest cannot be canonicalized") from error


def validate_landmark_extraction_manifest(
    document: ExtractionInput,
) -> LandmarkExtractionManifestV1:
    """Validate one self-digested extraction handoff without migration or coercion."""

    return _validate_model(document, LandmarkExtractionManifestV1, "landmark extraction manifest")


def assert_landmark_extraction_bound_to_raw_dataset(
    manifest: LandmarkExtractionManifestV1 | ExtractionInput,
    raw_dataset: RawDatasetManifestV1 | ExtractionInput,
) -> None:
    """Prove that extraction consumed the exact declared raw dataset manifest."""

    checked = (
        manifest
        if isinstance(manifest, LandmarkExtractionManifestV1)
        else validate_landmark_extraction_manifest(manifest)
    )
    raw = (
        raw_dataset
        if isinstance(raw_dataset, RawDatasetManifestV1)
        else validate_raw_dataset_manifest(raw_dataset)
    )
    expected = (
        raw.dataset_id,
        raw.version,
        raw.raw_data_sha256,
        raw_dataset_manifest_digest(raw),
    )
    actual = (
        checked.raw_dataset_id,
        checked.raw_dataset_version,
        checked.raw_data_sha256,
        checked.raw_dataset_manifest_sha256,
    )
    if actual != expected:
        raise ExtractionContractError("extraction manifest does not match the raw dataset identity")


__all__ = [
    "BODY_ANCHOR_NAMES",
    "HAND_LANDMARK_COUNT",
    "HAND_SLOT_IDS",
    "MEDIAPIPE_PACKAGE_VERSION",
    "TIMESTAMP_RULE",
    "TRACKING_ALGORITHM",
    "TRACKING_ALGORITHM_VERSION",
    "BodyAnchorName",
    "BodyAnchorV1",
    "ExtractionContractError",
    "HandSlotId",
    "HandSlotV1",
    "LandmarkExtractionManifestV1",
    "LandmarkFrameV1",
    "LandmarkFramesTableV1",
    "LandmarkObservationCountsV1",
    "LandmarkSequenceRefV1",
    "MediaPipeExtractionConfigV1",
    "MediaPipeTaskAssetV1",
    "Point3V1",
    "assert_landmark_extraction_bound_to_raw_dataset",
    "assert_landmark_sequence_ref_matches_table",
    "landmark_extraction_manifest_digest",
    "landmark_frames_table_digest",
    "landmark_observation_counts",
    "mediapipe_extraction_config_digest",
    "raw_dataset_manifest_digest",
    "validate_landmark_extraction_manifest",
    "validate_landmark_frames_table",
    "validate_mediapipe_extraction_config",
]
