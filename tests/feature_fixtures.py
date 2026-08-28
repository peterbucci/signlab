"""Reusable synthetic landmark evidence for portable-feature tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from signlab.contracts.core import ArtifactRefV1, ArtifactUriLocatorV1, ContractRefV1
from signlab.contracts.dataset import DerivedArtifactRowV1, RecordingRowV1, RecordingsTableV1
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    BodyAnchorV1,
    HandSlotId,
    HandSlotV1,
    LandmarkFramesTableV1,
    LandmarkFrameV1,
    LandmarkSequenceRefV1,
    Point3V1,
    landmark_frames_table_digest,
    landmark_observation_counts,
)
from signlab.contracts.features import (
    BodyRelativeRuleV1,
    FeatureRepresentation,
    HandLocalRuleV1,
    LandmarkFeaturePlanV1,
    LearnedStatisticsRuleV1,
    OptionalFeatureRuleV1,
    PaddingFeatureRuleV1,
    TemporalFeatureRuleV1,
    registered_feature_names,
)
from signlab.contracts.pipeline import SplitManifestV1, SplitPartitionV1
from signlab.contracts.quality import SequenceQualityReportV1
from signlab.datasets.resources import build_example_dataset_tables
from signlab.quality.policy import assess_landmark_sequence
from signlab.quality.resources import build_default_quality_policy

type HandRow = tuple[HandSlotV1, HandSlotV1]
type AnchorRow = tuple[
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
]
type Point = tuple[float, float, float]

EXTRACTION_CONFIG_SHA256 = "sha256:" + "e" * 64
_DERIVED_ID = "derived_artifact_00000000000000000000000000000222"
_DERIVED_SHA256 = "sha256:" + "d" * 64

# Deliberately non-collinear, nondegenerate MediaPipe-order hand geometry. These
# project-authored coordinates are synthetic and do not describe a real signer.
_HAND_SHAPE: tuple[Point, ...] = (
    (0.00, 0.00, 0.00),
    (-0.22, 0.12, 0.02),
    (-0.38, 0.27, 0.05),
    (-0.49, 0.46, 0.09),
    (-0.55, 0.66, 0.13),
    (-0.23, 0.38, 0.02),
    (-0.26, 0.73, 0.05),
    (-0.25, 1.02, 0.09),
    (-0.22, 1.29, 0.12),
    (0.00, 0.45, 0.01),
    (0.01, 0.84, 0.04),
    (0.04, 1.18, 0.08),
    (0.08, 1.47, 0.11),
    (0.22, 0.40, -0.02),
    (0.25, 0.77, 0.01),
    (0.29, 1.08, 0.05),
    (0.34, 1.34, 0.09),
    (0.40, 0.31, -0.03),
    (0.47, 0.61, 0.00),
    (0.53, 0.87, 0.04),
    (0.60, 1.09, 0.08),
)

_ANCHOR_POINTS: dict[str, Point] = {
    "left_shoulder": (0.30, 0.40, 0.00),
    "right_shoulder": (0.70, 0.40, 0.00),
    "left_elbow": (0.24, 0.57, 0.01),
    "right_elbow": (0.76, 0.57, 0.01),
    "left_wrist": (0.20, 0.73, 0.02),
    "right_wrist": (0.80, 0.73, 0.02),
}


@dataclass(frozen=True, slots=True)
class FeatureFixture:
    """One fully bound raw table, extraction reference, and quality report."""

    recording: RecordingRowV1
    table: LandmarkFramesTableV1
    sequence: LandmarkSequenceRefV1
    quality: SequenceQualityReportV1


def _point(coordinates: Point, *, scored: bool = False) -> Point3V1:
    return Point3V1(
        x=coordinates[0],
        y=coordinates[1],
        z=coordinates[2],
        visibility=0.99 if scored else None,
        presence=0.99 if scored else None,
    )


def _global_point(
    point: Point,
    *,
    scale: float,
    translation: Point,
) -> Point:
    return cast(
        Point,
        tuple(translation[axis] + scale * point[axis] for axis in range(3)),
    )


def _raw_mirror(point: Point, *, mirrored: bool, image: bool) -> Point:
    if not mirrored:
        return point
    return ((1.0 - point[0]) if image else -point[0], point[1], point[2])


def make_hand(
    slot_id: HandSlotId,
    *,
    present: bool = True,
    anatomical_handedness: Literal["left", "right"] = "right",
    detector_index: int = 0,
    timestamp_us: int = 0,
    image_center: tuple[float, float] = (0.45, 0.58),
    image_velocity_per_second: tuple[float, float] = (0.0, 0.0),
    image_global_scale: float = 1.0,
    image_global_translation: tuple[float, float] = (0.0, 0.0),
    world_global_scale: float = 1.0,
    world_global_translation: Point = (0.0, 0.0, 0.0),
    mirrored: bool = False,
) -> HandSlotV1:
    """Build one synthetic stable hand slot with nondegenerate geometry."""

    if not present:
        return HandSlotV1(
            slot_id=slot_id,
            present=False,
            detector_index=None,
            tracking_id=None,
            handedness=None,
            handedness_confidence=None,
            image_landmarks=None,
            world_landmarks=None,
        )
    handedness_sign = -1.0 if anatomical_handedness == "left" else 1.0
    vendor_handedness: Literal["left", "right"] = (
        anatomical_handedness
        if mirrored
        else ("right" if anatomical_handedness == "left" else "left")
    )
    elapsed_seconds = timestamp_us / 1_000_000
    canonical_world = tuple(
        (
            handedness_sign * item[0],
            item[1],
            item[2],
        )
        for item in _HAND_SHAPE
    )
    world = tuple(
        _point(
            _raw_mirror(
                _global_point(
                    item,
                    scale=world_global_scale,
                    translation=world_global_translation,
                ),
                mirrored=mirrored,
                image=False,
            )
        )
        for item in canonical_world
    )
    canonical_image = tuple(
        (
            image_center[0]
            + image_velocity_per_second[0] * elapsed_seconds
            + handedness_sign * item[0] * 0.08,
            image_center[1] + image_velocity_per_second[1] * elapsed_seconds - item[1] * 0.08,
            item[2] * 0.08,
        )
        for item in _HAND_SHAPE
    )
    image = tuple(
        _point(
            _raw_mirror(
                (
                    image_global_translation[0] + image_global_scale * item[0],
                    image_global_translation[1] + image_global_scale * item[1],
                    image_global_scale * item[2],
                ),
                mirrored=mirrored,
                image=True,
            )
        )
        for item in canonical_image
    )
    return HandSlotV1(
        slot_id=slot_id,
        present=True,
        detector_index=detector_index,
        tracking_id=slot_id,
        handedness=vendor_handedness,
        handedness_confidence=0.99,
        image_landmarks=image,
        world_landmarks=world,
    )


def make_hand_row(
    *,
    timestamp_us: int,
    two_hands: bool = False,
    first_present: bool = True,
    second_present: bool | None = None,
    first_anatomical_handedness: Literal["left", "right"] = "right",
    second_anatomical_handedness: Literal["left", "right"] = "left",
    first_center: tuple[float, float] = (0.42, 0.58),
    second_center: tuple[float, float] = (0.68, 0.58),
    image_velocity_per_second: tuple[float, float] = (0.0, 0.0),
    image_global_scale: float = 1.0,
    image_global_translation: tuple[float, float] = (0.0, 0.0),
    world_global_scale: float = 1.0,
    world_global_translation: Point = (0.0, 0.0, 0.0),
    mirrored: bool = False,
) -> HandRow:
    """Build the exact two-slot row required by extraction."""

    second_is_present = two_hands if second_present is None else second_present
    return (
        make_hand(
            "hand_0",
            present=first_present,
            anatomical_handedness=first_anatomical_handedness,
            detector_index=0,
            timestamp_us=timestamp_us,
            image_center=first_center,
            image_velocity_per_second=image_velocity_per_second,
            image_global_scale=image_global_scale,
            image_global_translation=image_global_translation,
            world_global_scale=world_global_scale,
            world_global_translation=world_global_translation,
            mirrored=mirrored,
        ),
        make_hand(
            "hand_1",
            present=second_is_present,
            anatomical_handedness=second_anatomical_handedness,
            detector_index=1,
            timestamp_us=timestamp_us,
            image_center=second_center,
            image_velocity_per_second=image_velocity_per_second,
            image_global_scale=image_global_scale,
            image_global_translation=image_global_translation,
            world_global_scale=world_global_scale,
            world_global_translation=world_global_translation,
            mirrored=mirrored,
        ),
    )


def make_anchor_row(
    *,
    present: bool = True,
    missing: frozenset[str] = frozenset(),
    image_global_scale: float = 1.0,
    image_global_translation: tuple[float, float] = (0.0, 0.0),
    mirrored: bool = False,
) -> AnchorRow:
    """Build six ordered pose anchors, optionally masking selected names."""

    anchors: list[BodyAnchorV1] = []
    for name in BODY_ANCHOR_NAMES:
        available = present and name not in missing
        canonical = _ANCHOR_POINTS[name]
        transformed = (
            image_global_translation[0] + image_global_scale * canonical[0],
            image_global_translation[1] + image_global_scale * canonical[1],
            image_global_scale * canonical[2],
        )
        raw = _raw_mirror(transformed, mirrored=mirrored, image=True)
        anchors.append(
            BodyAnchorV1(
                name=name,
                present=available,
                image_point=_point(raw, scored=True) if available else None,
                world_point=_point(raw, scored=True) if available else None,
            )
        )
    return cast(AnchorRow, tuple(anchors))


def make_recording(
    *,
    handedness: Literal["left", "right", "both", "unknown"] = "right",
    mirrored: bool = False,
    duration_us: int = 1_000_000,
) -> RecordingRowV1:
    """Reuse opaque example identities while changing only synthetic capture facts."""

    recordings = build_example_dataset_tables()["recordings"]
    assert isinstance(recordings, RecordingsTableV1)
    payload = recordings.rows[0].model_dump(mode="json", round_trip=True)
    payload.update(
        handedness=handedness,
        mirror_state="mirrored" if mirrored else "not_mirrored",
        duration_us=duration_us,
    )
    return RecordingRowV1.model_validate(payload, strict=True)


def make_table(
    recording: RecordingRowV1,
    timestamps_us: tuple[int, ...],
    hand_rows: tuple[HandRow, ...],
    anchor_rows: tuple[AnchorRow, ...],
) -> LandmarkFramesTableV1:
    """Build one exact extraction table over the supplied source timeline."""

    if not timestamps_us or timestamps_us[0] != 0:
        raise ValueError("synthetic feature timelines must start at zero")
    if len(hand_rows) != len(timestamps_us) or len(anchor_rows) != len(timestamps_us):
        raise ValueError("synthetic feature rows must cover every timestamp")
    rows: list[LandmarkFrameV1] = []
    previous_task_ms: int | None = None
    for index, (timestamp_us, hands, anchors) in enumerate(
        zip(timestamps_us, hand_rows, anchor_rows, strict=True)
    ):
        task_ms = max(
            timestamp_us // 1_000,
            0 if previous_task_ms is None else previous_task_ms + 1,
        )
        rows.append(
            LandmarkFrameV1(
                schema_version="landmark-frame/1",
                source_recording_id=recording.recording_id,
                frame_index=index,
                source_pts=1_000_000 + timestamp_us,
                source_time_base_numerator=1,
                source_time_base_denominator=1_000_000,
                relative_timestamp_us=timestamp_us,
                task_timestamp_ms=task_ms,
                invalid=False,
                invalid_reason=None,
                hands=hands,
                body_anchors=anchors,
                observed_hand_count=sum(hand.present for hand in hands),
                observed_body_anchor_count=sum(anchor.present for anchor in anchors),
            )
        )
        previous_task_ms = task_ms
    return LandmarkFramesTableV1(schema_version="landmark-frames-table/1", rows=tuple(rows))


def bind_feature_fixture(
    recording: RecordingRowV1,
    table: LandmarkFramesTableV1,
) -> FeatureFixture:
    """Bind strict extraction and quality evidence to a synthetic raw table."""

    digest = _DERIVED_SHA256.removeprefix("sha256:")
    artifact = ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=_DERIVED_ID,
        role="derived_data",
        media_type="application/vnd.apache.parquet",
        sha256=_DERIVED_SHA256,
        size_bytes=4096,
        locator=ArtifactUriLocatorV1(
            kind="artifact_uri",
            uri=f"signlab://objects/sha256/p-{digest[:2]}/sha256-{digest}/{_DERIVED_ID}",
        ),
    )
    lineage = DerivedArtifactRowV1(
        derived_artifact_id=_DERIVED_ID,
        derivation_kind="landmark_extraction",
        parent_artifact_ids=(recording.recording_id,),
        participant_id=recording.participant_id,
        session_id=recording.session_id,
        source_recording_id=recording.recording_id,
        clip_id=None,
        annotation_id=None,
        sample_id=None,
        label_id=None,
        split_id=None,
        partition=None,
        handedness=recording.handedness,
        mirror_state=recording.mirror_state,
        operation_id="mediapipe_tasks_video",
        operation_version="1.0.0",
        artifact=artifact,
    )
    sequence = LandmarkSequenceRefV1(
        schema_version="landmark-sequence-reference/1",
        lineage=lineage,
        source_media_sha256=recording.media.sha256,
        source_media_size_bytes=recording.media.size_bytes,
        source_rotation_degrees=recording.rotation_degrees,
        source_mirror_state=recording.mirror_state,
        frames_schema_version="landmark-frames-table/1",
        content_sha256=landmark_frames_table_digest(table),
        counts=landmark_observation_counts(table),
    )
    quality = assess_landmark_sequence(
        sequence,
        table,
        recording,
        build_default_quality_policy(),
    )
    return FeatureFixture(
        recording=recording,
        table=table,
        sequence=sequence,
        quality=quality,
    )


def make_feature_fixture(
    timestamps_us: tuple[int, ...] = (0, 33_333, 66_667),
    *,
    two_hands: bool = False,
    pose_present: bool = True,
    mirrored: bool = False,
    hand_rows: tuple[HandRow, ...] | None = None,
    anchor_rows: tuple[AnchorRow, ...] | None = None,
    image_velocity_per_second: tuple[float, float] = (0.0, 0.0),
    image_global_scale: float = 1.0,
    image_global_translation: tuple[float, float] = (0.0, 0.0),
    world_global_scale: float = 1.0,
    world_global_translation: Point = (0.0, 0.0, 0.0),
) -> FeatureFixture:
    """Build the normal one- or two-hand synthetic feature evidence path."""

    duration_us = max(1, timestamps_us[-1] + 33_333)
    recording = make_recording(
        handedness="both" if two_hands else "right",
        mirrored=mirrored,
        duration_us=duration_us,
    )
    hands = hand_rows or tuple(
        make_hand_row(
            timestamp_us=timestamp,
            two_hands=two_hands,
            image_velocity_per_second=image_velocity_per_second,
            image_global_scale=image_global_scale,
            image_global_translation=image_global_translation,
            world_global_scale=world_global_scale,
            world_global_translation=world_global_translation,
            mirrored=mirrored,
        )
        for timestamp in timestamps_us
    )
    anchors = anchor_rows or tuple(
        make_anchor_row(
            present=pose_present,
            image_global_scale=image_global_scale,
            image_global_translation=image_global_translation,
            mirrored=mirrored,
        )
        for _ in timestamps_us
    )
    return bind_feature_fixture(recording, make_table(recording, timestamps_us, hands, anchors))


def make_feature_plan(
    representation: FeatureRepresentation = "combined",
    *,
    target_frame_count: int = 4,
    include_velocity: bool = False,
    include_acceleration: bool = False,
    include_joint_angles: bool = False,
    include_tip_distances: bool = False,
    statistics_mode: Literal["none", "train_only_masked_zscore/1"] = "none",
) -> LandmarkFeaturePlanV1:
    """Build the exact registered portable feature plan used by tests."""

    optional = OptionalFeatureRuleV1(
        include_velocity=include_velocity,
        include_acceleration=include_acceleration,
        include_joint_angles=include_joint_angles,
        include_tip_distances=include_tip_distances,
        joint_angle_rule="five_registered_pip_flexion_angles_radians/1",
        tip_distance_rule="five_registered_wrist_to_fingertip_distances/1",
    )
    return LandmarkFeaturePlanV1(
        schema_version="landmark-feature-plan/1",
        plan_id="portable_landmark_features",
        version="1.0.0",
        representation=representation,
        compatible_runtimes=("python", "typescript"),
        hand_slots=("hand_0", "hand_1"),
        handedness_source="mediapipe_vendor_report_corrected_by_source_mirror_state",
        swap_rule="preserve_slots_never_repair",
        hand_local=HandLocalRuleV1(
            coordinate_space="hand_world_xyz",
            landmark_order=tuple(range(21)),
            center="wrist_landmark_0",
            scale="wrist_to_middle_mcp_landmark_9_euclidean",
            source_mirror_rule="undo_world_x_when_source_mirrored",
            anatomical_canonicalization=(
                "swap_vendor_label_when_not_mirrored_then_reflect_left_hand_x"
            ),
            zero_scale_rule="mask_hand_features",
        ),
        body_relative=BodyRelativeRuleV1(
            coordinate_space="image_xy",
            trajectory_points=("wrist", "palm_centroid"),
            palm_landmarks=(0, 5, 9, 17),
            center="shoulder_midpoint",
            scale="shoulder_width_xy_euclidean",
            source_mirror_rule="undo_image_x_when_source_mirrored",
            missing_anchor_rule="mask_body_keep_hand_local",
            zero_scale_rule="mask_body_features",
        ),
        temporal=TemporalFeatureRuleV1(
            clock="relative_timestamp_us",
            grid_rule="nominal_elapsed_time_append_final/1",
            target_rate_numerator=30,
            target_rate_denominator=1,
            interpolation="quality_report_approved_linear_coordinates_only",
            extrapolation_allowed=False,
            forward_fill_allowed=False,
            derivative_rule="backward_elapsed_time_finite_difference/1",
            derivative_application_order="resample_then_derive_then_select_then_pad/1",
        ),
        padding=PaddingFeatureRuleV1(
            target_frame_count=target_frame_count,
            long_sequence_rule="uniform_endpoint_preserving_index_selection/1",
            padding_side="right",
            padding_value_q=0,
            padding_mask_rule="all_feature_masks_false",
            padding_timestamp_rule="continue_nominal_grid",
        ),
        optional=optional,
        learned_statistics=LearnedStatisticsRuleV1(
            mode=statistics_mode,
            partition_evidence="explicit_train_membership_required",
            masked_value_rule="exclude_from_fit",
            zero_count_rule="mean_zero_scale_one",
            zero_variance_rule="scale_one",
        ),
        quantization_scale=1_000_000,
        quantization_rule="round_half_away_from_zero/1",
        interchange_values="signed_integer_divided_by_quantization_scale",
        feature_order=registered_feature_names(representation, optional),
    )


def make_split_manifest(
    *,
    train_recording_ids: tuple[str, ...],
    validation_recording_ids: tuple[str, ...],
    test_recording_ids: tuple[str, ...],
) -> SplitManifestV1:
    """Build a strict grouped split around caller-selected recording membership."""

    partitions: list[SplitPartitionV1] = []
    next_identity = 1
    for name, recording_ids in (
        ("train", train_recording_ids),
        ("validation", validation_recording_ids),
        ("test", test_recording_ids),
    ):
        count = len(recording_ids)
        identities = tuple(range(next_identity, next_identity + count))
        next_identity += count
        partitions.append(
            SplitPartitionV1(
                name=cast(Any, name),
                sample_ids=tuple(f"sample_{number:032x}" for number in identities),
                participant_ids=tuple(f"participant_{number:032x}" for number in identities),
                session_ids=tuple(f"session_{number:032x}" for number in identities),
                source_recording_ids=tuple(sorted(recording_ids)),
            )
        )
    return SplitManifestV1(
        schema_version="split-manifest/1",
        split_id="feature_fixture_split",
        version="1.0.0",
        dataset=ContractRefV1(
            schema_version="contract-reference/1",
            kind="dataset",
            contract_schema_version="dataset-manifest/2",
            contract_id="feature_fixture_dataset",
            contract_version="1.0.0",
            canonicalization="rfc8785/1",
            sha256="sha256:" + "a" * 64,
            locator=ArtifactUriLocatorV1(
                kind="artifact_uri",
                uri="signlab://fixtures/contracts/feature-dataset",
            ),
        ),
        dataset_data_sha256="sha256:" + "b" * 64,
        strategy="participant-and-session-grouped",
        random_seed=1729,
        partitions=cast(Any, tuple(partitions)),
    )
