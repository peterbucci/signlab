from __future__ import annotations

import math
from typing import Any, Literal, cast

import pytest

from signlab.contracts.core import ArtifactRefV1, ArtifactUriLocatorV1
from signlab.contracts.dataset import DerivedArtifactRowV1, RecordingRowV1, RecordingsTableV1
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    BodyAnchorV1,
    HandSlotV1,
    LandmarkFramesTableV1,
    LandmarkFrameV1,
    LandmarkSequenceRefV1,
    Point3V1,
    landmark_frames_table_digest,
    landmark_observation_counts,
)
from signlab.contracts.quality import SequenceQualityReportV1
from signlab.datasets.resources import build_example_dataset_tables
from signlab.quality.policy import (
    QualityPolicyError,
    aggregate_quality_reports,
    assess_landmark_sequence,
    build_elapsed_time_resampling_plan,
    elapsed_resampling_timestamps,
    interpolate_coordinate,
    interpolate_point_coordinates,
)
from signlab.quality.resources import build_default_quality_policy

DERIVED_ID = "derived_artifact_00000000000000000000000000000099"
DERIVED_SHA256 = "sha256:" + "a" * 64


def _point(
    x: float,
    *,
    y: float | None = None,
    z: float | None = None,
    visibility: float | None = None,
    presence: float | None = None,
) -> Point3V1:
    return Point3V1(
        x=x,
        y=x if y is None else y,
        z=x if z is None else z,
        visibility=visibility,
        presence=presence,
    )


def _recording(
    *,
    handedness: Literal["left", "right", "both", "unknown"] = "right",
) -> RecordingRowV1:
    tables = build_example_dataset_tables()
    recordings = tables["recordings"]
    assert isinstance(recordings, RecordingsTableV1)
    payload = recordings.rows[0].model_dump(mode="json", round_trip=True)
    payload["handedness"] = handedness
    return RecordingRowV1.model_validate(payload, strict=True)


def _hand(
    slot_id: Literal["hand_0", "hand_1"],
    *,
    present: bool,
    x: float = 0.25,
    handedness: Literal["left", "right"] = "right",
    confidence: float = 0.95,
    detector_index: int = 0,
) -> HandSlotV1:
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
    points = tuple(_point(x) for _ in range(21))
    return HandSlotV1(
        slot_id=slot_id,
        present=True,
        detector_index=detector_index,
        tracking_id=slot_id,
        handedness=handedness,
        handedness_confidence=confidence,
        image_landmarks=points,
        world_landmarks=points,
    )


def _hand_pair(
    *,
    first: bool = True,
    second: bool = False,
    first_x: float = 0.25,
    second_x: float = 0.75,
    first_label: Literal["left", "right"] = "right",
    second_label: Literal["left", "right"] = "left",
    first_confidence: float = 0.95,
    second_confidence: float = 0.95,
    first_detector_index: int = 0,
    second_detector_index: int = 1,
) -> tuple[HandSlotV1, HandSlotV1]:
    return (
        _hand(
            "hand_0",
            present=first,
            x=first_x,
            handedness=first_label,
            confidence=first_confidence,
            detector_index=first_detector_index,
        ),
        _hand(
            "hand_1",
            present=second,
            x=second_x,
            handedness=second_label,
            confidence=second_confidence,
            detector_index=second_detector_index,
        ),
    )


def _anchors(*, present: bool = True, confidence: float = 0.9) -> tuple[BodyAnchorV1, ...]:
    return tuple(
        BodyAnchorV1(
            name=name,
            present=present,
            image_point=(
                _point(0.5, visibility=confidence, presence=confidence) if present else None
            ),
            world_point=(
                _point(0.5, visibility=confidence, presence=confidence) if present else None
            ),
        )
        for name in BODY_ANCHOR_NAMES
    )


def _table(
    timestamps_us: tuple[int, ...],
    *,
    hand_rows: tuple[tuple[HandSlotV1, HandSlotV1], ...] | None = None,
    anchor_rows: tuple[tuple[BodyAnchorV1, ...], ...] | None = None,
    invalid_reasons: tuple[
        Literal["source_frame_invalid", "task_inference_failed"] | None,
        ...,
    ]
    | None = None,
) -> LandmarkFramesTableV1:
    recording = _recording()
    count = len(timestamps_us)
    hands = hand_rows or tuple(_hand_pair() for _ in range(count))
    anchors = anchor_rows or tuple(_anchors() for _ in range(count))
    reasons = invalid_reasons or (None,) * count
    assert len(hands) == len(anchors) == len(reasons) == count
    rows: list[LandmarkFrameV1] = []
    previous_task_ms: int | None = None
    for index, timestamp_us in enumerate(timestamps_us):
        reason = reasons[index]
        row_hands = (
            (_hand("hand_0", present=False), _hand("hand_1", present=False))
            if reason is not None
            else hands[index]
        )
        row_anchors = _anchors(present=False) if reason is not None else anchors[index]
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
                invalid=reason is not None,
                invalid_reason=reason,
                hands=row_hands,
                body_anchors=cast(Any, row_anchors),
                observed_hand_count=sum(hand.present for hand in row_hands),
                observed_body_anchor_count=sum(anchor.present for anchor in row_anchors),
            )
        )
        previous_task_ms = task_ms
    return LandmarkFramesTableV1(
        schema_version="landmark-frames-table/1",
        rows=tuple(rows),
    )


def _duplicate_timestamp_table() -> LandmarkFramesTableV1:
    recording = _recording()
    rows = tuple(
        LandmarkFrameV1(
            schema_version="landmark-frame/1",
            source_recording_id=recording.recording_id,
            frame_index=index,
            source_pts=1_000 + index,
            source_time_base_numerator=1,
            source_time_base_denominator=2_000_000,
            relative_timestamp_us=(0, 0, 1)[index],
            task_timestamp_ms=index,
            invalid=False,
            invalid_reason=None,
            hands=_hand_pair(),
            body_anchors=cast(Any, _anchors()),
            observed_hand_count=1,
            observed_body_anchor_count=6,
        )
        for index in range(3)
    )
    return LandmarkFramesTableV1(schema_version="landmark-frames-table/1", rows=rows)


def _sequence_reference(
    table: LandmarkFramesTableV1,
    recording: RecordingRowV1,
) -> LandmarkSequenceRefV1:
    digest = DERIVED_SHA256.removeprefix("sha256:")
    artifact = ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=DERIVED_ID,
        role="derived_data",
        media_type="application/vnd.apache.parquet",
        sha256=DERIVED_SHA256,
        size_bytes=4096,
        locator=ArtifactUriLocatorV1(
            kind="artifact_uri",
            uri=f"signlab://objects/sha256/p-{digest[:2]}/sha256-{digest}/{DERIVED_ID}",
        ),
    )
    lineage = DerivedArtifactRowV1(
        derived_artifact_id=DERIVED_ID,
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
    return LandmarkSequenceRefV1(
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


def _assess(
    table: LandmarkFramesTableV1,
    *,
    recording: RecordingRowV1 | None = None,
) -> SequenceQualityReportV1:
    source = _recording() if recording is None else recording
    return assess_landmark_sequence(
        _sequence_reference(table, source),
        table,
        source,
        build_default_quality_policy(),
    )


def test_elapsed_grid_uses_rational_time_and_preserves_exact_observed_duration() -> None:
    assert elapsed_resampling_timestamps(
        100_000,
        rate_numerator=30,
        rate_denominator=1,
    ) == (0, 33_333, 66_667, 100_000)
    assert elapsed_resampling_timestamps(
        90_000,
        rate_numerator=30,
        rate_denominator=1,
    ) == (0, 33_333, 66_667, 90_000)
    assert elapsed_resampling_timestamps(
        0,
        rate_numerator=30,
        rate_denominator=1,
    ) == (0,)


@pytest.mark.parametrize(
    ("duration", "numerator", "denominator"),
    [(-1, 30, 1), (1, 0, 1), (1, 30, 0), (2, 2_000_001, 1)],
)
def test_elapsed_grid_rejects_unrepresentable_or_invalid_inputs(
    duration: int,
    numerator: int,
    denominator: int,
) -> None:
    with pytest.raises(QualityPolicyError):
        elapsed_resampling_timestamps(
            duration,
            rate_numerator=numerator,
            rate_denominator=denominator,
        )


def test_coordinate_interpolation_uses_elapsed_time_not_frame_position() -> None:
    assert interpolate_coordinate(
        0.0,
        10.0,
        left_timestamp_us=10_000,
        right_timestamp_us=110_000,
        target_timestamp_us=35_000,
    ) == pytest.approx(2.5)
    assert interpolate_point_coordinates(
        _point(0.0, y=10.0, z=-4.0, visibility=0.1, presence=0.2),
        _point(10.0, y=20.0, z=4.0, visibility=0.9, presence=0.8),
        left_timestamp_us=10_000,
        right_timestamp_us=110_000,
        target_timestamp_us=35_000,
    ) == pytest.approx((2.5, 12.5, -2.0))


def test_coordinate_interpolation_refuses_extrapolation_and_invalid_values() -> None:
    common = {"left_timestamp_us": 10, "right_timestamp_us": 20}
    with pytest.raises(QualityPolicyError, match="extrapolation"):
        interpolate_coordinate(0.0, 1.0, target_timestamp_us=9, **common)
    with pytest.raises(QualityPolicyError, match="positive bracket"):
        interpolate_coordinate(
            0.0,
            1.0,
            left_timestamp_us=10,
            right_timestamp_us=10,
            target_timestamp_us=10,
        )
    with pytest.raises(QualityPolicyError, match="finite"):
        interpolate_coordinate(math.nan, 1.0, target_timestamp_us=15, **common)


def test_sequence_assessment_is_pure_deterministic_and_uses_all_frame_denominators() -> None:
    table = _table((0, 33_333, 66_667))
    recording = _recording()
    policy = build_default_quality_policy()
    table_before = table.model_dump_json(round_trip=True)
    recording_before = recording.model_dump_json(round_trip=True)

    first = assess_landmark_sequence(
        _sequence_reference(table, recording),
        table,
        recording,
        policy,
    )
    second = assess_landmark_sequence(
        _sequence_reference(table, recording),
        table,
        recording,
        policy,
    )

    assert first == second
    assert table.model_dump_json(round_trip=True) == table_before
    assert recording.model_dump_json(round_trip=True) == recording_before
    assert first.metrics.frame_count == 3
    assert first.metrics.expected_hand_count == 1
    assert first.metrics.expected_hand_observation_count == 3
    assert first.metrics.expected_hand_opportunity_count == 3
    assert first.metrics.expected_hand_coverage_ppm == 1_000_000
    assert first.metrics.minimum_pose_anchor_coverage_ppm == 1_000_000
    assert first.metrics.handedness_confidence_observation_count == 3
    assert first.metrics.low_handedness_confidence_fraction_ppm == 0
    assert first.metrics.pose_confidence_observation_count == 72
    assert first.disposition == "pass"
    assert first.findings == ()

    plan = first.resampling
    assert plan.observed_span_us == 66_667
    assert plan.declared_duration_us == recording.duration_us
    assert plan.declared_unobserved_tail_us == recording.duration_us - 66_667
    assert plan.declared_unobserved_tail_decision == "preserve_missing"
    assert plan.target_count == 3


def test_exhaustive_four_frame_masks_only_interpolate_internal_gaps() -> None:
    timestamps = (0, 25_000, 50_000, 75_000)
    for bits in range(1 << len(timestamps)):
        presence = tuple(bool(bits & (1 << index)) for index in range(len(timestamps)))
        table = _table(
            timestamps,
            hand_rows=tuple(_hand_pair(first=value) for value in presence),
        )
        report = _assess(table)
        hand_gaps = tuple(gap for gap in report.gaps if gap.signal == "hand_0")
        expected_runs: list[tuple[int, int]] = []
        index = 0
        while index < len(presence):
            if presence[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(presence) and not presence[index + 1]:
                index += 1
            expected_runs.append((start, index))
            index += 1
        assert tuple(
            (gap.first_missing_frame_index, gap.last_missing_frame_index) for gap in hand_gaps
        ) == tuple(expected_runs)
        for gap in hand_gaps:
            expected_decision = (
                "interpolate_linear"
                if gap.first_missing_frame_index > 0
                and gap.last_missing_frame_index < len(presence) - 1
                else "preserve_missing"
            )
            assert gap.decision == expected_decision


def test_gap_limits_are_inclusive_and_edges_long_runs_and_invalid_frames_are_preserved() -> None:
    exact_limit = _table(
        (0, 25_000, 50_000, 100_000),
        hand_rows=(
            _hand_pair(first=True),
            _hand_pair(first=False),
            _hand_pair(first=False),
            _hand_pair(first=True),
        ),
    )
    exact_gap = next(gap for gap in _assess(exact_limit).gaps if gap.signal == "hand_0")
    assert exact_gap.missing_frame_count == 2
    assert exact_gap.duration_us == 100_000
    assert exact_gap.decision == "interpolate_linear"
    assert exact_gap.reasons == ("eligible_short_internal_gap",)

    mixed = _table(
        (0, 33_333, 66_667, 100_000, 133_333, 166_667, 200_000, 233_333, 266_667),
        hand_rows=tuple(
            _hand_pair(first=value)
            for value in (False, True, False, True, False, False, False, True, False)
        ),
    )
    mixed_gaps = tuple(gap for gap in _assess(mixed).gaps if gap.signal == "hand_0")
    assert tuple(gap.boundary for gap in mixed_gaps) == (
        "leading",
        "internal",
        "internal",
        "trailing",
    )
    assert mixed_gaps[0].decision == "preserve_missing"
    assert mixed_gaps[1].decision == "interpolate_linear"
    assert mixed_gaps[2].decision == "preserve_missing"
    assert set(mixed_gaps[2].reasons) == {
        "bridge_too_long",
        "too_many_missing_frames",
    }
    assert mixed_gaps[3].decision == "preserve_missing"

    invalid = _table(
        (0, 33_333, 66_667),
        invalid_reasons=(None, "task_inference_failed", None),
    )
    invalid_gap = next(gap for gap in _assess(invalid).gaps if gap.signal == "hand_0")
    assert invalid_gap.boundary == "internal"
    assert invalid_gap.contains_invalid_frame is True
    assert invalid_gap.decision == "preserve_missing"
    assert "invalid_frame" in invalid_gap.reasons


def test_gap_confidence_and_identity_barriers_do_not_reclassify_present_pose() -> None:
    low_hand_boundary = _table(
        (0, 33_333, 66_667),
        hand_rows=(
            _hand_pair(first_confidence=0.7),
            _hand_pair(first=False),
            _hand_pair(),
        ),
    )
    low_gap = next(gap for gap in _assess(low_hand_boundary).gaps if gap.signal == "hand_0")
    assert low_gap.decision == "preserve_missing"
    assert low_gap.contains_identity_ambiguity is False
    assert low_gap.reasons == ("low_confidence",)

    label_conflict = _table(
        (0, 33_333, 66_667),
        hand_rows=(
            _hand_pair(first_label="right"),
            _hand_pair(first=False),
            _hand_pair(first_label="left"),
        ),
    )
    ambiguous = next(gap for gap in _assess(label_conflict).gaps if gap.signal == "hand_0")
    assert ambiguous.decision == "preserve_missing"
    assert ambiguous.contains_identity_ambiguity is True
    assert ambiguous.crosses_suspected_hand_swap is False
    assert ambiguous.reasons == ("identity_ambiguity",)

    low_pose = _table(
        (0, 33_333, 66_667),
        anchor_rows=tuple(_anchors(confidence=0.1) for _ in range(3)),
    )
    pose_report = _assess(low_pose)
    assert not any(gap.signal in BODY_ANCHOR_NAMES for gap in pose_report.gaps)
    assert pose_report.metrics.pose_anchor_presence_counts == (3, 3, 3, 3, 3, 3)
    assert pose_report.metrics.low_pose_confidence_observation_count == 72


def test_irregular_and_duplicate_timestamps_are_diagnostic_and_block_gap_interpolation() -> None:
    irregular = _table(
        (0, 33_333, 200_000, 233_333),
        hand_rows=(
            _hand_pair(),
            _hand_pair(first=False),
            _hand_pair(),
            _hand_pair(),
        ),
    )
    report = _assess(irregular)
    gap = next(gap for gap in report.gaps if gap.signal == "hand_0")
    assert report.metrics.timestamp_discontinuity_count == 1
    assert report.metrics.median_timestamp_delta_us == 33_333
    assert report.metrics.maximum_timestamp_delta_us == 166_667
    assert gap.crosses_timestamp_discontinuity is True
    assert gap.decision == "preserve_missing"
    assert "timestamp_discontinuity" in gap.reasons
    assert report.resampling.last_target_timestamp_us == 233_333

    duplicate_report = _assess(_duplicate_timestamp_table())
    assert duplicate_report.metrics.timestamp_delta_count == 2
    assert duplicate_report.metrics.median_timestamp_delta_us == 1
    assert duplicate_report.metrics.timestamp_discontinuity_count == 1
    assert duplicate_report.metrics.temporal_discontinuity_count == 0


def test_detector_index_reversal_is_not_a_swap_but_crossed_stable_slots_are_suspected() -> None:
    recording = _recording(handedness="both")
    stable = _table(
        (0, 33_333),
        hand_rows=(
            _hand_pair(
                second=True,
                first_x=0.2,
                second_x=0.8,
                first_label="left",
                second_label="right",
                first_detector_index=0,
                second_detector_index=1,
            ),
            _hand_pair(
                second=True,
                first_x=0.2,
                second_x=0.8,
                first_label="left",
                second_label="right",
                first_detector_index=1,
                second_detector_index=0,
            ),
        ),
    )
    assert _assess(stable, recording=recording).metrics.suspected_hand_swap_count == 0

    crossed = _table(
        (0, 33_333),
        hand_rows=(
            _hand_pair(
                second=True,
                first_x=0.2,
                second_x=0.8,
                first_label="left",
                second_label="right",
            ),
            _hand_pair(
                second=True,
                first_x=0.8,
                second_x=0.2,
                first_label="right",
                second_label="left",
            ),
        ),
    )
    swapped = _assess(crossed, recording=recording)
    assert swapped.metrics.suspected_hand_swap_count == 1
    assert any(
        finding.metric == "suspected_hand_swap_count" and finding.severity == "warning"
        for finding in swapped.findings
    )

    crossed_gap = _table(
        (0, 33_333, 66_667),
        hand_rows=(
            _hand_pair(
                second=True,
                first_x=0.2,
                second_x=0.8,
                first_label="left",
                second_label="right",
            ),
            _hand_pair(first=False, second=False),
            _hand_pair(
                second=True,
                first_x=0.8,
                second_x=0.2,
                first_label="right",
                second_label="left",
            ),
        ),
    )
    crossed_gap_report = _assess(crossed_gap, recording=recording)
    hand_gap = next(gap for gap in crossed_gap_report.gaps if gap.signal == "hand_0")
    assert crossed_gap_report.metrics.suspected_hand_swap_count == 1
    assert hand_gap.crosses_suspected_hand_swap is True
    assert hand_gap.contains_identity_ambiguity is True
    assert hand_gap.decision == "preserve_missing"


def test_pose_absence_allows_hand_only_quarantine_while_no_hands_and_zero_valid_reject() -> None:
    no_pose = _table(
        (0, 33_333, 66_667),
        anchor_rows=tuple(_anchors(present=False) for _ in range(3)),
    )
    hand_only = _assess(no_pose)
    assert hand_only.metrics.minimum_pose_anchor_coverage_ppm == 0
    assert hand_only.disposition == "quarantine"
    assert all(
        finding.severity != "reject"
        for finding in hand_only.findings
        if finding.metric == "minimum_pose_anchor_coverage_ppm"
    )

    no_hands = _table(
        (0, 33_333),
        hand_rows=tuple(_hand_pair(first=False) for _ in range(2)),
    )
    undetected = _assess(no_hands)
    assert undetected.metrics.valid_frame_count == 2
    assert undetected.metrics.expected_hand_coverage_ppm == 0
    assert undetected.disposition == "reject"

    zero_valid = _table(
        (0, 33_333),
        invalid_reasons=("source_frame_invalid", "task_inference_failed"),
    )
    invalid = _assess(zero_valid)
    assert invalid.metrics.valid_frame_count == 0
    assert invalid.metrics.invalid_frame_fraction_ppm == 1_000_000
    assert invalid.disposition == "reject"


def test_both_hand_cardinality_and_strict_threshold_boundaries_are_honest() -> None:
    recording = _recording(handedness="both")
    one_visible = _table((0, 33_333, 66_667))
    report = _assess(one_visible, recording=recording)

    assert report.metrics.expected_hand_count == 2
    assert report.metrics.expected_hand_observation_count == 3
    assert report.metrics.expected_hand_opportunity_count == 6
    assert report.metrics.expected_hand_coverage_ppm == 500_000
    coverage_finding = next(
        finding for finding in report.findings if finding.metric == "expected_hand_coverage_ppm"
    )
    assert coverage_finding.severity == "quarantine"
    assert coverage_finding.threshold == 800_000
    assert not any(finding.severity == "reject" for finding in report.findings)
    assert not any(finding.metric == "suspected_hand_swap_count" for finding in report.findings)


def test_dataset_aggregation_uses_weighted_denominators_and_rejects_bad_inputs() -> None:
    table = _table((0, 33_333, 66_667))
    policy = build_default_quality_policy()
    report = _assess(table)
    second_recording_payload = _recording().model_dump(mode="json", round_trip=True)
    second_recording_id = "recording_00000000000000000000000000000099"
    second_recording_payload["recording_id"] = second_recording_id
    second_recording_payload["consent_grant"]["recording_id"] = second_recording_id
    second_recording_payload["media"]["artifact_id"] = second_recording_id
    second_recording_payload["media"]["locator"]["path"] = (
        second_recording_payload["media"]["locator"]["path"].rsplit("/", 1)[0]
        + "/"
        + second_recording_id
    )
    second_recording = RecordingRowV1.model_validate(second_recording_payload, strict=True)
    second_table_payload = _table(
        (0, 33_333, 66_667, 100_000, 133_333, 166_667, 200_000),
        invalid_reasons=(None, None, None, None, None, None, "source_frame_invalid"),
    ).model_dump(mode="python", round_trip=True)
    for row in second_table_payload["rows"]:
        row["source_recording_id"] = second_recording_id
    second_table = LandmarkFramesTableV1.model_validate(second_table_payload, strict=True)
    second_report = _assess(second_table, recording=second_recording)

    aggregate = aggregate_quality_reports((report, second_report), policy)

    assert aggregate.sequence_count == 2
    assert (aggregate.pass_count, aggregate.quarantine_count) == (1, 1)
    assert aggregate.total_frame_count == 10
    assert aggregate.total_invalid_frame_count == 1
    assert aggregate.invalid_frame_fraction_ppm == 100_000
    assert aggregate.total_expected_hand_observation_count == 9
    assert aggregate.total_expected_hand_opportunity_count == 10
    assert aggregate.expected_hand_coverage_ppm == 900_000
    assert aggregate.minimum_pose_anchor_coverage_ppm == 857_143
    assert aggregate.longest_unfilled_internal_hand_gap_us == 0
    assert aggregate.timestamp_discontinuity_count == 0
    assert aggregate.temporal_discontinuity_count == 0
    assert aggregate.suspected_hand_swap_count == 0
    assert aggregate.status == "blocked"
    with pytest.raises(QualityPolicyError, match="at least one"):
        aggregate_quality_reports((), policy)
    with pytest.raises(QualityPolicyError, match="unique recording IDs"):
        aggregate_quality_reports((report, report), policy)


def test_resampling_plan_rejects_recording_identity_mismatch() -> None:
    table = _table((0, 33_333))
    payload = _recording().model_dump(mode="json", round_trip=True)
    payload["recording_id"] = "recording_00000000000000000000000000000099"
    payload["consent_grant"]["recording_id"] = payload["recording_id"]
    payload["media"]["artifact_id"] = payload["recording_id"]
    payload["media"]["locator"]["path"] = (
        payload["media"]["locator"]["path"].rsplit("/", 1)[0] + "/" + payload["recording_id"]
    )
    mismatched = RecordingRowV1.model_validate(payload, strict=True)
    with pytest.raises(QualityPolicyError, match="identities"):
        build_elapsed_time_resampling_plan(
            table,
            mismatched,
            build_default_quality_policy(),
        )
