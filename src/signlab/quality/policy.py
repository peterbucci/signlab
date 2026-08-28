"""Pure landmark-quality assessment and elapsed-time planning.

This module never reads or writes files. Storage adapters validate and load exact
artifacts before calling these services, then persist the returned strict contracts.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal, cast

from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.dataset import RecordingRowV1
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    HAND_SLOT_IDS,
    BodyAnchorV1,
    HandSlotV1,
    LandmarkFramesTableV1,
    LandmarkFrameV1,
    LandmarkSequenceRefV1,
    Point3V1,
    assert_landmark_sequence_ref_matches_table,
    landmark_frames_table_digest,
)
from signlab.contracts.quality import (
    QUALITY_METRIC_DIRECTIONS,
    DatasetQualityReportV1,
    LandmarkQualityPolicyV1,
    MissingIntervalReason,
    MissingIntervalV1,
    QualityContractError,
    QualityFindingSeverity,
    QualityFindingV1,
    QualityMetricName,
    QualityThresholdRuleV1,
    SequenceQualityMetricsV1,
    SequenceQualityReportV1,
    TemporalResamplingSummaryV1,
    elapsed_time_grid_commitment,
    elapsed_time_grid_shape,
    elapsed_time_grid_us,
    landmark_quality_policy_digest,
    ratio_ppm,
    sequence_quality_report_digest,
)

_PALM_INDICES = (0, 5, 9, 13, 17)
_HAND_SIGNALS = frozenset(HAND_SLOT_IDS)
type _SignalName = Literal[
    "hand_0",
    "hand_1",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
type _Observation = HandSlotV1 | BodyAnchorV1


class QualityPolicyError(ValueError):
    """Raised when otherwise valid inputs cannot form a quality assessment."""


def _round_ratio_half_up(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise QualityPolicyError("rounding operands must describe a non-negative ratio")
    return (2 * numerator + denominator) // (2 * denominator)


def elapsed_resampling_timestamps(
    final_observed_us: int,
    *,
    rate_numerator: int,
    rate_denominator: int,
) -> tuple[int, ...]:
    """Build a rational elapsed-time grid with the exact observed endpoint.

    Nominal samples use half-up integer microsecond rounding. If the final source
    timestamp is off-grid it is appended; observations are never stretched to make
    the nominal cadence divide their duration.
    """

    try:
        return elapsed_time_grid_us(
            final_observed_us,
            rate_numerator,
            rate_denominator,
        )
    except QualityContractError as error:
        raise QualityPolicyError(str(error)) from error


def interpolate_coordinate(
    left: float,
    right: float,
    *,
    left_timestamp_us: int,
    right_timestamp_us: int,
    target_timestamp_us: int,
) -> float:
    """Linearly interpolate one finite coordinate inside a closed time bracket."""

    if not (math.isfinite(left) and math.isfinite(right)):
        raise QualityPolicyError("interpolation endpoints must be finite")
    if left_timestamp_us < 0 or right_timestamp_us <= left_timestamp_us:
        raise QualityPolicyError("interpolation timestamps must form a positive bracket")
    if not left_timestamp_us <= target_timestamp_us <= right_timestamp_us:
        raise QualityPolicyError(
            "interpolation target must be bracketed; extrapolation is forbidden"
        )
    numerator = target_timestamp_us - left_timestamp_us
    denominator = right_timestamp_us - left_timestamp_us
    result = left + (right - left) * (numerator / denominator)
    if not math.isfinite(result):
        raise QualityPolicyError("interpolated coordinate is not finite")
    return result


def interpolate_point_coordinates(
    left: Point3V1,
    right: Point3V1,
    *,
    left_timestamp_us: int,
    right_timestamp_us: int,
    target_timestamp_us: int,
) -> tuple[float, float, float]:
    """Interpolate only XYZ, deliberately excluding confidence-like channels."""

    coordinates = tuple(
        interpolate_coordinate(
            first,
            second,
            left_timestamp_us=left_timestamp_us,
            right_timestamp_us=right_timestamp_us,
            target_timestamp_us=target_timestamp_us,
        )
        for first, second in ((left.x, right.x), (left.y, right.y), (left.z, right.z))
    )
    return (coordinates[0], coordinates[1], coordinates[2])


def build_elapsed_time_resampling_plan(
    table: LandmarkFramesTableV1,
    recording: RecordingRowV1,
    policy: LandmarkQualityPolicyV1,
) -> TemporalResamplingSummaryV1:
    """Summarize a nominal elapsed-time grid without materializing coordinates."""

    return _build_elapsed_time_resampling_plan(
        table,
        source_recording_id=recording.recording_id,
        declared_duration_us=recording.duration_us,
        policy=policy,
    )


def _build_elapsed_time_resampling_plan(
    table: LandmarkFramesTableV1,
    *,
    source_recording_id: str,
    declared_duration_us: int,
    policy: LandmarkQualityPolicyV1,
) -> TemporalResamplingSummaryV1:
    if table.rows[0].source_recording_id != source_recording_id:
        raise QualityPolicyError("landmark table and recording identities do not match")
    if declared_duration_us <= 0:
        raise QualityPolicyError("declared duration must be positive")
    observed_span_us = table.rows[-1].relative_timestamp_us
    target_count, last_target_timestamp_us = elapsed_time_grid_shape(
        observed_span_us,
        policy.target_rate_numerator,
        policy.target_rate_denominator,
    )
    return TemporalResamplingSummaryV1(
        schema_version="temporal-resampling-summary/1",
        clock="relative_timestamp_us",
        grid_rule=policy.resampling_rule,
        target_rate_numerator=policy.target_rate_numerator,
        target_rate_denominator=policy.target_rate_denominator,
        declared_duration_us=declared_duration_us,
        observed_span_us=observed_span_us,
        declared_unobserved_tail_us=max(declared_duration_us - observed_span_us, 0),
        declared_unobserved_tail_decision="preserve_missing",
        target_count=target_count,
        first_target_timestamp_us=0,
        last_target_timestamp_us=last_target_timestamp_us,
        target_grid_commitment_sha256=elapsed_time_grid_commitment(
            observed_span_us,
            policy.target_rate_numerator,
            policy.target_rate_denominator,
        ),
    )


def _is_low(score: float, threshold_ppm: int) -> bool:
    return score * 1_000_000 < threshold_ppm


def _pose_scores(
    anchor: BodyAnchorV1,
) -> tuple[tuple[Literal["visibility", "presence"], float], ...]:
    if not anchor.present:
        return ()
    assert anchor.image_point is not None
    assert anchor.world_point is not None
    scores: list[tuple[Literal["visibility", "presence"], float]] = []
    for point in (anchor.image_point, anchor.world_point):
        if point.visibility is not None:
            scores.append(("visibility", point.visibility))
        if point.presence is not None:
            scores.append(("presence", point.presence))
    return tuple(scores)


def _pose_is_low_confidence(anchor: BodyAnchorV1, policy: LandmarkQualityPolicyV1) -> bool:
    return any(
        _is_low(
            score,
            policy.pose_visibility_diagnostic_ppm
            if kind == "visibility"
            else policy.pose_presence_diagnostic_ppm,
        )
        for kind, score in _pose_scores(anchor)
    )


def _palm_centroid(hand: HandSlotV1) -> tuple[float, float, float]:
    assert hand.image_landmarks is not None
    points = tuple(hand.image_landmarks[index] for index in _PALM_INDICES)
    divisor = len(points)
    return (
        math.fsum(point.x for point in points) / divisor,
        math.fsum(point.y for point in points) / divisor,
        math.fsum(point.z for point in points) / divisor,
    )


def _squared_distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return math.fsum((first - second) ** 2 for first, second in zip(left, right, strict=True))


def _hand_transition_cost(left: HandSlotV1, right: HandSlotV1) -> float:
    assert left.image_landmarks is not None
    assert right.image_landmarks is not None
    left_wrist = left.image_landmarks[0]
    right_wrist = right.image_landmarks[0]
    return _squared_distance(
        (left_wrist.x, left_wrist.y, left_wrist.z),
        (right_wrist.x, right_wrist.y, right_wrist.z),
    ) + _squared_distance(_palm_centroid(left), _palm_centroid(right))


def _has_high_confidence_handedness(hand: HandSlotV1, policy: LandmarkQualityPolicyV1) -> bool:
    return (
        hand.present
        and hand.handedness is not None
        and hand.handedness_confidence is not None
        and not _is_low(
            hand.handedness_confidence,
            policy.handedness_confidence_diagnostic_ppm,
        )
    )


def _is_suspected_swap(
    left: LandmarkFrameV1,
    right: LandmarkFrameV1,
    policy: LandmarkQualityPolicyV1,
) -> bool:
    previous = left.hands
    current = right.hands
    if not all(_has_high_confidence_handedness(hand, policy) for hand in (*previous, *current)):
        return False
    assert all(hand.handedness is not None for hand in (*previous, *current))
    label_cross = (
        previous[0].handedness != previous[1].handedness
        and previous[0].handedness == current[1].handedness
        and previous[1].handedness == current[0].handedness
    )
    if not label_cross:
        return False
    stay = _hand_transition_cost(previous[0], current[0]) + _hand_transition_cost(
        previous[1], current[1]
    )
    cross = _hand_transition_cost(previous[0], current[1]) + _hand_transition_cost(
        previous[1], current[0]
    )
    margin = policy.suspected_swap_cost_margin_ppm / 1_000_000
    return cross + margin < stay


def _suspected_swap_spans(
    rows: tuple[LandmarkFrameV1, ...],
    policy: LandmarkQualityPolicyV1,
) -> tuple[tuple[int, int], ...]:
    two_hand_rows = tuple(row for row in rows if all(hand.present for hand in row.hands))
    return tuple(
        (left.frame_index, right.frame_index)
        for left, right in pairwise(two_hand_rows)
        if _is_suspected_swap(left, right, policy)
    )


def _timestamp_diagnostics(
    rows: tuple[LandmarkFrameV1, ...],
    policy: LandmarkQualityPolicyV1,
) -> tuple[tuple[int, ...], int | None, int | None]:
    deltas = tuple(
        right.relative_timestamp_us - left.relative_timestamp_us for left, right in pairwise(rows)
    )
    if not deltas:
        return (), None, None
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    median_us = (
        ordered[middle]
        if len(ordered) % 2
        else _round_ratio_half_up(ordered[middle - 1] + ordered[middle], 2)
    )
    discontinuities = tuple(
        index
        for index, delta in enumerate(deltas, start=1)
        if delta <= 0
        or (
            delta > policy.timestamp_discontinuity_absolute_us
            and delta * 1_000_000 > median_us * policy.timestamp_discontinuity_median_multiplier_ppm
        )
    )
    return discontinuities, median_us, max(deltas)


def _temporal_discontinuities(
    rows: tuple[LandmarkFrameV1, ...],
    policy: LandmarkQualityPolicyV1,
) -> tuple[int, ...]:
    events: list[int] = []
    for left, right in pairwise(rows):
        delta_us = right.relative_timestamp_us - left.relative_timestamp_us
        if delta_us <= 0:
            continue
        for slot_index in range(len(HAND_SLOT_IDS)):
            previous = left.hands[slot_index]
            current = right.hands[slot_index]
            if not (previous.present and current.present):
                continue
            speed = math.sqrt(_hand_transition_cost(previous, current)) * 1_000_000 / delta_us
            if speed > policy.max_palm_wrist_speed_units_per_second:
                events.append(right.frame_index)
                break
    return tuple(events)


def _signal_observation(frame: LandmarkFrameV1, signal: _SignalName) -> _Observation:
    if signal == "hand_0":
        return frame.hands[0]
    if signal == "hand_1":
        return frame.hands[1]
    anchor_index = BODY_ANCHOR_NAMES.index(signal)
    return frame.body_anchors[anchor_index]


def _is_present(observation: _Observation) -> bool:
    """Return the immutable raw mask without reclassifying confidence as absence."""

    return observation.present


def _gap_duration_us(
    rows: tuple[LandmarkFrameV1, ...],
    start: int,
    end: int,
    left_index: int | None,
    right_index: int | None,
) -> int:
    if left_index is not None and right_index is not None:
        return rows[right_index].relative_timestamp_us - rows[left_index].relative_timestamp_us
    if right_index is not None:
        return rows[right_index].relative_timestamp_us - rows[start].relative_timestamp_us
    if left_index is not None:
        return rows[end].relative_timestamp_us - rows[left_index].relative_timestamp_us
    return rows[end].relative_timestamp_us - rows[start].relative_timestamp_us


def _identity_is_ambiguous(
    signal: _SignalName,
    rows: tuple[LandmarkFrameV1, ...],
    left_index: int | None,
    right_index: int | None,
    policy: LandmarkQualityPolicyV1,
) -> bool:
    if signal not in _HAND_SIGNALS or left_index is None or right_index is None:
        return False
    left = cast(HandSlotV1, _signal_observation(rows[left_index], signal))
    right = cast(HandSlotV1, _signal_observation(rows[right_index], signal))
    if not (
        _has_high_confidence_handedness(left, policy)
        and _has_high_confidence_handedness(right, policy)
    ):
        return False
    return left.handedness != right.handedness


def _boundary_has_low_confidence(
    signal: _SignalName,
    rows: tuple[LandmarkFrameV1, ...],
    left_index: int | None,
    right_index: int | None,
    policy: LandmarkQualityPolicyV1,
) -> bool:
    """Return whether an observed interpolation boundary fails confidence policy."""

    indexes = tuple(index for index in (left_index, right_index) if index is not None)
    if signal in _HAND_SIGNALS:
        return any(
            not _has_high_confidence_handedness(
                cast(HandSlotV1, _signal_observation(rows[index], signal)),
                policy,
            )
            for index in indexes
        )
    return any(
        _pose_is_low_confidence(
            cast(BodyAnchorV1, _signal_observation(rows[index], signal)),
            policy,
        )
        for index in indexes
    )


def _build_gap(
    signal: _SignalName,
    rows: tuple[LandmarkFrameV1, ...],
    start: int,
    end: int,
    policy: LandmarkQualityPolicyV1,
    timestamp_discontinuities: frozenset[int],
    suspected_swap_spans: tuple[tuple[int, int], ...],
) -> MissingIntervalV1:
    left_index = start - 1 if start > 0 else None
    right_index = end + 1 if end + 1 < len(rows) else None
    boundary: Literal["leading", "internal", "trailing", "entire_sequence"]
    if left_index is None and right_index is None:
        boundary = "entire_sequence"
    elif left_index is None:
        boundary = "leading"
    elif right_index is None:
        boundary = "trailing"
    else:
        boundary = "internal"
    duration_us = _gap_duration_us(rows, start, end, left_index, right_index)
    contains_invalid = any(row.invalid for row in rows[start : end + 1])
    transition_start = start if left_index is not None else start + 1
    transition_end = right_index if right_index is not None else end
    crosses_timestamp = any(
        transition_start <= event <= transition_end for event in timestamp_discontinuities
    )
    crosses_swap = signal in _HAND_SIGNALS and any(
        left <= start and end < right for left, right in suspected_swap_spans
    )
    reasons: set[MissingIntervalReason] = set()
    boundary_reason: MissingIntervalReason | None = {
        "leading": "leading_gap",
        "internal": None,
        "trailing": "trailing_gap",
        "entire_sequence": "entire_sequence_missing",
    }[boundary]
    if boundary_reason is not None:
        reasons.add(boundary_reason)
    missing_count = end - start + 1
    if missing_count > policy.max_interpolated_missing_frames:
        reasons.add("too_many_missing_frames")
    if duration_us > policy.max_interpolation_bridge_us:
        reasons.add("bridge_too_long")
    if contains_invalid:
        reasons.add("invalid_frame")
    if crosses_timestamp:
        reasons.add("timestamp_discontinuity")
    identity_ambiguous = crosses_swap or _identity_is_ambiguous(
        signal,
        rows,
        left_index,
        right_index,
        policy,
    )
    if identity_ambiguous:
        reasons.add("identity_ambiguity")
    if _boundary_has_low_confidence(
        signal,
        rows,
        left_index,
        right_index,
        policy,
    ):
        reasons.add("low_confidence")
    eligible = boundary == "internal" and not reasons
    if eligible:
        reasons.add("eligible_short_internal_gap")
    return MissingIntervalV1(
        schema_version="missing-interval/1",
        gap_id=f"quality_gap_{signal}_{start:08d}",
        signal=signal,
        boundary=boundary,
        first_missing_frame_index=start,
        last_missing_frame_index=end,
        first_missing_timestamp_us=rows[start].relative_timestamp_us,
        last_missing_timestamp_us=rows[end].relative_timestamp_us,
        missing_frame_count=missing_count,
        duration_us=duration_us,
        left_observed_frame_index=left_index,
        left_observed_timestamp_us=(
            None if left_index is None else rows[left_index].relative_timestamp_us
        ),
        right_observed_frame_index=right_index,
        right_observed_timestamp_us=(
            None if right_index is None else rows[right_index].relative_timestamp_us
        ),
        contains_invalid_frame=contains_invalid,
        crosses_timestamp_discontinuity=crosses_timestamp,
        crosses_suspected_hand_swap=crosses_swap,
        contains_identity_ambiguity=identity_ambiguous,
        decision="interpolate_linear" if eligible else "preserve_missing",
        reasons=tuple(sorted(reasons)),
    )


def _missing_intervals(
    rows: tuple[LandmarkFrameV1, ...],
    policy: LandmarkQualityPolicyV1,
    timestamp_discontinuities: tuple[int, ...],
    suspected_swap_spans: tuple[tuple[int, int], ...],
) -> tuple[MissingIntervalV1, ...]:
    gaps: list[MissingIntervalV1] = []
    signals = cast(tuple[_SignalName, ...], (*HAND_SLOT_IDS, *BODY_ANCHOR_NAMES))
    for signal in signals:
        usable = tuple(_is_present(_signal_observation(row, signal)) for row in rows)
        index = 0
        while index < len(rows):
            if usable[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(rows) and not usable[index + 1]:
                index += 1
            gaps.append(
                _build_gap(
                    signal,
                    rows,
                    start,
                    index,
                    policy,
                    frozenset(timestamp_discontinuities),
                    suspected_swap_spans,
                )
            )
            index += 1
    return tuple(gaps)


def _metric_value(
    metric: QualityMetricName,
    metrics: SequenceQualityMetricsV1,
) -> int | None:
    return cast(int | None, getattr(metrics, metric))


def _highest_violation(
    rule: QualityThresholdRuleV1,
    observed: int,
) -> tuple[QualityFindingSeverity, int] | None:
    result: tuple[QualityFindingSeverity, int] | None = None
    for severity, threshold in (
        ("warning", rule.warning),
        ("quarantine", rule.quarantine),
        ("reject", rule.reject),
    ):
        if threshold is None:
            continue
        violated = (
            observed > threshold if rule.direction == "higher_is_worse" else observed < threshold
        )
        if violated:
            result = (cast(QualityFindingSeverity, severity), threshold)
    return result


def _quality_findings(
    metrics: SequenceQualityMetricsV1,
    policy: LandmarkQualityPolicyV1,
) -> tuple[QualityFindingV1, ...]:
    findings: list[QualityFindingV1] = []
    for rule in policy.threshold_rules:
        observed = _metric_value(rule.metric, metrics)
        if observed is None:
            continue
        violation = _highest_violation(rule, observed)
        if violation is None:
            continue
        severity, threshold = violation
        findings.append(
            QualityFindingV1(
                schema_version="quality-finding/1",
                rule_id=rule.rule_id,
                metric=rule.metric,
                direction=QUALITY_METRIC_DIRECTIONS[rule.metric],
                severity=severity,
                observed_value=observed,
                threshold=threshold,
            )
        )
    return tuple(sorted(findings, key=lambda finding: finding.metric))


def _expected_hand_count(recording: RecordingRowV1) -> Literal[1, 2]:
    return 2 if recording.handedness == "both" else 1


def _validate_source_binding(
    reference: LandmarkSequenceRefV1,
    table: LandmarkFramesTableV1,
    recording: RecordingRowV1,
) -> None:
    try:
        assert_landmark_sequence_ref_matches_table(reference, table)
    except ValueError as error:
        raise QualityPolicyError("sequence reference does not match landmark rows") from error
    expected = (
        recording.recording_id,
        recording.participant_id,
        recording.session_id,
        recording.handedness,
        recording.mirror_state,
        recording.rotation_degrees,
        recording.media.sha256,
        recording.media.size_bytes,
    )
    actual = (
        reference.lineage.source_recording_id,
        reference.lineage.participant_id,
        reference.lineage.session_id,
        reference.lineage.handedness,
        reference.source_mirror_state,
        reference.source_rotation_degrees,
        reference.source_media_sha256,
        reference.source_media_size_bytes,
    )
    if actual != expected:
        raise QualityPolicyError("sequence reference does not match raw recording facts")


def assess_landmark_sequence(
    reference: LandmarkSequenceRefV1,
    table: LandmarkFramesTableV1,
    recording: RecordingRowV1,
    policy: LandmarkQualityPolicyV1,
) -> SequenceQualityReportV1:
    """Assess one exact raw sequence without mutating observations or masks."""

    _validate_source_binding(reference, table, recording)
    return assess_landmark_source(
        table,
        policy,
        source_recording_id=recording.recording_id,
        source_sequence_content_sha256=reference.content_sha256,
        source_landmark_parquet_sha256=reference.lineage.artifact.sha256,
        declared_duration_us=recording.duration_us,
        expected_hand_count=_expected_hand_count(recording),
    )


def assess_landmark_source(
    table: LandmarkFramesTableV1,
    policy: LandmarkQualityPolicyV1,
    *,
    source_recording_id: str,
    source_sequence_content_sha256: str,
    source_landmark_parquet_sha256: str,
    declared_duration_us: int,
    expected_hand_count: Literal[1, 2],
) -> SequenceQualityReportV1:
    """Assess verified landmarks without requiring participant-consent records."""

    if table.rows[0].source_recording_id != source_recording_id:
        raise QualityPolicyError("landmark table and recording identities do not match")
    if landmark_frames_table_digest(table) != source_sequence_content_sha256:
        raise QualityPolicyError("landmark table content identity does not match")
    if expected_hand_count not in (1, 2):
        raise QualityPolicyError("expected hand count must be one or two")
    rows = table.rows
    timestamp_events, median_delta, maximum_delta = _timestamp_diagnostics(rows, policy)
    swap_spans = _suspected_swap_spans(rows, policy)
    temporal_events = _temporal_discontinuities(rows, policy)
    gaps = _missing_intervals(rows, policy, timestamp_events, swap_spans)

    invalid_source = sum(row.invalid_reason == "source_frame_invalid" for row in rows)
    invalid_task = sum(row.invalid_reason == "task_inference_failed" for row in rows)
    valid_count = len(rows) - invalid_source - invalid_task
    expected_hands = expected_hand_count
    expected_observations = sum(min(row.observed_hand_count, expected_hands) for row in rows)
    expected_opportunities = len(rows) * expected_hands
    hand_confidences = tuple(
        hand.handedness_confidence
        for row in rows
        for hand in row.hands
        if hand.present and hand.handedness_confidence is not None
    )
    low_hand_confidences = sum(
        _is_low(score, policy.handedness_confidence_diagnostic_ppm) for score in hand_confidences
    )
    pose_presence_counts = tuple(
        sum(row.body_anchors[index].present for row in rows)
        for index in range(len(BODY_ANCHOR_NAMES))
    )
    pose_scores = tuple(
        score for row in rows for anchor in row.body_anchors for kind, score in _pose_scores(anchor)
    )
    low_pose_scores = sum(
        _is_low(
            score,
            policy.pose_visibility_diagnostic_ppm
            if kind == "visibility"
            else policy.pose_presence_diagnostic_ppm,
        )
        for row in rows
        for anchor in row.body_anchors
        for kind, score in _pose_scores(anchor)
    )
    pose_coverage = cast(
        tuple[int, int, int, int, int, int],
        tuple(ratio_ppm(count, len(rows)) for count in pose_presence_counts),
    )
    longest_unfilled_hand_gap = max(
        (
            gap.duration_us
            for gap in gaps
            if gap.signal in _HAND_SIGNALS
            and gap.boundary == "internal"
            and gap.decision == "preserve_missing"
        ),
        default=0,
    )
    metrics = SequenceQualityMetricsV1(
        frame_count=len(rows),
        valid_frame_count=valid_count,
        source_invalid_frame_count=invalid_source,
        task_inference_failed_frame_count=invalid_task,
        invalid_frame_fraction_ppm=ratio_ppm(invalid_source + invalid_task, len(rows)),
        expected_hand_count=expected_hands,
        expected_hand_observation_count=expected_observations,
        expected_hand_opportunity_count=expected_opportunities,
        expected_hand_coverage_ppm=ratio_ppm(expected_observations, expected_opportunities),
        handedness_confidence_observation_count=len(hand_confidences),
        low_handedness_confidence_observation_count=low_hand_confidences,
        low_handedness_confidence_fraction_ppm=(
            None if not hand_confidences else ratio_ppm(low_hand_confidences, len(hand_confidences))
        ),
        pose_anchor_presence_counts=cast(
            tuple[int, int, int, int, int, int],
            pose_presence_counts,
        ),
        pose_anchor_coverage_ppm=pose_coverage,
        minimum_pose_anchor_coverage_ppm=min(pose_coverage),
        pose_confidence_observation_count=len(pose_scores),
        low_pose_confidence_observation_count=low_pose_scores,
        interpolated_gap_count=sum(gap.decision == "interpolate_linear" for gap in gaps),
        preserved_gap_count=sum(gap.decision == "preserve_missing" for gap in gaps),
        longest_unfilled_internal_hand_gap_us=longest_unfilled_hand_gap,
        timestamp_delta_count=len(rows) - 1,
        median_timestamp_delta_us=median_delta,
        maximum_timestamp_delta_us=maximum_delta,
        timestamp_discontinuity_count=len(timestamp_events),
        temporal_discontinuity_count=len(temporal_events),
        suspected_hand_swap_count=len(swap_spans),
    )
    findings = _quality_findings(metrics, policy)
    if (valid_count == 0 or expected_observations == 0) and not any(
        finding.severity == "reject" for finding in findings
    ):
        raise QualityPolicyError(
            "quality policy must reject zero-valid and fully undetected sequences"
        )
    disposition = cast(
        Literal["pass", "warning", "quarantine", "reject"],
        max(
            (finding.severity for finding in findings),
            key={"warning": 1, "quarantine": 2, "reject": 3}.__getitem__,
            default="pass",
        ),
    )
    report_payload: dict[str, object] = {
        "schema_version": "sequence-quality-report/1",
        "source_recording_id": source_recording_id,
        "source_sequence_content_sha256": source_sequence_content_sha256,
        "source_landmark_parquet_sha256": source_landmark_parquet_sha256,
        "policy_sha256": landmark_quality_policy_digest(policy),
        "metrics": metrics.model_dump(mode="json", round_trip=True),
        "gaps": [gap.model_dump(mode="json", round_trip=True) for gap in gaps],
        "resampling": _build_elapsed_time_resampling_plan(
            table,
            source_recording_id=source_recording_id,
            declared_duration_us=declared_duration_us,
            policy=policy,
        ).model_dump(mode="json", round_trip=True),
        "findings": [finding.model_dump(mode="json", round_trip=True) for finding in findings],
        "disposition": disposition,
    }
    report_payload["report_sha256"] = sequence_quality_report_digest(report_payload)
    return SequenceQualityReportV1.model_validate_json(
        canonical_json_bytes(report_payload),
        strict=True,
    )


def aggregate_quality_reports(
    reports: tuple[SequenceQualityReportV1, ...],
    policy: LandmarkQualityPolicyV1,
) -> DatasetQualityReportV1:
    """Build one weighted dataset summary over canonical ordered reports."""

    if not reports:
        raise QualityPolicyError("dataset quality aggregation requires at least one report")
    recording_ids = tuple(report.source_recording_id for report in reports)
    if recording_ids != tuple(sorted(set(recording_ids))):
        raise QualityPolicyError("quality reports must have unique recording IDs in order")
    policy_sha256 = landmark_quality_policy_digest(policy)
    if any(report.policy_sha256 != policy_sha256 for report in reports):
        raise QualityPolicyError("quality reports do not share the requested policy")
    dispositions = tuple(report.disposition for report in reports)
    total_frames = sum(report.metrics.frame_count for report in reports)
    total_invalid = sum(
        report.metrics.source_invalid_frame_count + report.metrics.task_inference_failed_frame_count
        for report in reports
    )
    hand_observations = sum(report.metrics.expected_hand_observation_count for report in reports)
    hand_opportunities = sum(report.metrics.expected_hand_opportunity_count for report in reports)
    status: Literal["ready", "ready_with_warnings", "blocked"] = (
        "blocked"
        if dispositions.count("quarantine") or dispositions.count("reject")
        else "ready_with_warnings"
        if dispositions.count("warning")
        else "ready"
    )
    return DatasetQualityReportV1(
        schema_version="dataset-quality-report/1",
        sequence_count=len(reports),
        pass_count=dispositions.count("pass"),
        warning_count=dispositions.count("warning"),
        quarantine_count=dispositions.count("quarantine"),
        reject_count=dispositions.count("reject"),
        total_frame_count=total_frames,
        total_invalid_frame_count=total_invalid,
        invalid_frame_fraction_ppm=ratio_ppm(total_invalid, total_frames),
        total_expected_hand_observation_count=hand_observations,
        total_expected_hand_opportunity_count=hand_opportunities,
        expected_hand_coverage_ppm=ratio_ppm(hand_observations, hand_opportunities),
        minimum_pose_anchor_coverage_ppm=min(
            report.metrics.minimum_pose_anchor_coverage_ppm for report in reports
        ),
        longest_unfilled_internal_hand_gap_us=max(
            report.metrics.longest_unfilled_internal_hand_gap_us for report in reports
        ),
        timestamp_discontinuity_count=sum(
            report.metrics.timestamp_discontinuity_count for report in reports
        ),
        temporal_discontinuity_count=sum(
            report.metrics.temporal_discontinuity_count for report in reports
        ),
        suspected_hand_swap_count=sum(
            report.metrics.suspected_hand_swap_count for report in reports
        ),
        status=status,
    )


__all__ = [
    "QualityPolicyError",
    "aggregate_quality_reports",
    "assess_landmark_sequence",
    "assess_landmark_source",
    "build_elapsed_time_resampling_plan",
    "elapsed_resampling_timestamps",
    "interpolate_coordinate",
    "interpolate_point_coordinates",
]
