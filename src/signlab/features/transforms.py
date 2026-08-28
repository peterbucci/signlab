"""Pure deterministic transforms from raw landmarks to portable feature tensors."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal, cast

from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.dataset import MirrorState
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    BodyAnchorV1,
    LandmarkFramesTableV1,
    LandmarkSequenceRefV1,
    Point3V1,
    assert_landmark_sequence_ref_matches_table,
    landmark_frames_table_digest,
)
from signlab.contracts.features import (
    FEATURE_HAND_SLOTS,
    FEATURE_QUANTIZATION_SCALE,
    FeatureStatisticsV1,
    LandmarkFeaturePlanV1,
    PortableFeatureSequenceV1,
    landmark_feature_plan_digest,
    portable_feature_sequence_digest,
)
from signlab.contracts.quality import (
    MissingIntervalV1,
    SequenceQualityReportV1,
    assert_sequence_quality_report_matches_table,
)
from signlab.quality.policy import elapsed_resampling_timestamps, interpolate_coordinate

type _Point = tuple[float, float, float]
type _Evidence = Literal["observed", "interpolated"]

_ANGLE_TRIPLETS = ((1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
_TIP_INDEXES = (4, 8, 12, 16, 20)


class FeatureTransformError(ValueError):
    """Raised when bound source evidence cannot satisfy a feature plan."""


@dataclass(frozen=True, slots=True)
class _SampledHand:
    world: tuple[_Point, ...]
    image: tuple[_Point, ...]
    handedness: Literal["left", "right"]
    evidence: _Evidence


@dataclass(frozen=True, slots=True)
class _SampledAnchor:
    image: _Point
    evidence: _Evidence


@dataclass(frozen=True, slots=True)
class _FeatureRow:
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    observed: tuple[bool, ...]
    interpolated: tuple[bool, ...]
    hand_present: tuple[bool, bool]
    body_available: bool


def _point_tuple(point: Point3V1) -> _Point:
    return (point.x, point.y, point.z)


def _interpolate_value(
    left: float,
    right: float,
    *,
    left_us: int,
    right_us: int,
    target_us: int,
) -> float:
    return interpolate_coordinate(
        left,
        right,
        left_timestamp_us=left_us,
        right_timestamp_us=right_us,
        target_timestamp_us=target_us,
    )


def _interpolate_points(
    left: tuple[Point3V1, ...],
    right: tuple[Point3V1, ...],
    *,
    left_us: int,
    right_us: int,
    target_us: int,
) -> tuple[_Point, ...]:
    return tuple(
        cast(
            _Point,
            tuple(
                _interpolate_value(
                    first,
                    second,
                    left_us=left_us,
                    right_us=right_us,
                    target_us=target_us,
                )
                for first, second in zip(
                    _point_tuple(left_point), _point_tuple(right_point), strict=True
                )
            ),
        )
        for left_point, right_point in zip(left, right, strict=True)
    )


def _approved_gap(
    gaps: tuple[MissingIntervalV1, ...],
    *,
    target_us: int,
) -> MissingIntervalV1 | None:
    for gap in gaps:
        if (
            gap.decision == "interpolate_linear"
            and gap.left_observed_timestamp_us is not None
            and gap.right_observed_timestamp_us is not None
            and gap.left_observed_timestamp_us <= target_us <= gap.right_observed_timestamp_us
        ):
            return gap
    return None


def _sample_hand(
    table: LandmarkFramesTableV1,
    *,
    slot_index: int,
    target_us: int,
    timestamps: tuple[int, ...],
    gaps: tuple[MissingIntervalV1, ...],
) -> _SampledHand | None:
    position = bisect_left(timestamps, target_us)
    if position < len(timestamps) and timestamps[position] == target_us:
        exact = table.rows[position].hands[slot_index]
        if exact.present:
            assert exact.world_landmarks is not None
            assert exact.image_landmarks is not None
            assert exact.handedness is not None
            return _SampledHand(
                world=tuple(_point_tuple(point) for point in exact.world_landmarks),
                image=tuple(_point_tuple(point) for point in exact.image_landmarks),
                handedness=exact.handedness,
                evidence="observed",
            )
    gap = _approved_gap(gaps, target_us=target_us)
    if gap is not None:
        assert gap.left_observed_frame_index is not None
        assert gap.right_observed_frame_index is not None
        left_index = gap.left_observed_frame_index
        right_index = gap.right_observed_frame_index
    else:
        left_index = position - 1
        right_index = position
    if left_index < 0 or right_index >= len(table.rows):
        return None
    left = table.rows[left_index].hands[slot_index]
    right = table.rows[right_index].hands[slot_index]
    if not left.present or not right.present or left.handedness != right.handedness:
        return None
    assert left.world_landmarks is not None
    assert right.world_landmarks is not None
    assert left.image_landmarks is not None
    assert right.image_landmarks is not None
    assert left.handedness is not None
    left_us = timestamps[left_index]
    right_us = timestamps[right_index]
    if not left_us <= target_us <= right_us:
        return None
    return _SampledHand(
        world=_interpolate_points(
            left.world_landmarks,
            right.world_landmarks,
            left_us=left_us,
            right_us=right_us,
            target_us=target_us,
        ),
        image=_interpolate_points(
            left.image_landmarks,
            right.image_landmarks,
            left_us=left_us,
            right_us=right_us,
            target_us=target_us,
        ),
        handedness=left.handedness,
        evidence="interpolated",
    )


def _anchor_point(anchor: BodyAnchorV1) -> Point3V1 | None:
    return anchor.image_point if anchor.present else None


def _sample_anchor(
    table: LandmarkFramesTableV1,
    *,
    anchor_index: int,
    target_us: int,
    timestamps: tuple[int, ...],
    gaps: tuple[MissingIntervalV1, ...],
) -> _SampledAnchor | None:
    position = bisect_left(timestamps, target_us)
    if position < len(timestamps) and timestamps[position] == target_us:
        point = _anchor_point(table.rows[position].body_anchors[anchor_index])
        if point is not None:
            return _SampledAnchor(image=_point_tuple(point), evidence="observed")
    gap = _approved_gap(gaps, target_us=target_us)
    if gap is not None:
        assert gap.left_observed_frame_index is not None
        assert gap.right_observed_frame_index is not None
        left_index = gap.left_observed_frame_index
        right_index = gap.right_observed_frame_index
    else:
        left_index = position - 1
        right_index = position
    if left_index < 0 or right_index >= len(table.rows):
        return None
    left = _anchor_point(table.rows[left_index].body_anchors[anchor_index])
    right = _anchor_point(table.rows[right_index].body_anchors[anchor_index])
    if left is None or right is None:
        return None
    left_us = timestamps[left_index]
    right_us = timestamps[right_index]
    if not left_us <= target_us <= right_us:
        return None
    return _SampledAnchor(
        image=cast(
            _Point,
            tuple(
                _interpolate_value(
                    first,
                    second,
                    left_us=left_us,
                    right_us=right_us,
                    target_us=target_us,
                )
                for first, second in zip(_point_tuple(left), _point_tuple(right), strict=True)
            ),
        ),
        evidence="interpolated",
    )


def _unmirror_world(point: _Point, mirrored: bool) -> _Point:
    return (-point[0], point[1], point[2]) if mirrored else point


def _unmirror_image(point: _Point, mirrored: bool) -> _Point:
    return (1.0 - point[0], point[1], point[2]) if mirrored else point


def _distance(left: _Point, right: _Point) -> float:
    return math.sqrt(sum((first - second) ** 2 for first, second in zip(left, right, strict=True)))


def _local_points(hand: _SampledHand, *, mirrored: bool) -> tuple[_Point, ...] | None:
    points = tuple(_unmirror_world(point, mirrored) for point in hand.world)
    wrist = points[0]
    scale = _distance(wrist, points[9])
    if scale <= 0.0 or not math.isfinite(scale):
        return None
    corrected_handedness: Literal["left", "right"] = (
        hand.handedness if mirrored else ("right" if hand.handedness == "left" else "left")
    )
    handedness_sign = -1.0 if corrected_handedness == "left" else 1.0
    return tuple(
        (
            handedness_sign * (point[0] - wrist[0]) / scale,
            (point[1] - wrist[1]) / scale,
            (point[2] - wrist[2]) / scale,
        )
        for point in points
    )


def _palm(points: tuple[_Point, ...]) -> _Point:
    indexes = (0, 5, 9, 17)
    return cast(
        _Point,
        tuple(sum(points[index][axis] for index in indexes) / len(indexes) for axis in range(3)),
    )


def _angle(first: _Point, middle: _Point, last: _Point) -> float | None:
    left = tuple(first[axis] - middle[axis] for axis in range(3))
    right = tuple(last[axis] - middle[axis] for axis in range(3))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _append(
    values: list[float],
    valid: list[bool],
    observed: list[bool],
    interpolated: list[bool],
    item: float | None,
    evidence: _Evidence | None,
) -> None:
    usable = item is not None and math.isfinite(item)
    values.append(0.0 if not usable else cast(float, item))
    valid.append(usable)
    observed.append(usable and evidence == "observed")
    interpolated.append(usable and evidence == "interpolated")


def _row_features(
    plan: LandmarkFeaturePlanV1,
    hands: tuple[_SampledHand | None, _SampledHand | None],
    shoulders: tuple[_SampledAnchor | None, _SampledAnchor | None],
    *,
    mirrored: bool,
) -> _FeatureRow:
    values: list[float] = []
    valid: list[bool] = []
    observed: list[bool] = []
    interpolated: list[bool] = []
    local = tuple(
        None if hand is None else _local_points(hand, mirrored=mirrored) for hand in hands
    )
    if plan.representation in {"hand_local", "combined"}:
        for hand, points in zip(hands, local, strict=True):
            evidence = None if hand is None else hand.evidence
            for landmark_index in range(21):
                point = None if points is None else points[landmark_index]
                for axis in range(3):
                    _append(
                        values,
                        valid,
                        observed,
                        interpolated,
                        None if point is None else point[axis],
                        evidence,
                    )
    left_shoulder, right_shoulder = shoulders
    body_center: tuple[float, float] | None = None
    body_scale: float | None = None
    body_evidence: _Evidence | None = None
    if left_shoulder is not None and right_shoulder is not None:
        left = _unmirror_image(left_shoulder.image, mirrored)
        right = _unmirror_image(right_shoulder.image, mirrored)
        scale = math.hypot(right[0] - left[0], right[1] - left[1])
        if scale > 0.0 and math.isfinite(scale):
            body_center = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
            body_scale = scale
            body_evidence = (
                "observed"
                if left_shoulder.evidence == right_shoulder.evidence == "observed"
                else "interpolated"
            )
    if plan.representation in {"body_relative", "combined"}:
        for hand in hands:
            trajectory_points: tuple[_Point, _Point] | None = None
            trajectory_evidence: _Evidence | None = None
            if hand is not None and body_center is not None and body_scale is not None:
                image = tuple(_unmirror_image(point, mirrored) for point in hand.image)
                trajectory_points = (image[0], _palm(image))
                trajectory_evidence = (
                    "observed" if hand.evidence == body_evidence == "observed" else "interpolated"
                )
            for point_index in range(2):
                point = None if trajectory_points is None else trajectory_points[point_index]
                for axis in range(2):
                    item = (
                        None
                        if point is None or body_center is None or body_scale is None
                        else (point[axis] - body_center[axis]) / body_scale
                    )
                    _append(
                        values,
                        valid,
                        observed,
                        interpolated,
                        item,
                        trajectory_evidence,
                    )
    if plan.optional.include_joint_angles:
        for hand, points in zip(hands, local, strict=True):
            evidence = None if hand is None else hand.evidence
            for first, middle, last in _ANGLE_TRIPLETS:
                item = (
                    None if points is None else _angle(points[first], points[middle], points[last])
                )
                _append(values, valid, observed, interpolated, item, evidence)
    if plan.optional.include_tip_distances:
        for hand, points in zip(hands, local, strict=True):
            evidence = None if hand is None else hand.evidence
            for tip in _TIP_INDEXES:
                item = None if points is None else _distance(points[0], points[tip])
                _append(values, valid, observed, interpolated, item, evidence)
    return _FeatureRow(
        values=tuple(values),
        valid=tuple(valid),
        observed=tuple(observed),
        interpolated=tuple(interpolated),
        hand_present=cast(tuple[bool, bool], tuple(hand is not None for hand in hands)),
        body_available=body_center is not None,
    )


def _with_derivatives(
    rows: tuple[_FeatureRow, ...],
    timestamps_us: tuple[int, ...],
    plan: LandmarkFeaturePlanV1,
    gaps_by_signal: dict[str, tuple[MissingIntervalV1, ...]],
) -> tuple[_FeatureRow, ...]:
    position_count = 0
    position_signals: list[tuple[str, ...]] = []
    if plan.representation in {"hand_local", "combined"}:
        position_count += 2 * 21 * 3
        for slot in FEATURE_HAND_SLOTS:
            position_signals.extend(((slot,),) * (21 * 3))
    if plan.representation in {"body_relative", "combined"}:
        position_count += 2 * 2 * 2
        for slot in FEATURE_HAND_SLOTS:
            position_signals.extend(((slot, "left_shoulder", "right_shoulder"),) * (2 * 2))
    if not plan.optional.include_velocity:
        return rows
    if len(position_signals) != position_count:  # pragma: no cover - registered-plan invariant
        raise FeatureTransformError("derivative signal dependencies do not match positions")

    preserved_gap_barriers = {
        signal: tuple(
            False
            if index == 0
            else any(
                gap.decision == "preserve_missing"
                and gap.first_missing_timestamp_us <= timestamps_us[index]
                and gap.last_missing_timestamp_us >= timestamps_us[index - 1]
                for gap in gaps
            )
            for index in range(len(timestamps_us))
        )
        for signal, gaps in gaps_by_signal.items()
    }
    velocity_values: list[list[float]] = []
    velocity_valid: list[list[bool]] = []
    for index, row in enumerate(rows):
        values = [0.0] * position_count
        valid = [False] * position_count
        if index > 0:
            elapsed_seconds = (timestamps_us[index] - timestamps_us[index - 1]) / 1_000_000
            for feature_index in range(position_count):
                crosses_preserved_gap = any(
                    preserved_gap_barriers[signal][index]
                    for signal in position_signals[feature_index]
                )
                if (
                    not crosses_preserved_gap
                    and row.valid[feature_index]
                    and rows[index - 1].valid[feature_index]
                ):
                    values[feature_index] = (
                        row.values[feature_index] - rows[index - 1].values[feature_index]
                    ) / elapsed_seconds
                    valid[feature_index] = math.isfinite(values[feature_index])
                    if not valid[feature_index]:
                        values[feature_index] = 0.0
        velocity_values.append(values)
        velocity_valid.append(valid)
    result: list[_FeatureRow] = []
    for index, row in enumerate(rows):
        derivative_values = list(velocity_values[index])
        derivative_valid = list(velocity_valid[index])
        if plan.optional.include_acceleration:
            acceleration = [0.0] * position_count
            acceleration_valid = [False] * position_count
            if index > 1:
                elapsed_seconds = (timestamps_us[index] - timestamps_us[index - 1]) / 1_000_000
                for feature_index in range(position_count):
                    if (
                        velocity_valid[index][feature_index]
                        and velocity_valid[index - 1][feature_index]
                    ):
                        acceleration[feature_index] = (
                            velocity_values[index][feature_index]
                            - velocity_values[index - 1][feature_index]
                        ) / elapsed_seconds
                        acceleration_valid[feature_index] = math.isfinite(
                            acceleration[feature_index]
                        )
                        if not acceleration_valid[feature_index]:
                            acceleration[feature_index] = 0.0
            derivative_values.extend(acceleration)
            derivative_valid.extend(acceleration_valid)
        result.append(
            _FeatureRow(
                values=(*row.values, *derivative_values),
                valid=(*row.valid, *derivative_valid),
                observed=(*row.observed, *((False,) * len(derivative_values))),
                interpolated=(*row.interpolated, *((False,) * len(derivative_values))),
                hand_present=row.hand_present,
                body_available=row.body_available,
            )
        )
    return tuple(result)


def _round_half_away_from_zero(value: float) -> int:
    scaled = abs(value) * FEATURE_QUANTIZATION_SCALE
    rounded = math.floor(scaled + 0.5)
    return rounded if value >= 0.0 else -rounded


def _selection_indices(source_count: int, target_count: int) -> tuple[int, ...]:
    if source_count <= target_count:
        return tuple(range(source_count))
    denominator = target_count - 1
    return tuple(
        (2 * index * (source_count - 1) + denominator) // (2 * denominator)
        for index in range(target_count)
    )


def _padded_timestamps(
    selected: tuple[int, ...],
    source_timestamps: tuple[int, ...],
    target_count: int,
    plan: LandmarkFeaturePlanV1,
) -> tuple[int, ...]:
    timestamps = [source_timestamps[index] for index in selected]
    missing = target_count - len(timestamps)
    if missing:
        last = timestamps[-1]
        numerator = 1_000_000 * plan.temporal.target_rate_denominator
        denominator = plan.temporal.target_rate_numerator
        timestamps.extend(
            last + (2 * offset * numerator + denominator) // (2 * denominator)
            for offset in range(1, missing + 1)
        )
    return tuple(timestamps)


def _validate_source_bindings(
    table: LandmarkFramesTableV1,
    quality: SequenceQualityReportV1,
    plan: LandmarkFeaturePlanV1,
    *,
    source_recording_id: str,
    source_landmarks_sha256: str,
    source_landmark_parquet_sha256: str,
) -> tuple[int, ...]:
    try:
        assert_sequence_quality_report_matches_table(quality, table)
    except ValueError as error:
        raise FeatureTransformError("feature inputs do not share one source sequence") from error
    if (
        table.rows[0].source_recording_id != source_recording_id
        or quality.source_recording_id != source_recording_id
        or landmark_frames_table_digest(table) != source_landmarks_sha256
        or quality.source_sequence_content_sha256 != source_landmarks_sha256
        or quality.source_landmark_parquet_sha256 != source_landmark_parquet_sha256
    ):
        raise FeatureTransformError("feature quality evidence does not match extraction lineage")
    if quality.metrics.timestamp_discontinuity_count:
        raise FeatureTransformError(
            "feature resampling cannot locate reported timestamp discontinuities"
        )
    expected_rate = (
        plan.temporal.target_rate_numerator,
        plan.temporal.target_rate_denominator,
    )
    report_rate = (
        quality.resampling.target_rate_numerator,
        quality.resampling.target_rate_denominator,
    )
    if report_rate != expected_rate:
        raise FeatureTransformError("feature and quality resampling rates differ")
    grid = elapsed_resampling_timestamps(
        quality.resampling.observed_span_us,
        rate_numerator=plan.temporal.target_rate_numerator,
        rate_denominator=plan.temporal.target_rate_denominator,
    )
    if len(grid) != quality.resampling.target_count:
        raise FeatureTransformError("quality resampling commitment is inconsistent")
    return grid


def derive_feature_sequence(
    table: LandmarkFramesTableV1,
    sequence: LandmarkSequenceRefV1,
    quality: SequenceQualityReportV1,
    plan: LandmarkFeaturePlanV1,
    *,
    extraction_config_sha256: str,
    statistics: FeatureStatisticsV1 | None = None,
) -> PortableFeatureSequenceV1:
    """Derive one fixed-shape feature sequence without mutating raw extraction."""

    if not isinstance(sequence, LandmarkSequenceRefV1):
        raise FeatureTransformError("feature source must be a validated contract")
    try:
        sequence = LandmarkSequenceRefV1.model_validate(sequence, strict=True)
        assert_landmark_sequence_ref_matches_table(sequence, table)
    except ValueError as error:
        raise FeatureTransformError("feature source does not match landmark rows") from error
    return derive_feature_source(
        table,
        quality,
        plan,
        source_recording_id=sequence.lineage.source_recording_id,
        source_media_sha256=sequence.source_media_sha256,
        source_landmarks_sha256=sequence.content_sha256,
        source_landmark_parquet_sha256=sequence.lineage.artifact.sha256,
        source_mirror_state=sequence.source_mirror_state,
        extraction_config_sha256=extraction_config_sha256,
        statistics=statistics,
    )


def derive_feature_source(
    table: LandmarkFramesTableV1,
    quality: SequenceQualityReportV1,
    plan: LandmarkFeaturePlanV1,
    *,
    source_recording_id: str,
    source_media_sha256: str,
    source_landmarks_sha256: str,
    source_landmark_parquet_sha256: str,
    source_mirror_state: MirrorState,
    extraction_config_sha256: str,
    statistics: FeatureStatisticsV1 | None = None,
) -> PortableFeatureSequenceV1:
    """Derive features from verified source facts without participant records."""

    if not isinstance(table, LandmarkFramesTableV1):
        raise FeatureTransformError("feature source must be a validated contract")
    if not isinstance(quality, SequenceQualityReportV1) or not isinstance(
        plan, LandmarkFeaturePlanV1
    ):
        raise FeatureTransformError("feature policy inputs must be validated contracts")
    try:
        table = LandmarkFramesTableV1.model_validate(table, strict=True)
        quality = SequenceQualityReportV1.model_validate(quality, strict=True)
        plan = LandmarkFeaturePlanV1.model_validate(plan, strict=True)
        if statistics is not None:
            if not isinstance(statistics, FeatureStatisticsV1):
                raise ValueError
            statistics = FeatureStatisticsV1.model_validate(statistics, strict=True)
    except ValueError:
        raise FeatureTransformError("feature inputs must be valid immutable contracts") from None

    grid = _validate_source_bindings(
        table,
        quality,
        plan,
        source_recording_id=source_recording_id,
        source_landmarks_sha256=source_landmarks_sha256,
        source_landmark_parquet_sha256=source_landmark_parquet_sha256,
    )
    source_timestamps = tuple(frame.relative_timestamp_us for frame in table.rows)
    gaps_by_signal = {
        signal: tuple(gap for gap in quality.gaps if gap.signal == signal)
        for signal in (*FEATURE_HAND_SLOTS, "left_shoulder", "right_shoulder")
    }
    mirrored = source_mirror_state == "mirrored"
    raw_rows: list[_FeatureRow] = []
    for target_us in grid:
        hands = cast(
            tuple[_SampledHand | None, _SampledHand | None],
            tuple(
                _sample_hand(
                    table,
                    slot_index=slot_index,
                    target_us=target_us,
                    timestamps=source_timestamps,
                    gaps=gaps_by_signal[slot_id],
                )
                for slot_index, slot_id in enumerate(FEATURE_HAND_SLOTS)
            ),
        )
        shoulders = cast(
            tuple[_SampledAnchor | None, _SampledAnchor | None],
            tuple(
                _sample_anchor(
                    table,
                    anchor_index=BODY_ANCHOR_NAMES.index(anchor_name),
                    target_us=target_us,
                    timestamps=source_timestamps,
                    gaps=gaps_by_signal[anchor_name],
                )
                for anchor_name in ("left_shoulder", "right_shoulder")
            ),
        )
        raw_rows.append(_row_features(plan, hands, shoulders, mirrored=mirrored))
    rows = _with_derivatives(tuple(raw_rows), grid, plan, gaps_by_signal)
    if any(len(row.values) != len(plan.feature_order) for row in rows):
        raise FeatureTransformError("derived feature width does not match the plan")
    target_count = plan.padding.target_frame_count
    selected = _selection_indices(len(rows), target_count)
    selected_rows = tuple(rows[index] for index in selected)
    padding_count = target_count - len(selected_rows)
    width = len(plan.feature_order)
    neutral = _FeatureRow(
        values=(0.0,) * width,
        valid=(False,) * width,
        observed=(False,) * width,
        interpolated=(False,) * width,
        hand_present=(False, False),
        body_available=False,
    )
    output_rows = (*selected_rows, *((neutral,) * padding_count))
    payload: dict[str, object] = {
        "schema_version": "portable-feature-sequence/1",
        "source_recording_id": source_recording_id,
        "source_media_sha256": source_media_sha256,
        "source_landmarks_sha256": source_landmarks_sha256,
        "extraction_config_sha256": extraction_config_sha256,
        "quality_policy_sha256": quality.policy_sha256,
        "quality_report_sha256": quality.report_sha256,
        "feature_plan_sha256": landmark_feature_plan_digest(plan),
        "statistics_sha256": None,
        "feature_names": list(plan.feature_order),
        "quantization_scale": FEATURE_QUANTIZATION_SCALE,
        "source_grid_frame_count": len(grid),
        "selected_source_indices": list(selected),
        "timestamps_us": list(_padded_timestamps(selected, grid, target_count, plan)),
        "values_q": [
            [
                _round_half_away_from_zero(value) if valid else 0
                for value, valid in zip(row.values, row.valid, strict=True)
            ]
            for row in output_rows
        ],
        "valid_mask": [list(row.valid) for row in output_rows],
        "observed_mask": [list(row.observed) for row in output_rows],
        "interpolated_mask": [list(row.interpolated) for row in output_rows],
        "hand_present_mask": [list(row.hand_present) for row in output_rows],
        "body_available_mask": [row.body_available for row in output_rows],
        "padding_mask": [False] * len(selected_rows) + [True] * padding_count,
    }
    payload["sequence_sha256"] = portable_feature_sequence_digest(payload)
    try:
        derived = PortableFeatureSequenceV1.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except ValueError as error:
        raise FeatureTransformError(
            "derived feature sequence is outside the portable contract"
        ) from error
    if statistics is None:
        return derived
    if plan.learned_statistics.mode != "train_only_masked_zscore/1":
        raise FeatureTransformError("feature statistics are disabled by the plan")
    from signlab.features.statistics import apply_feature_statistics

    try:
        return apply_feature_statistics(derived, statistics, plan)
    except ValueError as error:
        raise FeatureTransformError(
            "feature statistics do not match the derived sequence"
        ) from error


__all__ = ["FeatureTransformError", "derive_feature_sequence", "derive_feature_source"]
