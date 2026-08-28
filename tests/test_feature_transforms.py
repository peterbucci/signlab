from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace

import pytest

from feature_fixtures import (
    EXTRACTION_CONFIG_SHA256,
    FeatureFixture,
    make_feature_fixture,
    make_feature_plan,
    make_hand_row,
)
from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.features import LandmarkFeaturePlanV1, PortableFeatureSequenceV1
from signlab.contracts.quality import SequenceQualityReportV1, sequence_quality_report_digest
from signlab.features.transforms import FeatureTransformError, derive_feature_sequence


def _derive(
    fixture: FeatureFixture,
    plan: LandmarkFeaturePlanV1,
) -> PortableFeatureSequenceV1:
    return derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        plan,
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )


def _indexes_with_prefix(names: tuple[str, ...], prefix: str) -> tuple[int, ...]:
    return tuple(index for index, name in enumerate(names) if name.startswith(prefix))


def _assert_rows_close(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
    *,
    tolerance_q: int = 1,
) -> None:
    assert len(left) == len(right)
    for left_row, right_row in zip(left, right, strict=True):
        assert len(left_row) == len(right_row)
        assert all(
            abs(left_value - right_value) <= tolerance_q
            for left_value, right_value in zip(left_row, right_row, strict=True)
        )


@pytest.mark.parametrize(
    ("two_hands", "expected_presence"),
    [(False, (True, False)), (True, (True, True))],
)
def test_one_and_two_hand_inputs_preserve_two_fixed_slots(
    two_hands: bool,
    expected_presence: tuple[bool, bool],
) -> None:
    result = _derive(
        make_feature_fixture(two_hands=two_hands),
        make_feature_plan("hand_local", target_frame_count=3),
    )
    hand_0 = _indexes_with_prefix(result.feature_names, "hand_0.local.")
    hand_1 = _indexes_with_prefix(result.feature_names, "hand_1.local.")

    assert len(result.feature_names) == 126
    assert len(hand_0) == len(hand_1) == 63
    assert result.hand_present_mask == (expected_presence,) * 3
    assert all(result.valid_mask[0][index] for index in hand_0)
    assert all(result.valid_mask[0][index] is two_hands for index in hand_1)
    if two_hands:
        assert any(result.values_q[0][index] != 0 for index in hand_1)
    else:
        assert all(result.values_q[0][index] == 0 for index in hand_1)


def test_hand_and_body_normalization_are_translation_and_scale_invariant() -> None:
    baseline = _derive(
        make_feature_fixture(),
        make_feature_plan("combined", target_frame_count=3),
    )
    transformed = _derive(
        make_feature_fixture(
            image_global_scale=1.7,
            image_global_translation=(0.23, -0.14),
            world_global_scale=2.75,
            world_global_translation=(4.0, -3.0, 1.25),
        ),
        make_feature_plan("combined", target_frame_count=3),
    )

    assert transformed.valid_mask == baseline.valid_mask
    assert transformed.observed_mask == baseline.observed_mask
    assert transformed.interpolated_mask == baseline.interpolated_mask
    _assert_rows_close(transformed.values_q, baseline.values_q)


def test_recorded_mirror_state_produces_equivalent_canonical_features() -> None:
    unmirrored_fixture = make_feature_fixture(mirrored=False)
    mirrored_fixture = make_feature_fixture(mirrored=True)
    unmirrored = _derive(
        unmirrored_fixture,
        make_feature_plan("combined", target_frame_count=3),
    )
    mirrored = _derive(
        mirrored_fixture,
        make_feature_plan("combined", target_frame_count=3),
    )

    assert unmirrored_fixture.table.rows[0].hands[0].handedness == "left"
    assert mirrored_fixture.table.rows[0].hands[0].handedness == "right"
    assert mirrored.valid_mask == unmirrored.valid_mask
    assert mirrored.hand_present_mask == unmirrored.hand_present_mask
    assert mirrored.body_available_mask == unmirrored.body_available_mask
    _assert_rows_close(mirrored.values_q, unmirrored.values_q)


def test_quality_parquet_digest_must_match_extraction_artifact() -> None:
    fixture = make_feature_fixture()
    payload = fixture.quality.model_dump(mode="json", round_trip=True)
    payload["source_landmark_parquet_sha256"] = "sha256:" + "f" * 64
    payload.pop("report_sha256")
    payload["report_sha256"] = sequence_quality_report_digest(payload)
    mismatched_quality = SequenceQualityReportV1.model_validate_json(
        canonical_json_bytes(payload),
        strict=True,
    )

    with pytest.raises(
        FeatureTransformError,
        match="quality evidence does not match extraction lineage",
    ):
        _derive(
            replace(fixture, quality=mismatched_quality),
            make_feature_plan("combined", target_frame_count=3),
        )


def test_reported_timestamp_discontinuity_fails_closed_before_resampling() -> None:
    fixture = make_feature_fixture((0, 33_333, 66_667, 566_667))

    assert fixture.quality.metrics.timestamp_discontinuity_count == 1
    with pytest.raises(FeatureTransformError, match="timestamp discontinuities"):
        _derive(
            fixture,
            make_feature_plan("combined", target_frame_count=4),
        )


def test_only_quality_approved_short_hand_gap_is_interpolated() -> None:
    timestamps = (0, 33_333, 66_667)
    approved = make_feature_fixture(
        timestamps,
        hand_rows=(
            make_hand_row(timestamp_us=timestamps[0]),
            make_hand_row(timestamp_us=timestamps[1], first_present=False),
            make_hand_row(timestamp_us=timestamps[2]),
        ),
    )
    approved_gap = next(gap for gap in approved.quality.gaps if gap.signal == "hand_0")
    approved_result = _derive(
        approved,
        make_feature_plan("hand_local", target_frame_count=3),
    )
    hand_0 = _indexes_with_prefix(approved_result.feature_names, "hand_0.local.")

    assert approved_gap.decision == "interpolate_linear"
    assert approved_gap.reasons == ("eligible_short_internal_gap",)
    assert all(approved_result.valid_mask[1][index] for index in hand_0)
    assert all(approved_result.interpolated_mask[1][index] for index in hand_0)
    assert not any(approved_result.observed_mask[1][index] for index in hand_0)

    blocked = make_feature_fixture(
        timestamps,
        two_hands=True,
        hand_rows=(
            make_hand_row(timestamp_us=timestamps[0], two_hands=True),
            make_hand_row(
                timestamp_us=timestamps[1],
                two_hands=True,
                first_present=False,
            ),
            make_hand_row(
                timestamp_us=timestamps[2],
                two_hands=True,
                first_anatomical_handedness="left",
                second_anatomical_handedness="right",
                first_center=(0.68, 0.58),
                second_center=(0.42, 0.58),
            ),
        ),
    )
    blocked_gap = next(gap for gap in blocked.quality.gaps if gap.signal == "hand_0")
    blocked_result = _derive(
        blocked,
        make_feature_plan("hand_local", target_frame_count=3),
    )

    assert blocked.quality.metrics.suspected_hand_swap_count == 1
    assert blocked_gap.decision == "preserve_missing"
    assert blocked_gap.crosses_suspected_hand_swap
    assert blocked_gap.contains_identity_ambiguity
    assert not any(blocked_result.valid_mask[1][index] for index in hand_0)
    assert not any(blocked_result.interpolated_mask[1][index] for index in hand_0)
    assert blocked_result.hand_present_mask[1] == (False, True)


@pytest.mark.parametrize(("missing_index", "boundary"), [(0, "leading"), (2, "trailing")])
def test_boundary_gaps_are_never_extrapolated_or_forward_filled(
    missing_index: int,
    boundary: str,
) -> None:
    timestamps = (0, 33_333, 66_667)
    fixture = make_feature_fixture(
        timestamps,
        hand_rows=tuple(
            make_hand_row(timestamp_us=timestamp, first_present=index != missing_index)
            for index, timestamp in enumerate(timestamps)
        ),
    )
    result = _derive(
        fixture,
        make_feature_plan("hand_local", target_frame_count=3),
    )
    gap = next(gap for gap in fixture.quality.gaps if gap.signal == "hand_0")
    hand_0 = _indexes_with_prefix(result.feature_names, "hand_0.local.")

    assert gap.boundary == boundary
    assert gap.decision == "preserve_missing"
    assert not any(result.valid_mask[missing_index][index] for index in hand_0)
    assert not any(result.interpolated_mask[missing_index][index] for index in hand_0)
    assert result.hand_present_mask[missing_index] == (False, False)


def test_absent_pose_masks_body_channels_but_keeps_hand_local_shape() -> None:
    result = _derive(
        make_feature_fixture(pose_present=False),
        make_feature_plan("combined", target_frame_count=3),
    )
    hand_0 = _indexes_with_prefix(result.feature_names, "hand_0.local.")
    body = tuple(index for index, name in enumerate(result.feature_names) if ".body." in name)

    assert result.body_available_mask == (False, False, False)
    assert all(result.valid_mask[0][index] for index in hand_0)
    assert body
    assert not any(result.valid_mask[0][index] for index in body)
    assert all(result.values_q[0][index] == 0 for index in body)


def test_irregular_elapsed_time_drives_optional_geometry_and_derivatives() -> None:
    fixture = make_feature_fixture(
        (0, 20_000, 70_000, 100_000),
        image_velocity_per_second=(0.2, 0.0),
    )
    result = _derive(
        fixture,
        make_feature_plan(
            "combined",
            target_frame_count=4,
            include_velocity=True,
            include_acceleration=True,
            include_joint_angles=True,
            include_tip_distances=True,
        ),
    )
    wrist_x = result.feature_names.index("hand_0.body.wrist.x")
    velocity = result.feature_names.index("hand_0.body.wrist.x.velocity")
    acceleration = result.feature_names.index("hand_0.body.wrist.x.acceleration")
    angle = result.feature_names.index("hand_0.geometry.index.angle")
    distance = result.feature_names.index("hand_0.geometry.index.tip_distance")

    assert result.timestamps_us == (0, 33_333, 66_667, 100_000)
    assert result.observed_mask[0][wrist_x]
    assert result.interpolated_mask[1][wrist_x]
    assert result.interpolated_mask[2][wrist_x]
    assert result.observed_mask[3][wrist_x]
    assert tuple(row[velocity] for row in result.valid_mask) == (False, True, True, True)
    assert tuple(row[acceleration] for row in result.valid_mask) == (False, False, True, True)
    assert all(abs(result.values_q[index][velocity] - 500_000) <= 2 for index in range(1, 4))
    assert all(abs(result.values_q[index][acceleration]) <= 4 for index in range(2, 4))
    assert result.valid_mask[0][angle]
    assert result.values_q[0][angle] > 0
    assert result.valid_mask[0][distance]
    assert result.values_q[0][distance] > 0
    assert not any(
        result.observed_mask[index][velocity]
        or result.interpolated_mask[index][velocity]
        or result.observed_mask[index][acceleration]
        or result.interpolated_mask[index][acceleration]
        for index in range(4)
    )


def test_derivatives_do_not_cross_a_preserved_gap_between_target_frames() -> None:
    timestamps = (0, 16_667, 33_333, 66_667)
    fixture = make_feature_fixture(
        timestamps,
        two_hands=True,
        hand_rows=(
            make_hand_row(timestamp_us=timestamps[0], two_hands=True),
            make_hand_row(
                timestamp_us=timestamps[1],
                two_hands=True,
                first_present=False,
            ),
            make_hand_row(
                timestamp_us=timestamps[2],
                two_hands=True,
                first_anatomical_handedness="left",
                second_anatomical_handedness="right",
                first_center=(0.68, 0.58),
                second_center=(0.42, 0.58),
            ),
            make_hand_row(
                timestamp_us=timestamps[3],
                two_hands=True,
                first_anatomical_handedness="left",
                second_anatomical_handedness="right",
                first_center=(0.68, 0.58),
                second_center=(0.42, 0.58),
            ),
        ),
    )
    result = _derive(
        fixture,
        make_feature_plan(
            "hand_local",
            target_frame_count=3,
            include_velocity=True,
            include_acceleration=True,
        ),
    )
    gap = next(gap for gap in fixture.quality.gaps if gap.signal == "hand_0")
    hand_0_velocity = result.feature_names.index("hand_0.local.landmark_01.x.velocity")
    hand_0_acceleration = result.feature_names.index("hand_0.local.landmark_01.x.acceleration")
    hand_1_velocity = result.feature_names.index("hand_1.local.landmark_01.x.velocity")
    hand_1_acceleration = result.feature_names.index("hand_1.local.landmark_01.x.acceleration")

    assert gap.decision == "preserve_missing"
    assert gap.contains_identity_ambiguity
    assert gap.first_missing_timestamp_us not in result.timestamps_us
    assert result.timestamps_us == (0, 33_333, 66_667)
    assert tuple(row[hand_0_velocity] for row in result.valid_mask) == (
        False,
        False,
        True,
    )
    assert tuple(row[hand_0_acceleration] for row in result.valid_mask) == (
        False,
        False,
        False,
    )
    assert tuple(row[hand_1_velocity] for row in result.valid_mask) == (
        False,
        True,
        True,
    )
    assert tuple(row[hand_1_acceleration] for row in result.valid_mask) == (
        False,
        False,
        True,
    )


def test_derivatives_are_computed_before_long_sequence_selection() -> None:
    timestamps = (0, 33_333, 66_667, 100_000)
    centers = (0.42, 0.44, 0.50, 0.62)
    fixture = make_feature_fixture(
        timestamps,
        hand_rows=tuple(
            make_hand_row(
                timestamp_us=timestamp,
                first_center=(center, 0.58),
            )
            for timestamp, center in zip(timestamps, centers, strict=True)
        ),
    )
    result = _derive(
        fixture,
        make_feature_plan(
            "body_relative",
            target_frame_count=3,
            include_velocity=True,
        ),
    )
    velocity = result.feature_names.index("hand_0.body.wrist.x.velocity")
    full_grid_velocity = (0.06 / 0.4) / (33_334 / 1_000_000)
    selected_first_velocity = (0.08 / 0.4) / (66_667 / 1_000_000)
    expected_q = math.floor(full_grid_velocity * 1_000_000 + 0.5)
    selected_first_q = math.floor(selected_first_velocity * 1_000_000 + 0.5)

    assert result.selected_source_indices == (0, 2, 3)
    assert result.values_q[1][velocity] == expected_q
    assert result.values_q[1][velocity] != selected_first_q


def _all_false(values: Iterable[bool]) -> bool:
    return not any(values)


def test_long_selection_and_right_padding_are_deterministic_neutral_and_pure() -> None:
    long_fixture = make_feature_fixture((0, 100_000, 200_000, 300_000))
    raw_before = (
        long_fixture.table.model_dump_json(),
        long_fixture.sequence.model_dump_json(),
        long_fixture.quality.model_dump_json(),
    )
    first = _derive(
        long_fixture,
        make_feature_plan("hand_local", target_frame_count=4),
    )
    second = _derive(
        long_fixture,
        make_feature_plan("hand_local", target_frame_count=4),
    )

    assert first == second
    assert first.sequence_sha256 == second.sequence_sha256
    assert first.source_grid_frame_count == 10
    assert first.selected_source_indices == (0, 3, 6, 9)
    assert first.padding_mask == (False, False, False, False)
    assert raw_before == (
        long_fixture.table.model_dump_json(),
        long_fixture.sequence.model_dump_json(),
        long_fixture.quality.model_dump_json(),
    )

    short = _derive(
        make_feature_fixture((0, 50_000)),
        make_feature_plan("hand_local", target_frame_count=5),
    )
    assert short.source_grid_frame_count == 3
    assert short.selected_source_indices == (0, 1, 2)
    assert short.timestamps_us == (0, 33_333, 50_000, 83_333, 116_667)
    assert short.padding_mask == (False, False, False, True, True)
    for index in (3, 4):
        assert set(short.values_q[index]) == {0}
        assert _all_false(short.valid_mask[index])
        assert _all_false(short.observed_mask[index])
        assert _all_false(short.interpolated_mask[index])
        assert short.hand_present_mask[index] == (False, False)
        assert not short.body_available_mask[index]
