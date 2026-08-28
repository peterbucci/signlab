from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from feature_fixtures import make_feature_fixture, make_hand_row
from signlab.contracts.extraction import LandmarkFramesTableV1, landmark_frames_table_digest
from signlab.contracts.quality import landmark_quality_policy_digest
from signlab.datasets.popsign_window import (
    POPSIGN_WINDOW_RULE_ID,
    PopSignWindowError,
    materialize_popsign_window,
    select_popsign_window,
)
from signlab.quality.policy import assess_landmark_source
from signlab.quality.resources import build_default_quality_policy


def _table(
    presence: tuple[bool, ...],
    *,
    centers: tuple[tuple[float, float], ...] | None = None,
) -> LandmarkFramesTableV1:
    timestamps = tuple(index * 33_333 for index in range(len(presence)))
    hand_rows = tuple(
        make_hand_row(
            timestamp_us=timestamp,
            first_present=present,
            first_center=(0.42, 0.58) if centers is None else centers[index],
        )
        for index, (timestamp, present) in enumerate(zip(timestamps, presence, strict=True))
    )
    return make_feature_fixture(timestamps, hand_rows=hand_rows).table


def _quality(table: LandmarkFramesTableV1) -> str:
    digest = landmark_frames_table_digest(table)
    return assess_landmark_source(
        table,
        build_default_quality_policy(),
        source_recording_id=table.rows[0].source_recording_id,
        source_sequence_content_sha256=digest,
        source_landmark_parquet_sha256="sha256:" + "a" * 64,
        declared_duration_us=max(table.rows[-1].relative_timestamp_us + 33_333, 1),
        expected_hand_count=1,
    ).disposition


def test_window_removes_inactive_edges_without_changing_quality_policy() -> None:
    source = _table((False, False, True, True, True, True, False, False, False))
    source_digest = landmark_frames_table_digest(source)
    policy_digest = landmark_quality_policy_digest(build_default_quality_policy())

    window = select_popsign_window(source)
    materialized = materialize_popsign_window(source, window)

    assert window.rule_id == POPSIGN_WINDOW_RULE_ID
    assert window.reason == "selected_longest_detected_hand_episode"
    assert window.source_table_sha256 == source_digest
    assert (window.first_source_frame_index, window.last_source_frame_index) == (2, 5)
    assert _quality(source) == "reject"
    assert _quality(materialized) == "pass"
    assert landmark_frames_table_digest(source) == source_digest
    assert policy_digest == (
        "sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545"
    )


def test_window_returns_coded_failures_for_no_hands_and_short_detection() -> None:
    no_hands = select_popsign_window(_table((False, False, False, False)))
    short = select_popsign_window(_table((False, True, True, False)))

    assert no_hands.reason == "no_hand_observations"
    assert no_hands.selected is False
    assert no_hands.first_source_frame_index is None
    assert short.reason == "episode_too_short"
    assert short.observed_hand_frame_count == 2
    assert short.selected is False
    with pytest.raises(PopSignWindowError):
        materialize_popsign_window(_table((False, True, True, False)), short)


def test_window_bridges_only_the_existing_policy_sized_gap() -> None:
    source = _table((False, True, True, False, True, True, False, False, False, True, True, True))

    window = select_popsign_window(source)

    assert window.episode_count == 2
    assert window.observed_hand_frame_count == 4
    assert (window.first_source_frame_index, window.last_source_frame_index) == (1, 5)


def test_window_uses_motion_then_earliest_for_equal_length_episodes() -> None:
    presence = (True, True, True, False, False, False, True, True, True)
    centers = (
        (0.42, 0.58),
        (0.42, 0.58),
        (0.42, 0.58),
        (0.42, 0.58),
        (0.42, 0.58),
        (0.42, 0.58),
        (0.42, 0.58),
        (0.48, 0.58),
        (0.54, 0.58),
    )
    moving = select_popsign_window(_table(presence, centers=centers))
    tied = select_popsign_window(_table(presence))

    assert (moving.first_source_frame_index, moving.last_source_frame_index) == (6, 8)
    assert (tied.first_source_frame_index, tied.last_source_frame_index) == (0, 2)


def test_window_excludes_a_short_trailing_reappearance_and_rebases_the_view() -> None:
    source = _table((False, True, True, True, True, False, False, False, False, True))

    window = select_popsign_window(source)
    materialized = materialize_popsign_window(source, window)

    assert window.episode_count == 2
    assert (window.first_source_frame_index, window.last_source_frame_index) == (1, 4)
    assert window.first_source_pts == source.rows[1].source_pts
    assert window.last_source_pts == source.rows[4].source_pts
    assert tuple(row.frame_index for row in materialized.rows) == (0, 1, 2, 3)
    assert materialized.rows[0].relative_timestamp_us == 0
    assert materialized.rows[0].task_timestamp_ms == 0
    assert materialized.rows[0].source_pts == source.rows[1].source_pts
    assert materialized.rows[-1].source_pts == source.rows[4].source_pts


def test_window_cannot_be_applied_after_its_source_changes() -> None:
    source = _table((False, True, True, True, False))
    window = select_popsign_window(source)
    changed_content = _table(
        (False, True, True, True, False),
        centers=((0.42, 0.58), (0.43, 0.58), (0.44, 0.58), (0.45, 0.58), (0.42, 0.58)),
    )

    with pytest.raises(PopSignWindowError):
        select_popsign_window(cast(Any, object()))
    with pytest.raises(PopSignWindowError):
        materialize_popsign_window(source, replace(window, source_frame_count=99))
    with pytest.raises(PopSignWindowError):
        materialize_popsign_window(changed_content, window)
    with pytest.raises(PopSignWindowError):
        materialize_popsign_window(source, replace(window, first_source_frame_index=None))
    with pytest.raises(PopSignWindowError):
        materialize_popsign_window(source, replace(window, first_source_pts=-1))
