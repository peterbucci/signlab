"""One bounded active-sign window for isolated PopSign landmark sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from signlab.contracts.extraction import (
    HandSlotV1,
    LandmarkFramesTableV1,
    landmark_frames_table_digest,
)

POPSIGN_WINDOW_RULE_ID: Final = "popsign_longest_detected_hand_episode/1"

_MAX_BRIDGED_MISSING_FRAMES: Final = 2
_MAX_BRIDGE_US: Final = 100_000
_MIN_OBSERVED_HAND_FRAMES: Final = 3
_PALM_LANDMARK_INDICES: Final = (0, 5, 9, 17)
_COORDINATE_QUANTIZATION: Final = 1_000_000

type PopSignWindowReason = Literal[
    "selected_longest_detected_hand_episode",
    "no_hand_observations",
    "episode_too_short",
]


class PopSignWindowError(ValueError):
    """Raised when a window decision cannot be applied to its source table."""


@dataclass(frozen=True, slots=True)
class PopSignWindow:
    """An auditable decision over original source-frame coordinates."""

    rule_id: str
    reason: PopSignWindowReason
    source_table_sha256: str
    source_frame_count: int
    episode_count: int
    observed_hand_frame_count: int
    motion_l1_q: int
    first_source_frame_index: int | None
    last_source_frame_index: int | None
    first_source_pts: int | None
    last_source_pts: int | None
    first_source_timestamp_us: int | None
    last_source_timestamp_us: int | None

    @property
    def selected(self) -> bool:
        """Return whether this decision contains a safe inclusive window."""

        return self.reason == "selected_longest_detected_hand_episode"


def _bridged_presence(table: LandmarkFramesTableV1) -> tuple[bool, ...]:
    rows = table.rows
    present = tuple(row.observed_hand_count > 0 for row in rows)
    bridged = list(present)
    index = 0
    while index < len(rows):
        if bridged[index]:
            index += 1
            continue
        end = index
        while end < len(rows) and not bridged[end]:
            end += 1
        if (
            index > 0
            and end < len(rows)
            and end - index <= _MAX_BRIDGED_MISSING_FRAMES
            and rows[end].relative_timestamp_us - rows[index - 1].relative_timestamp_us
            <= _MAX_BRIDGE_US
        ):
            bridged[index:end] = [True] * (end - index)
        index = end
    return tuple(bridged)


def _episodes(presence: tuple[bool, ...]) -> tuple[tuple[int, int], ...]:
    episodes: list[tuple[int, int]] = []
    index = 0
    while index < len(presence):
        if not presence[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(presence) and presence[end + 1]:
            end += 1
        episodes.append((index, end))
        index = end + 1
    return tuple(episodes)


def _palm_centroid_q(hand: HandSlotV1) -> tuple[int, int, int]:
    if not hand.present or hand.image_landmarks is None:
        raise PopSignWindowError("motion requires an observed hand")
    points = tuple(hand.image_landmarks[index] for index in _PALM_LANDMARK_INDICES)
    return tuple(
        round(
            sum(getattr(point, axis) for point in points) * _COORDINATE_QUANTIZATION / len(points)
        )
        for axis in ("x", "y", "z")
    )


def _episode_motion_l1_q(
    table: LandmarkFramesTableV1,
    start: int,
    end: int,
) -> int:
    motion = 0
    for slot_index in range(2):
        previous: tuple[int, int, int] | None = None
        for row in table.rows[start : end + 1]:
            hand = row.hands[slot_index]
            if not hand.present:
                continue
            current = _palm_centroid_q(hand)
            if previous is not None:
                motion += sum(
                    abs(right - left) for left, right in zip(previous, current, strict=True)
                )
            previous = current
    return motion


def select_popsign_window(table: LandmarkFramesTableV1) -> PopSignWindow:
    """Select one label-blind hand episode without changing the source table."""

    if not isinstance(table, LandmarkFramesTableV1):
        raise PopSignWindowError("PopSign window input must be a validated landmark table")
    source_table_sha256 = landmark_frames_table_digest(table)
    source_presence = tuple(row.observed_hand_count > 0 for row in table.rows)
    episodes = _episodes(_bridged_presence(table))
    if not episodes:
        return PopSignWindow(
            rule_id=POPSIGN_WINDOW_RULE_ID,
            reason="no_hand_observations",
            source_table_sha256=source_table_sha256,
            source_frame_count=len(table.rows),
            episode_count=0,
            observed_hand_frame_count=0,
            motion_l1_q=0,
            first_source_frame_index=None,
            last_source_frame_index=None,
            first_source_pts=None,
            last_source_pts=None,
            first_source_timestamp_us=None,
            last_source_timestamp_us=None,
        )

    ranked = tuple(
        (
            sum(source_presence[start : end + 1]),
            _episode_motion_l1_q(table, start, end),
            end - start + 1,
            -start,
            start,
            end,
        )
        for start, end in episodes
    )
    observed_count, motion_l1_q, _span, _earliest, start, end = max(ranked)
    if observed_count < _MIN_OBSERVED_HAND_FRAMES:
        return PopSignWindow(
            rule_id=POPSIGN_WINDOW_RULE_ID,
            reason="episode_too_short",
            source_table_sha256=source_table_sha256,
            source_frame_count=len(table.rows),
            episode_count=len(episodes),
            observed_hand_frame_count=observed_count,
            motion_l1_q=motion_l1_q,
            first_source_frame_index=None,
            last_source_frame_index=None,
            first_source_pts=None,
            last_source_pts=None,
            first_source_timestamp_us=None,
            last_source_timestamp_us=None,
        )

    first = table.rows[start]
    last = table.rows[end]
    return PopSignWindow(
        rule_id=POPSIGN_WINDOW_RULE_ID,
        reason="selected_longest_detected_hand_episode",
        source_table_sha256=source_table_sha256,
        source_frame_count=len(table.rows),
        episode_count=len(episodes),
        observed_hand_frame_count=observed_count,
        motion_l1_q=motion_l1_q,
        first_source_frame_index=first.frame_index,
        last_source_frame_index=last.frame_index,
        first_source_pts=first.source_pts,
        last_source_pts=last.source_pts,
        first_source_timestamp_us=first.relative_timestamp_us,
        last_source_timestamp_us=last.relative_timestamp_us,
    )


def materialize_popsign_window(
    table: LandmarkFramesTableV1,
    window: PopSignWindow,
) -> LandmarkFramesTableV1:
    """Create the explicit rebased view consumed by unchanged quality and feature code."""

    if not window.selected:
        raise PopSignWindowError("a failed PopSign window decision cannot be materialized")
    if window.source_table_sha256 != landmark_frames_table_digest(table):
        raise PopSignWindowError("PopSign window source content changed")
    if window.source_frame_count != len(table.rows):
        raise PopSignWindowError("PopSign window source frame count changed")
    start = window.first_source_frame_index
    end = window.last_source_frame_index
    if start is None or end is None or not 0 <= start <= end < len(table.rows):
        raise PopSignWindowError("PopSign window bounds are invalid")
    if (
        table.rows[start].source_pts != window.first_source_pts
        or table.rows[end].source_pts != window.last_source_pts
        or table.rows[start].relative_timestamp_us != window.first_source_timestamp_us
        or table.rows[end].relative_timestamp_us != window.last_source_timestamp_us
    ):
        raise PopSignWindowError("PopSign window does not match its source table")

    first = table.rows[start]
    rows = []
    previous_task_ms: int | None = None
    for frame_index, row in enumerate(table.rows[start : end + 1]):
        relative_timestamp_us = (
            (row.source_pts - first.source_pts)
            * row.source_time_base_numerator
            * 1_000_000
            // row.source_time_base_denominator
        )
        task_timestamp_ms = max(
            relative_timestamp_us // 1_000,
            0 if previous_task_ms is None else previous_task_ms + 1,
        )
        rows.append(
            row.model_copy(
                update={
                    "frame_index": frame_index,
                    "relative_timestamp_us": relative_timestamp_us,
                    "task_timestamp_ms": task_timestamp_ms,
                }
            )
        )
        previous_task_ms = task_timestamp_ms
    return LandmarkFramesTableV1(
        schema_version="landmark-frames-table/1",
        rows=tuple(rows),
    )


__all__ = [
    "POPSIGN_WINDOW_RULE_ID",
    "PopSignWindow",
    "PopSignWindowError",
    "materialize_popsign_window",
    "select_popsign_window",
]
