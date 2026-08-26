"""Deterministic two-hand identity tracking over vendor-neutral detections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations

from signlab.extraction.types import (
    HandDetection,
    HandTrackingResult,
    LandmarkPoint,
    TrackedHand,
)

_SLOT_IDS = ("hand_0", "hand_1")
_PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)


@dataclass(frozen=True, slots=True)
class HandTrackingConfig:
    """Versionable algorithm facts used to decide whether identities continue.

    These values govern identity assignment only. They are not dataset-quality
    thresholds and do not reject or transform source observations.
    """

    max_spatial_cost: float = 0.25
    handedness_disagreement_penalty: float = 0.05
    ambiguity_margin: float = 1e-9

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_spatial_cost", self.max_spatial_cost),
            ("handedness_disagreement_penalty", self.handedness_disagreement_penalty),
            ("ambiguity_margin", self.ambiguity_margin),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class _Assignment:
    pairs: tuple[tuple[int, int], ...]
    cost: float
    signature: tuple[tuple[str, tuple[float | str | int, ...]], ...]


def _centroid(points: tuple[LandmarkPoint, ...]) -> tuple[float, float, float]:
    count = len(points)
    return (
        sum(point.x for point in points) / count,
        sum(point.y for point in points) / count,
        sum(point.z for point in points) / count,
    )


def _palm_centroid(detection: HandDetection) -> tuple[float, float, float]:
    return _centroid(tuple(detection.image_landmarks[index] for index in _PALM_LANDMARK_INDICES))


def _squared_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return sum((left - right) ** 2 for left, right in zip(first, second, strict=True))


def _spatial_cost(previous: HandDetection, current: HandDetection) -> float:
    previous_wrist = previous.image_landmarks[0]
    current_wrist = current.image_landmarks[0]
    return _squared_distance(
        (previous_wrist.x, previous_wrist.y, previous_wrist.z),
        (current_wrist.x, current_wrist.y, current_wrist.z),
    ) + _squared_distance(_palm_centroid(previous), _palm_centroid(current))


def _assignment_cost(
    previous: HandDetection,
    current: HandDetection,
    config: HandTrackingConfig,
) -> tuple[float, float]:
    spatial = _spatial_cost(previous, current)
    disagrees = (
        previous.reported_handedness is not None
        and current.reported_handedness is not None
        and previous.reported_handedness != current.reported_handedness
    )
    total = spatial + (config.handedness_disagreement_penalty if disagrees else 0.0)
    return spatial, total


def _detection_sort_key(detection: HandDetection) -> tuple[float | str | int, ...]:
    """Order by untouched image geometry, then lexical and detector-index facts."""

    wrist = detection.image_landmarks[0]
    palm = _palm_centroid(detection)
    return (
        wrist.x,
        wrist.y,
        wrist.z,
        palm[0],
        palm[1],
        palm[2],
        detection.reported_handedness or "",
        detection.detector_index,
    )


def _enumerate_assignments(
    previous: tuple[HandDetection | None, HandDetection | None],
    detections: tuple[HandDetection, ...],
    config: HandTrackingConfig,
) -> tuple[_Assignment, ...]:
    active_slots = tuple(index for index, detection in enumerate(previous) if detection is not None)
    assignments: list[_Assignment] = []
    for pair_count in range(min(len(active_slots), len(detections)) + 1):
        for selected_slots in combinations(active_slots, pair_count):
            for selected_detections in permutations(range(len(detections)), pair_count):
                pairs = tuple(zip(selected_slots, selected_detections, strict=True))
                costs: list[float] = []
                valid = True
                for slot_index, detection_position in pairs:
                    prior = previous[slot_index]
                    assert prior is not None
                    spatial, total = _assignment_cost(
                        prior,
                        detections[detection_position],
                        config,
                    )
                    if spatial > config.max_spatial_cost:
                        valid = False
                        break
                    costs.append(total)
                if not valid:
                    continue
                signature = tuple(
                    (_SLOT_IDS[slot_index], _detection_sort_key(detections[detection_position]))
                    for slot_index, detection_position in pairs
                )
                assignments.append(
                    _Assignment(pairs=pairs, cost=math.fsum(costs), signature=signature)
                )
    return tuple(assignments)


def _unambiguous_pairs(
    assignments: tuple[_Assignment, ...],
    config: HandTrackingConfig,
) -> tuple[tuple[int, int], ...]:
    """Keep only pairs shared by every competitively optimal assignment."""

    max_pair_count = max(len(assignment.pairs) for assignment in assignments)
    fullest = tuple(
        assignment for assignment in assignments if len(assignment.pairs) == max_pair_count
    )
    ordered = sorted(fullest, key=lambda assignment: (assignment.cost, assignment.signature))
    best_cost = ordered[0].cost
    competitive = tuple(
        assignment
        for assignment in ordered
        if assignment.cost <= best_cost + config.ambiguity_margin
    )
    common_pairs = set(competitive[0].pairs)
    for assignment in competitive[1:]:
        common_pairs.intersection_update(assignment.pairs)
    return tuple(sorted(common_pairs))


def _fallback_pairs(
    previous: tuple[HandDetection | None, HandDetection | None],
    detections: tuple[HandDetection, ...],
    available_slots: tuple[int, ...],
    detection_positions: tuple[int, ...],
    config: HandTrackingConfig,
) -> tuple[tuple[int, int], ...]:
    """Place every unmatched observation without claiming confident continuity."""

    candidates: list[
        tuple[
            int,
            float,
            tuple[tuple[str, tuple[float | str | int, ...]], ...],
            tuple[tuple[int, int], ...],
        ]
    ] = []
    for selected_slots in combinations(available_slots, len(detection_positions)):
        for ordered_positions in permutations(detection_positions):
            pairs = tuple(zip(selected_slots, ordered_positions, strict=True))
            overwritten_slots = sum(previous[slot_index] is not None for slot_index, _ in pairs)
            costs = []
            for slot_index, detection_position in pairs:
                prior = previous[slot_index]
                if prior is not None:
                    _, total = _assignment_cost(prior, detections[detection_position], config)
                    costs.append(total)
            signature = tuple(
                (_SLOT_IDS[slot_index], _detection_sort_key(detections[detection_position]))
                for slot_index, detection_position in pairs
            )
            candidates.append((overwritten_slots, math.fsum(costs), signature, pairs))
    return min(candidates)[-1]


def _result(detections: list[HandDetection | None]) -> HandTrackingResult:
    return HandTrackingResult(
        slots=(
            TrackedHand(slot_id="hand_0", detection=detections[0]),
            TrackedHand(slot_id="hand_1", detection=detections[1]),
        )
    )


class HandIdentityTracker:
    """Assign at most two detections to stable, deterministic output slots.

    Initialization uses unmodified image-space geometry rather than detector order.
    Subsequent frames exhaustively compare the possible continuity assignments.
    Unambiguous in-range pairs retain their slots. Every remaining observation is
    still emitted through a deterministic geometry/index fallback; tracking
    thresholds never become observation-retention or quality policy.
    """

    def __init__(self, config: HandTrackingConfig | None = None) -> None:
        self.config = config or HandTrackingConfig()
        self._previous: tuple[HandDetection | None, HandDetection | None] = (None, None)

    def reset(self) -> None:
        """Forget prior identities so the next frame initializes from geometry."""

        self._previous = (None, None)

    def track(self, detections: tuple[HandDetection, ...]) -> HandTrackingResult:
        """Track one frame and return both ordered slots, including absences."""

        if len(detections) > 2:
            raise ValueError("HandIdentityTracker accepts at most two detections per frame")
        detector_indices = tuple(detection.detector_index for detection in detections)
        if len(set(detector_indices)) != len(detector_indices):
            raise ValueError("detector_index values must be unique within a frame")

        if not detections:
            # A detector can miss every hand for an individual frame. Emit the
            # absence without ending identities that belong to this recording;
            # the recording boundary calls ``reset`` explicitly.
            return _result([None, None])

        active_slots = tuple(
            index for index, detection in enumerate(self._previous) if detection is not None
        )
        if not active_slots:
            initialized: list[HandDetection | None] = [None, None]
            for slot_index, detection in enumerate(sorted(detections, key=_detection_sort_key)):
                initialized[slot_index] = detection
            self._previous = (initialized[0], initialized[1])
            return _result(initialized)

        assignments = _enumerate_assignments(self._previous, detections, self.config)
        pairs = _unambiguous_pairs(assignments, self.config)
        tracked: list[HandDetection | None] = [None, None]
        next_previous = list(self._previous)
        used_positions: set[int] = set()
        used_slots: set[int] = set()
        for slot_index, detection_position in pairs:
            detection = detections[detection_position]
            tracked[slot_index] = detection
            next_previous[slot_index] = detection
            used_positions.add(detection_position)
            used_slots.add(slot_index)

        fallback_positions = tuple(
            position for position in range(len(detections)) if position not in used_positions
        )
        fallback_slots = tuple(index for index in range(2) if index not in used_slots)
        for slot_index, detection_position in _fallback_pairs(
            self._previous,
            detections,
            fallback_slots,
            fallback_positions,
            self.config,
        ):
            detection = detections[detection_position]
            tracked[slot_index] = detection
            next_previous[slot_index] = detection

        self._previous = (next_previous[0], next_previous[1])
        return _result(tracked)


__all__ = ["HandIdentityTracker", "HandTrackingConfig"]
