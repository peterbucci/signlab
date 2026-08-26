from __future__ import annotations

import math
from typing import cast

import pytest

from signlab.extraction.tracking import HandIdentityTracker, HandTrackingConfig
from signlab.extraction.types import (
    HandDetection,
    HandTrackingResult,
    LandmarkPoint,
    ReportedHandedness,
    TrackedHand,
)


def _hand(
    detector_index: int,
    x: float,
    handedness: ReportedHandedness | None,
    *,
    y: float = 0.5,
    z: float = 0.0,
) -> HandDetection:
    points = tuple(LandmarkPoint(x=x, y=y, z=z) for _ in range(21))
    return HandDetection(
        detector_index=detector_index,
        image_landmarks=points,
        world_landmarks=points,
        reported_handedness=handedness,
        handedness_score=None if handedness is None else 0.9,
    )


def _indices(result: HandTrackingResult) -> tuple[int | None, int | None]:
    def detector_index(slot_index: int) -> int | None:
        detection = result.slots[slot_index].detection
        return None if detection is None else detection.detector_index

    return detector_index(0), detector_index(1)


def test_landmark_and_detection_types_reject_invalid_observations() -> None:
    with pytest.raises(ValueError, match="finite"):
        LandmarkPoint(x=math.nan, y=0.0, z=0.0)
    with pytest.raises(ValueError, match="between zero and one"):
        LandmarkPoint(x=0.0, y=0.0, z=0.0, visibility=1.1)
    with pytest.raises(ValueError, match="between zero and one"):
        LandmarkPoint(x=0.0, y=0.0, z=0.0, presence=math.inf)

    points = tuple(LandmarkPoint(0.0, 0.0, 0.0) for _ in range(21))
    short = points[:-1]
    with pytest.raises(ValueError, match="non-negative"):
        HandDetection(detector_index=-1, image_landmarks=points)
    with pytest.raises(ValueError, match="exactly 21"):
        HandDetection(detector_index=0, image_landmarks=short)
    with pytest.raises(ValueError, match="exactly 21"):
        HandDetection(
            detector_index=0,
            image_landmarks=(*points, LandmarkPoint(0.0, 0.0, 0.0)),
        )
    with pytest.raises(TypeError, match="image_landmarks"):
        HandDetection(
            detector_index=0,
            image_landmarks=cast(tuple[LandmarkPoint, ...], (*points[:-1], "invalid")),
        )
    with pytest.raises(ValueError, match="exactly 21"):
        HandDetection(detector_index=0, image_landmarks=points, world_landmarks=points[:-1])
    with pytest.raises(TypeError, match="world_landmarks"):
        HandDetection(
            detector_index=0,
            image_landmarks=points,
            world_landmarks=cast(tuple[LandmarkPoint, ...], (*points[:-1], "invalid")),
        )
    with pytest.raises(ValueError, match="left, right"):
        HandDetection(
            detector_index=0,
            image_landmarks=points,
            reported_handedness=cast(ReportedHandedness, "unknown"),
            handedness_score=0.5,
        )
    with pytest.raises(ValueError, match="between zero and one"):
        HandDetection(
            detector_index=0,
            image_landmarks=points,
            reported_handedness="left",
            handedness_score=-0.1,
        )
    with pytest.raises(ValueError, match="appear together"):
        HandDetection(
            detector_index=0,
            image_landmarks=points,
            reported_handedness="left",
        )

    with pytest.raises(ValueError, match="ordered"):
        HandTrackingResult(
            slots=(
                TrackedHand(slot_id="hand_1", detection=None),
                TrackedHand(slot_id="hand_0", detection=None),
            )
        )


def test_initialization_uses_unmodified_geometry_not_detector_order() -> None:
    image_left = _hand(9, 0.2, "left")
    image_right = _hand(3, 0.8, "right")

    first = HandIdentityTracker().track((image_right, image_left))
    second = HandIdentityTracker().track((image_left, image_right))

    assert _indices(first) == (9, 3)
    assert _indices(second) == (9, 3)
    assert all(slot.present for slot in first.slots)


def test_initialization_has_lexical_then_detector_index_tie_breaks() -> None:
    same_place_right = _hand(1, 0.5, "right")
    same_place_left_high_index = _hand(8, 0.5, "left")
    same_place_left_low_index = _hand(2, 0.5, "left")

    lexical = HandIdentityTracker().track((same_place_right, same_place_left_high_index))
    detector_index = HandIdentityTracker().track(
        (same_place_left_high_index, same_place_left_low_index)
    )

    assert _indices(lexical) == (8, 1)
    assert _indices(detector_index) == (2, 8)


def test_reverse_detector_order_does_not_swap_existing_tracks() -> None:
    tracker = HandIdentityTracker()
    tracker.track((_hand(0, 0.2, "left"), _hand(1, 0.8, "right")))

    reversed_result = tracker.track((_hand(0, 0.78, "right"), _hand(1, 0.22, "left")))

    assert _indices(reversed_result) == (1, 0)
    assert reversed_result.detection_for("hand_0") is reversed_result.slots[0].detection
    assert reversed_result.detection_for("hand_1") is reversed_result.slots[1].detection


def test_handedness_penalty_resolves_equal_spatial_assignments() -> None:
    tracker = HandIdentityTracker(
        HandTrackingConfig(
            max_spatial_cost=1.0,
            handedness_disagreement_penalty=0.2,
            ambiguity_margin=0.0,
        )
    )
    tracker.track((_hand(0, 0.4, "left"), _hand(1, 0.6, "right")))

    result = tracker.track((_hand(0, 0.5, "right"), _hand(1, 0.5, "left")))

    assert _indices(result) == (1, 0)


def test_ambiguous_continuity_preserves_both_detections_with_stable_fallback() -> None:
    tracker = HandIdentityTracker(HandTrackingConfig(ambiguity_margin=0.0))
    tracker.track((_hand(0, 0.25, None), _hand(1, 0.75, None)))

    ambiguous = tracker.track((_hand(7, 0.5, None), _hand(4, 0.5, None)))

    assert _indices(ambiguous) == (4, 7)
    assert all(slot.present for slot in ambiguous.slots)

    repeated = tracker.track((_hand(7, 0.7, None), _hand(4, 0.3, None)))
    assert _indices(repeated) == (4, 7)


def test_too_far_detections_are_reanchored_without_becoming_quality_rejections() -> None:
    tracker = HandIdentityTracker(HandTrackingConfig(max_spatial_cost=0.01))
    tracker.track((_hand(0, 0.2, "left"), _hand(1, 0.8, "right")))

    too_far = tracker.track((_hand(0, 2.0, "left"), _hand(1, 3.0, "right")))

    assert _indices(too_far) == (0, 1)


def test_fallback_preserves_unaffected_prior_slot_state_across_partial_detection() -> None:
    tracker = HandIdentityTracker(HandTrackingConfig(max_spatial_cost=0.01))
    tracker.track((_hand(0, 0.2, "left"), _hand(1, 0.8, "right")))

    far_detection = tracker.track((_hand(8, -1.0, "left"),))
    returning_prior_hand = tracker.track((_hand(9, 0.81, "right"),))

    assert _indices(far_detection) == (8, None)
    assert _indices(returning_prior_hand) == (None, 9)


def test_fallback_prefers_empty_slot_before_replacing_a_retained_identity() -> None:
    tracker = HandIdentityTracker(HandTrackingConfig(max_spatial_cost=0.01))
    tracker.track((_hand(0, 0.2, "left"),))

    new_far_hand = tracker.track((_hand(8, 2.0, "right"),))
    returning_first_hand = tracker.track((_hand(9, 0.21, "left"),))

    assert _indices(new_far_hand) == (None, 8)
    assert _indices(returning_first_hand) == (9, None)


def test_confident_pair_keeps_its_slot_while_far_detection_fills_remaining_slot() -> None:
    tracker = HandIdentityTracker(HandTrackingConfig(max_spatial_cost=0.01))
    tracker.track((_hand(0, 0.2, "left"), _hand(1, 0.8, "right")))

    mixed = tracker.track((_hand(8, 3.0, "right"), _hand(7, 0.21, "left")))

    assert _indices(mixed) == (7, 8)
    assert {
        slot.detection.detector_index for slot in mixed.slots if slot.detection is not None
    } == {
        7,
        8,
    }


def test_new_second_hand_uses_empty_slot_without_reassigning_first() -> None:
    tracker = HandIdentityTracker()
    one_hand = tracker.track((_hand(4, 0.2, "left"),))
    two_hands = tracker.track((_hand(8, 0.8, "right"), _hand(7, 0.22, "left")))

    assert _indices(one_hand) == (4, None)
    assert _indices(two_hands) == (7, 8)


def test_empty_frame_retains_identity_until_explicit_recording_reset() -> None:
    tracker = HandIdentityTracker()
    tracker.track((_hand(0, 0.2, "left"), _hand(1, 0.8, "right")))

    absent = tracker.track(())
    after_absence = tracker.track((_hand(5, 0.78, "right"), _hand(6, 0.22, "left")))
    tracker.reset()
    after_reset = tracker.track((_hand(9, 0.9, "left"), _hand(3, 0.1, "right")))

    assert _indices(absent) == (None, None)
    assert _indices(after_absence) == (6, 5)
    assert _indices(after_reset) == (3, 9)


def test_tracker_rejects_more_than_two_or_duplicate_detector_indices() -> None:
    tracker = HandIdentityTracker()
    with pytest.raises(ValueError, match="at most two"):
        tracker.track((_hand(0, 0.1, None), _hand(1, 0.2, None), _hand(2, 0.3, None)))
    with pytest.raises(ValueError, match="unique"):
        tracker.track((_hand(0, 0.1, None), _hand(0, 0.2, None)))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_spatial_cost", -1.0),
        ("handedness_disagreement_penalty", math.inf),
        ("ambiguity_margin", math.nan),
    ],
)
def test_tracking_config_rejects_invalid_algorithm_facts(
    field_name: str,
    value: float,
) -> None:
    values = {
        "max_spatial_cost": 0.25,
        "handedness_disagreement_penalty": 0.05,
        "ambiguity_margin": 1e-9,
    }
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        HandTrackingConfig(**values)
