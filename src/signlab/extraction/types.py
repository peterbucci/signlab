"""Dependency-light observations shared by extraction backends and tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

type ReportedHandedness = Literal["left", "right"]
type HandSlotId = Literal["hand_0", "hand_1"]


def _require_unit_interval(value: float | None, field_name: str) -> None:
    if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{field_name} must be finite and between zero and one")


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """A vendor-neutral three-dimensional landmark observation.

    ``visibility`` and ``presence`` stay nullable because not every task exposes
    those signals. Coordinates are deliberately not clipped: image landmarks may
    legitimately fall outside the normalized image bounds.
    """

    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None

    def __post_init__(self) -> None:
        if not all(math.isfinite(coordinate) for coordinate in (self.x, self.y, self.z)):
            raise ValueError("landmark coordinates must be finite")
        _require_unit_interval(self.visibility, "visibility")
        _require_unit_interval(self.presence, "presence")


@dataclass(frozen=True, slots=True)
class HandDetection:
    """One detector output before a stable tracking identity is assigned.

    ``detector_index`` is the source detector's index for this frame. It is kept
    intact through tracking so extraction output can be audited back to the raw
    detector result. Image landmarks remain in their original coordinate system;
    the tracker never mirrors, rotates, interpolates, or transforms them.
    """

    detector_index: int
    image_landmarks: tuple[LandmarkPoint, ...]
    world_landmarks: tuple[LandmarkPoint, ...] = ()
    reported_handedness: ReportedHandedness | None = None
    handedness_score: float | None = None

    def __post_init__(self) -> None:
        if self.detector_index < 0:
            raise ValueError("detector_index must be non-negative")
        if len(self.image_landmarks) != 21:
            raise ValueError("a hand detection must contain exactly 21 image landmarks")
        if not all(isinstance(point, LandmarkPoint) for point in self.image_landmarks):
            raise TypeError("image_landmarks must contain LandmarkPoint values")
        if self.world_landmarks and len(self.world_landmarks) != 21:
            raise ValueError("world_landmarks must be empty or contain exactly 21 landmarks")
        if not all(isinstance(point, LandmarkPoint) for point in self.world_landmarks):
            raise TypeError("world_landmarks must contain LandmarkPoint values")
        if self.reported_handedness not in (None, "left", "right"):
            raise ValueError("reported_handedness must be left, right, or None")
        _require_unit_interval(self.handedness_score, "handedness_score")
        if (self.reported_handedness is None) != (self.handedness_score is None):
            raise ValueError("reported_handedness and handedness_score must appear together")


@dataclass(frozen=True, slots=True)
class TrackedHand:
    """One stable output slot, with ``None`` representing an explicit absence."""

    slot_id: HandSlotId
    detection: HandDetection | None

    @property
    def present(self) -> bool:
        """Whether this slot has a detector observation for the current frame."""

        return self.detection is not None


@dataclass(frozen=True, slots=True)
class HandTrackingResult:
    """The two ordered tracking slots emitted for every processed frame."""

    slots: tuple[TrackedHand, TrackedHand]

    def __post_init__(self) -> None:
        if tuple(slot.slot_id for slot in self.slots) != ("hand_0", "hand_1"):
            raise ValueError("tracking slots must be ordered as hand_0, hand_1")

    def detection_for(self, slot_id: HandSlotId) -> HandDetection | None:
        """Return the current detector observation for a named slot."""

        return self.slots[0 if slot_id == "hand_0" else 1].detection


__all__ = [
    "HandDetection",
    "HandSlotId",
    "HandTrackingResult",
    "LandmarkPoint",
    "ReportedHandedness",
    "TrackedHand",
]
