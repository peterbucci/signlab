"""Deterministic candidate-event segmentation over timestamped motion signals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    NonNegativeSafeInteger,
    PositiveSafeInteger,
    SemanticVersion,
    StableId,
    StrictContractModel,
)
from signlab.contracts.extraction import HandSlotV1, LandmarkFramesTableV1
from signlab.contracts.taxonomy import Sha256Digest

CandidateState = Literal["inactive", "arming", "recording", "finalizing", "cooldown"]
CandidateEndReason = Literal["settled", "signal_gap", "max_duration", "stream_end"]

_CONFIG_DOMAIN: Final = "signlab-candidate-event-config/1"
_PALM_LANDMARK_INDICES: Final = (0, 5, 9, 17)
_COORDINATE_QUANTIZATION: Final = 1_000_000
_PPM: Final = 1_000_000


class CandidateEventError(ValueError):
    """Raised for invalid configuration, input, or observation order."""


class CandidateEventConfigV1(StrictContractModel):
    """Portable engineering parameters for the candidate-event state machine."""

    format: Literal["signlab-candidate-event-config/1"]
    config_id: StableId
    version: SemanticVersion
    quality_policy_sha256: Sha256Digest
    smoothing_alpha_ppm: Annotated[PositiveSafeInteger, Field(le=_PPM)]
    start_motion_q: PositiveSafeInteger
    stop_motion_q: NonNegativeSafeInteger
    maximum_motion_q: Annotated[PositiveSafeInteger, Field(le=12_000_000)]
    pre_roll_us: PositiveSafeInteger
    arming_duration_us: PositiveSafeInteger
    minimum_event_duration_us: PositiveSafeInteger
    maximum_event_duration_us: PositiveSafeInteger
    finalization_duration_us: PositiveSafeInteger
    maximum_gap_frames: Annotated[NonNegativeSafeInteger, Field(le=2)]
    maximum_gap_us: Annotated[NonNegativeSafeInteger, Field(le=100_000)]
    cooldown_duration_us: PositiveSafeInteger

    @model_validator(mode="after")
    def _require_safe_hysteresis_and_policy_bounds(self) -> Self:
        from signlab.quality.resources import PUBLISHED_DEFAULT_QUALITY_POLICY_SEMANTIC_DIGEST

        if not self.stop_motion_q < self.start_motion_q < self.maximum_motion_q:
            raise ValueError("motion thresholds must be strictly ordered")
        if self.minimum_event_duration_us >= self.maximum_event_duration_us:
            raise ValueError("minimum event duration must be lower than maximum duration")
        if self.quality_policy_sha256 != PUBLISHED_DEFAULT_QUALITY_POLICY_SEMANTIC_DIGEST:
            raise ValueError("candidate-event config must bind the published quality policy")
        return self


def candidate_event_config_digest(config: CandidateEventConfigV1) -> str:
    """Return the portable identity of one validated detector configuration."""

    return canonical_sha256(config, domain=_CONFIG_DOMAIN)


def load_candidate_event_config(path: str | Path) -> CandidateEventConfigV1:
    """Load one canonical checked-in detector configuration."""

    try:
        raw = Path(path).read_bytes()
        config = CandidateEventConfigV1.model_validate_json(
            canonical_json_bytes(parse_json_object(raw)), strict=True
        )
        if raw != canonical_json_bytes(config) + b"\n":
            raise ValueError("candidate-event config is not canonical")
        return config
    except (CanonicalizationError, OSError, ValidationError, ValueError) as error:
        raise CandidateEventError("candidate-event config is invalid") from error


class CandidateObservation(StrictContractModel):
    """The small runtime-neutral signal consumed for one frame."""

    timestamp_us: NonNegativeSafeInteger
    hand_present: bool
    quality_ok: bool
    motion_q: NonNegativeSafeInteger


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    """One inclusive observed-sample boundary with configuration identity."""

    first_frame_index: int
    last_frame_index: int
    first_timestamp_us: int
    last_timestamp_us: int
    termination_reason: CandidateEndReason
    config_sha256: str


def _palm_centroid_q(hand: HandSlotV1) -> tuple[int, int]:
    if not hand.present or hand.image_landmarks is None:
        raise CandidateEventError("palm motion requires a present hand")
    points = tuple(hand.image_landmarks[index] for index in _PALM_LANDMARK_INDICES)
    return tuple(
        round(sum(getattr(point, axis) for point in points) * _COORDINATE_QUANTIZATION / 4)
        for axis in ("x", "y")
    )


def project_candidate_observations(
    table: LandmarkFramesTableV1,
) -> tuple[CandidateObservation, ...]:
    """Project validated raw landmark rows without using confidence as quality."""

    if not isinstance(table, LandmarkFramesTableV1):
        raise CandidateEventError("candidate projection requires a validated landmark table")
    previous: list[tuple[tuple[int, int], int] | None] = [None, None]
    observations: list[CandidateObservation] = []
    for frame in table.rows:
        if frame.invalid:
            observations.append(
                CandidateObservation(
                    timestamp_us=frame.relative_timestamp_us,
                    hand_present=False,
                    quality_ok=False,
                    motion_q=0,
                )
            )
            continue
        speeds: list[int] = []
        for slot_index, hand in enumerate(frame.hands):
            if not hand.present:
                previous[slot_index] = None
                continue
            current = _palm_centroid_q(hand)
            prior = previous[slot_index]
            previous[slot_index] = (current, frame.relative_timestamp_us)
            if prior is None:
                continue
            prior_point, prior_us = prior
            elapsed_us = frame.relative_timestamp_us - prior_us
            distance_q = sum(
                abs(right - left) for left, right in zip(prior_point, current, strict=True)
            )
            speeds.append((distance_q * _PPM + elapsed_us // 2) // elapsed_us)
        hand_present = frame.observed_hand_count > 0
        observations.append(
            CandidateObservation(
                timestamp_us=frame.relative_timestamp_us,
                hand_present=hand_present,
                quality_ok=True,
                motion_q=max(speeds, default=0),
            )
        )
    return tuple(observations)


class CandidateEventDetector:
    """A small online state machine; classification is intentionally out of scope."""

    def __init__(self, config: CandidateEventConfigV1) -> None:
        if not isinstance(config, CandidateEventConfigV1):
            raise CandidateEventError("candidate-event config must be validated")
        self.config = CandidateEventConfigV1.model_validate(config, strict=True)
        self.config_sha256 = candidate_event_config_digest(self.config)
        self.state: CandidateState = "inactive"
        self._next_index = 0
        self._last_timestamp_us: int | None = None
        self._smoothed_motion_q: int | None = None
        self._pre_roll: deque[tuple[int, int]] = deque()
        self._arm_since_us: int | None = None
        self._event_start: tuple[int, int] | None = None
        self._last_usable: tuple[int, int] | None = None
        self._last_active: tuple[int, int] | None = None
        self._quiet_since_us: int | None = None
        self._gap_frames = 0
        self._cooldown_quiet_since_us: int | None = None
        self._finished = False

    def push(self, observation: CandidateObservation) -> CandidateEvent | None:
        """Consume exactly one later observation and optionally emit one event."""

        if self._finished:
            raise CandidateEventError("candidate-event stream is already finished")
        if not isinstance(observation, CandidateObservation):
            raise CandidateEventError("candidate observation must be validated")
        if (
            self._last_timestamp_us is not None
            and observation.timestamp_us <= self._last_timestamp_us
        ):
            raise CandidateEventError("candidate timestamps must increase strictly")
        index = self._next_index
        self._next_index += 1
        self._last_timestamp_us = observation.timestamp_us
        usable = self._update_motion(observation)

        if self.state == "inactive":
            self._consume_inactive(observation, index, usable)
        elif self.state == "arming":
            self._consume_arming(observation, index, usable)
        elif self.state in {"recording", "finalizing"}:
            return self._consume_active(observation, index, usable)
        else:
            self._consume_cooldown(observation, usable)
        return None

    def finish(self) -> CandidateEvent | None:
        """Close a confirmed event at end-of-stream and discard incomplete arming."""

        if self._finished:
            return None
        self._finished = True
        event: CandidateEvent | None = None
        if self.state in {"recording", "finalizing"} and self._last_usable is not None:
            end = self._last_active if self.state == "finalizing" else self._last_usable
            if end is not None:
                event = self._build_event(end, "stream_end")
        self._reset("inactive", reset_motion=True)
        return event

    def _update_motion(self, observation: CandidateObservation) -> bool:
        usable = (
            observation.hand_present
            and observation.quality_ok
            and observation.motion_q <= self.config.maximum_motion_q
        )
        if usable:
            alpha = self.config.smoothing_alpha_ppm
            previous = self._smoothed_motion_q
            self._smoothed_motion_q = (
                observation.motion_q
                if previous is None
                else (alpha * observation.motion_q + (_PPM - alpha) * previous + _PPM // 2) // _PPM
            )
        return usable

    def _consume_inactive(
        self, observation: CandidateObservation, index: int, usable: bool
    ) -> None:
        if not usable:
            if not observation.hand_present:
                self._pre_roll.clear()
                self._last_usable = None
                self._smoothed_motion_q = None
            return
        current = (index, observation.timestamp_us)
        if (
            self._last_usable is not None
            and observation.timestamp_us - self._last_usable[1] > self.config.maximum_gap_us
        ):
            self._pre_roll.clear()
            self._smoothed_motion_q = 0
            self._last_usable = current
            self._pre_roll.append(current)
            return
        self._last_usable = current
        self._pre_roll.append((index, observation.timestamp_us))
        while observation.timestamp_us - self._pre_roll[0][1] > self.config.pre_roll_us:
            self._pre_roll.popleft()
        if self._motion_at_least(self.config.start_motion_q):
            self._event_start = self._pre_roll[0]
            self._arm_since_us = observation.timestamp_us
            self._last_usable = (index, observation.timestamp_us)
            self._gap_frames = 0
            self.state = "arming"

    def _consume_arming(self, observation: CandidateObservation, index: int, usable: bool) -> None:
        if not usable:
            self._gap_frames += 1
            if self._hard_gap(observation.timestamp_us):
                self._reset("inactive", reset_motion=True)
            return
        if self._hard_gap(observation.timestamp_us):
            self._reset("inactive", reset_motion=True)
            return
        self._gap_frames = 0
        if self._motion_at_most(self.config.stop_motion_q):
            self._reset("inactive", reset_motion=True)
            return
        if self._arm_since_us is None or self._event_start is None:
            raise RuntimeError("arming state lost its boundary")
        if observation.timestamp_us - self._arm_since_us >= self.config.arming_duration_us:
            self._last_usable = (index, observation.timestamp_us)
            self._last_active = (index, observation.timestamp_us)
            self.state = "recording"
        else:
            self._last_usable = (index, observation.timestamp_us)

    def _consume_active(
        self, observation: CandidateObservation, index: int, usable: bool
    ) -> CandidateEvent | None:
        if self._event_start is None or self._last_usable is None:
            raise RuntimeError("active state lost its event boundary")
        deadline_us = self._event_start[1] + self.config.maximum_event_duration_us
        if not usable:
            self._gap_frames += 1
            if self._hard_gap(observation.timestamp_us):
                return self._terminate(self._last_usable, "signal_gap", observation, usable)
            if observation.timestamp_us > deadline_us:
                return self._terminate(self._last_usable, "max_duration", observation, usable)
            return None
        if self._hard_gap(observation.timestamp_us):
            return self._terminate(self._last_usable, "signal_gap", observation, usable)
        if observation.timestamp_us > deadline_us:
            return self._terminate(self._last_usable, "max_duration", observation, usable)
        self._gap_frames = 0
        current = (index, observation.timestamp_us)
        self._last_usable = current
        if observation.timestamp_us == deadline_us:
            return self._terminate(current, "max_duration", observation, usable)

        if self.state == "recording":
            if self._motion_at_most(self.config.stop_motion_q):
                self._quiet_since_us = observation.timestamp_us
                self.state = "finalizing"
            else:
                self._last_active = current
            return None

        if self._quiet_since_us is None or self._last_active is None:
            raise RuntimeError("finalizing state lost its boundary")
        if observation.timestamp_us - self._quiet_since_us >= self.config.finalization_duration_us:
            return self._terminate(self._last_active, "settled", observation, usable)
        if self._motion_at_least(self.config.start_motion_q):
            self._last_active = current
            self._quiet_since_us = None
            self.state = "recording"
        return None

    def _consume_cooldown(self, observation: CandidateObservation, usable: bool) -> None:
        stable_quiet = observation.quality_ok and (
            not observation.hand_present
            or (usable and self._motion_at_most(self.config.stop_motion_q))
        )
        if not stable_quiet:
            self._cooldown_quiet_since_us = None
            return
        if self._cooldown_quiet_since_us is None:
            self._cooldown_quiet_since_us = observation.timestamp_us
        if (
            observation.timestamp_us - self._cooldown_quiet_since_us
            >= self.config.cooldown_duration_us
        ):
            self._reset("inactive", reset_motion=True)

    def _hard_gap(self, timestamp_us: int) -> bool:
        reference = self._last_usable or self._event_start
        return reference is None or (
            self._gap_frames > self.config.maximum_gap_frames
            or timestamp_us - reference[1] > self.config.maximum_gap_us
        )

    def _motion_at_least(self, threshold: int) -> bool:
        return self._smoothed_motion_q is not None and self._smoothed_motion_q >= threshold

    def _motion_at_most(self, threshold: int) -> bool:
        return self._smoothed_motion_q is not None and self._smoothed_motion_q <= threshold

    def _terminate(
        self,
        end: tuple[int, int],
        reason: CandidateEndReason,
        observation: CandidateObservation,
        usable: bool,
    ) -> CandidateEvent | None:
        event = self._build_event(end, reason)
        self._reset("cooldown")
        if observation.quality_ok and (
            not observation.hand_present
            or (usable and self._motion_at_most(self.config.stop_motion_q))
        ):
            self._cooldown_quiet_since_us = observation.timestamp_us
        return event

    def _build_event(
        self, end: tuple[int, int], reason: CandidateEndReason
    ) -> CandidateEvent | None:
        if self._event_start is None:
            raise RuntimeError("event boundary is unavailable")
        first_index, first_us = self._event_start
        last_index, last_us = end
        if last_us - first_us < self.config.minimum_event_duration_us:
            return None
        return CandidateEvent(
            first_frame_index=first_index,
            last_frame_index=last_index,
            first_timestamp_us=first_us,
            last_timestamp_us=last_us,
            termination_reason=reason,
            config_sha256=self.config_sha256,
        )

    def _reset(self, state: CandidateState, *, reset_motion: bool = False) -> None:
        self.state = state
        self._pre_roll.clear()
        self._arm_since_us = None
        self._event_start = None
        self._last_usable = None
        self._last_active = None
        self._quiet_since_us = None
        self._gap_frames = 0
        self._cooldown_quiet_since_us = None
        if reset_motion:
            self._smoothed_motion_q = None


__all__ = [
    "CandidateEndReason",
    "CandidateEvent",
    "CandidateEventConfigV1",
    "CandidateEventDetector",
    "CandidateEventError",
    "CandidateObservation",
    "CandidateState",
    "candidate_event_config_digest",
    "load_candidate_event_config",
    "project_candidate_observations",
]
