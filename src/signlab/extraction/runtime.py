"""Fail-closed adapters for deterministic MediaPipe video extraction.

This module owns the vendor boundary only: model-byte verification, presentation-time
decoding, MediaPipe task construction, and normalization into SignLab types.  The
optional native dependencies are deliberately imported only when the boundary is used.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import os
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    HAND_LANDMARK_COUNT,
    MEDIAPIPE_PACKAGE_VERSION,
)
from signlab.extraction.types import HandDetection, LandmarkPoint

MEDIAPIPE_VERSION = MEDIAPIPE_PACKAGE_VERSION
AV_VERSION = "18.1.0"


@dataclass(frozen=True, slots=True)
class ModelAssetSpec:
    """The immutable byte identity expected for one external task model."""

    filename: str
    size_bytes: int
    sha256: str


HAND_MODEL_SPEC = ModelAssetSpec(
    filename="hand_landmarker.task",
    size_bytes=7_819_105,
    sha256="sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
)
POSE_MODEL_SPEC = ModelAssetSpec(
    filename="pose_landmarker_lite.task",
    size_bytes=5_777_746,
    sha256="sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
)

type RuntimeErrorCode = Literal[
    "extraction.config.invalid",
    "extraction.decoder.invalid",
    "extraction.decoder.timestamp.invalid",
    "extraction.dependency.unavailable",
    "extraction.dependency.version.invalid",
    "extraction.models.invalid",
    "extraction.result.invalid",
    "extraction.runtime.closed",
    "extraction.runtime.inference.failed",
    "extraction.runtime.initialization.failed",
]

_ERROR_MESSAGES: dict[RuntimeErrorCode, str] = {
    "extraction.config.invalid": "extraction runtime configuration is invalid",
    "extraction.decoder.invalid": "video could not be decoded safely",
    "extraction.decoder.timestamp.invalid": "video timestamps are invalid",
    "extraction.dependency.unavailable": "an extraction runtime dependency is unavailable",
    "extraction.dependency.version.invalid": "the MediaPipe runtime version is unsupported",
    "extraction.models.invalid": "extraction model assets could not be verified",
    "extraction.result.invalid": "the extraction runtime returned an invalid result",
    "extraction.runtime.closed": "the extraction runtime is closed",
    "extraction.runtime.inference.failed": "frame inference failed",
    "extraction.runtime.initialization.failed": "the extraction runtime could not be initialized",
}


class ExtractionRuntimeError(ValueError):
    """A stable, path-free error raised by the optional extraction boundary."""

    def __init__(self, code: RuntimeErrorCode) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class VerifiedModelAssets:
    """Verified task bytes detached from their private filesystem location."""

    hand_model_bytes: bytes = field(repr=False)
    pose_model_bytes: bytes = field(repr=False)
    hand_model_sha256: str
    pose_model_sha256: str


@dataclass(frozen=True, slots=True)
class ExtractionRuntimeConfig:
    """Explicit detector thresholds; execution shape is intentionally fixed."""

    hand_detection_confidence: float = 0.5
    hand_presence_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
    pose_detection_confidence: float = 0.5
    pose_presence_confidence: float = 0.5
    pose_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.hand_detection_confidence,
            self.hand_presence_confidence,
            self.hand_tracking_confidence,
            self.pose_detection_confidence,
            self.pose_presence_confidence,
            self.pose_tracking_confidence,
        )
        if any(not _is_unit_interval(value) for value in values):
            raise ExtractionRuntimeError("extraction.config.invalid")


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One RGB frame with exact source and derived task timing."""

    frame_index: int
    source_pts: int
    source_time_base_numerator: int
    source_time_base_denominator: int
    relative_timestamp_us: int
    task_timestamp_ms: int
    rgb: object = field(repr=False)
    source_valid: bool = True


type PoseAnchorName = Literal[
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


@dataclass(frozen=True, slots=True)
class PoseAnchorObservation:
    """Image and metric coordinates for one selected upper-body anchor."""

    name: PoseAnchorName
    landmark_index: int
    image_landmark: LandmarkPoint
    world_landmark: LandmarkPoint


@dataclass(frozen=True, slots=True)
class FrameInference:
    """Normalized detector output joined to path-free frame provenance."""

    frame_index: int
    source_pts: int
    source_time_base_numerator: int
    source_time_base_denominator: int
    relative_timestamp_us: int
    task_timestamp_ms: int
    hands: tuple[HandDetection, ...]
    pose_anchors: tuple[PoseAnchorObservation, ...]


_POSE_ANCHORS: tuple[tuple[PoseAnchorName, int], ...] = (
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
)

assert tuple(name for name, _ in _POSE_ANCHORS) == BODY_ANCHOR_NAMES


def _is_unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _is_link(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _verify_asset(root: Path, spec: ModelAssetSpec) -> bytes:
    candidate = root / spec.filename
    if _is_link(candidate) or not candidate.is_file():
        raise ExtractionRuntimeError("extraction.models.invalid")
    if candidate.stat().st_size != spec.size_bytes:
        raise ExtractionRuntimeError("extraction.models.invalid")
    content = candidate.read_bytes()
    digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
    if len(content) != spec.size_bytes or digest != spec.sha256:
        raise ExtractionRuntimeError("extraction.models.invalid")
    return content


def verify_model_assets(model_root: str | Path) -> VerifiedModelAssets:
    """Read and verify the two registered external model assets without networking."""

    try:
        root = Path(model_root)
        if _is_link(root) or not root.is_dir():
            raise ExtractionRuntimeError("extraction.models.invalid")
        hand = _verify_asset(root, HAND_MODEL_SPEC)
        pose = _verify_asset(root, POSE_MODEL_SPEC)
        return VerifiedModelAssets(
            hand_model_bytes=hand,
            pose_model_bytes=pose,
            hand_model_sha256=HAND_MODEL_SPEC.sha256,
            pose_model_sha256=POSE_MODEL_SPEC.sha256,
        )
    except ExtractionRuntimeError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionRuntimeError("extraction.models.invalid") from None


def _installed_mediapipe_version() -> str:
    return importlib.metadata.version("mediapipe")


def _installed_av_version() -> str:
    return importlib.metadata.version("av")


def _load_mediapipe_module() -> Any:
    try:
        if _installed_mediapipe_version() != MEDIAPIPE_VERSION:
            raise ExtractionRuntimeError("extraction.dependency.version.invalid")
        return cast(Any, importlib.import_module("mediapipe"))
    except ExtractionRuntimeError:
        raise
    except (ImportError, importlib.metadata.PackageNotFoundError):
        raise ExtractionRuntimeError("extraction.dependency.unavailable") from None
    except Exception:
        raise ExtractionRuntimeError("extraction.dependency.unavailable") from None


def _load_av_module() -> Any:
    try:
        if _installed_av_version() != AV_VERSION:
            raise ExtractionRuntimeError("extraction.dependency.version.invalid")
        return cast(Any, importlib.import_module("av"))
    except ExtractionRuntimeError:
        raise
    except Exception:
        raise ExtractionRuntimeError("extraction.dependency.unavailable") from None


def _frame_timing(frame: Any) -> tuple[int, int, int, Fraction]:
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    numerator = getattr(time_base, "numerator", None)
    denominator = getattr(time_base, "denominator", None)
    if (
        not isinstance(pts, int)
        or isinstance(pts, bool)
        or not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or numerator <= 0
        or denominator <= 0
    ):
        raise ExtractionRuntimeError("extraction.decoder.timestamp.invalid")
    return pts, numerator, denominator, Fraction(numerator, denominator)


def iter_decoded_frames(source: str | Path) -> Iterator[DecodedFrame]:
    """Decode one video stream in presentation order with integer-derived timing."""

    av_module = _load_av_module()
    try:
        with av_module.open(str(Path(source)), mode="r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1:
                raise ExtractionRuntimeError("extraction.decoder.invalid")
            origin: Fraction | None = None
            previous_presentation: Fraction | None = None
            previous_task_ms = -1
            for frame_index, vendor_frame in enumerate(container.decode(streams[0])):
                pts, time_base_numerator, time_base_denominator, time_base = _frame_timing(
                    vendor_frame
                )
                presentation = pts * time_base
                if previous_presentation is not None and presentation <= previous_presentation:
                    raise ExtractionRuntimeError("extraction.decoder.timestamp.invalid")
                if origin is None:
                    origin = presentation
                relative = presentation - origin
                relative_us = relative.numerator * 1_000_000 // relative.denominator
                natural_task_ms = relative_us // 1_000
                task_timestamp_ms = max(natural_task_ms, previous_task_ms + 1)
                try:
                    rgb = vendor_frame.to_ndarray(format="rgb24")
                    source_valid = True
                except Exception:
                    # PTS/time-base provenance remains usable even when a decoded
                    # frame cannot be converted to the canonical RGB input. The
                    # batch layer records an explicit source-frame invalid mask.
                    rgb = None
                    source_valid = False
                yield DecodedFrame(
                    frame_index=frame_index,
                    source_pts=pts,
                    source_time_base_numerator=time_base_numerator,
                    source_time_base_denominator=time_base_denominator,
                    relative_timestamp_us=relative_us,
                    task_timestamp_ms=task_timestamp_ms,
                    rgb=rgb,
                    source_valid=source_valid,
                )
                previous_presentation = presentation
                previous_task_ms = task_timestamp_ms
            if origin is None:
                raise ExtractionRuntimeError("extraction.decoder.invalid")
    except ExtractionRuntimeError:
        raise
    except Exception:
        raise ExtractionRuntimeError("extraction.decoder.invalid") from None


def _sequence(value: object) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ExtractionRuntimeError("extraction.result.invalid")
    return value


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtractionRuntimeError("extraction.result.invalid")
    converted = float(value)
    if not math.isfinite(converted):
        raise ExtractionRuntimeError("extraction.result.invalid")
    return converted


def _optional_finite_float(value: object) -> float | None:
    return None if value is None else _finite_float(value)


def _normalize_landmark(vendor_landmark: object) -> LandmarkPoint:
    try:
        landmark = cast(Any, vendor_landmark)
        return LandmarkPoint(
            x=_finite_float(landmark.x),
            y=_finite_float(landmark.y),
            z=_finite_float(landmark.z),
            visibility=_optional_finite_float(getattr(landmark, "visibility", None)),
            presence=_optional_finite_float(getattr(landmark, "presence", None)),
        )
    except ExtractionRuntimeError:
        raise
    except (AttributeError, TypeError, ValueError):
        raise ExtractionRuntimeError("extraction.result.invalid") from None


def _normalize_landmark_list(value: object, *, expected_count: int) -> tuple[LandmarkPoint, ...]:
    vendor_landmarks = _sequence(value)
    if len(vendor_landmarks) != expected_count:
        raise ExtractionRuntimeError("extraction.result.invalid")
    return tuple(_normalize_landmark(item) for item in vendor_landmarks)


def normalize_hand_result(result: object) -> tuple[HandDetection, ...]:
    """Normalize one MediaPipe hand result without retaining vendor objects."""

    try:
        vendor_result = cast(Any, result)
        image_hands = _sequence(vendor_result.hand_landmarks)
        world_hands = _sequence(vendor_result.hand_world_landmarks)
        handedness = _sequence(vendor_result.handedness)
        if len(image_hands) > 2 or not (len(image_hands) == len(world_hands) == len(handedness)):
            raise ExtractionRuntimeError("extraction.result.invalid")
        normalized: list[HandDetection] = []
        for detector_index, (image, world, categories_value) in enumerate(
            zip(image_hands, world_hands, handedness, strict=True)
        ):
            categories = _sequence(categories_value)
            if not categories:
                raise ExtractionRuntimeError("extraction.result.invalid")
            category = categories[0]
            raw_label = getattr(category, "category_name", None)
            if not isinstance(raw_label, str) or raw_label.casefold() not in {"left", "right"}:
                raise ExtractionRuntimeError("extraction.result.invalid")
            label = cast(Literal["left", "right"], raw_label.casefold())
            score = _finite_float(getattr(category, "score", None))
            if not 0.0 <= score <= 1.0:
                raise ExtractionRuntimeError("extraction.result.invalid")
            normalized.append(
                HandDetection(
                    detector_index=detector_index,
                    image_landmarks=_normalize_landmark_list(
                        image, expected_count=HAND_LANDMARK_COUNT
                    ),
                    world_landmarks=_normalize_landmark_list(
                        world, expected_count=HAND_LANDMARK_COUNT
                    ),
                    reported_handedness=label,
                    handedness_score=score,
                )
            )
        return tuple(normalized)
    except ExtractionRuntimeError:
        raise
    except (AttributeError, TypeError, ValueError):
        raise ExtractionRuntimeError("extraction.result.invalid") from None


def normalize_pose_result(result: object) -> tuple[PoseAnchorObservation, ...]:
    """Select and normalize the six upper-body pose anchors used by SignLab."""

    try:
        vendor_result = cast(Any, result)
        image_poses = _sequence(vendor_result.pose_landmarks)
        world_poses = _sequence(vendor_result.pose_world_landmarks)
        if len(image_poses) != len(world_poses) or len(image_poses) > 1:
            raise ExtractionRuntimeError("extraction.result.invalid")
        if not image_poses:
            return ()
        image = _normalize_landmark_list(image_poses[0], expected_count=33)
        world = _normalize_landmark_list(world_poses[0], expected_count=33)
        return tuple(
            PoseAnchorObservation(
                name=name,
                landmark_index=index,
                image_landmark=image[index],
                world_landmark=world[index],
            )
            for name, index in _POSE_ANCHORS
        )
    except ExtractionRuntimeError:
        raise
    except (AttributeError, TypeError, ValueError):
        raise ExtractionRuntimeError("extraction.result.invalid") from None


class MediaPipeVideoRuntime:
    """CPU-only MediaPipe Tasks VIDEO runtime with verified in-memory assets."""

    def __init__(
        self,
        assets: VerifiedModelAssets,
        config: ExtractionRuntimeConfig | None = None,
    ) -> None:
        self._assets = assets
        self._config = config or ExtractionRuntimeConfig()
        self._mp: Any | None = None
        self._hand_landmarker: Any | None = None
        self._pose_landmarker: Any | None = None
        self._closed = False

    def __enter__(self) -> MediaPipeVideoRuntime:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    def _require_asset_identity(self) -> None:
        try:
            for content, claimed_digest, spec in (
                (
                    self._assets.hand_model_bytes,
                    self._assets.hand_model_sha256,
                    HAND_MODEL_SPEC,
                ),
                (
                    self._assets.pose_model_bytes,
                    self._assets.pose_model_sha256,
                    POSE_MODEL_SPEC,
                ),
            ):
                actual_digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
                if (
                    len(content) != spec.size_bytes
                    or claimed_digest != spec.sha256
                    or actual_digest != spec.sha256
                ):
                    raise ExtractionRuntimeError("extraction.models.invalid")
        except ExtractionRuntimeError:
            raise
        except (AttributeError, TypeError, ValueError):
            raise ExtractionRuntimeError("extraction.models.invalid") from None

    def open(self) -> None:
        """Initialize both MediaPipe tasks exactly once."""

        if self._closed:
            raise ExtractionRuntimeError("extraction.runtime.closed")
        if self._hand_landmarker is not None and self._pose_landmarker is not None:
            return
        self._require_asset_identity()
        mp = _load_mediapipe_module()
        hand_landmarker: Any | None = None
        try:
            base_options = mp.tasks.BaseOptions
            delegate = base_options.Delegate.CPU
            running_mode = mp.tasks.vision.RunningMode.VIDEO
            hand_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options(
                    model_asset_buffer=self._assets.hand_model_bytes,
                    delegate=delegate,
                ),
                running_mode=running_mode,
                num_hands=2,
                min_hand_detection_confidence=self._config.hand_detection_confidence,
                min_hand_presence_confidence=self._config.hand_presence_confidence,
                min_tracking_confidence=self._config.hand_tracking_confidence,
            )
            pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options(
                    model_asset_buffer=self._assets.pose_model_bytes,
                    delegate=delegate,
                ),
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=self._config.pose_detection_confidence,
                min_pose_presence_confidence=self._config.pose_presence_confidence,
                min_tracking_confidence=self._config.pose_tracking_confidence,
                output_segmentation_masks=False,
            )
            hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
            pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        except Exception:
            if hand_landmarker is not None:
                with suppress(Exception):
                    hand_landmarker.close()
            raise ExtractionRuntimeError("extraction.runtime.initialization.failed") from None
        self._mp = mp
        self._hand_landmarker = hand_landmarker
        self._pose_landmarker = pose_landmarker

    def infer_frame(self, frame: DecodedFrame) -> FrameInference:
        """Run both tasks for a decoded frame and discard all vendor result objects."""

        if not frame.source_valid or frame.rgb is None:
            raise ExtractionRuntimeError("extraction.result.invalid")
        if self._closed:
            raise ExtractionRuntimeError("extraction.runtime.closed")
        if self._hand_landmarker is None or self._pose_landmarker is None or self._mp is None:
            self.open()
        mp = self._mp
        hand_landmarker = self._hand_landmarker
        pose_landmarker = self._pose_landmarker
        if mp is None or hand_landmarker is None or pose_landmarker is None:
            raise ExtractionRuntimeError("extraction.runtime.initialization.failed")
        try:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.rgb)
            hand_result = hand_landmarker.detect_for_video(image, frame.task_timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(image, frame.task_timestamp_ms)
            hands = normalize_hand_result(hand_result)
            pose_anchors = normalize_pose_result(pose_result)
        except ExtractionRuntimeError:
            raise
        except Exception:
            raise ExtractionRuntimeError("extraction.runtime.inference.failed") from None
        return FrameInference(
            frame_index=frame.frame_index,
            source_pts=frame.source_pts,
            source_time_base_numerator=frame.source_time_base_numerator,
            source_time_base_denominator=frame.source_time_base_denominator,
            relative_timestamp_us=frame.relative_timestamp_us,
            task_timestamp_ms=frame.task_timestamp_ms,
            hands=hands,
            pose_anchors=pose_anchors,
        )

    def close(self) -> None:
        """Close both native tasks; repeated calls are safe."""

        if self._closed:
            return
        self._closed = True
        for landmarker in (self._hand_landmarker, self._pose_landmarker):
            if landmarker is not None:
                with suppress(Exception):
                    landmarker.close()
        self._hand_landmarker = None
        self._pose_landmarker = None
        self._mp = None


__all__ = [
    "AV_VERSION",
    "HAND_MODEL_SPEC",
    "MEDIAPIPE_VERSION",
    "POSE_MODEL_SPEC",
    "DecodedFrame",
    "ExtractionRuntimeConfig",
    "ExtractionRuntimeError",
    "FrameInference",
    "MediaPipeVideoRuntime",
    "ModelAssetSpec",
    "PoseAnchorName",
    "PoseAnchorObservation",
    "VerifiedModelAssets",
    "iter_decoded_frames",
    "normalize_hand_result",
    "normalize_pose_result",
    "verify_model_assets",
]
