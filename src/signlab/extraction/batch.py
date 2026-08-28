"""Synthetic-eligibility-gated, atomic publication of raw landmark observations."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, Protocol, cast

from signlab.contracts.canonical import CanonicalizationError, canonical_json_bytes
from signlab.contracts.core import ArtifactRefV1, WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import DerivedArtifactRowV1, RecordingRowV1, RecordingsTableV1
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    BodyAnchorName,
    BodyAnchorV1,
    ExtractionContractError,
    HandSlotId,
    HandSlotV1,
    LandmarkExtractionManifestV1,
    LandmarkFramesTableV1,
    LandmarkFrameV1,
    LandmarkSequenceRefV1,
    MediaPipeExtractionConfigV1,
    Point3V1,
    assert_landmark_extraction_bound_to_raw_dataset,
    assert_landmark_sequence_ref_matches_table,
    landmark_extraction_manifest_digest,
    landmark_observation_counts,
    mediapipe_extraction_config_digest,
    raw_dataset_manifest_digest,
    validate_landmark_extraction_manifest,
    validate_mediapipe_extraction_config,
)
from signlab.contracts.ingest import RawDatasetManifestV1
from signlab.datasets.raw_bundle import (
    RawDatasetBundleError,
    RawDatasetManifestInput,
    ValidatedRawDatasetBundle,
    validate_raw_dataset_bundle,
)
from signlab.extraction.parquet import (
    LandmarkParquetError,
    read_landmark_frames,
    write_landmark_frames,
)
from signlab.extraction.runtime import (
    DecodedFrame,
    ExtractionRuntimeConfig,
    ExtractionRuntimeError,
    FrameInference,
    MediaPipeVideoRuntime,
    PoseAnchorObservation,
    VerifiedModelAssets,
    iter_decoded_frames,
    verify_model_assets,
)
from signlab.extraction.tracking import HandIdentityTracker, HandTrackingConfig
from signlab.extraction.types import (
    HandDetection,
    HandTrackingResult,
    LandmarkPoint,
    TrackedHand,
)

LANDMARK_EXTRACTION_MANIFEST_FILENAME: Final = "landmark-extraction-manifest.json"
LANDMARK_EXTRACTION_ID: Final = "mediapipe_tasks_landmark_extraction"
LANDMARK_EXTRACTION_VERSION: Final = "1.0.0"
_MAX_MANIFEST_BYTES: Final = 16 * 1024 * 1024
_SOURCE_HASH_CHUNK_BYTES: Final = 1024 * 1024

type _SourceFileIdentity = tuple[int, int, int, int, int]

type ExtractionBatchStatus = Literal["published", "unchanged"]
type ExtractionManifestInput = (
    LandmarkExtractionManifestV1 | str | bytes | bytearray | Mapping[str, object]
)
type ExtractionConfigInput = (
    MediaPipeExtractionConfigV1 | str | bytes | bytearray | Mapping[str, object]
)
type ExtractionBatchErrorCategory = Literal[
    "bundle.invalid",
    "config.invalid",
    "consent.unauthorized",
    "destination.conflict",
    "destination.invalid",
    "execution.failed",
    "models.invalid",
    "publication.failed",
    "raw_bundle.invalid",
]

_ERROR_MESSAGES: Final[dict[ExtractionBatchErrorCategory, str]] = {
    "bundle.invalid": "landmark extraction bundle is invalid",
    "config.invalid": "landmark extraction configuration is invalid",
    "consent.unauthorized": "landmark extraction is not authorized",
    "destination.conflict": "landmark extraction destination conflicts with this run",
    "destination.invalid": "landmark extraction destination is invalid",
    "execution.failed": "landmark extraction could not be completed",
    "models.invalid": "landmark extraction model assets are invalid",
    "publication.failed": "landmark extraction bundle could not be published",
    "raw_bundle.invalid": "raw dataset bundle is invalid for landmark extraction",
}


class ExtractionBatchError(ValueError):
    """A stable, path-free batch extraction or validation failure."""

    def __init__(self, category: ExtractionBatchErrorCategory) -> None:
        self.category = category
        self.code = f"extraction.batch.{category}"
        super().__init__(_ERROR_MESSAGES[category])


@dataclass(frozen=True, slots=True)
class ExtractionBundleValidationResult:
    """Positive evidence returned only after every bundle layer is reconciled."""

    raw_bundle_integrity: Literal["verified"]
    consent_boundary: Literal["synthetic_fixture_only"]
    manifest_integrity: Literal["verified"]
    registered_model_identity: Literal["verified"]
    parquet_byte_integrity: Literal["verified"]
    semantic_integrity: Literal["verified"]
    exact_inventory: Literal["verified"]
    sequence_count: int
    frame_count: int
    invalid_frame_count: int


@dataclass(frozen=True, slots=True)
class ValidatedLandmarkExtractionBundle:
    """A fully checked manifest and its decoded, path-free semantic tables."""

    manifest: LandmarkExtractionManifestV1
    tables: Mapping[str, LandmarkFramesTableV1]
    validation: ExtractionBundleValidationResult


@dataclass(frozen=True, slots=True)
class ExtractionBatchResult:
    """The published or already-identical landmark extraction bundle."""

    status: ExtractionBatchStatus
    manifest: LandmarkExtractionManifestV1
    manifest_path: Path
    validation: ExtractionBundleValidationResult


class VideoInferenceRuntime(Protocol):
    """Small injectable surface used by the batch service."""

    def __enter__(self) -> VideoInferenceRuntime: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None: ...

    def infer_frame(self, frame: DecodedFrame) -> FrameInference: ...


class BatchHandTracker(Protocol):
    """Injectable deterministic tracking surface."""

    def reset(self) -> None: ...

    def track(self, detections: tuple[HandDetection, ...]) -> HandTrackingResult: ...


type DecoderFactory = Callable[[str | Path], Iterable[DecodedFrame]]
type RuntimeFactory = Callable[
    [VerifiedModelAssets, ExtractionRuntimeConfig], VideoInferenceRuntime
]
type TrackerFactory = Callable[[HandTrackingConfig], BatchHandTracker]


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _is_same_or_descendant(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
    except ValueError:
        return False
    return True


def _nonoverlapping_extraction_paths(
    raw_bundle_root: str | Path,
    destination: str | Path,
) -> tuple[Path, Path]:
    try:
        raw_input = Path(raw_bundle_root)
        if _is_linklike(raw_input):
            raise ExtractionBatchError("raw_bundle.invalid")
        raw_root = raw_input.resolve(strict=True)
        if not raw_root.is_dir():
            raise ExtractionBatchError("raw_bundle.invalid")
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("raw_bundle.invalid") from None
    try:
        output_input = Path(destination)
        if _is_linklike(output_input):
            raise ExtractionBatchError("destination.invalid")
        output = output_input.resolve(strict=False)
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("destination.invalid") from None
    if _is_same_or_descendant(output, raw_root) or _is_same_or_descendant(raw_root, output):
        raise ExtractionBatchError("destination.invalid")
    return raw_root, output


def _recordings(raw: ValidatedRawDatasetBundle) -> tuple[RecordingRowV1, ...]:
    table = raw.tables.get("recordings")
    if not isinstance(table, RecordingsTableV1):
        raise ExtractionBatchError("raw_bundle.invalid")
    rows = table.rows
    if not rows or tuple(row.recording_id for row in rows) != tuple(
        sorted({row.recording_id for row in rows})
    ):
        raise ExtractionBatchError("raw_bundle.invalid")
    return rows


def _require_eligible_synthetic_fixture(
    raw: ValidatedRawDatasetBundle,
    recordings: tuple[RecordingRowV1, ...],
) -> None:
    # The public boundary intentionally has no caller-supplied authorization bypass.
    if raw.sidecar.fixture_only is not True:
        raise ExtractionBatchError("consent.unauthorized")
    if any(not recording.consent_grant.scope.derived_features for recording in recordings):
        raise ExtractionBatchError("consent.unauthorized")


def _validate_raw_first(
    raw_manifest: RawDatasetManifestInput,
    raw_bundle_root: str | Path,
) -> tuple[ValidatedRawDatasetBundle, tuple[RecordingRowV1, ...]]:
    try:
        raw = validate_raw_dataset_bundle(raw_manifest, raw_bundle_root)
        recordings = _recordings(raw)
        _require_eligible_synthetic_fixture(raw, recordings)
        return raw, recordings
    except ExtractionBatchError:
        raise
    except (RawDatasetBundleError, OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("raw_bundle.invalid") from None


def _validate_config(config: ExtractionConfigInput) -> MediaPipeExtractionConfigV1:
    try:
        return validate_mediapipe_extraction_config(config)
    except (ExtractionContractError, TypeError, ValueError):
        raise ExtractionBatchError("config.invalid") from None


def _runtime_config(config: MediaPipeExtractionConfigV1) -> ExtractionRuntimeConfig:
    return ExtractionRuntimeConfig(
        hand_detection_confidence=config.min_hand_detection_confidence,
        hand_presence_confidence=config.min_hand_presence_confidence,
        hand_tracking_confidence=config.min_hand_tracking_confidence,
        pose_detection_confidence=config.min_pose_detection_confidence,
        pose_presence_confidence=config.min_pose_presence_confidence,
        pose_tracking_confidence=config.min_pose_tracking_confidence,
    )


def _tracking_config(config: MediaPipeExtractionConfigV1) -> HandTrackingConfig:
    return HandTrackingConfig(
        max_spatial_cost=config.max_spatial_cost,
        handedness_disagreement_penalty=config.handedness_disagreement_penalty,
        ambiguity_margin=config.ambiguity_margin,
    )


def _verify_models(
    model_root: str | Path,
    config: MediaPipeExtractionConfigV1,
) -> VerifiedModelAssets:
    try:
        assets = verify_model_assets(model_root)
        if (
            assets.hand_model_sha256 != config.hand_task_asset.sha256
            or assets.pose_model_sha256 != config.pose_task_asset.sha256
            or len(assets.hand_model_bytes) != config.hand_task_asset.size_bytes
            or len(assets.pose_model_bytes) != config.pose_task_asset.size_bytes
        ):
            raise ExtractionBatchError("models.invalid")
        return assets
    except ExtractionBatchError:
        raise
    except (ExtractionRuntimeError, OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("models.invalid") from None


def _derived_artifact_id(recording_id: str, config_sha256: str) -> str:
    identity = f"landmark-extraction/1\0{recording_id}\0{config_sha256}".encode("ascii")
    return f"derived_artifact_{hashlib.sha256(identity).hexdigest()[:32]}"


def _point(point: LandmarkPoint) -> Point3V1:
    return Point3V1(
        x=point.x,
        y=point.y,
        z=point.z,
        visibility=point.visibility,
        presence=point.presence,
    )


def _absent_hand(slot_id: HandSlotId) -> HandSlotV1:
    return HandSlotV1(
        slot_id=slot_id,
        present=False,
        detector_index=None,
        tracking_id=None,
        handedness=None,
        handedness_confidence=None,
        image_landmarks=None,
        world_landmarks=None,
    )


def _hand_slot(slot: TrackedHand) -> HandSlotV1:
    detection = slot.detection
    if detection is None:
        return _absent_hand(slot.slot_id)
    if (
        detection.reported_handedness is None
        or detection.handedness_score is None
        or len(detection.world_landmarks) != len(detection.image_landmarks)
    ):
        raise ExtractionBatchError("execution.failed")
    return HandSlotV1(
        slot_id=slot.slot_id,
        present=True,
        detector_index=detection.detector_index,
        tracking_id=slot.slot_id,
        handedness=detection.reported_handedness,
        handedness_confidence=detection.handedness_score,
        image_landmarks=tuple(_point(point) for point in detection.image_landmarks),
        world_landmarks=tuple(_point(point) for point in detection.world_landmarks),
    )


def _hand_slots(tracking: HandTrackingResult) -> tuple[HandSlotV1, HandSlotV1]:
    return (_hand_slot(tracking.slots[0]), _hand_slot(tracking.slots[1]))


def _absent_anchors() -> tuple[
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
]:
    return cast(
        tuple[
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
        ],
        tuple(
            BodyAnchorV1(name=name, present=False, image_point=None, world_point=None)
            for name in BODY_ANCHOR_NAMES
        ),
    )


def _body_anchors(
    observations: tuple[PoseAnchorObservation, ...],
) -> tuple[
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
    BodyAnchorV1,
]:
    by_name: dict[BodyAnchorName, PoseAnchorObservation] = {}
    for observation in observations:
        name = observation.name
        if name not in BODY_ANCHOR_NAMES or name in by_name:
            raise ExtractionBatchError("execution.failed")
        by_name[name] = observation
    return cast(
        tuple[
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
            BodyAnchorV1,
        ],
        tuple(
            BodyAnchorV1(
                name=name,
                present=name in by_name,
                image_point=_point(by_name[name].image_landmark) if name in by_name else None,
                world_point=_point(by_name[name].world_landmark) if name in by_name else None,
            )
            for name in BODY_ANCHOR_NAMES
        ),
    )


def _provenance(frame: DecodedFrame) -> tuple[int, int, int, int, int, int]:
    return (
        frame.frame_index,
        frame.source_pts,
        frame.source_time_base_numerator,
        frame.source_time_base_denominator,
        frame.relative_timestamp_us,
        frame.task_timestamp_ms,
    )


def _inference_provenance(inference: FrameInference) -> tuple[int, int, int, int, int, int]:
    return (
        inference.frame_index,
        inference.source_pts,
        inference.source_time_base_numerator,
        inference.source_time_base_denominator,
        inference.relative_timestamp_us,
        inference.task_timestamp_ms,
    )


def _invalid_frame(
    recording_id: str,
    frame: DecodedFrame,
    reason: Literal["source_frame_invalid", "task_inference_failed"],
) -> LandmarkFrameV1:
    return LandmarkFrameV1(
        schema_version="landmark-frame/1",
        source_recording_id=recording_id,
        frame_index=frame.frame_index,
        source_pts=frame.source_pts,
        source_time_base_numerator=frame.source_time_base_numerator,
        source_time_base_denominator=frame.source_time_base_denominator,
        relative_timestamp_us=frame.relative_timestamp_us,
        task_timestamp_ms=frame.task_timestamp_ms,
        invalid=True,
        invalid_reason=reason,
        hands=(_absent_hand("hand_0"), _absent_hand("hand_1")),
        body_anchors=_absent_anchors(),
        observed_hand_count=0,
        observed_body_anchor_count=0,
    )


def _valid_frame(
    recording_id: str,
    frame: DecodedFrame,
    inference: FrameInference,
    tracker: BatchHandTracker,
) -> LandmarkFrameV1:
    if _inference_provenance(inference) != _provenance(frame):
        raise ExtractionBatchError("execution.failed")
    tracking = tracker.track(inference.hands)
    hands = _hand_slots(tracking)
    anchors = _body_anchors(inference.pose_anchors)
    return LandmarkFrameV1(
        schema_version="landmark-frame/1",
        source_recording_id=recording_id,
        frame_index=frame.frame_index,
        source_pts=frame.source_pts,
        source_time_base_numerator=frame.source_time_base_numerator,
        source_time_base_denominator=frame.source_time_base_denominator,
        relative_timestamp_us=frame.relative_timestamp_us,
        task_timestamp_ms=frame.task_timestamp_ms,
        invalid=False,
        invalid_reason=None,
        hands=hands,
        body_anchors=anchors,
        observed_hand_count=sum(slot.present for slot in hands),
        observed_body_anchor_count=sum(anchor.present for anchor in anchors),
    )


def _extract_frames(
    recording_id: str,
    source: Path,
    assets: VerifiedModelAssets,
    runtime_config: ExtractionRuntimeConfig,
    tracker: BatchHandTracker,
    decoder_factory: DecoderFactory,
    runtime_factory: RuntimeFactory,
) -> LandmarkFramesTableV1:
    tracker.reset()
    rows: list[LandmarkFrameV1] = []
    try:
        with runtime_factory(assets, runtime_config) as detector:
            for frame in decoder_factory(source):
                if not frame.source_valid:
                    tracker.track(())
                    rows.append(_invalid_frame(recording_id, frame, "source_frame_invalid"))
                    continue
                try:
                    inference = detector.infer_frame(frame)
                except ExtractionRuntimeError as error:
                    if error.code not in {
                        "extraction.result.invalid",
                        "extraction.runtime.inference.failed",
                    }:
                        raise ExtractionBatchError("execution.failed") from None
                    tracker.track(())
                    rows.append(_invalid_frame(recording_id, frame, "task_inference_failed"))
                else:
                    rows.append(_valid_frame(recording_id, frame, inference, tracker))
        return LandmarkFramesTableV1(
            schema_version="landmark-frames-table/1",
            rows=tuple(rows),
        )
    except ExtractionBatchError:
        raise
    except (ExtractionRuntimeError, OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("execution.failed") from None


def extract_media_landmarks(
    recording_id: str,
    source: str | Path,
    *,
    assets: VerifiedModelAssets,
    config: ExtractionConfigInput,
    decoder_factory: DecoderFactory = iter_decoded_frames,
    runtime_factory: RuntimeFactory = MediaPipeVideoRuntime,
    tracker_factory: TrackerFactory = HandIdentityTracker,
) -> LandmarkFramesTableV1:
    """Run the existing extractor for one already-authorized media object.

    Authorization, lineage, and source-byte verification remain the caller's
    responsibility.  This small entry point lets licensed public data reuse the
    exact participant extraction implementation without manufacturing consent or
    session records.
    """

    checked_config = _validate_config(config)
    tracker = tracker_factory(_tracking_config(checked_config))
    return _extract_frames(
        recording_id,
        Path(source),
        assets,
        _runtime_config(checked_config),
        tracker,
        decoder_factory,
        runtime_factory,
    )


def _source_media_path(recording: RecordingRowV1, raw_root: Path) -> Path:
    locator = recording.media.locator
    if not isinstance(locator, WorkspaceRelativeLocatorV1):
        raise ExtractionBatchError("raw_bundle.invalid")
    try:
        candidate = raw_root.joinpath(*locator.path.split("/"))
        if _is_linklike(candidate):
            raise ExtractionBatchError("raw_bundle.invalid")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(raw_root) or not resolved.is_file():
            raise ExtractionBatchError("raw_bundle.invalid")
        if resolved.stat().st_size != recording.media.size_bytes:
            raise ExtractionBatchError("raw_bundle.invalid")
        return resolved
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("raw_bundle.invalid") from None


def _source_file_identity(metadata: os.stat_result) -> _SourceFileIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _verify_source_media_bytes(
    recording: RecordingRowV1,
    source: Path,
    raw_root: Path,
    *,
    expected_identity: _SourceFileIdentity | None = None,
) -> _SourceFileIdentity:
    try:
        if _is_linklike(source) or source.resolve(strict=True) != source:
            raise ExtractionBatchError("raw_bundle.invalid")
        if not source.is_relative_to(raw_root) or not source.is_file():
            raise ExtractionBatchError("raw_bundle.invalid")
        before = source.stat()
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_SOURCE_HASH_CHUNK_BYTES), b""):
                size += len(chunk)
                digest.update(chunk)
        after = source.stat()
        before_identity = _source_file_identity(before)
        after_identity = _source_file_identity(after)
        if (
            before_identity != after_identity
            or (expected_identity is not None and after_identity != expected_identity)
            or size != recording.media.size_bytes
            or size != after.st_size
            or f"sha256:{digest.hexdigest()}" != recording.media.sha256
            or _is_linklike(source)
            or source.resolve(strict=True) != source
        ):
            raise ExtractionBatchError("raw_bundle.invalid")
        return after_identity
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("raw_bundle.invalid") from None


def _artifact_path(sha256: str, artifact_id: str) -> str:
    digest = sha256.removeprefix("sha256:")
    return f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{artifact_id}"


def _sequence_reference(
    recording: RecordingRowV1,
    config: MediaPipeExtractionConfigV1,
    config_sha256: str,
    table: LandmarkFramesTableV1,
    parquet_sha256: str,
    parquet_size_bytes: int,
    content_sha256: str,
) -> LandmarkSequenceRefV1:
    artifact_id = _derived_artifact_id(recording.recording_id, config_sha256)
    relative_path = _artifact_path(parquet_sha256, artifact_id)
    artifact = ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=artifact_id,
        role="derived_data",
        media_type="application/vnd.apache.parquet",
        sha256=parquet_sha256,
        size_bytes=parquet_size_bytes,
        locator=WorkspaceRelativeLocatorV1(kind="workspace_relative", path=relative_path),
    )
    lineage = DerivedArtifactRowV1(
        derived_artifact_id=artifact_id,
        derivation_kind="landmark_extraction",
        parent_artifact_ids=(recording.recording_id,),
        participant_id=recording.participant_id,
        session_id=recording.session_id,
        source_recording_id=recording.recording_id,
        clip_id=None,
        annotation_id=None,
        sample_id=None,
        label_id=None,
        split_id=None,
        partition=None,
        handedness=recording.handedness,
        mirror_state=recording.mirror_state,
        operation_id=config.config_id,
        operation_version=config.version,
        artifact=artifact,
    )
    return LandmarkSequenceRefV1(
        schema_version="landmark-sequence-reference/1",
        lineage=lineage,
        source_media_sha256=recording.media.sha256,
        source_media_size_bytes=recording.media.size_bytes,
        source_rotation_degrees=recording.rotation_degrees,
        source_mirror_state=recording.mirror_state,
        frames_schema_version="landmark-frames-table/1",
        content_sha256=content_sha256,
        counts=landmark_observation_counts(table),
    )


def _manifest(
    raw_manifest: RawDatasetManifestV1,
    config: MediaPipeExtractionConfigV1,
    sequences: tuple[LandmarkSequenceRefV1, ...],
) -> LandmarkExtractionManifestV1:
    config_sha256 = mediapipe_extraction_config_digest(config)
    payload: dict[str, object] = {
        "schema_version": "landmark-extraction-manifest/1",
        "extraction_id": LANDMARK_EXTRACTION_ID,
        "version": LANDMARK_EXTRACTION_VERSION,
        "raw_dataset_id": raw_manifest.dataset_id,
        "raw_dataset_version": raw_manifest.version,
        "raw_data_sha256": raw_manifest.raw_data_sha256,
        "raw_dataset_manifest_sha256": raw_dataset_manifest_digest(raw_manifest),
        "config": config.model_dump(mode="json", round_trip=True),
        "config_sha256": config_sha256,
        "sequences": [sequence.model_dump(mode="json", round_trip=True) for sequence in sequences],
    }
    payload["manifest_sha256"] = landmark_extraction_manifest_digest(payload)
    return validate_landmark_extraction_manifest(payload)


def _write_manifest_durably(path: Path, manifest: LandmarkExtractionManifestV1) -> None:
    try:
        captured = canonical_json_bytes(manifest) + b"\n"
        with path.open("xb") as stream:
            stream.write(captured)
            stream.flush()
            os.fsync(stream.fileno())
    except (CanonicalizationError, OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("publication.failed") from None


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_bytes(root: Path) -> bytes:
    path = root / LANDMARK_EXTRACTION_MANIFEST_FILENAME
    try:
        if _is_linklike(path):
            raise ExtractionBatchError("bundle.invalid")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ExtractionBatchError("bundle.invalid")
        if resolved.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ExtractionBatchError("bundle.invalid")
        return resolved.read_bytes()
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise ExtractionBatchError("bundle.invalid") from None


def _resolved_directory(root_input: str | Path) -> Path:
    try:
        candidate = Path(root_input)
        if _is_linklike(candidate):
            raise ExtractionBatchError("bundle.invalid")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise ExtractionBatchError("bundle.invalid")
        return root
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ExtractionBatchError("bundle.invalid") from None


def _expected_inventory(manifest: LandmarkExtractionManifestV1) -> tuple[set[str], set[str]]:
    files = {LANDMARK_EXTRACTION_MANIFEST_FILENAME}
    directories: set[str] = set()
    for sequence in manifest.sequences:
        locator = sequence.lineage.artifact.locator
        if not isinstance(locator, WorkspaceRelativeLocatorV1):
            raise ExtractionBatchError("bundle.invalid")
        relative = locator.path
        files.add(relative)
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _verify_exact_inventory(root: Path, manifest: LandmarkExtractionManifestV1) -> None:
    expected_files, expected_directories = _expected_inventory(manifest)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    pending = [root]
    try:
        while pending:
            current = pending.pop()
            for entry in current.iterdir():
                if _is_linklike(entry):
                    raise ExtractionBatchError("bundle.invalid")
                relative = entry.relative_to(root).as_posix()
                if entry.is_dir():
                    actual_directories.add(relative)
                    pending.append(entry)
                elif entry.is_file():
                    actual_files.add(relative)
                else:
                    raise ExtractionBatchError("bundle.invalid")
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise ExtractionBatchError("bundle.invalid") from None
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ExtractionBatchError("bundle.invalid")


def _reconcile_sequence(
    sequence: LandmarkSequenceRefV1,
    recording: RecordingRowV1,
    config_sha256: str,
) -> None:
    lineage = sequence.lineage
    expected = (
        recording.participant_id,
        recording.session_id,
        recording.recording_id,
        recording.handedness,
        recording.mirror_state,
        _derived_artifact_id(recording.recording_id, config_sha256),
        recording.media.sha256,
        recording.media.size_bytes,
        recording.rotation_degrees,
        recording.mirror_state,
    )
    actual = (
        lineage.participant_id,
        lineage.session_id,
        lineage.source_recording_id,
        lineage.handedness,
        lineage.mirror_state,
        lineage.derived_artifact_id,
        sequence.source_media_sha256,
        sequence.source_media_size_bytes,
        sequence.source_rotation_degrees,
        sequence.source_mirror_state,
    )
    if actual != expected:
        raise ExtractionBatchError("bundle.invalid")


def _validate_landmark_extraction_bundle_against_raw(
    manifest: ExtractionManifestInput,
    workspace_root: str | Path,
    raw: ValidatedRawDatasetBundle,
    recordings: tuple[RecordingRowV1, ...],
) -> ValidatedLandmarkExtractionBundle:
    try:
        checked = validate_landmark_extraction_manifest(manifest)
        assert_landmark_extraction_bound_to_raw_dataset(checked, raw.manifest)
        root = _resolved_directory(workspace_root)
        disk_bytes = _manifest_bytes(root)
        disk_manifest = validate_landmark_extraction_manifest(disk_bytes)
        if disk_manifest != checked or disk_bytes != canonical_json_bytes(disk_manifest) + b"\n":
            raise ExtractionBatchError("bundle.invalid")
        if len(checked.sequences) != len(recordings):
            raise ExtractionBatchError("bundle.invalid")
        _verify_exact_inventory(root, checked)
        config_sha256 = mediapipe_extraction_config_digest(checked.config)
        by_recording = {recording.recording_id: recording for recording in recordings}
        tables: dict[str, LandmarkFramesTableV1] = {}
        for sequence in checked.sequences:
            recording_id = sequence.lineage.source_recording_id
            recording = by_recording.get(recording_id)
            if recording is None:
                raise ExtractionBatchError("bundle.invalid")
            _reconcile_sequence(sequence, recording, config_sha256)
            locator = sequence.lineage.artifact.locator
            if not isinstance(locator, WorkspaceRelativeLocatorV1):
                raise ExtractionBatchError("bundle.invalid")
            artifact_path = root.joinpath(*locator.path.split("/"))
            if _is_linklike(artifact_path):
                raise ExtractionBatchError("bundle.invalid")
            resolved_artifact = artifact_path.resolve(strict=True)
            if not resolved_artifact.is_relative_to(root) or not resolved_artifact.is_file():
                raise ExtractionBatchError("bundle.invalid")
            table = read_landmark_frames(
                resolved_artifact,
                expected_size_bytes=sequence.lineage.artifact.size_bytes,
                expected_sha256=sequence.lineage.artifact.sha256,
                expected_content_sha256=sequence.content_sha256,
                expected_row_count=sequence.counts.frame_count,
            )
            assert_landmark_sequence_ref_matches_table(sequence, table)
            tables[recording_id] = table
        if set(tables) != set(by_recording):
            raise ExtractionBatchError("bundle.invalid")
        validation = ExtractionBundleValidationResult(
            raw_bundle_integrity="verified",
            consent_boundary="synthetic_fixture_only",
            manifest_integrity="verified",
            registered_model_identity="verified",
            parquet_byte_integrity="verified",
            semantic_integrity="verified",
            exact_inventory="verified",
            sequence_count=len(checked.sequences),
            frame_count=sum(len(table.rows) for table in tables.values()),
            invalid_frame_count=sum(
                frame.invalid for table in tables.values() for frame in table.rows
            ),
        )
        return ValidatedLandmarkExtractionBundle(
            manifest=checked,
            tables=MappingProxyType(tables),
            validation=validation,
        )
    except ExtractionBatchError:
        raise
    except (
        CanonicalizationError,
        ExtractionContractError,
        LandmarkParquetError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise ExtractionBatchError("bundle.invalid") from None


def validate_landmark_extraction_bundle(
    manifest: ExtractionManifestInput,
    workspace_root: str | Path,
    *,
    raw_manifest: RawDatasetManifestInput,
    raw_bundle_root: str | Path,
) -> ValidatedLandmarkExtractionBundle:
    """Verify the raw input, consent, manifest, inventory, and every Parquet row."""

    raw, recordings = _validate_raw_first(raw_manifest, raw_bundle_root)
    return _validate_landmark_extraction_bundle_against_raw(
        manifest,
        workspace_root,
        raw,
        recordings,
    )


def _destination_state(destination: Path) -> Literal["absent", "empty", "occupied"]:
    try:
        if _is_linklike(destination) or (destination.exists() and not destination.is_dir()):
            raise ExtractionBatchError("destination.invalid")
        if not destination.exists():
            return "absent"
        return "occupied" if any(destination.iterdir()) else "empty"
    except ExtractionBatchError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise ExtractionBatchError("destination.invalid") from None


def _existing_result(
    destination: Path,
    config: MediaPipeExtractionConfigV1,
    raw: ValidatedRawDatasetBundle,
    recordings: tuple[RecordingRowV1, ...],
) -> ExtractionBatchResult:
    try:
        validated = _validate_landmark_extraction_bundle_against_raw(
            _manifest_bytes(_resolved_directory(destination)),
            destination,
            raw,
            recordings,
        )
    except ExtractionBatchError:
        raise ExtractionBatchError("destination.conflict") from None
    if (
        validated.manifest.config != config
        or validated.manifest.extraction_id != LANDMARK_EXTRACTION_ID
        or validated.manifest.version != LANDMARK_EXTRACTION_VERSION
    ):
        raise ExtractionBatchError("destination.conflict")
    return ExtractionBatchResult(
        status="unchanged",
        manifest=validated.manifest,
        manifest_path=destination / LANDMARK_EXTRACTION_MANIFEST_FILENAME,
        validation=validated.validation,
    )


def _publish_or_reconcile(
    staging: Path,
    destination: Path,
    manifest: LandmarkExtractionManifestV1,
    config: MediaPipeExtractionConfigV1,
    raw: ValidatedRawDatasetBundle,
    recordings: tuple[RecordingRowV1, ...],
) -> ExtractionBatchResult:
    state = _destination_state(destination)
    if state == "occupied":
        existing = _existing_result(destination, config, raw, recordings)
        if existing.manifest != manifest:
            raise ExtractionBatchError("destination.conflict")
        return existing
    removed_empty = False
    try:
        if state == "empty":
            destination.rmdir()
            removed_empty = True
        staging.rename(destination)
        with suppress(OSError):
            _fsync_directory(destination.parent)
        validated = _validate_landmark_extraction_bundle_against_raw(
            manifest,
            destination,
            raw,
            recordings,
        )
        return ExtractionBatchResult(
            status="published",
            manifest=validated.manifest,
            manifest_path=destination / LANDMARK_EXTRACTION_MANIFEST_FILENAME,
            validation=validated.validation,
        )
    except ExtractionBatchError:
        if removed_empty and not destination.exists():
            with suppress(OSError):
                destination.mkdir()
        if destination.exists():
            existing = _existing_result(destination, config, raw, recordings)
            if existing.manifest == manifest:
                return existing
        raise
    except (OSError, RuntimeError, ValueError):
        if removed_empty and not destination.exists():
            with suppress(OSError):
                destination.mkdir()
        if destination.exists():
            try:
                existing = _existing_result(destination, config, raw, recordings)
                if existing.manifest == manifest:
                    return existing
            except ExtractionBatchError:
                pass
        raise ExtractionBatchError("publication.failed") from None


def extract_raw_dataset(
    raw_manifest: RawDatasetManifestInput,
    *,
    raw_bundle_root: str | Path,
    model_root: str | Path,
    config: ExtractionConfigInput,
    destination: str | Path,
    decoder_factory: DecoderFactory = iter_decoded_frames,
    runtime_factory: RuntimeFactory = MediaPipeVideoRuntime,
    tracker_factory: TrackerFactory = HandIdentityTracker,
) -> ExtractionBatchResult:
    """Extract every eligible synthetic recording and atomically publish one bundle."""

    raw, recordings = _validate_raw_first(raw_manifest, raw_bundle_root)
    checked_config = _validate_config(config)
    raw_root, output = _nonoverlapping_extraction_paths(raw_bundle_root, destination)
    state = _destination_state(output)
    if state == "occupied":
        return _existing_result(output, checked_config, raw, recordings)
    assets = _verify_models(model_root, checked_config)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        config_sha256 = mediapipe_extraction_config_digest(checked_config)
        runtime_config = _runtime_config(checked_config)
        tracker = tracker_factory(_tracking_config(checked_config))
        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.staging-",
        ) as temporary_directory:
            staging = Path(temporary_directory)
            work = staging / "work"
            work.mkdir()
            sequences: list[LandmarkSequenceRefV1] = []
            for recording in recordings:
                source = _source_media_path(recording, raw_root)
                source_identity = _verify_source_media_bytes(recording, source, raw_root)
                table = _extract_frames(
                    recording.recording_id,
                    source,
                    assets,
                    runtime_config,
                    tracker,
                    decoder_factory,
                    runtime_factory,
                )
                _verify_source_media_bytes(
                    recording,
                    source,
                    raw_root,
                    expected_identity=source_identity,
                )
                artifact_id = _derived_artifact_id(recording.recording_id, config_sha256)
                temporary_parquet = work / artifact_id
                written = write_landmark_frames(table, temporary_parquet)
                reference = _sequence_reference(
                    recording,
                    checked_config,
                    config_sha256,
                    table,
                    written.sha256,
                    written.size_bytes,
                    written.content_sha256,
                )
                locator = reference.lineage.artifact.locator
                if not isinstance(locator, WorkspaceRelativeLocatorV1):
                    raise ExtractionBatchError("execution.failed")
                final_path = staging.joinpath(*locator.path.split("/"))
                final_path.parent.mkdir(parents=True, exist_ok=True)
                written.path.replace(final_path)
                _fsync_file(final_path)
                sequences.append(reference)
            work.rmdir()
            checked_manifest = _manifest(raw.manifest, checked_config, tuple(sequences))
            for directory in sorted(
                (path for path in staging.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _fsync_directory(directory)
            # The manifest is the completion marker and is always staged last.
            _write_manifest_durably(
                staging / LANDMARK_EXTRACTION_MANIFEST_FILENAME,
                checked_manifest,
            )
            _fsync_directory(staging)
            _validate_landmark_extraction_bundle_against_raw(
                checked_manifest,
                staging,
                raw,
                recordings,
            )
            return _publish_or_reconcile(
                staging,
                output,
                checked_manifest,
                checked_config,
                raw,
                recordings,
            )
    except ExtractionBatchError:
        raise
    except (
        ExtractionContractError,
        ExtractionRuntimeError,
        LandmarkParquetError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise ExtractionBatchError("execution.failed") from None


__all__ = [
    "LANDMARK_EXTRACTION_ID",
    "LANDMARK_EXTRACTION_MANIFEST_FILENAME",
    "LANDMARK_EXTRACTION_VERSION",
    "BatchHandTracker",
    "DecoderFactory",
    "ExtractionBatchError",
    "ExtractionBatchErrorCategory",
    "ExtractionBatchResult",
    "ExtractionBatchStatus",
    "ExtractionBundleValidationResult",
    "ExtractionConfigInput",
    "ExtractionManifestInput",
    "RuntimeFactory",
    "TrackerFactory",
    "ValidatedLandmarkExtractionBundle",
    "VideoInferenceRuntime",
    "extract_media_landmarks",
    "extract_raw_dataset",
    "validate_landmark_extraction_bundle",
]
