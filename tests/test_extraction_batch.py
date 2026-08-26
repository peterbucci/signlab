from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import pytest

import signlab.extraction.batch as batch
from signlab.contracts.canonical import CanonicalizationError, canonical_json_bytes
from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import RecordingsTableV1
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    LandmarkExtractionManifestV1,
    MediaPipeExtractionConfigV1,
    landmark_extraction_manifest_digest,
    validate_landmark_extraction_manifest,
)
from signlab.contracts.ingest import RawDatasetManifestV1
from signlab.datasets.importer import import_collection_sidecar
from signlab.datasets.raw_bundle import (
    ValidatedRawDatasetBundle,
    validate_raw_dataset_bundle,
)
from signlab.extraction.batch import (
    ExtractionBatchError,
    ExtractionBatchResult,
    extract_raw_dataset,
    validate_landmark_extraction_bundle,
)
from signlab.extraction.runtime import (
    DecodedFrame,
    ExtractionRuntimeConfig,
    ExtractionRuntimeError,
    FrameInference,
    PoseAnchorName,
    PoseAnchorObservation,
    VerifiedModelAssets,
)
from signlab.extraction.tracking import HandIdentityTracker, HandTrackingConfig
from signlab.extraction.types import HandDetection, LandmarkPoint
from test_dataset_importer import _FIXTURE_ROOT, _public_fixture
from test_extraction_contracts import _config

_SCRIPTED_RESULTS_PATH: Final = (
    Path(__file__).parent / "fixtures" / "public" / "extraction" / "scripted-results-v1.json"
)


def _scripted_document() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(_SCRIPTED_RESULTS_PATH.read_text(encoding="utf-8")),
    )


def _scripted_frames() -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _scripted_document()["frames"])


def _publish_raw(tmp_path: Path) -> tuple[Path, RawDatasetManifestV1]:
    sidecar, source_map = _public_fixture()
    root = tmp_path / "raw"
    result = import_collection_sidecar(
        sidecar,
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=root,
    )
    return root, result.manifest


def _recording_source(raw_root: Path, raw_manifest: RawDatasetManifestV1) -> Path:
    loaded = validate_raw_dataset_bundle(raw_manifest, raw_root)
    recordings = loaded.tables["recordings"]
    assert isinstance(recordings, RecordingsTableV1)
    locator = recordings.rows[0].media.locator
    assert isinstance(locator, WorkspaceRelativeLocatorV1)
    return raw_root.joinpath(*locator.path.split("/")).resolve(strict=True)


def _same_size_mutation(content: bytes) -> bytes:
    assert content
    return bytes((content[0] ^ 1,)) + content[1:]


def _tree_snapshot(root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    directories = tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir())
    )
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return directories, files


def _point(seed: float, *, wrist_x: float | None = None, index: int = 0) -> LandmarkPoint:
    x = wrist_x if wrist_x is not None and index == 0 else seed + index / 1_000.0
    return LandmarkPoint(x=x, y=0.5, z=-0.01, visibility=None, presence=None)


def _hand(
    detector_index: int,
    handedness: Literal["left", "right"],
    score: float,
    seed: float,
    wrist_x: float,
) -> HandDetection:
    points = tuple(_point(seed, wrist_x=wrist_x, index=index) for index in range(21))
    return HandDetection(
        detector_index=detector_index,
        image_landmarks=points,
        world_landmarks=points,
        reported_handedness=handedness,
        handedness_score=score,
    )


def _pose() -> tuple[PoseAnchorObservation, ...]:
    observations: list[PoseAnchorObservation] = []
    for offset, (name, index) in enumerate(zip(BODY_ANCHOR_NAMES, range(11, 17), strict=True)):
        point = LandmarkPoint(
            x=0.1 + offset / 10,
            y=0.2,
            z=-0.1,
            visibility=0.9,
            presence=0.8,
        )
        observations.append(
            PoseAnchorObservation(
                name=cast(PoseAnchorName, name),
                landmark_index=index,
                image_landmark=point,
                world_landmark=point,
            )
        )
    return tuple(observations)


def _decoded_frames(_source: str | Path) -> tuple[DecodedFrame, ...]:
    previous_ms = -1
    frames: list[DecodedFrame] = []
    scripted = _scripted_frames()
    first_pts = cast(int, scripted[0]["pts"])
    for index, item in enumerate(scripted):
        pts = cast(int, item["pts"])
        numerator = cast(int, item["time_base_numerator"])
        denominator = cast(int, item["time_base_denominator"])
        relative_us = (pts - first_pts) * numerator * 1_000_000 // denominator
        task_ms = max(relative_us // 1_000, previous_ms + 1)
        source_valid = cast(bool, item["source_valid"])
        frames.append(
            DecodedFrame(
                frame_index=index,
                source_pts=pts,
                source_time_base_numerator=numerator,
                source_time_base_denominator=denominator,
                relative_timestamp_us=relative_us,
                task_timestamp_ms=task_ms,
                rgb=object() if source_valid else None,
                source_valid=source_valid,
            )
        )
        previous_ms = task_ms
    return tuple(frames)


class _FakeRuntime:
    def __init__(self) -> None:
        self.inferred_indexes: list[int] = []
        self.closed = False

    def __enter__(self) -> _FakeRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.closed = True

    def infer_frame(self, frame: DecodedFrame) -> FrameInference:
        self.inferred_indexes.append(frame.frame_index)
        scripted = _scripted_frames()[frame.frame_index]
        if scripted.get("task_inference_failure") is True:
            raise ExtractionRuntimeError("extraction.runtime.inference.failed")
        hands: list[HandDetection] = []
        for detector_index, hand in enumerate(cast(list[dict[str, Any]], scripted["hands"])):
            handedness = cast(str, hand["handedness"]).casefold()
            assert handedness in {"left", "right"}
            wrist = cast(list[float], hand["wrist"])
            hands.append(
                _hand(
                    detector_index,
                    cast(Literal["left", "right"], handedness),
                    cast(float, hand["score"]),
                    cast(float, hand["point_seed"]),
                    wrist[0],
                )
            )
        return FrameInference(
            frame_index=frame.frame_index,
            source_pts=frame.source_pts,
            source_time_base_numerator=frame.source_time_base_numerator,
            source_time_base_denominator=frame.source_time_base_denominator,
            relative_timestamp_us=frame.relative_timestamp_us,
            task_timestamp_ms=frame.task_timestamp_ms,
            hands=tuple(hands),
            pose_anchors=_pose() if scripted["pose_present"] is True else (),
        )


class _RuntimeFactory:
    def __init__(self) -> None:
        self.instances: list[_FakeRuntime] = []

    def __call__(
        self,
        _assets: VerifiedModelAssets,
        _config: ExtractionRuntimeConfig,
    ) -> _FakeRuntime:
        instance = _FakeRuntime()
        self.instances.append(instance)
        return instance


class _CountingTracker(HandIdentityTracker):
    def __init__(self, config: HandTrackingConfig) -> None:
        super().__init__(config)
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1
        super().reset()


class _TrackerFactory:
    def __init__(self) -> None:
        self.instances: list[_CountingTracker] = []

    def __call__(self, config: HandTrackingConfig) -> _CountingTracker:
        instance = _CountingTracker(config)
        self.instances.append(instance)
        return instance


def _fake_models(
    _model_root: str | Path,
    config: MediaPipeExtractionConfigV1,
) -> VerifiedModelAssets:
    return VerifiedModelAssets(
        hand_model_bytes=b"synthetic-hand",
        pose_model_bytes=b"synthetic-pose",
        hand_model_sha256=config.hand_task_asset.sha256,
        pose_model_sha256=config.pose_task_asset.sha256,
    )


def _extract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    destination_name: str = "landmarks",
) -> tuple[
    ExtractionBatchResult,
    Path,
    RawDatasetManifestV1,
    _RuntimeFactory,
    _TrackerFactory,
]:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    monkeypatch.setattr(batch, "_verify_models", _fake_models)
    runtime_factory = _RuntimeFactory()
    tracker_factory = _TrackerFactory()
    destination = tmp_path / destination_name
    result = extract_raw_dataset(
        raw_manifest,
        raw_bundle_root=raw_root,
        model_root=tmp_path / "external-models",
        config=_config(),
        destination=destination,
        decoder_factory=_decoded_frames,
        runtime_factory=runtime_factory,
        tracker_factory=tracker_factory,
    )
    return result, raw_root, raw_manifest, runtime_factory, tracker_factory


def test_scripted_fixture_is_synthetic_versioned_and_covers_both_invalid_frame_kinds() -> None:
    document = _scripted_document()
    frames = _scripted_frames()

    assert document["schema_version"] == "scripted-landmarker-fixture/1"
    assert document["fixture_only"] is True
    assert sum(frame["source_valid"] is False for frame in frames) == 1
    assert sum(frame.get("task_inference_failure") is True for frame in frames) == 1
    assert [frame["pts"] for frame in frames] == list(range(len(frames)))


def test_batch_publishes_content_addressed_lineage_and_explicit_frame_masks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _raw_root, _raw_manifest, runtimes, trackers = _extract(monkeypatch, tmp_path)
    destination = tmp_path / "landmarks"

    assert result.status == "published"
    assert result.manifest_path == destination / batch.LANDMARK_EXTRACTION_MANIFEST_FILENAME
    assert result.validation.sequence_count == 1
    assert result.validation.frame_count == 6
    assert result.validation.invalid_frame_count == 2
    assert result.validation.consent_boundary == "synthetic_fixture_only"
    assert result.validation.registered_model_identity == "verified"
    sequence = result.manifest.sequences[0]
    assert sequence.counts.valid_frame_count == 4
    assert sequence.counts.invalid_frame_count == 2
    assert sequence.counts.zero_hand_frame_count == 1
    assert sequence.counts.one_hand_frame_count == 1
    assert sequence.counts.two_hand_frame_count == 2
    assert sequence.counts.hand_observation_count == 5
    assert sequence.counts.body_anchor_observation_count == 12
    locator = sequence.lineage.artifact.locator
    assert isinstance(locator, WorkspaceRelativeLocatorV1)
    digest = sequence.lineage.artifact.sha256.removeprefix("sha256:")
    assert locator.path == (
        f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{sequence.lineage.derived_artifact_id}"
    )
    assert runtimes.instances[0].inferred_indexes == [0, 1, 2, 3, 5]
    assert runtimes.instances[0].closed
    assert trackers.instances[0].reset_count == 1

    table = result.manifest.sequences[0]
    validated = validate_landmark_extraction_bundle(
        result.manifest,
        destination,
        raw_manifest=_raw_manifest,
        raw_bundle_root=_raw_root,
    )
    rows = validated.tables[table.lineage.source_recording_id].rows
    assert rows[3].invalid is False
    assert rows[3].observed_hand_count == 0
    assert rows[4].invalid_reason == "source_frame_invalid"
    assert rows[5].invalid_reason == "task_inference_failed"
    assert all(not slot.present for row in rows[4:] for slot in row.hands)
    assert all(not anchor.present for row in rows[4:] for anchor in row.body_anchors)


def test_identical_replay_validates_and_returns_unchanged_without_reextracting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, raw_root, raw_manifest, _runtimes, _trackers = _extract(monkeypatch, tmp_path)

    def forbidden_models(_root: str | Path, _config: MediaPipeExtractionConfigV1) -> object:
        raise AssertionError("existing output must not reopen external model bytes")

    def forbidden_decoder(_source: str | Path) -> tuple[DecodedFrame, ...]:
        raise AssertionError("existing output must not decode source media")

    monkeypatch.setattr(batch, "_verify_models", forbidden_models)
    replay = extract_raw_dataset(
        raw_manifest,
        raw_bundle_root=raw_root,
        model_root=tmp_path / "missing-models",
        config=_config(),
        destination=tmp_path / "landmarks",
        decoder_factory=forbidden_decoder,
        runtime_factory=_RuntimeFactory(),
        tracker_factory=_TrackerFactory(),
    )

    assert replay.status == "unchanged"
    assert replay.manifest == first.manifest


def test_golden_scripted_fixture_has_cross_platform_manifest_and_parquet_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, raw_root, raw_manifest, _runtimes, _trackers = _extract(monkeypatch, tmp_path)
    second = extract_raw_dataset(
        raw_manifest,
        raw_bundle_root=raw_root,
        model_root=tmp_path / "external-models",
        config=_config(),
        destination=tmp_path / "landmarks-copy",
        decoder_factory=_decoded_frames,
        runtime_factory=_RuntimeFactory(),
        tracker_factory=_TrackerFactory(),
    )

    assert second.status == "published"
    assert second.manifest == first.manifest
    assert first.manifest.manifest_sha256 == (
        "sha256:921433db648f7146b05783159b5063c78e71596154ada499eaa990b132637236"
    )
    sequence = first.manifest.sequences[0]
    assert sequence.content_sha256 == (
        "sha256:73869bf8eb6441eb299ba17c11e2412194b99a1a7534029b9937f8f567b61984"
    )
    assert sequence.lineage.artifact.sha256 == (
        "sha256:b591391fe8f1401e82a9400c16c5411f33012e7e1cfe32b798ed841a7fb781a9"
    )
    assert sequence.lineage.artifact.size_bytes == 21_119


def test_fatal_runtime_state_aborts_instead_of_publishing_an_all_invalid_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    monkeypatch.setattr(batch, "_verify_models", _fake_models)

    def fatal_inference(_self: _FakeRuntime, _frame: DecodedFrame) -> FrameInference:
        raise ExtractionRuntimeError("extraction.dependency.unavailable")

    monkeypatch.setattr(_FakeRuntime, "infer_frame", fatal_inference)
    destination = tmp_path / "landmarks"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "external-models",
            config=_config(),
            destination=destination,
            decoder_factory=_decoded_frames,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )

    assert captured.value.category == "execution.failed"
    assert not destination.exists()


def test_source_media_is_not_loaded_into_memory_after_streaming_raw_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    loaded = validate_raw_dataset_bundle(raw_manifest, raw_root)
    recordings = loaded.tables["recordings"]
    assert isinstance(recordings, RecordingsTableV1)
    locator = recordings.rows[0].media.locator
    assert isinstance(locator, WorkspaceRelativeLocatorV1)
    source = raw_root.joinpath(*locator.path.split("/")).resolve(strict=True)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve(strict=True) == source:
            raise AssertionError("batch extraction must not buffer the already-verified video")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(batch, "_verify_models", _fake_models)

    result = extract_raw_dataset(
        raw_manifest,
        raw_bundle_root=raw_root,
        model_root=tmp_path / "external-models",
        config=_config(),
        destination=tmp_path / "landmarks",
        decoder_factory=_decoded_frames,
        runtime_factory=_RuntimeFactory(),
        tracker_factory=_TrackerFactory(),
    )

    assert result.status == "published"


def test_same_size_source_change_after_initial_validation_fails_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    source = _recording_source(raw_root, raw_manifest)
    original_content = source.read_bytes()
    original_validate = validate_raw_dataset_bundle

    def validate_then_mutate(
        manifest: RawDatasetManifestV1,
        workspace_root: str | Path,
    ) -> ValidatedRawDatasetBundle:
        validated = original_validate(manifest, workspace_root)
        source.write_bytes(_same_size_mutation(original_content))
        return validated

    def forbidden_decoder(_source: str | Path) -> tuple[DecodedFrame, ...]:
        raise AssertionError("changed source must fail before decode")

    monkeypatch.setattr(batch, "validate_raw_dataset_bundle", validate_then_mutate)
    monkeypatch.setattr(batch, "_verify_models", _fake_models)
    destination = tmp_path / "landmarks"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
            decoder_factory=forbidden_decoder,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )

    assert captured.value.category == "raw_bundle.invalid"
    assert str(source) not in str(captured.value)
    assert source.stat().st_size == len(original_content)
    assert source.read_bytes() != original_content
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".landmarks.staging-*"))


def test_same_size_source_mutation_during_decode_fails_after_frame_consumption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    source = _recording_source(raw_root, raw_manifest)
    original_content = source.read_bytes()
    consumed: list[int] = []

    def mutating_decoder(source_input: str | Path) -> Iterable[DecodedFrame]:
        frames = _decoded_frames(source_input)
        for index, frame in enumerate(frames):
            consumed.append(frame.frame_index)
            yield frame
            if index == 0:
                Path(source_input).write_bytes(_same_size_mutation(original_content))

    monkeypatch.setattr(batch, "_verify_models", _fake_models)
    runtime_factory = _RuntimeFactory()
    destination = tmp_path / "landmarks"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
            decoder_factory=mutating_decoder,
            runtime_factory=runtime_factory,
            tracker_factory=_TrackerFactory(),
        )

    assert captured.value.category == "raw_bundle.invalid"
    assert str(source) not in str(captured.value)
    assert consumed == list(range(len(_scripted_frames())))
    assert runtime_factory.instances[0].closed
    assert source.stat().st_size == len(original_content)
    assert source.read_bytes() != original_content
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".landmarks.staging-*"))


@pytest.mark.parametrize("tamper", ["manifest", "parquet", "extra_file", "extra_directory"])
def test_validation_rejects_tampered_or_nonexact_bundle_inventory_without_path_disclosure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    result, raw_root, raw_manifest, _runtimes, _trackers = _extract(monkeypatch, tmp_path)
    destination = tmp_path / "landmarks"
    if tamper == "manifest":
        target = destination / batch.LANDMARK_EXTRACTION_MANIFEST_FILENAME
        target.write_bytes(target.read_bytes() + b" ")
    elif tamper == "parquet":
        locator = result.manifest.sequences[0].lineage.artifact.locator
        assert isinstance(locator, WorkspaceRelativeLocatorV1)
        target = destination.joinpath(*locator.path.split("/"))
        target.write_bytes(target.read_bytes() + b"tampered")
    elif tamper == "extra_file":
        (destination / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    else:
        (destination / "unexpected").mkdir()

    with pytest.raises(ExtractionBatchError) as captured:
        validate_landmark_extraction_bundle(
            result.manifest,
            destination,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
        )

    assert captured.value.category == "bundle.invalid"
    assert str(destination) not in str(captured.value)

    with pytest.raises(ExtractionBatchError) as conflict:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
            decoder_factory=_decoded_frames,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )

    assert conflict.value.category == "destination.conflict"
    assert str(destination) not in str(conflict.value)


def test_occupied_output_with_another_valid_configuration_fails_as_a_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _result, raw_root, raw_manifest, _runtimes, _trackers = _extract(monkeypatch, tmp_path)
    changed = _config().model_copy(update={"config_id": "different_mediapipe_config"})

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "external-models",
            config=changed,
            destination=tmp_path / "landmarks",
            decoder_factory=_decoded_frames,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )

    assert captured.value.category == "destination.conflict"


@pytest.mark.parametrize("authorization_failure", ["non_fixture", "scope"])
def test_public_batch_boundary_has_no_bypass_for_real_or_unauthorized_media(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    authorization_failure: str,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    loaded = validate_raw_dataset_bundle(raw_manifest, raw_root)
    if authorization_failure == "non_fixture":
        unauthorized = replace(
            loaded,
            sidecar=loaded.sidecar.model_copy(update={"fixture_only": False}),
        )
    else:
        recordings = loaded.tables["recordings"]
        assert isinstance(recordings, RecordingsTableV1)
        recording = recordings.rows[0]
        scope = recording.consent_grant.scope.model_copy(update={"derived_features": False})
        grant = recording.consent_grant.model_copy(update={"scope": scope})
        changed_recording = recording.model_copy(update={"consent_grant": grant})
        changed_table = recordings.model_copy(update={"rows": (changed_recording,)})
        tables = dict(loaded.tables)
        tables["recordings"] = changed_table
        unauthorized = replace(loaded, tables=MappingProxyType(tables))
    monkeypatch.setattr(batch, "validate_raw_dataset_bundle", lambda *_args: unauthorized)

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "external-models",
            config=_config(),
            destination=tmp_path / "landmarks",
            decoder_factory=_decoded_frames,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )

    assert captured.value.category == "consent.unauthorized"


def test_validator_reconciles_orientation_mirror_and_lineage_to_raw_recording(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, raw_root, raw_manifest, _runtimes, _trackers = _extract(monkeypatch, tmp_path)
    destination = tmp_path / "landmarks"
    payload = result.manifest.model_dump(mode="json", round_trip=True)
    sequence = payload["sequences"][0]
    sequence["source_rotation_degrees"] = 90
    payload["manifest_sha256"] = landmark_extraction_manifest_digest(payload)
    changed = validate_landmark_extraction_manifest(payload)
    manifest_path = destination / batch.LANDMARK_EXTRACTION_MANIFEST_FILENAME
    manifest_path.write_bytes(canonical_json_bytes(changed) + b"\n")

    with pytest.raises(ExtractionBatchError) as captured:
        validate_landmark_extraction_bundle(
            changed,
            destination,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
        )

    assert captured.value.category == "bundle.invalid"


def test_invalid_raw_bundle_is_rejected_before_config_models_or_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden_models(_root: str | Path, _config: MediaPipeExtractionConfigV1) -> object:
        raise AssertionError("model verification must follow raw-bundle validation")

    monkeypatch.setattr(batch, "_verify_models", forbidden_models)
    destination = tmp_path / "private-output"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            {},
            raw_bundle_root=tmp_path / "missing-raw",
            model_root=tmp_path / "models",
            config={"invalid": True},
            destination=destination,
        )

    assert captured.value.category == "raw_bundle.invalid"
    assert str(tmp_path) not in str(captured.value)
    assert not destination.exists()


@pytest.mark.parametrize("raw_shape", ["missing_recordings", "empty_recordings"])
def test_defensive_raw_shape_checks_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_shape: str,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    loaded = validate_raw_dataset_bundle(raw_manifest, raw_root)
    tables = dict(loaded.tables)
    if raw_shape == "missing_recordings":
        del tables["recordings"]
    else:
        recordings = tables["recordings"]
        assert isinstance(recordings, RecordingsTableV1)
        tables["recordings"] = RecordingsTableV1.model_construct(
            schema_version="recordings-table/1",
            rows=(),
        )
    malformed = replace(loaded, tables=MappingProxyType(tables))
    monkeypatch.setattr(batch, "validate_raw_dataset_bundle", lambda *_args: malformed)

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=tmp_path / "landmarks",
        )

    assert captured.value.category == "raw_bundle.invalid"


def test_invalid_configuration_is_rejected_before_model_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)

    def forbidden_models(_root: str | Path, _config: MediaPipeExtractionConfigV1) -> object:
        raise AssertionError("invalid config must not reach model verification")

    monkeypatch.setattr(batch, "_verify_models", forbidden_models)
    invalid_config = _config().model_dump(mode="json", round_trip=True)
    invalid_config["delegate"] = "GPU"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=invalid_config,
            destination=tmp_path / "landmarks",
        )

    assert captured.value.category == "config.invalid"


@pytest.mark.parametrize("model_failure", ["mismatch", "unavailable"])
def test_external_model_verification_failures_are_sanitized_and_do_not_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_failure: str,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    config = _config()
    if model_failure == "mismatch":
        monkeypatch.setattr(
            batch,
            "verify_model_assets",
            lambda _root: VerifiedModelAssets(
                hand_model_bytes=b"wrong",
                pose_model_bytes=b"wrong",
                hand_model_sha256=config.hand_task_asset.sha256,
                pose_model_sha256=config.pose_task_asset.sha256,
            ),
        )
    else:

        def unavailable(_root: str | Path) -> VerifiedModelAssets:
            raise ExtractionRuntimeError("extraction.dependency.unavailable")

        monkeypatch.setattr(batch, "verify_model_assets", unavailable)
    destination = tmp_path / "landmarks"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "private-model-root",
            config=config,
            destination=destination,
        )

    assert captured.value.category == "models.invalid"
    assert str(tmp_path) not in str(captured.value)
    assert not destination.exists()


@pytest.mark.parametrize("destination_kind", ["file", "directory_link"])
def test_unsafe_destination_shapes_are_rejected_before_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    destination_kind: str,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    destination = tmp_path / "landmarks"
    if destination_kind == "file":
        destination.write_text("occupied", encoding="utf-8")
    else:
        target = tmp_path / "elsewhere"
        target.mkdir()
        try:
            destination.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("directory links are unavailable for this account")

    def forbidden_models(_root: str | Path, _config: MediaPipeExtractionConfigV1) -> object:
        raise AssertionError("invalid destination must not reach model verification")

    monkeypatch.setattr(batch, "_verify_models", forbidden_models)

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
        )

    assert captured.value.category == "destination.invalid"
    assert str(destination) not in str(captured.value)


@pytest.mark.parametrize(
    "relationship",
    ["equal", "descendant", "normalized_descendant", "ancestor", "normalized_equal"],
)
def test_destination_and_raw_bundle_overlap_is_rejected_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relationship: str,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    if relationship == "equal":
        destination = raw_root
    elif relationship == "descendant":
        destination = raw_root / "derived" / "landmarks"
    elif relationship == "normalized_descendant":
        destination = raw_root / "unused" / ".." / "landmarks"
    elif relationship == "ancestor":
        destination = tmp_path
    else:
        destination = raw_root / "unused" / ".."
    before = _tree_snapshot(tmp_path)

    def forbidden_models(
        _model_root: str | Path,
        _config_value: MediaPipeExtractionConfigV1,
    ) -> VerifiedModelAssets:
        raise AssertionError("overlapping paths must fail before model verification")

    monkeypatch.setattr(batch, "_verify_models", forbidden_models)

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
            decoder_factory=_decoded_frames,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )

    assert captured.value.category == "destination.invalid"
    assert str(raw_root) not in str(captured.value)
    assert str(destination) not in str(captured.value)
    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "execution_failure",
    [
        "empty_decoder",
        "runtime_factory",
        "tracker_factory",
        "provenance",
        "missing_world_landmarks",
        "duplicate_anchor",
    ],
)
def test_malformed_or_failed_injected_execution_aborts_without_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    execution_failure: str,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    monkeypatch.setattr(batch, "_verify_models", _fake_models)
    decoder_factory: Any = _decoded_frames
    runtime_factory: Any = _RuntimeFactory()
    tracker_factory: Any = _TrackerFactory()
    original_inference = _FakeRuntime.infer_frame

    if execution_failure == "empty_decoder":

        def empty_decoder(_source: str | Path) -> tuple[DecodedFrame, ...]:
            return ()

        decoder_factory = empty_decoder
    elif execution_failure == "runtime_factory":

        def failed_runtime_factory(
            _assets: VerifiedModelAssets,
            _config_value: ExtractionRuntimeConfig,
        ) -> _FakeRuntime:
            raise RuntimeError("private runtime detail")

        runtime_factory = failed_runtime_factory
    elif execution_failure == "tracker_factory":

        def failed_tracker_factory(_config_value: HandTrackingConfig) -> _CountingTracker:
            raise RuntimeError("private tracker detail")

        tracker_factory = failed_tracker_factory
    elif execution_failure == "provenance":

        def wrong_provenance(self: _FakeRuntime, frame: DecodedFrame) -> FrameInference:
            inference = original_inference(self, frame)
            return replace(inference, source_pts=inference.source_pts + 1)

        monkeypatch.setattr(_FakeRuntime, "infer_frame", wrong_provenance)
    elif execution_failure == "missing_world_landmarks":

        def missing_world(self: _FakeRuntime, frame: DecodedFrame) -> FrameInference:
            inference = original_inference(self, frame)
            detection = replace(inference.hands[0], world_landmarks=())
            return replace(inference, hands=(detection,))

        monkeypatch.setattr(_FakeRuntime, "infer_frame", missing_world)
    else:

        def duplicate_anchor(self: _FakeRuntime, frame: DecodedFrame) -> FrameInference:
            inference = original_inference(self, frame)
            anchor = inference.pose_anchors[0]
            return replace(inference, pose_anchors=(anchor, anchor))

        monkeypatch.setattr(_FakeRuntime, "infer_frame", duplicate_anchor)
    destination = tmp_path / "landmarks"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
            decoder_factory=decoder_factory,
            runtime_factory=runtime_factory,
            tracker_factory=tracker_factory,
        )

    assert captured.value.category == "execution.failed"
    assert "private" not in str(captured.value)
    assert not destination.exists()


def _rewrite_manifest(
    result: ExtractionBatchResult,
    destination: Path,
    mutation: str,
) -> LandmarkExtractionManifestV1:
    payload = result.manifest.model_dump(mode="json", round_trip=True)
    sequence = payload["sequences"][0]
    if mutation == "participant":
        sequence["lineage"]["participant_id"] = "participant_ffffffffffffffffffffffffffffffff"
    elif mutation == "counts":
        sequence["counts"].update(
            {
                "zero_hand_frame_count": 2,
                "one_hand_frame_count": 0,
                "two_hand_frame_count": 2,
                "hand_observation_count": 4,
            }
        )
    else:
        payload["raw_dataset_manifest_sha256"] = "sha256:" + "0" * 64
    payload["manifest_sha256"] = landmark_extraction_manifest_digest(payload)
    changed = validate_landmark_extraction_manifest(payload)
    manifest_path = destination / batch.LANDMARK_EXTRACTION_MANIFEST_FILENAME
    manifest_path.write_bytes(canonical_json_bytes(changed) + b"\n")
    return changed


@pytest.mark.parametrize("mismatch", ["participant", "counts", "raw_binding"])
def test_full_validation_rejects_lineage_counts_and_raw_binding_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    result, raw_root, raw_manifest, _runtimes, _trackers = _extract(monkeypatch, tmp_path)
    destination = tmp_path / "landmarks"
    changed = _rewrite_manifest(result, destination, mismatch)

    with pytest.raises(ExtractionBatchError) as captured:
        validate_landmark_extraction_bundle(
            changed,
            destination,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
        )

    assert captured.value.category == "bundle.invalid"


@pytest.mark.parametrize("link_target", ["manifest", "parquet", "root"])
def test_bundle_validation_rejects_links_at_trust_boundaries_when_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    link_target: str,
) -> None:
    result, raw_root, raw_manifest, _runtimes, _trackers = _extract(monkeypatch, tmp_path)
    destination = tmp_path / "landmarks"
    validation_root = destination
    if link_target == "root":
        validation_root = tmp_path / "landmarks-link"
        try:
            validation_root.symlink_to(destination, target_is_directory=True)
        except OSError:
            pytest.skip("directory links are unavailable for this account")
    else:
        if link_target == "manifest":
            target = destination / batch.LANDMARK_EXTRACTION_MANIFEST_FILENAME
        else:
            locator = result.manifest.sequences[0].lineage.artifact.locator
            assert isinstance(locator, WorkspaceRelativeLocatorV1)
            target = destination.joinpath(*locator.path.split("/"))
        outside = tmp_path / f"outside-{link_target}"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        try:
            target.symlink_to(outside)
        except OSError:
            pytest.skip("file links are unavailable for this account")

    with pytest.raises(ExtractionBatchError) as captured:
        validate_landmark_extraction_bundle(
            result.manifest,
            validation_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
        )

    assert captured.value.category == "bundle.invalid"


def test_empty_destination_is_replaced_by_complete_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    monkeypatch.setattr(batch, "_verify_models", _fake_models)
    destination = tmp_path / "landmarks"
    destination.mkdir()

    result = extract_raw_dataset(
        raw_manifest,
        raw_bundle_root=raw_root,
        model_root=tmp_path / "models",
        config=_config(),
        destination=destination,
        decoder_factory=_decoded_frames,
        runtime_factory=_RuntimeFactory(),
        tracker_factory=_TrackerFactory(),
    )

    assert result.status == "published"
    assert (destination / batch.LANDMARK_EXTRACTION_MANIFEST_FILENAME).is_file()


def test_manifest_write_failure_cleans_staging_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root, raw_manifest = _publish_raw(tmp_path)
    monkeypatch.setattr(batch, "_verify_models", _fake_models)

    def failed_canonicalization(_value: object) -> bytes:
        raise CanonicalizationError("private manifest detail")

    monkeypatch.setattr(batch, "canonical_json_bytes", failed_canonicalization)
    destination = tmp_path / "landmarks"

    with pytest.raises(ExtractionBatchError) as captured:
        extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
            decoder_factory=_decoded_frames,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )

    assert captured.value.category == "publication.failed"
    assert "private" not in str(captured.value)
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".landmarks.staging-*"))


@pytest.mark.parametrize("race", ["identical", "conflict", "empty_cleanup"])
def test_atomic_rename_race_reconciliation_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    race: str,
) -> None:
    winner, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
        destination_name="winner",
    )
    destination = tmp_path / "landmarks"
    if race == "empty_cleanup":
        destination.mkdir()
    original_rename = Path.rename

    def raced_rename(source: Path, target: str | Path) -> Path:
        target_path = Path(target)
        if target_path == destination and ".staging-" in source.name:
            if race == "identical":
                shutil.copytree(tmp_path / "winner", destination)
            elif race == "conflict":
                destination.mkdir()
                (destination / "conflict").write_text("occupied", encoding="utf-8")
            raise OSError("private rename failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", raced_rename)

    if race == "identical":
        result = extract_raw_dataset(
            raw_manifest,
            raw_bundle_root=raw_root,
            model_root=tmp_path / "models",
            config=_config(),
            destination=destination,
            decoder_factory=_decoded_frames,
            runtime_factory=_RuntimeFactory(),
            tracker_factory=_TrackerFactory(),
        )
        assert result.status == "unchanged"
        assert result.manifest == winner.manifest
    else:
        with pytest.raises(ExtractionBatchError) as captured:
            extract_raw_dataset(
                raw_manifest,
                raw_bundle_root=raw_root,
                model_root=tmp_path / "models",
                config=_config(),
                destination=destination,
                decoder_factory=_decoded_frames,
                runtime_factory=_RuntimeFactory(),
                tracker_factory=_TrackerFactory(),
            )
        assert captured.value.category == "publication.failed"
        assert "private" not in str(captured.value)
        if race == "empty_cleanup":
            assert destination.is_dir()
            assert not any(destination.iterdir())
