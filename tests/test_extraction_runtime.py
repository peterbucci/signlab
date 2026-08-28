from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest

import signlab.extraction.runtime as runtime
from signlab.extraction.runtime import (
    DecodedFrame,
    ExtractionRuntimeConfig,
    ExtractionRuntimeError,
    MediaPipeVideoRuntime,
    ModelAssetSpec,
    VerifiedModelAssets,
    normalize_hand_result,
    normalize_pose_result,
    verify_model_assets,
)


def _spec(filename: str, content: bytes) -> ModelAssetSpec:
    return ModelAssetSpec(
        filename=filename,
        size_bytes=len(content),
        sha256=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def _tiny_model_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    hand = b"synthetic hand task"
    pose = b"synthetic pose task"
    hand_spec = _spec("hand_landmarker.task", hand)
    pose_spec = _spec("pose_landmarker_lite.task", pose)
    monkeypatch.setattr(runtime, "HAND_MODEL_SPEC", hand_spec)
    monkeypatch.setattr(runtime, "POSE_MODEL_SPEC", pose_spec)
    root = tmp_path / "private-model-root"
    root.mkdir()
    (root / hand_spec.filename).write_bytes(hand)
    (root / pose_spec.filename).write_bytes(pose)
    return root


def test_model_assets_are_verified_and_detached_from_the_source_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _tiny_model_root(monkeypatch, tmp_path)

    assets = verify_model_assets(root)

    assert assets.hand_model_bytes == b"synthetic hand task"
    assert assets.pose_model_bytes == b"synthetic pose task"
    assert assets.hand_model_sha256 == runtime.HAND_MODEL_SPEC.sha256
    assert assets.pose_model_sha256 == runtime.POSE_MODEL_SPEC.sha256
    assert str(root) not in repr(assets)
    assert "synthetic" not in repr(assets)


@pytest.mark.parametrize("failure", ["missing", "size", "digest"])
def test_invalid_model_assets_fail_without_disclosing_the_model_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    root = _tiny_model_root(monkeypatch, tmp_path)
    hand_path = root / runtime.HAND_MODEL_SPEC.filename
    if failure == "missing":
        hand_path.unlink()
    elif failure == "size":
        hand_path.write_bytes(b"wrong size")
    else:
        hand_path.write_bytes(b"X" * runtime.HAND_MODEL_SPEC.size_bytes)

    with pytest.raises(ExtractionRuntimeError) as captured:
        verify_model_assets(root)

    assert captured.value.code == "extraction.models.invalid"
    assert str(root) not in str(captured.value)
    assert runtime.HAND_MODEL_SPEC.filename not in str(captured.value)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_thresholds_must_be_finite_unit_interval_values(value: float) -> None:
    with pytest.raises(ExtractionRuntimeError) as captured:
        ExtractionRuntimeConfig(hand_detection_confidence=value)

    assert captured.value.code == "extraction.config.invalid"


class _FakeVideoFrame:
    def __init__(self, pts: int | None, time_base: object, rgb: object) -> None:
        self.pts = pts
        self.time_base = time_base
        self.rgb = rgb
        self.formats: list[str] = []

    def to_ndarray(self, *, format: str) -> object:
        self.formats.append(format)
        return self.rgb


class _FakeContainer:
    def __init__(self, frames: tuple[_FakeVideoFrame, ...], *, stream_count: int = 1) -> None:
        self._frames = frames
        self._streams = tuple(object() for _ in range(stream_count))
        self.streams = SimpleNamespace(video=self._streams)
        self.closed = False

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.closed = True

    def decode(self, stream: object) -> Iterator[_FakeVideoFrame]:
        assert stream is self._streams[0]
        yield from self._frames


class _FakeAv:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container
        self.calls: list[tuple[str, str]] = []

    def open(self, source: str, *, mode: str) -> _FakeContainer:
        self.calls.append((source, mode))
        return self.container


def test_decoder_preserves_source_timing_and_makes_task_milliseconds_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rgb_values = (object(), object(), object())
    frames = tuple(
        _FakeVideoFrame(pts, Fraction(1, 2_000_000), rgb)
        for pts, rgb in zip((100, 101, 2_101), rgb_values, strict=True)
    )
    fake_av = _FakeAv(_FakeContainer(frames))
    monkeypatch.setattr(runtime, "_load_av_module", lambda: fake_av)

    decoded = tuple(runtime.iter_decoded_frames(tmp_path / "private-video.mp4"))

    assert [frame.source_pts for frame in decoded] == [100, 101, 2_101]
    assert [frame.source_time_base_numerator for frame in decoded] == [1, 1, 1]
    assert [frame.source_time_base_denominator for frame in decoded] == [2_000_000] * 3
    assert [frame.relative_timestamp_us for frame in decoded] == [0, 0, 1_000]
    assert [frame.task_timestamp_ms for frame in decoded] == [0, 1, 2]
    assert [frame.rgb for frame in decoded] == list(rgb_values)
    assert all(frame.formats == ["rgb24"] for frame in frames)
    assert fake_av.calls[0][1] == "r"
    assert fake_av.container.closed


def test_decoder_detaches_padded_rgb_rows_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    padded = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    rgb = padded[:, :3, :]
    assert not rgb.flags.c_contiguous
    frame = _FakeVideoFrame(100, Fraction(1, 30), rgb)
    monkeypatch.setattr(
        runtime,
        "_load_av_module",
        lambda: _FakeAv(_FakeContainer((frame,))),
    )

    decoded = tuple(runtime.iter_decoded_frames(tmp_path / "private-video.mp4"))

    assert len(decoded) == 1
    assert isinstance(decoded[0].rgb, np.ndarray)
    assert decoded[0].rgb.flags.c_contiguous
    assert np.array_equal(decoded[0].rgb, rgb)


def test_decoder_retains_timing_when_rgb_conversion_marks_source_frame_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _FakeVideoFrame(100, Fraction(1, 30), object())

    def fail_conversion(*, format: str) -> object:
        assert format == "rgb24"
        raise ValueError("private decoder detail")

    monkeypatch.setattr(frame, "to_ndarray", fail_conversion)
    monkeypatch.setattr(
        runtime,
        "_load_av_module",
        lambda: _FakeAv(_FakeContainer((frame,))),
    )

    decoded = tuple(runtime.iter_decoded_frames(tmp_path / "private-video.mp4"))

    assert len(decoded) == 1
    assert decoded[0].source_valid is False
    assert decoded[0].rgb is None
    assert decoded[0].task_timestamp_ms == 0


@pytest.mark.parametrize(
    ("frames", "stream_count"),
    [
        (
            (
                _FakeVideoFrame(2, Fraction(1, 1_000), object()),
                _FakeVideoFrame(1, Fraction(1, 1_000), object()),
            ),
            1,
        ),
        (
            (
                _FakeVideoFrame(1, Fraction(1, 1_000), object()),
                _FakeVideoFrame(1, Fraction(1, 1_000), object()),
            ),
            1,
        ),
        ((_FakeVideoFrame(None, Fraction(1, 1_000), object()),), 1),
        ((), 1),
        ((_FakeVideoFrame(1, Fraction(1, 1_000), object()),), 2),
    ],
)
def test_decoder_rejects_invalid_stream_or_timestamp_state_without_source_disclosure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frames: tuple[_FakeVideoFrame, ...],
    stream_count: int,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_av_module",
        lambda: _FakeAv(_FakeContainer(frames, stream_count=stream_count)),
    )
    source = tmp_path / "private-sentinel.mp4"

    with pytest.raises(ExtractionRuntimeError) as captured:
        tuple(runtime.iter_decoded_frames(source))

    assert captured.value.code in {
        "extraction.decoder.invalid",
        "extraction.decoder.timestamp.invalid",
    }
    assert str(source) not in str(captured.value)


def _landmark(index: int, *, pose: bool = False) -> SimpleNamespace:
    confidence = float(index % 10) / 10.0
    return SimpleNamespace(
        x=float(index),
        y=float(index) + 0.25,
        z=-float(index),
        visibility=confidence if pose else None,
        presence=(1.0 - confidence) if pose else None,
    )


def _hand_result() -> SimpleNamespace:
    return SimpleNamespace(
        hand_landmarks=[[_landmark(index) for index in range(21)]],
        hand_world_landmarks=[[_landmark(index) for index in range(21)]],
        handedness=[[SimpleNamespace(category_name="Left", score=0.9)]],
    )


def _pose_result() -> SimpleNamespace:
    return SimpleNamespace(
        pose_landmarks=[[_landmark(index, pose=True) for index in range(33)]],
        pose_world_landmarks=[[_landmark(index, pose=True) for index in range(33)]],
    )


class _BaseOptions:
    class Delegate:
        CPU = "CPU"

    def __init__(self, *, model_asset_buffer: bytes, delegate: object) -> None:
        self.model_asset_buffer = model_asset_buffer
        self.delegate = delegate


class _HandOptions:
    last: ClassVar[_HandOptions | None] = None

    def __init__(self, **values: object) -> None:
        self.values = values
        type(self).last = self


class _PoseOptions:
    last: ClassVar[_PoseOptions | None] = None

    def __init__(self, **values: object) -> None:
        self.values = values
        type(self).last = self


class _FakeTask:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[object, int]] = []
        self.closed = False

    def detect_for_video(self, image: object, timestamp_ms: int) -> object:
        self.calls.append((image, timestamp_ms))
        return self.result

    def close(self) -> None:
        self.closed = True


class _HandFactory:
    instance: ClassVar[_FakeTask | None] = None

    @classmethod
    def create_from_options(cls, options: object) -> _FakeTask:
        assert options is _HandOptions.last
        cls.instance = _FakeTask(_hand_result())
        return cls.instance


class _PoseFactory:
    instance: ClassVar[_FakeTask | None] = None

    @classmethod
    def create_from_options(cls, options: object) -> _FakeTask:
        assert options is _PoseOptions.last
        cls.instance = _FakeTask(_pose_result())
        return cls.instance


class _Image:
    calls: ClassVar[list[tuple[object, object]]] = []

    def __init__(self, *, image_format: object, data: object) -> None:
        self.calls.append((image_format, data))


def _fake_mediapipe() -> SimpleNamespace:
    vision = SimpleNamespace(
        RunningMode=SimpleNamespace(VIDEO="VIDEO"),
        HandLandmarkerOptions=_HandOptions,
        PoseLandmarkerOptions=_PoseOptions,
        HandLandmarker=_HandFactory,
        PoseLandmarker=_PoseFactory,
    )
    return SimpleNamespace(
        tasks=SimpleNamespace(BaseOptions=_BaseOptions, vision=vision),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
        Image=_Image,
    )


def test_runtime_builds_pinned_cpu_video_tasks_and_normalizes_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _tiny_model_root(monkeypatch, tmp_path)
    assets = verify_model_assets(root)
    monkeypatch.setattr(runtime, "_load_mediapipe_module", _fake_mediapipe)
    config = ExtractionRuntimeConfig(
        hand_detection_confidence=0.51,
        hand_presence_confidence=0.52,
        hand_tracking_confidence=0.53,
        pose_detection_confidence=0.61,
        pose_presence_confidence=0.62,
        pose_tracking_confidence=0.63,
    )
    rgb = object()
    frame = DecodedFrame(4, 40, 1, 30, 1_333_333, 1_333, rgb)

    with MediaPipeVideoRuntime(assets, config) as detector:
        result = detector.infer_frame(frame)

    assert _HandOptions.last is not None
    assert _PoseOptions.last is not None
    hand_values = _HandOptions.last.values
    pose_values = _PoseOptions.last.values
    assert hand_values["running_mode"] == "VIDEO"
    assert hand_values["num_hands"] == 2
    assert hand_values["min_hand_detection_confidence"] == 0.51
    assert hand_values["min_hand_presence_confidence"] == 0.52
    assert hand_values["min_tracking_confidence"] == 0.53
    assert pose_values["running_mode"] == "VIDEO"
    assert pose_values["num_poses"] == 1
    assert pose_values["min_pose_detection_confidence"] == 0.61
    assert pose_values["min_pose_presence_confidence"] == 0.62
    assert pose_values["min_tracking_confidence"] == 0.63
    assert pose_values["output_segmentation_masks"] is False
    assert isinstance(hand_values["base_options"], _BaseOptions)
    assert isinstance(pose_values["base_options"], _BaseOptions)
    assert hand_values["base_options"].delegate == "CPU"
    assert pose_values["base_options"].delegate == "CPU"
    assert result.frame_index == 4
    assert result.task_timestamp_ms == 1_333
    assert result.hands[0].reported_handedness == "left"
    assert result.hands[0].handedness_score == 0.9
    assert len(result.hands[0].image_landmarks) == 21
    assert [anchor.name for anchor in result.pose_anchors] == [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]
    assert [anchor.landmark_index for anchor in result.pose_anchors] == [11, 12, 13, 14, 15, 16]
    assert result.pose_anchors[0].image_landmark.visibility == 0.1
    assert _HandFactory.instance is not None
    assert _HandFactory.instance.closed
    assert _PoseFactory.instance is not None
    assert _PoseFactory.instance.closed
    assert _HandFactory.instance.calls[0][1] == frame.task_timestamp_ms
    assert _PoseFactory.instance.calls[0][1] == frame.task_timestamp_ms
    assert _Image.calls[-1] == ("SRGB", rgb)


def test_runtime_refuses_an_invalid_source_frame_before_initializing_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    detector = MediaPipeVideoRuntime(assets)
    frame = DecodedFrame(0, 100, 1, 30, 0, 0, None, source_valid=False)

    with pytest.raises(ExtractionRuntimeError) as captured:
        detector.infer_frame(frame)

    assert captured.value.code == "extraction.result.invalid"


def test_runtime_rejects_wrong_mediapipe_version_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_installed_mediapipe_version", lambda: "1.0.2")

    with pytest.raises(ExtractionRuntimeError) as captured:
        runtime._load_mediapipe_module()

    assert captured.value.code == "extraction.dependency.version.invalid"


def test_decoder_rejects_wrong_pyav_version_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_installed_av_version", lambda: "18.1.1")

    with pytest.raises(ExtractionRuntimeError) as captured:
        runtime._load_av_module()

    assert captured.value.code == "extraction.dependency.version.invalid"


def test_vendor_results_fail_closed_on_shape_or_nonfinite_data() -> None:
    invalid_hand = _hand_result()
    invalid_hand.hand_landmarks[0][0].x = float("nan")
    invalid_pose = _pose_result()
    invalid_pose.pose_world_landmarks[0].pop()

    with pytest.raises(ExtractionRuntimeError) as hand_error:
        normalize_hand_result(invalid_hand)
    with pytest.raises(ExtractionRuntimeError) as pose_error:
        normalize_pose_result(invalid_pose)

    assert hand_error.value.code == "extraction.result.invalid"
    assert pose_error.value.code == "extraction.result.invalid"


def test_closed_runtime_cannot_be_reopened(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    detector = MediaPipeVideoRuntime(assets)
    detector.close()

    with pytest.raises(ExtractionRuntimeError) as captured:
        detector.open()

    assert captured.value.code == "extraction.runtime.closed"


def test_model_root_and_asset_links_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-model-root"
    with pytest.raises(ExtractionRuntimeError) as missing:
        verify_model_assets(missing_root)
    assert missing.value.code == "extraction.models.invalid"

    regular_file = tmp_path / "model-root-file"
    regular_file.write_bytes(b"not a directory")
    with pytest.raises(ExtractionRuntimeError) as not_directory:
        verify_model_assets(regular_file)
    assert not_directory.value.code == "extraction.models.invalid"

    root = _tiny_model_root(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "_is_link", lambda path: path == root)
    with pytest.raises(ExtractionRuntimeError) as linked_root:
        verify_model_assets(root)
    assert linked_root.value.code == "extraction.models.invalid"

    monkeypatch.setattr(
        runtime,
        "_is_link",
        lambda path: path.name == runtime.HAND_MODEL_SPEC.filename,
    )
    with pytest.raises(ExtractionRuntimeError) as linked_asset:
        verify_model_assets(root)
    assert linked_asset.value.code == "extraction.models.invalid"


def test_model_asset_io_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _tiny_model_root(monkeypatch, tmp_path)
    original_read_bytes = Path.read_bytes

    def fail_private_read(path: Path) -> bytes:
        if path.name == runtime.HAND_MODEL_SPEC.filename:
            raise OSError("private model filesystem detail")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_private_read)
    with pytest.raises(ExtractionRuntimeError) as captured:
        verify_model_assets(root)

    assert captured.value.code == "extraction.models.invalid"
    assert "private model" not in str(captured.value)
    assert str(root) not in str(captured.value)


def test_dependency_version_probes_and_successful_lazy_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"mediapipe": runtime.MEDIAPIPE_VERSION, "av": runtime.AV_VERSION}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    assert runtime._installed_mediapipe_version() == runtime.MEDIAPIPE_VERSION
    assert runtime._installed_av_version() == runtime.AV_VERSION

    imported: list[str] = []
    sentinel = object()

    def fake_import(name: str) -> object:
        imported.append(name)
        return sentinel

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert runtime._load_mediapipe_module() is sentinel
    assert runtime._load_av_module() is sentinel
    assert imported == ["mediapipe", "av"]


def test_missing_and_broken_optional_dependencies_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_package() -> str:
        raise importlib.metadata.PackageNotFoundError("private package detail")

    monkeypatch.setattr(runtime, "_installed_mediapipe_version", missing_package)
    with pytest.raises(ExtractionRuntimeError) as missing_mediapipe:
        runtime._load_mediapipe_module()
    assert missing_mediapipe.value.code == "extraction.dependency.unavailable"
    monkeypatch.undo()

    monkeypatch.setattr(runtime, "_installed_mediapipe_version", lambda: runtime.MEDIAPIPE_VERSION)

    def broken_import(_name: str) -> object:
        raise RuntimeError("private import detail")

    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ExtractionRuntimeError) as broken_mediapipe:
        runtime._load_mediapipe_module()
    assert broken_mediapipe.value.code == "extraction.dependency.unavailable"
    assert "private import" not in str(broken_mediapipe.value)
    monkeypatch.undo()

    monkeypatch.setattr(runtime, "_installed_av_version", missing_package)
    with pytest.raises(ExtractionRuntimeError) as missing_av:
        runtime._load_av_module()
    assert missing_av.value.code == "extraction.dependency.unavailable"


class _FailingOpenAv:
    def open(self, _source: str, *, mode: str) -> object:
        assert mode == "r"
        raise OSError("private decoder open detail")


class _FailingDecodeContainer(_FakeContainer):
    def decode(self, stream: object) -> Iterator[_FakeVideoFrame]:
        assert stream is self._streams[0]
        raise RuntimeError("private decode detail")
        yield  # pragma: no cover


def test_decoder_open_and_decode_failures_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "private-video.mp4"
    monkeypatch.setattr(runtime, "_load_av_module", _FailingOpenAv)
    with pytest.raises(ExtractionRuntimeError) as open_error:
        tuple(runtime.iter_decoded_frames(source))
    assert open_error.value.code == "extraction.decoder.invalid"
    assert str(source) not in str(open_error.value)

    container = _FailingDecodeContainer(())
    monkeypatch.setattr(runtime, "_load_av_module", lambda: _FakeAv(container))
    with pytest.raises(ExtractionRuntimeError) as decode_error:
        tuple(runtime.iter_decoded_frames(source))
    assert decode_error.value.code == "extraction.decoder.invalid"
    assert container.closed
    assert "private decode" not in str(decode_error.value)


@pytest.mark.parametrize(
    ("pts", "numerator", "denominator"),
    [
        (True, 1, 1_000),
        (1.5, 1, 1_000),
        (1, True, 1_000),
        (1, 0, 1_000),
        (1, 1, True),
        (1, 1, 0),
        (1, None, None),
    ],
)
def test_decoder_rejects_invalid_pts_and_time_base_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pts: object,
    numerator: object,
    denominator: object,
) -> None:
    time_base = SimpleNamespace(numerator=numerator, denominator=denominator)
    frame = _FakeVideoFrame(cast(Any, pts), time_base, object())
    monkeypatch.setattr(
        runtime,
        "_load_av_module",
        lambda: _FakeAv(_FakeContainer((frame,))),
    )

    with pytest.raises(ExtractionRuntimeError) as captured:
        tuple(runtime.iter_decoded_frames(tmp_path / "private-video.mp4"))
    assert captured.value.code == "extraction.decoder.timestamp.invalid"


def _assert_invalid_vendor_result(
    normalizer: Callable[[object], object],
    value: object,
) -> None:
    with pytest.raises(ExtractionRuntimeError) as captured:
        normalizer(value)
    assert captured.value.code == "extraction.result.invalid"


def test_hand_normalization_rejects_malformed_vendor_shapes_labels_and_scores() -> None:
    too_many = SimpleNamespace(
        hand_landmarks=[[_landmark(index) for index in range(21)]] * 3,
        hand_world_landmarks=[[_landmark(index) for index in range(21)]] * 3,
        handedness=[[SimpleNamespace(category_name="Left", score=0.9)]] * 3,
    )
    mismatch = _hand_result()
    mismatch.hand_world_landmarks = []
    empty_categories = _hand_result()
    empty_categories.handedness = [[]]
    invalid_label = _hand_result()
    invalid_label.handedness[0][0].category_name = "unknown"
    boolean_score = _hand_result()
    boolean_score.handedness[0][0].score = True
    out_of_range_score = _hand_result()
    out_of_range_score.handedness[0][0].score = 1.1
    short_landmarks = _hand_result()
    short_landmarks.hand_landmarks[0].pop()
    missing_coordinate = _hand_result()
    missing_coordinate.hand_landmarks[0][0] = SimpleNamespace(x=0.0, y=0.0)
    non_sequence = _hand_result()
    non_sequence.hand_landmarks = "vendor string is not a landmark sequence"

    for result in (
        too_many,
        mismatch,
        empty_categories,
        invalid_label,
        boolean_score,
        out_of_range_score,
        short_landmarks,
        missing_coordinate,
        non_sequence,
        SimpleNamespace(),
    ):
        _assert_invalid_vendor_result(normalize_hand_result, result)

    assert (
        normalize_hand_result(
            SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])
        )
        == ()
    )


def test_pose_normalization_handles_absence_and_rejects_malformed_vendor_shapes() -> None:
    assert normalize_pose_result(SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])) == ()

    mismatch = _pose_result()
    mismatch.pose_world_landmarks = []
    too_many = SimpleNamespace(
        pose_landmarks=[[_landmark(index, pose=True) for index in range(33)]] * 2,
        pose_world_landmarks=[[_landmark(index, pose=True) for index in range(33)]] * 2,
    )
    short_pose = _pose_result()
    short_pose.pose_landmarks[0].pop()
    missing_coordinate = _pose_result()
    missing_coordinate.pose_world_landmarks[0][0] = SimpleNamespace(x=0.0, y=0.0)

    for result in (mismatch, too_many, short_pose, missing_coordinate, SimpleNamespace()):
        _assert_invalid_vendor_result(normalize_pose_result, result)


def test_runtime_initialization_failure_closes_partial_native_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    mp = _fake_mediapipe()

    class FailingPoseFactory:
        @classmethod
        def create_from_options(cls, _options: object) -> object:
            raise RuntimeError("private pose initialization detail")

    _HandFactory.instance = None
    mp.tasks.vision.PoseLandmarker = FailingPoseFactory
    monkeypatch.setattr(runtime, "_load_mediapipe_module", lambda: mp)
    detector = MediaPipeVideoRuntime(assets)

    with pytest.raises(ExtractionRuntimeError) as captured:
        detector.open()

    assert captured.value.code == "extraction.runtime.initialization.failed"
    assert "private pose" not in str(captured.value)
    assert _HandFactory.instance is not None
    assert _HandFactory.instance.closed


def test_runtime_open_and_close_are_idempotent_and_closed_inference_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    monkeypatch.setattr(runtime, "_load_mediapipe_module", _fake_mediapipe)
    detector = MediaPipeVideoRuntime(assets)

    detector.open()
    hand = _HandFactory.instance
    pose = _PoseFactory.instance
    detector.open()
    assert _HandFactory.instance is hand
    assert _PoseFactory.instance is pose

    detector.close()
    detector.close()
    frame = DecodedFrame(0, 1, 1, 30, 0, 0, object())
    with pytest.raises(ExtractionRuntimeError) as captured:
        detector.infer_frame(frame)
    assert captured.value.code == "extraction.runtime.closed"


def test_runtime_close_suppresses_one_native_close_failure_and_closes_the_other(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    detector = MediaPipeVideoRuntime(assets)
    other = _FakeTask(_pose_result())

    class BrokenCloser:
        def close(self) -> None:
            raise RuntimeError("private close detail")

    detector._hand_landmarker = BrokenCloser()
    detector._pose_landmarker = other
    detector.close()

    assert other.closed


def test_runtime_revalidates_detached_asset_identity_and_malformed_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    changed = VerifiedModelAssets(
        hand_model_bytes=assets.hand_model_bytes + b"changed",
        pose_model_bytes=assets.pose_model_bytes,
        hand_model_sha256=assets.hand_model_sha256,
        pose_model_sha256=assets.pose_model_sha256,
    )
    with pytest.raises(ExtractionRuntimeError) as changed_error:
        MediaPipeVideoRuntime(changed).open()
    assert changed_error.value.code == "extraction.models.invalid"

    malformed = cast(VerifiedModelAssets, SimpleNamespace())
    with pytest.raises(ExtractionRuntimeError) as malformed_error:
        MediaPipeVideoRuntime(malformed).open()
    assert malformed_error.value.code == "extraction.models.invalid"


def test_runtime_detects_impossible_partial_state_after_idempotent_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    detector = MediaPipeVideoRuntime(assets)
    detector._hand_landmarker = object()
    detector._pose_landmarker = object()
    frame = DecodedFrame(0, 1, 1, 30, 0, 0, object())

    with pytest.raises(ExtractionRuntimeError) as captured:
        detector.infer_frame(frame)
    assert captured.value.code == "extraction.runtime.initialization.failed"


def test_runtime_auto_initializes_and_sanitizes_vendor_inference_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    monkeypatch.setattr(runtime, "_load_mediapipe_module", _fake_mediapipe)
    detector = MediaPipeVideoRuntime(assets)
    frame = DecodedFrame(0, 1, 1, 30, 0, 0, object())

    successful = detector.infer_frame(frame)
    assert successful.frame_index == 0
    assert _HandFactory.instance is not None

    def fail_inference(_image: object, _timestamp_ms: int) -> object:
        raise RuntimeError("private inference detail")

    monkeypatch.setattr(_HandFactory.instance, "detect_for_video", fail_inference)
    with pytest.raises(ExtractionRuntimeError) as captured:
        detector.infer_frame(frame)
    assert captured.value.code == "extraction.runtime.inference.failed"
    assert "private inference" not in str(captured.value)
    detector.close()


def test_runtime_preserves_normalizer_error_code_during_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = verify_model_assets(_tiny_model_root(monkeypatch, tmp_path))
    monkeypatch.setattr(runtime, "_load_mediapipe_module", _fake_mediapipe)
    detector = MediaPipeVideoRuntime(assets)
    detector.open()
    assert _HandFactory.instance is not None
    _HandFactory.instance.result = SimpleNamespace()

    with pytest.raises(ExtractionRuntimeError) as captured:
        detector.infer_frame(DecodedFrame(0, 1, 1, 30, 0, 0, object()))
    assert captured.value.code == "extraction.result.invalid"
    detector.close()
