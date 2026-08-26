from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import ValidationError

from signlab.contracts.core import ArtifactRefV1, ArtifactUriLocatorV1
from signlab.contracts.dataset import DerivedArtifactRowV1
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    HAND_SLOT_IDS,
    BodyAnchorV1,
    ExtractionContractError,
    HandSlotV1,
    LandmarkExtractionManifestV1,
    LandmarkFramesTableV1,
    LandmarkFrameV1,
    LandmarkObservationCountsV1,
    LandmarkSequenceRefV1,
    MediaPipeExtractionConfigV1,
    MediaPipeTaskAssetV1,
    Point3V1,
    assert_landmark_extraction_bound_to_raw_dataset,
    assert_landmark_sequence_ref_matches_table,
    landmark_extraction_manifest_digest,
    landmark_frames_table_digest,
    landmark_observation_counts,
    mediapipe_extraction_config_digest,
    raw_dataset_manifest_digest,
    validate_landmark_extraction_manifest,
    validate_landmark_frames_table,
    validate_mediapipe_extraction_config,
)
from signlab.contracts.ingest import (
    RawDatasetContentV1,
    RawDatasetManifestV1,
    raw_dataset_content_digest,
)
from signlab.datasets.resources import build_example_dataset_manifest
from signlab.governance.resources import build_governance_policy

PARTICIPANT_ID = "participant_00000000000000000000000000000001"
SESSION_ID = "session_00000000000000000000000000000001"
RECORDING_ID = "recording_00000000000000000000000000000001"
DERIVED_ID = "derived_artifact_00000000000000000000000000000001"
ZERO_DIGEST = "sha256:" + "0" * 64


def _json(model: object) -> dict[str, Any]:
    assert hasattr(model, "model_dump_json")
    return cast(dict[str, Any], json.loads(model.model_dump_json(round_trip=True)))


def _task_asset(kind: str) -> MediaPipeTaskAssetV1:
    if kind == "hand_landmarker":
        values = (
            "mediapipe-hand-landmarker-full",
            "hand_landmarker.task",
            "sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
            7_819_105,
        )
    else:
        values = (
            "mediapipe-pose-landmarker-lite",
            "pose_landmarker_lite.task",
            "sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
            5_777_746,
        )
    model_id, filename, sha256, size_bytes = values
    return MediaPipeTaskAssetV1(
        schema_version="mediapipe-task-asset/1",
        task_kind=cast(Any, kind),
        model_id=cast(Any, model_id),
        model_revision="1.0.0",
        filename=cast(Any, filename),
        sha256=sha256,
        size_bytes=size_bytes,
        compatible_runtimes=("browser", "python"),
    )


def _config() -> MediaPipeExtractionConfigV1:
    return MediaPipeExtractionConfigV1(
        schema_version="mediapipe-extraction-config/1",
        config_id="mediapipe_tasks_video",
        version="1.0.0",
        python_package="mediapipe",
        python_package_version="1.0.1",
        browser_package="@mediapipe/tasks-vision",
        browser_package_version="1.0.1",
        decoder_package="av",
        decoder_package_version="18.1.0",
        delegate="CPU",
        running_mode="VIDEO",
        num_hands=2,
        num_poses=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_hand_tracking_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_pose_tracking_confidence=0.5,
        body_anchors=cast(Any, BODY_ANCHOR_NAMES),
        timestamp_rule="source_pts_relative_us_then_strict_monotonic_floor_ms/1",
        tracking_algorithm="deterministic_wrist_mcp_centroid_minimum_cost",
        tracking_algorithm_version="1.0.0",
        max_spatial_cost=0.25,
        handedness_disagreement_penalty=0.05,
        ambiguity_margin=1e-9,
        hand_task_asset=_task_asset("hand_landmarker"),
        pose_task_asset=_task_asset("pose_landmarker"),
    )


def _point(*, pose: bool = False) -> Point3V1:
    return Point3V1(
        x=0.25,
        y=0.5,
        z=-0.125,
        visibility=0.9 if pose else None,
        presence=0.8 if pose else None,
    )


def _present_hand(slot_id: str = "hand_0") -> HandSlotV1:
    points = tuple(_point() for _ in range(21))
    return HandSlotV1(
        slot_id=cast(Any, slot_id),
        present=True,
        detector_index=0,
        tracking_id=cast(Any, slot_id),
        handedness="right",
        handedness_confidence=0.95,
        image_landmarks=points,
        world_landmarks=points,
    )


def _absent_hand(slot_id: str = "hand_1") -> HandSlotV1:
    return HandSlotV1(
        slot_id=cast(Any, slot_id),
        present=False,
        detector_index=None,
        tracking_id=None,
        handedness=None,
        handedness_confidence=None,
        image_landmarks=None,
        world_landmarks=None,
    )


def _anchors(*, present: bool) -> tuple[BodyAnchorV1, ...]:
    return tuple(
        BodyAnchorV1(
            name=cast(Any, name),
            present=present,
            image_point=_point(pose=True) if present else None,
            world_point=_point(pose=True) if present else None,
        )
        for name in BODY_ANCHOR_NAMES
    )


def _frame(
    index: int,
    *,
    source_pts: int,
    invalid: bool = False,
) -> LandmarkFrameV1:
    relative_us = (source_pts - 100) * 1_000_000 // 30
    hands = (
        (_absent_hand("hand_0"), _absent_hand("hand_1"))
        if invalid
        else (_present_hand(), _absent_hand())
    )
    anchors = _anchors(present=not invalid)
    return LandmarkFrameV1(
        schema_version="landmark-frame/1",
        source_recording_id=RECORDING_ID,
        frame_index=index,
        source_pts=source_pts,
        source_time_base_numerator=1,
        source_time_base_denominator=30,
        relative_timestamp_us=relative_us,
        task_timestamp_ms=relative_us // 1_000,
        invalid=invalid,
        invalid_reason="task_inference_failed" if invalid else None,
        hands=cast(Any, hands),
        body_anchors=cast(Any, anchors),
        observed_hand_count=0 if invalid else 1,
        observed_body_anchor_count=0 if invalid else 6,
    )


def _table() -> LandmarkFramesTableV1:
    return LandmarkFramesTableV1(
        schema_version="landmark-frames-table/1",
        rows=(
            _frame(0, source_pts=100),
            _frame(1, source_pts=101, invalid=True),
        ),
    )


def _artifact(
    *,
    artifact_id: str = DERIVED_ID,
    media_type: str = "application/vnd.apache.parquet",
) -> ArtifactRefV1:
    sha256 = "sha256:" + "a" * 64
    digest = sha256.removeprefix("sha256:")
    return ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=artifact_id,
        role="derived_data",
        media_type=media_type,
        sha256=sha256,
        size_bytes=4096,
        locator=ArtifactUriLocatorV1(
            kind="artifact_uri",
            uri=(f"signlab://objects/sha256/p-{digest[:2]}/sha256-{digest}/{artifact_id}"),
        ),
    )


def _lineage(
    *,
    recording_id: str = RECORDING_ID,
    derived_id: str = DERIVED_ID,
) -> DerivedArtifactRowV1:
    return DerivedArtifactRowV1(
        derived_artifact_id=cast(Any, derived_id),
        derivation_kind="landmark_extraction",
        parent_artifact_ids=(recording_id,),
        participant_id=PARTICIPANT_ID,
        session_id=SESSION_ID,
        source_recording_id=recording_id,
        clip_id=None,
        annotation_id=None,
        sample_id=None,
        label_id=None,
        split_id=None,
        partition=None,
        handedness="right",
        mirror_state="mirrored",
        operation_id=_config().config_id,
        operation_version=_config().version,
        artifact=_artifact(artifact_id=derived_id),
    )


def _sequence_ref() -> LandmarkSequenceRefV1:
    table = _table()
    return LandmarkSequenceRefV1(
        schema_version="landmark-sequence-reference/1",
        lineage=_lineage(),
        source_media_sha256="sha256:" + "b" * 64,
        source_media_size_bytes=8192,
        source_rotation_degrees=0,
        source_mirror_state="mirrored",
        frames_schema_version="landmark-frames-table/1",
        content_sha256=landmark_frames_table_digest(table),
        counts=landmark_observation_counts(table),
    )


def _raw_manifest() -> RawDatasetManifestV1:
    policy = build_governance_policy()
    content = RawDatasetContentV1(
        schema_version="raw-dataset-content/1",
        taxonomy=policy.taxonomy,
        governance_policy=policy.policy_document,
        lineage_inventory_sha256="sha256:" + "c" * 64,
        collection_sidecar_sha256="sha256:" + "d" * 64,
        tables=build_example_dataset_manifest().content.tables,
    )
    return RawDatasetManifestV1(
        schema_version="raw-dataset-manifest/1",
        dataset_id="dataset_00000000000000000000000000000001",
        version="1.0.0",
        content=content,
        raw_data_sha256=raw_dataset_content_digest(content),
    )


def _manifest_payload() -> dict[str, Any]:
    raw = _raw_manifest()
    payload: dict[str, Any] = {
        "schema_version": "landmark-extraction-manifest/1",
        "extraction_id": "mediapipe_tasks_extraction",
        "version": "1.0.0",
        "raw_dataset_id": raw.dataset_id,
        "raw_dataset_version": raw.version,
        "raw_data_sha256": raw.raw_data_sha256,
        "raw_dataset_manifest_sha256": raw_dataset_manifest_digest(raw),
        "config": _json(_config()),
        "config_sha256": mediapipe_extraction_config_digest(_config()),
        "sequences": [_json(_sequence_ref())],
        "manifest_sha256": ZERO_DIGEST,
    }
    payload["manifest_sha256"] = landmark_extraction_manifest_digest(payload)
    return payload


def test_registered_assets_and_configuration_are_exact_strict_and_stable() -> None:
    config = _config()
    encoded = config.model_dump_json(round_trip=True)

    assert validate_mediapipe_extraction_config(encoded) == config
    assert mediapipe_extraction_config_digest(encoded) == mediapipe_extraction_config_digest(config)
    assert config.body_anchors == BODY_ANCHOR_NAMES
    assert config.hand_task_asset.compatible_runtimes == ("browser", "python")
    with pytest.raises(ValidationError, match="frozen"):
        config.num_hands = 1  # type: ignore[assignment]

    for field, value in (
        ("delegate", "GPU"),
        ("running_mode", "LIVE_STREAM"),
        ("num_hands", 1),
        ("min_hand_detection_confidence", 0.6),
        ("max_spatial_cost", 0.5),
        ("browser_package_version", "1.0.0"),
        ("decoder_package_version", "18.0.0"),
    ):
        invalid = _json(config)
        invalid[field] = value
        with pytest.raises(ExtractionContractError, match="invalid MediaPipe"):
            validate_mediapipe_extraction_config(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", ZERO_DIGEST),
        ("size_bytes", 1),
        ("model_id", "mediapipe-pose-landmarker-lite"),
        ("filename", "pose_landmarker_lite.task"),
        ("compatible_runtimes", ["python", "browser"]),
    ],
)
def test_task_asset_rejects_any_unregistered_identity_change(field: str, value: object) -> None:
    payload = _json(_task_asset("hand_landmarker"))
    payload[field] = value
    with pytest.raises(ValidationError):
        MediaPipeTaskAssetV1.model_validate_json(json.dumps(payload), strict=True)


def test_configuration_rejects_swapped_task_roles_and_anchor_order() -> None:
    payload = _json(_config())
    payload["hand_task_asset"] = _json(_task_asset("pose_landmarker"))
    with pytest.raises(ValidationError, match="hand_task_asset"):
        MediaPipeExtractionConfigV1.model_validate_json(json.dumps(payload), strict=True)

    payload = _json(_config())
    payload["pose_task_asset"] = _json(_task_asset("hand_landmarker"))
    with pytest.raises(ValidationError, match="pose_task_asset"):
        MediaPipeExtractionConfigV1.model_validate_json(json.dumps(payload), strict=True)

    payload = _json(_config())
    payload["body_anchors"] = list(reversed(BODY_ANCHOR_NAMES))
    with pytest.raises(ValidationError, match="six-anchor order"):
        MediaPipeExtractionConfigV1.model_validate_json(json.dumps(payload), strict=True)


def test_points_hands_and_body_anchors_are_finite_and_explicitly_masked() -> None:
    assert _present_hand().tracking_id == "hand_0"
    assert _absent_hand().image_landmarks is None
    assert _anchors(present=True)[0].image_point is not None

    for field, value in (("x", float("nan")), ("visibility", 1.1)):
        point = _json(_point(pose=True))
        point[field] = value
        with pytest.raises(ValidationError):
            Point3V1.model_validate_json(json.dumps(point), strict=True)

    present = _json(_present_hand())
    present["world_landmarks"] = None
    absent = _json(_absent_hand())
    absent["handedness"] = "left"
    wrong_track = _json(_present_hand())
    wrong_track["tracking_id"] = "hand_1"
    too_few = _json(_present_hand())
    too_few["image_landmarks"] = too_few["image_landmarks"][:-1]
    implicit_absence = _json(_absent_hand())
    implicit_absence.pop("world_landmarks")
    for payload in (present, absent, wrong_track, too_few, implicit_absence):
        with pytest.raises(ValidationError):
            HandSlotV1.model_validate_json(json.dumps(payload), strict=True)

    present_anchor = _json(_anchors(present=True)[0])
    present_anchor["world_point"] = None
    absent_anchor = _json(_anchors(present=False)[0])
    absent_anchor["image_point"] = _json(_point(pose=True))
    for payload in (present_anchor, absent_anchor):
        with pytest.raises(ValidationError):
            BodyAnchorV1.model_validate_json(json.dumps(payload), strict=True)


def test_frames_enforce_slot_anchor_mask_count_and_timestamp_invariants() -> None:
    frame = _frame(0, source_pts=100)
    assert tuple(hand.slot_id for hand in frame.hands) == HAND_SLOT_IDS
    assert frame.observed_hand_count == 1

    mutations: list[dict[str, Any]] = []
    mutators: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda item: item.update(hands=list(reversed(item["hands"]))),
        lambda item: item.update(body_anchors=list(reversed(item["body_anchors"]))),
        lambda item: item.update(observed_hand_count=2),
        lambda item: item.update(observed_body_anchor_count=5),
        lambda item: item.update(invalid=True),
        lambda item: item.update(invalid_reason="task_inference_failed"),
        lambda item: item.update(relative_timestamp_us=1_000),
    )
    for mutate in mutators:
        payload = _json(frame)
        mutate(payload)
        mutations.append(payload)
    invalid_with_points = _json(_frame(1, source_pts=101, invalid=True))
    invalid_with_points["hands"] = _json(frame)["hands"]
    invalid_with_points["observed_hand_count"] = 1
    mutations.append(invalid_with_points)

    for payload in mutations:
        with pytest.raises(ValidationError):
            LandmarkFrameV1.model_validate_json(json.dumps(payload), strict=True)


def test_frame_table_has_one_exact_monotonic_source_timeline_and_stable_digest() -> None:
    table = _table()
    encoded = table.model_dump_json(round_trip=True)

    assert validate_landmark_frames_table(encoded) == table
    assert landmark_frames_table_digest(encoded) == landmark_frames_table_digest(table)

    mutations: list[dict[str, Any]] = []
    for row_index, field, value in (
        (1, "frame_index", 2),
        (1, "source_recording_id", "recording_ffffffffffffffffffffffffffffffff"),
        (1, "source_time_base_denominator", 60),
        (0, "relative_timestamp_us", 1),
        (1, "source_pts", 100),
        (1, "relative_timestamp_us", 33334),
        (1, "task_timestamp_ms", 0),
    ):
        payload = _json(table)
        payload["rows"][row_index][field] = value
        if field == "relative_timestamp_us":
            assert isinstance(value, int)
            payload["rows"][row_index]["task_timestamp_ms"] = value // 1_000
        mutations.append(payload)
    for payload in mutations:
        with pytest.raises(ExtractionContractError, match="invalid landmark frames table"):
            validate_landmark_frames_table(payload)

    collision = _json(table)
    for row in collision["rows"]:
        row["source_time_base_denominator"] = 2_000_000
    collision["rows"][1]["relative_timestamp_us"] = 0
    collision["rows"][1]["task_timestamp_ms"] = 1
    assert validate_landmark_frames_table(collision).rows[1].task_timestamp_ms == 1
    collision["rows"][1]["task_timestamp_ms"] = 0
    with pytest.raises(ExtractionContractError, match="invalid landmark frames table"):
        validate_landmark_frames_table(collision)


def test_observation_counts_are_derived_without_quality_judgments() -> None:
    counts = landmark_observation_counts(_table())
    assert counts == LandmarkObservationCountsV1(
        frame_count=2,
        valid_frame_count=1,
        invalid_frame_count=1,
        zero_hand_frame_count=0,
        one_hand_frame_count=1,
        two_hand_frame_count=0,
        hand_observation_count=1,
        body_anchor_observation_count=6,
        body_anchor_presence_counts=(1, 1, 1, 1, 1, 1),
    )

    for update in (
        {"invalid_frame_count": 0},
        {"hand_observation_count": 2},
        {"zero_hand_frame_count": 1},
        {"body_anchor_observation_count": 7},
        {"body_anchor_presence_counts": [2, 1, 1, 1, 1, 0]},
    ):
        payload = _json(counts)
        payload.update(update)
        with pytest.raises(ValidationError):
            LandmarkObservationCountsV1.model_validate_json(json.dumps(payload), strict=True)


def test_sequence_reference_reuses_lineage_and_binds_semantic_rows() -> None:
    reference = _sequence_ref()
    table = _table()
    assert_landmark_sequence_ref_matches_table(reference, table)

    wrong_source = _json(table)
    for row in wrong_source["rows"]:
        row["source_recording_id"] = "recording_ffffffffffffffffffffffffffffffff"
    wrong_content = _json(reference)
    wrong_content["content_sha256"] = ZERO_DIGEST
    wrong_counts = _json(reference)
    wrong_counts["counts"]["hand_observation_count"] = 0
    wrong_counts["counts"]["one_hand_frame_count"] = 0
    wrong_counts["counts"]["zero_hand_frame_count"] = 1
    with pytest.raises(ExtractionContractError, match="source recording"):
        assert_landmark_sequence_ref_matches_table(
            reference,
            LandmarkFramesTableV1.model_validate_json(json.dumps(wrong_source), strict=True),
        )
    with pytest.raises(ExtractionContractError, match="semantic digest"):
        assert_landmark_sequence_ref_matches_table(
            LandmarkSequenceRefV1.model_validate_json(json.dumps(wrong_content), strict=True),
            table,
        )
    with pytest.raises(ExtractionContractError, match="observation counts"):
        assert_landmark_sequence_ref_matches_table(
            LandmarkSequenceRefV1.model_validate_json(json.dumps(wrong_counts), strict=True),
            table,
        )


def test_sequence_reference_rejects_nonraw_or_incompatible_lineage() -> None:
    reference = _json(_sequence_ref())
    cases: list[dict[str, Any]] = []
    mutators: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda item: item["lineage"].update(derivation_kind="feature_extraction"),
        lambda item: item["lineage"].update(parent_artifact_ids=["parent_recording"]),
        lambda item: item.update(source_mirror_state="not_mirrored"),
        lambda item: item["lineage"]["artifact"].update(media_type="application/json"),
    )
    for mutate in mutators:
        payload = copy.deepcopy(reference)
        mutate(payload)
        cases.append(payload)

    sample_bound = copy.deepcopy(reference)
    sample_bound["lineage"].update(
        clip_id="clip_00000000000000000000000000000001",
        annotation_id="annotation_00000000000000000000000000000001",
        sample_id="sample_00000000000000000000000000000001",
        label_id="hello",
        split_id="grouped_split",
        partition="train",
    )
    sample_bound["lineage"]["artifact"] = _json(
        _artifact(artifact_id="sample_00000000000000000000000000000001")
    )
    sample_bound["lineage"]["artifact"]["role"] = "sample_data"
    cases.append(sample_bound)

    for payload in cases:
        with pytest.raises(ValidationError):
            LandmarkSequenceRefV1.model_validate_json(json.dumps(payload), strict=True)


def test_manifest_is_self_digested_ordered_and_bound_to_exact_config() -> None:
    payload = _manifest_payload()
    checked = validate_landmark_extraction_manifest(json.dumps(payload))

    assert isinstance(checked, LandmarkExtractionManifestV1)
    assert checked.manifest_sha256 == landmark_extraction_manifest_digest(checked)

    wrong_digest = copy.deepcopy(payload)
    wrong_digest["manifest_sha256"] = ZERO_DIGEST
    with pytest.raises(ExtractionContractError, match="invalid landmark extraction manifest"):
        validate_landmark_extraction_manifest(wrong_digest)

    wrong_operation = copy.deepcopy(payload)
    wrong_operation["sequences"][0]["lineage"]["operation_id"] = "other_extractor"
    wrong_operation["manifest_sha256"] = landmark_extraction_manifest_digest(wrong_operation)
    with pytest.raises(ExtractionContractError):
        validate_landmark_extraction_manifest(wrong_operation)

    wrong_config_digest = copy.deepcopy(payload)
    wrong_config_digest["config_sha256"] = ZERO_DIGEST
    wrong_config_digest["manifest_sha256"] = landmark_extraction_manifest_digest(
        wrong_config_digest
    )
    with pytest.raises(ExtractionContractError):
        validate_landmark_extraction_manifest(wrong_config_digest)

    duplicate = copy.deepcopy(payload)
    duplicate["sequences"].append(copy.deepcopy(duplicate["sequences"][0]))
    duplicate["manifest_sha256"] = landmark_extraction_manifest_digest(duplicate)
    with pytest.raises(ExtractionContractError):
        validate_landmark_extraction_manifest(duplicate)


def test_manifest_rejects_duplicate_artifact_ids_even_across_unique_recordings() -> None:
    payload = _manifest_payload()
    second = copy.deepcopy(payload["sequences"][0])
    second_recording = "recording_ffffffffffffffffffffffffffffffff"
    second["lineage"]["source_recording_id"] = second_recording
    second["lineage"]["parent_artifact_ids"] = [second_recording]
    payload["sequences"].append(second)
    payload["manifest_sha256"] = landmark_extraction_manifest_digest(payload)

    with pytest.raises(ExtractionContractError):
        validate_landmark_extraction_manifest(payload)


def test_manifest_raw_dataset_binding_is_exact_and_reader_fails_closed() -> None:
    raw = _raw_manifest()
    manifest = validate_landmark_extraction_manifest(_manifest_payload())

    assert raw_dataset_manifest_digest(raw.model_dump_json(round_trip=True)) == (
        raw_dataset_manifest_digest(raw)
    )
    assert_landmark_extraction_bound_to_raw_dataset(manifest, raw)

    changed = _manifest_payload()
    changed["raw_data_sha256"] = ZERO_DIGEST
    changed["manifest_sha256"] = landmark_extraction_manifest_digest(changed)
    mismatched = validate_landmark_extraction_manifest(changed)
    with pytest.raises(ExtractionContractError, match="does not match"):
        assert_landmark_extraction_bound_to_raw_dataset(mismatched, raw)

    with pytest.raises(ExtractionContractError, match="invalid raw dataset"):
        raw_dataset_manifest_digest({"schema_version": "raw-dataset-manifest/2"})
    with pytest.raises(ExtractionContractError, match="invalid landmark frames table"):
        validate_landmark_frames_table({"schema_version": "landmark-frames-table/2"})


def test_public_readers_reject_extra_fields_and_non_object_json() -> None:
    config = _json(_config())
    config["unregistered"] = True
    with pytest.raises(ExtractionContractError):
        validate_mediapipe_extraction_config(config)
    with pytest.raises(ExtractionContractError):
        validate_landmark_extraction_manifest("[]")
    with pytest.raises(ExtractionContractError):
        landmark_extraction_manifest_digest({"bad": object()})
