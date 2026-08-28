from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Literal, cast

import pytest

from feature_fixtures import make_split_manifest
from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.features import (
    BodyRelativeRuleV1,
    FeatureCacheKeyV1,
    HandLocalRuleV1,
    LandmarkFeaturePlanV1,
    LearnedStatisticsRuleV1,
    OptionalFeatureRuleV1,
    PaddingFeatureRuleV1,
    PortableFeatureSequenceV1,
    TemporalFeatureRuleV1,
    TrainingFeatureSequenceV1,
    portable_feature_sequence_digest,
    registered_feature_names,
    validate_feature_cache_key,
)
from signlab.contracts.pipeline import split_manifest_digest
from signlab.contracts.quality import SequenceQualityReportV1, sequence_quality_report_digest
from signlab.features.cache import (
    FeatureCacheError,
    build_feature_cache_key,
    feature_cache_key_from_sequence,
    feature_cache_path,
    load_cached_feature,
    store_cached_feature,
)
from signlab.features.statistics import fit_feature_statistics
from test_extraction_contracts import _sequence_ref
from test_quality_contracts import _sequence_report


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _plan(
    *,
    statistics_mode: Literal["none", "train_only_masked_zscore/1"] = "none",
    target_frame_count: int = 2,
) -> LandmarkFeaturePlanV1:
    optional = OptionalFeatureRuleV1(
        include_velocity=False,
        include_acceleration=False,
        include_joint_angles=False,
        include_tip_distances=False,
        joint_angle_rule="five_registered_pip_flexion_angles_radians/1",
        tip_distance_rule="five_registered_wrist_to_fingertip_distances/1",
    )
    representation: Literal["body_relative"] = "body_relative"
    return LandmarkFeaturePlanV1(
        schema_version="landmark-feature-plan/1",
        plan_id=("body_relative_zscore" if statistics_mode != "none" else "body_relative_raw"),
        version="1.0.0",
        representation=representation,
        compatible_runtimes=("python", "typescript"),
        hand_slots=("hand_0", "hand_1"),
        handedness_source="mediapipe_vendor_report_corrected_by_source_mirror_state",
        swap_rule="preserve_slots_never_repair",
        hand_local=HandLocalRuleV1(
            coordinate_space="hand_world_xyz",
            landmark_order=tuple(range(21)),
            center="wrist_landmark_0",
            scale="wrist_to_middle_mcp_landmark_9_euclidean",
            source_mirror_rule="undo_world_x_when_source_mirrored",
            anatomical_canonicalization=(
                "swap_vendor_label_when_not_mirrored_then_reflect_left_hand_x"
            ),
            zero_scale_rule="mask_hand_features",
        ),
        body_relative=BodyRelativeRuleV1(
            coordinate_space="image_xy",
            trajectory_points=("wrist", "palm_centroid"),
            palm_landmarks=(0, 5, 9, 17),
            center="shoulder_midpoint",
            scale="shoulder_width_xy_euclidean",
            source_mirror_rule="undo_image_x_when_source_mirrored",
            missing_anchor_rule="mask_body_keep_hand_local",
            zero_scale_rule="mask_body_features",
        ),
        temporal=TemporalFeatureRuleV1(
            clock="relative_timestamp_us",
            grid_rule="nominal_elapsed_time_append_final/1",
            target_rate_numerator=30,
            target_rate_denominator=1,
            interpolation="quality_report_approved_linear_coordinates_only",
            extrapolation_allowed=False,
            forward_fill_allowed=False,
            derivative_rule="backward_elapsed_time_finite_difference/1",
            derivative_application_order="resample_then_derive_then_select_then_pad/1",
        ),
        padding=PaddingFeatureRuleV1(
            target_frame_count=target_frame_count,
            long_sequence_rule="uniform_endpoint_preserving_index_selection/1",
            padding_side="right",
            padding_value_q=0,
            padding_mask_rule="all_feature_masks_false",
            padding_timestamp_rule="continue_nominal_grid",
        ),
        optional=optional,
        learned_statistics=LearnedStatisticsRuleV1(
            mode=statistics_mode,
            partition_evidence="explicit_train_membership_required",
            masked_value_rule="exclude_from_fit",
            zero_count_rule="mean_zero_scale_one",
            zero_variance_rule="scale_one",
        ),
        quantization_scale=1_000_000,
        quantization_rule="round_half_away_from_zero/1",
        interchange_values="signed_integer_divided_by_quantization_scale",
        feature_order=registered_feature_names(representation, optional),
    )


def _key_and_sequence(
    *,
    plan: LandmarkFeaturePlanV1 | None = None,
) -> tuple[FeatureCacheKeyV1, PortableFeatureSequenceV1]:
    checked_plan = plan or _plan()
    key = build_feature_cache_key(
        _sequence_ref(),
        _sequence_report(),
        checked_plan,
        extraction_config_sha256=_sha("e"),
    )
    assert validate_feature_cache_key(canonical_json_bytes(key)) == key
    width = len(checked_plan.feature_order)
    payload: dict[str, object] = {
        "schema_version": "portable-feature-sequence/1",
        "source_recording_id": key.source_recording_id,
        "source_media_sha256": key.source_media_sha256,
        "source_landmarks_sha256": key.source_landmarks_sha256,
        "extraction_config_sha256": key.extraction_config_sha256,
        "quality_policy_sha256": key.quality_policy_sha256,
        "quality_report_sha256": key.quality_report_sha256,
        "feature_plan_sha256": key.feature_plan_sha256,
        "statistics_sha256": key.statistics_sha256,
        "feature_names": checked_plan.feature_order,
        "quantization_scale": 1_000_000,
        "source_grid_frame_count": 2,
        "selected_source_indices": (0, 1),
        "timestamps_us": (0, 33_333),
        "values_q": (tuple(range(width)), tuple(range(width, 2 * width))),
        "valid_mask": ((True,) * width,) * 2,
        "observed_mask": ((True,) * width,) * 2,
        "interpolated_mask": ((False,) * width,) * 2,
        "hand_present_mask": ((True, True),) * 2,
        "body_available_mask": (True, True),
        "padding_mask": (False, False),
    }
    json_payload = cast(dict[str, object], json.loads(json.dumps(payload)))
    json_payload["sequence_sha256"] = portable_feature_sequence_digest(json_payload)
    return key, PortableFeatureSequenceV1.model_validate_json(
        canonical_json_bytes(json_payload), strict=True
    )


def test_cache_key_binds_upstream_inputs_and_reconstructs_from_sequence() -> None:
    key, sequence = _key_and_sequence()

    assert feature_cache_key_from_sequence(sequence) == key
    assert key.source_landmarks_sha256 == _sequence_ref().content_sha256
    assert key.quality_report_sha256 == _sequence_report().report_sha256

    changed = sequence.model_dump(mode="json", round_trip=True)
    changed["quality_report_sha256"] = _sha("f")
    changed.pop("sequence_sha256")
    changed["sequence_sha256"] = portable_feature_sequence_digest(changed)
    changed_sequence = PortableFeatureSequenceV1.model_validate_json(
        canonical_json_bytes(changed), strict=True
    )
    assert feature_cache_key_from_sequence(changed_sequence) != key


def test_cache_key_validates_quality_and_optional_statistics_bindings() -> None:
    plan = _plan(statistics_mode="train_only_masked_zscore/1")
    raw_key, raw_sequence = _key_and_sequence(plan=plan)
    split_manifest = make_split_manifest(
        train_recording_ids=(raw_sequence.source_recording_id,),
        validation_recording_ids=("recording_" + "e" * 32,),
        test_recording_ids=("recording_" + "f" * 32,),
    )
    training = TrainingFeatureSequenceV1(
        schema_version="training-feature-sequence/1",
        partition="train",
        split_manifest_sha256=split_manifest_digest(split_manifest),
        sequence=raw_sequence,
    )
    statistics = fit_feature_statistics((training,), split_manifest, plan)

    fitted_key = build_feature_cache_key(
        _sequence_ref(),
        _sequence_report(),
        plan,
        extraction_config_sha256=_sha("e"),
        statistics=statistics,
    )

    assert fitted_key.statistics_sha256 == statistics.statistics_sha256
    assert fitted_key != raw_key

    with pytest.raises(FeatureCacheError, match="statistics-free"):
        build_feature_cache_key(
            _sequence_ref(),
            _sequence_report(),
            _plan(),
            extraction_config_sha256=_sha("e"),
            statistics=statistics,
        )

    wrong_quality_payload = _sequence_report().model_dump(mode="json", round_trip=True)
    wrong_quality_payload["source_recording_id"] = "recording_ffffffffffffffffffffffffffffffff"
    wrong_quality_payload.pop("report_sha256")
    wrong_quality_payload["report_sha256"] = sequence_quality_report_digest(wrong_quality_payload)
    wrong_quality = SequenceQualityReportV1.model_validate_json(
        canonical_json_bytes(wrong_quality_payload), strict=True
    )
    with pytest.raises(FeatureCacheError, match="quality evidence"):
        build_feature_cache_key(
            _sequence_ref(),
            wrong_quality,
            plan,
            extraction_config_sha256=_sha("e"),
        )


def test_cache_revalidates_copied_contract_instances_before_use(tmp_path: Path) -> None:
    plan = _plan(statistics_mode="train_only_masked_zscore/1")
    _, raw_sequence = _key_and_sequence(plan=plan)
    split_manifest = make_split_manifest(
        train_recording_ids=(raw_sequence.source_recording_id,),
        validation_recording_ids=("recording_" + "e" * 32,),
        test_recording_ids=("recording_" + "f" * 32,),
    )
    training = TrainingFeatureSequenceV1(
        schema_version="training-feature-sequence/1",
        partition="train",
        split_manifest_sha256=split_manifest_digest(split_manifest),
        sequence=raw_sequence,
    )
    statistics = fit_feature_statistics((training,), split_manifest, plan)

    invalid_reference = _sequence_ref().model_copy(update={"content_sha256": "not-a-digest"})
    with pytest.raises(FeatureCacheError, match="sources must be validated"):
        build_feature_cache_key(
            invalid_reference,
            _sequence_report(),
            plan,
            extraction_config_sha256=_sha("e"),
        )

    invalid_report = _sequence_report().model_copy(
        update={"source_recording_id": "recording_ffffffffffffffffffffffffffffffff"}
    )
    with pytest.raises(FeatureCacheError, match="sources must be validated"):
        build_feature_cache_key(
            _sequence_ref(),
            invalid_report,
            plan,
            extraction_config_sha256=_sha("e"),
        )

    invalid_plan = plan.model_copy(update={"feature_order": tuple(reversed(plan.feature_order))})
    with pytest.raises(FeatureCacheError, match="preprocessing must be validated"):
        build_feature_cache_key(
            _sequence_ref(),
            _sequence_report(),
            invalid_plan,
            extraction_config_sha256=_sha("e"),
        )

    invalid_statistics = statistics.model_copy(
        update={"mean_q": (statistics.mean_q[0] + 1, *statistics.mean_q[1:])}
    )
    with pytest.raises(FeatureCacheError, match="preprocessing must be validated"):
        build_feature_cache_key(
            _sequence_ref(),
            _sequence_report(),
            plan,
            extraction_config_sha256=_sha("e"),
            statistics=invalid_statistics,
        )

    key, sequence = _key_and_sequence()
    changed_first_row = (sequence.values_q[0][0] + 1, *sequence.values_q[0][1:])
    invalid_sequence = sequence.model_copy(
        update={"values_q": (changed_first_row, *sequence.values_q[1:])}
    )
    with pytest.raises(FeatureCacheError, match="validated sequence"):
        feature_cache_key_from_sequence(invalid_sequence)

    cache_root = tmp_path / "cache"
    with pytest.raises(FeatureCacheError, match="validated sequence"):
        store_cached_feature(cache_root, key, invalid_sequence)
    assert not cache_root.exists()

    invalid_key = key.model_copy(update={"source_media_sha256": _sha("f")})
    with pytest.raises(FeatureCacheError, match="key is invalid"):
        feature_cache_path(cache_root, invalid_key)


def test_cache_store_is_atomic_canonical_idempotent_and_content_addressed(
    tmp_path: Path,
) -> None:
    key, sequence = _key_and_sequence()
    cache_root = tmp_path / "cache"

    first_path = store_cached_feature(cache_root, key, sequence)
    second_path = store_cached_feature(cache_root, key, sequence)

    assert second_path == first_path == feature_cache_path(cache_root, key)
    assert key.cache_key_sha256.removeprefix("sha256:") in first_path.as_posix()
    assert sequence.source_recording_id not in first_path.as_posix()
    assert first_path.read_bytes() == canonical_json_bytes(sequence) + b"\n"
    assert load_cached_feature(cache_root, key) == sequence


@pytest.mark.parametrize("contended_component", ["root", "objects"])
def test_concurrent_first_publication_reconciles_shared_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contended_component: str,
) -> None:
    key, sequence = _key_and_sequence()
    cache_root = tmp_path / "cache"
    if contended_component == "objects":
        cache_root.mkdir()
    contended_path = cache_root if contended_component == "root" else cache_root / "objects"
    rendezvous = Barrier(2)
    original_mkdir = Path.mkdir

    def synchronized_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path == contended_path:
            rendezvous.wait(timeout=5)
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", synchronized_mkdir)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(store_cached_feature, cache_root, key, sequence) for _ in range(2)
        )
        stored_paths = tuple(future.result(timeout=10) for future in futures)

    expected = feature_cache_path(cache_root, key)
    assert stored_paths == (expected, expected)
    assert load_cached_feature(cache_root, key) == sequence


def test_cache_rejects_key_mismatch_conflicts_and_noncanonical_content(tmp_path: Path) -> None:
    key, sequence = _key_and_sequence()
    cache_root = tmp_path / "cache"

    changed_payload = sequence.model_dump(mode="json", round_trip=True)
    values = changed_payload["values_q"]
    assert isinstance(values, list)
    first_row = values[0]
    assert isinstance(first_row, list)
    first_row[0] = 99
    changed_payload.pop("sequence_sha256")
    changed_payload["sequence_sha256"] = portable_feature_sequence_digest(changed_payload)
    changed = PortableFeatureSequenceV1.model_validate_json(
        canonical_json_bytes(changed_payload), strict=True
    )

    store_cached_feature(cache_root, key, sequence)
    with pytest.raises(FeatureCacheError, match="different content"):
        store_cached_feature(cache_root, key, changed)

    path = feature_cache_path(cache_root, key)
    path.write_bytes(canonical_json_bytes(sequence) + b" \n")
    with pytest.raises(FeatureCacheError, match="not canonical"):
        load_cached_feature(cache_root, key)

    other_key, other_sequence = _key_and_sequence(plan=_plan(target_frame_count=3))
    with pytest.raises(FeatureCacheError, match="does not match its key"):
        store_cached_feature(tmp_path / "other", other_key, sequence)
    assert other_sequence != sequence


def test_cache_rejects_symlinked_content_and_hierarchy(tmp_path: Path) -> None:
    key, sequence = _key_and_sequence()
    cache_root = tmp_path / "cache"
    path = store_cached_feature(cache_root, key, sequence)
    outside = tmp_path / "outside.json"
    outside.write_bytes(canonical_json_bytes(sequence) + b"\n")
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(FeatureCacheError, match="content is invalid"):
        load_cached_feature(cache_root, key)

    linked_root = tmp_path / "linked-cache"
    linked_root.mkdir()
    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (linked_root / "objects").symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(FeatureCacheError, match="cannot contain links"):
        store_cached_feature(linked_root, key, sequence)


def test_cache_load_fails_closed_for_missing_roots_and_extra_inventory(tmp_path: Path) -> None:
    key, sequence = _key_and_sequence()
    with pytest.raises(FeatureCacheError, match="does not exist"):
        load_cached_feature(tmp_path / "missing", key)

    cache_root = tmp_path / "cache"
    path = store_cached_feature(cache_root, key, sequence)
    (path.parent / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FeatureCacheError, match="inventory"):
        load_cached_feature(cache_root, key)
