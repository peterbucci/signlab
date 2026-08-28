from __future__ import annotations

import json
from typing import cast

import pytest

from feature_fixtures import make_feature_plan, make_split_manifest
from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.features import (
    FEATURE_QUANTIZATION_SCALE,
    PortableFeatureSequenceV1,
    TrainingFeatureSequenceV1,
    landmark_feature_plan_digest,
    portable_feature_sequence_digest,
    validate_feature_statistics,
)
from signlab.contracts.pipeline import SplitManifestV1, split_manifest_digest
from signlab.features.statistics import (
    FeatureStatisticsError,
    apply_feature_statistics,
    fit_feature_statistics,
    round_ratio_half_away_from_zero,
)

STATISTICS_PLAN = make_feature_plan(
    "body_relative",
    target_frame_count=3,
    statistics_mode="train_only_masked_zscore/1",
)
FEATURE_NAMES = STATISTICS_PLAN.feature_order
_EMPTY_TAIL = (0,) * (len(FEATURE_NAMES) - 2)
_INVALID_TAIL = (False,) * (len(FEATURE_NAMES) - 2)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _sequence(
    identity: str,
    *,
    values: tuple[tuple[int, ...], tuple[int, ...]] = (
        (0, 3_000_000, *_EMPTY_TAIL),
        (2_000_000, 3_000_000, *_EMPTY_TAIL),
    ),
    valid: tuple[tuple[bool, ...], tuple[bool, ...]] = (
        (True, True, *_INVALID_TAIL),
        (True, True, *_INVALID_TAIL),
    ),
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    plan_sha256: str | None = None,
) -> PortableFeatureSequenceV1:
    width = len(feature_names)
    payload: dict[str, object] = {
        "schema_version": "portable-feature-sequence/1",
        "source_recording_id": f"recording_{identity * 32}",
        "source_media_sha256": _sha(identity),
        "source_landmarks_sha256": _sha("a"),
        "extraction_config_sha256": _sha("b"),
        "quality_policy_sha256": _sha("c"),
        "quality_report_sha256": _sha("d"),
        "feature_plan_sha256": plan_sha256 or landmark_feature_plan_digest(STATISTICS_PLAN),
        "statistics_sha256": None,
        "feature_names": feature_names,
        "quantization_scale": FEATURE_QUANTIZATION_SCALE,
        "source_grid_frame_count": 2,
        "selected_source_indices": (0, 1),
        "timestamps_us": (0, 33_333, 66_666),
        "values_q": (*values, (0,) * width),
        "valid_mask": (*valid, (False,) * width),
        "observed_mask": (*valid, (False,) * width),
        "interpolated_mask": ((False,) * width,) * 3,
        "hand_present_mask": ((True, False), (True, False), (False, False)),
        "body_available_mask": (True, True, False),
        "padding_mask": (False, False, True),
    }
    json_payload = cast(dict[str, object], json.loads(json.dumps(payload)))
    json_payload["sequence_sha256"] = portable_feature_sequence_digest(json_payload)
    return PortableFeatureSequenceV1.model_validate_json(
        canonical_json_bytes(json_payload), strict=True
    )


def _split(*training_sequences: PortableFeatureSequenceV1) -> SplitManifestV1:
    return make_split_manifest(
        train_recording_ids=tuple(sequence.source_recording_id for sequence in training_sequences),
        validation_recording_ids=("recording_" + "3" * 32,),
        test_recording_ids=("recording_" + "4" * 32,),
    )


def _training(
    sequence: PortableFeatureSequenceV1,
    split_manifest: SplitManifestV1,
) -> TrainingFeatureSequenceV1:
    return TrainingFeatureSequenceV1(
        schema_version="training-feature-sequence/1",
        partition="train",
        split_manifest_sha256=split_manifest_digest(split_manifest),
        sequence=sequence,
    )


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(1, 2, 1), (-1, 2, -1), (1, 3, 0), (-1, 3, 0), (5, 2, 3), (-5, 2, -3)],
)
def test_exact_rounding_uses_half_away_from_zero(
    numerator: int,
    denominator: int,
    expected: int,
) -> None:
    assert round_ratio_half_away_from_zero(numerator, denominator) == expected

    with pytest.raises(FeatureStatisticsError, match="positive denominator"):
        round_ratio_half_away_from_zero(numerator, 0)


def test_fit_is_train_only_masked_order_independent_and_uses_safe_fallbacks() -> None:
    first_sequence = _sequence("1")
    second_sequence = _sequence("2")
    split_manifest = _split(first_sequence, second_sequence)
    first = _training(first_sequence, split_manifest)
    second = _training(second_sequence, split_manifest)

    forward = fit_feature_statistics((first, second), split_manifest, STATISTICS_PLAN)
    reverse = fit_feature_statistics((second, first), split_manifest, STATISTICS_PLAN)

    assert reverse == forward
    assert forward.split_manifest_sha256 == split_manifest_digest(split_manifest)
    assert forward.training_sequence_sha256 == tuple(
        sorted((first.sequence.sequence_sha256, second.sequence.sequence_sha256))
    )
    assert forward.observation_count == (4, 4, *(0,) * (len(FEATURE_NAMES) - 2))
    assert forward.mean_q == (1_000_000, 3_000_000, *_EMPTY_TAIL)
    assert forward.standard_deviation_q == (
        1_000_000,
        *(FEATURE_QUANTIZATION_SCALE,) * (len(FEATURE_NAMES) - 1),
    )
    assert validate_feature_statistics(canonical_json_bytes(forward)) == forward


def test_apply_standardizes_only_valid_unpadded_values_and_rehashes() -> None:
    source = _sequence("1")
    second = _sequence("2")
    split_manifest = _split(source, second)
    statistics = fit_feature_statistics(
        (_training(source, split_manifest), _training(second, split_manifest)),
        split_manifest,
        STATISTICS_PLAN,
    )

    standardized = apply_feature_statistics(source, statistics, STATISTICS_PLAN)

    assert source.statistics_sha256 is None
    assert standardized.statistics_sha256 == statistics.statistics_sha256
    assert standardized.values_q == (
        (-1_000_000, 0, *_EMPTY_TAIL),
        (1_000_000, 0, *_EMPTY_TAIL),
        (0,) * len(FEATURE_NAMES),
    )
    assert standardized.valid_mask == source.valid_mask
    assert standardized.padding_mask == source.padding_mask
    assert standardized.sequence_sha256 == portable_feature_sequence_digest(standardized)
    assert standardized.sequence_sha256 != source.sequence_sha256

    with pytest.raises(FeatureStatisticsError, match="already been applied"):
        apply_feature_statistics(standardized, statistics, STATISTICS_PLAN)

    tampered_statistics = statistics.model_copy(
        update={"mean_q": (statistics.mean_q[0] + 1, *statistics.mean_q[1:])}
    )
    with pytest.raises(FeatureStatisticsError, match="validated feature contracts"):
        apply_feature_statistics(source, tampered_statistics, STATISTICS_PLAN)


def test_fit_rejects_nontraining_duplicate_mixed_and_preprocessed_inputs() -> None:
    first = _sequence("1")
    second = _sequence("2")
    split_manifest = _split(first, second)

    with pytest.raises(FeatureStatisticsError, match="at least one"):
        fit_feature_statistics((), split_manifest, STATISTICS_PLAN)
    with pytest.raises(FeatureStatisticsError, match="identities must be unique"):
        fit_feature_statistics(
            (_training(first, split_manifest), _training(first, split_manifest)),
            split_manifest,
            STATISTICS_PLAN,
        )

    nontraining = TrainingFeatureSequenceV1.model_construct(
        schema_version="training-feature-sequence/1",
        partition="validation",
        split_manifest_sha256=split_manifest_digest(split_manifest),
        sequence=first,
    )
    with pytest.raises(FeatureStatisticsError, match="validated feature sequences"):
        fit_feature_statistics((nontraining,), split_manifest, STATISTICS_PLAN)

    wrong_split_input = _training(first, split_manifest).model_copy(
        update={"split_manifest_sha256": _sha("9")}
    )
    with pytest.raises(FeatureStatisticsError, match="different split manifest"):
        fit_feature_statistics((wrong_split_input,), split_manifest, STATISTICS_PLAN)

    tampered_sequence = first.model_copy(
        update={
            "values_q": (
                (first.values_q[0][0] + 1, *first.values_q[0][1:]),
                *first.values_q[1:],
            )
        }
    )
    tampered_training = _training(first, split_manifest).model_copy(
        update={"sequence": tampered_sequence}
    )
    with pytest.raises(FeatureStatisticsError, match="validated feature sequences"):
        fit_feature_statistics((tampered_training,), split_manifest, STATISTICS_PLAN)

    validation_sequence = _sequence("3")
    relabeled_validation = _training(validation_sequence, split_manifest)
    with pytest.raises(FeatureStatisticsError, match="not a member"):
        fit_feature_statistics((relabeled_validation,), split_manifest, STATISTICS_PLAN)

    different_plan = _sequence("2", plan_sha256=_sha("f"))
    with pytest.raises(FeatureStatisticsError, match="different preprocessing plan"):
        fit_feature_statistics(
            (
                _training(first, split_manifest),
                _training(different_plan, split_manifest),
            ),
            split_manifest,
            STATISTICS_PLAN,
        )

    different_names = _sequence(
        "2",
        feature_names=(FEATURE_NAMES[1], FEATURE_NAMES[0], *FEATURE_NAMES[2:]),
    )
    with pytest.raises(FeatureStatisticsError, match="different feature order"):
        fit_feature_statistics(
            (
                _training(first, split_manifest),
                _training(different_names, split_manifest),
            ),
            split_manifest,
            STATISTICS_PLAN,
        )

    invalid_scale = second.model_copy(
        update={
            "quantization_scale": 100,
            "sequence_sha256": _sha("f"),
        }
    )
    invalid_scale_input = TrainingFeatureSequenceV1.model_construct(
        schema_version="training-feature-sequence/1",
        partition="train",
        split_manifest_sha256=split_manifest_digest(split_manifest),
        sequence=invalid_scale,
    )
    with pytest.raises(FeatureStatisticsError, match="validated feature sequences"):
        fit_feature_statistics(
            (_training(first, split_manifest), invalid_scale_input),
            split_manifest,
            STATISTICS_PLAN,
        )

    preprocessed_payload = first.model_dump(mode="json", round_trip=True)
    preprocessed_payload["statistics_sha256"] = _sha("9")
    preprocessed_payload.pop("sequence_sha256")
    preprocessed_payload["sequence_sha256"] = portable_feature_sequence_digest(preprocessed_payload)
    preprocessed = PortableFeatureSequenceV1.model_validate_json(
        canonical_json_bytes(preprocessed_payload), strict=True
    )
    with pytest.raises(FeatureStatisticsError, match="cannot be refitted"):
        fit_feature_statistics(
            (_training(preprocessed, split_manifest),),
            split_manifest,
            STATISTICS_PLAN,
        )

    with pytest.raises(FeatureStatisticsError, match="validated split manifest"):
        fit_feature_statistics(
            (_training(first, split_manifest),),
            cast(SplitManifestV1, {"schema_version": "split-manifest/1"}),
            STATISTICS_PLAN,
        )

    disabled_plan = make_feature_plan("body_relative", target_frame_count=3)
    disabled_sequence = _sequence(
        "1",
        plan_sha256=landmark_feature_plan_digest(disabled_plan),
    )
    disabled_split = _split(disabled_sequence)
    with pytest.raises(FeatureStatisticsError, match="does not enable"):
        fit_feature_statistics(
            (_training(disabled_sequence, disabled_split),),
            disabled_split,
            disabled_plan,
        )
    enabled_statistics = fit_feature_statistics(
        (_training(first, split_manifest),),
        split_manifest,
        STATISTICS_PLAN,
    )
    with pytest.raises(FeatureStatisticsError, match="does not enable"):
        apply_feature_statistics(disabled_sequence, enabled_statistics, disabled_plan)


def test_apply_rejects_statistics_from_another_plan() -> None:
    source = _sequence("1")
    other_plan = make_feature_plan(
        "body_relative",
        target_frame_count=4,
        statistics_mode="train_only_masked_zscore/1",
    )
    other = _sequence("2", plan_sha256=landmark_feature_plan_digest(other_plan))
    split_manifest = _split(other)
    statistics = fit_feature_statistics(
        (_training(other, split_manifest),),
        split_manifest,
        other_plan,
    )

    with pytest.raises(FeatureStatisticsError, match="different preprocessing plan"):
        apply_feature_statistics(source, statistics, STATISTICS_PLAN)
