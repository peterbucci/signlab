"""Deterministic split-verified fitting and application of feature statistics.

The feature contract stores quantized integers so fitting can use exact integer
arithmetic.  This module deliberately has no NumPy or storage dependency.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Final, cast

from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.features import (
    FeatureContractError,
    FeatureStatisticsV1,
    LandmarkFeaturePlanV1,
    PortableFeatureSequenceV1,
    TrainingFeatureSequenceV1,
    feature_statistics_digest,
    landmark_feature_plan_digest,
    portable_feature_sequence_digest,
    validate_portable_feature_sequence,
)
from signlab.contracts.pipeline import (
    PipelineContractError,
    SplitManifestV1,
    split_manifest_digest,
    validate_split_manifest,
)

DEFAULT_STATISTICS_ID: Final = "landmark_feature_statistics"
DEFAULT_STATISTICS_VERSION: Final = "1.0.0"


class FeatureStatisticsError(ValueError):
    """Raised when feature statistics cannot be fitted or safely applied."""


def round_ratio_half_away_from_zero(numerator: int, denominator: int) -> int:
    """Round an exact rational number to the nearest integer, away at ties."""

    if denominator <= 0:
        raise FeatureStatisticsError("statistics rounding requires a positive denominator")
    magnitude, remainder = divmod(abs(numerator), denominator)
    if remainder * 2 >= denominator:
        magnitude += 1
    return -magnitude if numerator < 0 else magnitude


def _rounded_population_standard_deviation(
    *,
    count: int,
    total: int,
    total_squares: int,
    quantization_scale: int,
) -> int:
    """Return an exactly rounded population standard deviation in input quanta."""

    if count <= 0:
        return quantization_scale
    variance_numerator = count * total_squares - total * total
    if variance_numerator < 0:  # pragma: no cover - defensive integer invariant
        raise FeatureStatisticsError("feature variance cannot be negative")
    if variance_numerator == 0:
        return quantization_scale

    variance_denominator = count * count
    floor_value = math.isqrt(variance_numerator // variance_denominator)
    midpoint_twice = 2 * floor_value + 1
    rounded = floor_value + (
        4 * variance_numerator >= variance_denominator * midpoint_twice * midpoint_twice
    )
    # A non-zero variance can be smaller than half of one quantization unit.  The
    # contract requires a positive divisor, so retain the smallest representable
    # scale instead of manufacturing a divide-by-zero condition.
    return max(1, rounded)


def _ordered_training_sequences(
    training_sequences: Iterable[TrainingFeatureSequenceV1],
    *,
    split_manifest_sha256: str,
    training_recording_ids: frozenset[str],
    plan: LandmarkFeaturePlanV1,
    plan_sha256: str,
) -> tuple[TrainingFeatureSequenceV1, ...]:
    sequences = tuple(training_sequences)
    if not sequences:
        raise FeatureStatisticsError("at least one training feature sequence is required")
    if any(not isinstance(item, TrainingFeatureSequenceV1) for item in sequences):
        raise FeatureStatisticsError("statistics inputs must be training feature sequences")
    try:
        sequences = tuple(
            TrainingFeatureSequenceV1.model_validate(item, strict=True) for item in sequences
        )
    except ValueError:
        raise FeatureStatisticsError(
            "statistics inputs must be validated feature sequences"
        ) from None

    ordered = tuple(sorted(sequences, key=lambda item: item.sequence.sequence_sha256))
    identities = tuple(item.sequence.sequence_sha256 for item in ordered)
    if len(identities) != len(set(identities)):
        raise FeatureStatisticsError("training feature sequence identities must be unique")

    for item in ordered:
        sequence = item.sequence
        if item.partition != "train":  # pragma: no cover - enforced by the strict contract
            raise FeatureStatisticsError("feature statistics may use only training inputs")
        if item.split_manifest_sha256 != split_manifest_sha256:
            raise FeatureStatisticsError("training input uses a different split manifest")
        if sequence.source_recording_id not in training_recording_ids:
            raise FeatureStatisticsError(
                "training input recording is not a member of the split's train partition"
            )
        if sequence.statistics_sha256 is not None:
            raise FeatureStatisticsError("statistics cannot be refitted from standardized features")
        if sequence.feature_plan_sha256 != plan_sha256:
            raise FeatureStatisticsError("training features use a different preprocessing plan")
        if sequence.feature_names != plan.feature_order:
            raise FeatureStatisticsError("training features use a different feature order")
        if sequence.quantization_scale != plan.quantization_scale:
            raise FeatureStatisticsError("training features use a different quantization scale")
    return ordered


def _validated_statistics_plan(plan: LandmarkFeaturePlanV1) -> tuple[LandmarkFeaturePlanV1, str]:
    if not isinstance(plan, LandmarkFeaturePlanV1):
        raise FeatureStatisticsError("statistics require a validated preprocessing plan")
    try:
        checked = LandmarkFeaturePlanV1.model_validate(plan, strict=True)
    except ValueError:
        raise FeatureStatisticsError("statistics require a validated preprocessing plan") from None
    if checked.learned_statistics.mode != "train_only_masked_zscore/1":
        raise FeatureStatisticsError("preprocessing plan does not enable learned statistics")
    return checked, landmark_feature_plan_digest(checked)


def fit_feature_statistics(
    training_sequences: Iterable[TrainingFeatureSequenceV1],
    split_manifest: SplitManifestV1,
    plan: LandmarkFeaturePlanV1,
    *,
    statistics_id: str = DEFAULT_STATISTICS_ID,
    version: str = DEFAULT_STATISTICS_VERSION,
) -> FeatureStatisticsV1:
    """Fit masked population z-score statistics from the split's training inputs.

    The supplied split is revalidated and each wrapped recording must belong to its
    train partition. Invalid values and every padded frame are excluded. Inputs are
    sorted by their immutable sequence digest before accumulation, making the result
    independent of caller iteration order.
    """

    checked_plan, plan_sha256 = _validated_statistics_plan(plan)
    if not isinstance(split_manifest, SplitManifestV1):
        raise FeatureStatisticsError("statistics fitting requires a validated split manifest")
    try:
        checked_split = validate_split_manifest(split_manifest)
        split_sha256 = split_manifest_digest(checked_split)
    except PipelineContractError:
        raise FeatureStatisticsError(
            "statistics fitting requires a validated split manifest"
        ) from None
    train_partition = checked_split.partitions[0]
    ordered = _ordered_training_sequences(
        training_sequences,
        split_manifest_sha256=split_sha256,
        training_recording_ids=frozenset(train_partition.source_recording_ids),
        plan=checked_plan,
        plan_sha256=plan_sha256,
    )
    first = ordered[0].sequence
    width = len(first.feature_names)
    counts = [0] * width
    totals = [0] * width
    total_squares = [0] * width

    for item in ordered:
        sequence = item.sequence
        for frame_index, values in enumerate(sequence.values_q):
            if sequence.padding_mask[frame_index]:
                continue
            validity = sequence.valid_mask[frame_index]
            for feature_index, value in enumerate(values):
                if not validity[feature_index]:
                    continue
                counts[feature_index] += 1
                totals[feature_index] += value
                total_squares[feature_index] += value * value

    means = tuple(
        0 if count == 0 else round_ratio_half_away_from_zero(total, count)
        for count, total in zip(counts, totals, strict=True)
    )
    standard_deviations = tuple(
        _rounded_population_standard_deviation(
            count=count,
            total=total,
            total_squares=squares,
            quantization_scale=first.quantization_scale,
        )
        for count, total, squares in zip(counts, totals, total_squares, strict=True)
    )
    payload: dict[str, object] = {
        "schema_version": "feature-statistics/1",
        "statistics_id": statistics_id,
        "version": version,
        "feature_plan_sha256": plan_sha256,
        "split_manifest_sha256": split_sha256,
        "feature_names": list(first.feature_names),
        "quantization_scale": first.quantization_scale,
        "training_sequence_sha256": [item.sequence.sequence_sha256 for item in ordered],
        "observation_count": counts,
        "mean_q": list(means),
        "standard_deviation_q": list(standard_deviations),
    }
    payload["statistics_sha256"] = feature_statistics_digest(payload)
    try:
        return FeatureStatisticsV1.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except (FeatureContractError, TypeError, ValueError):
        raise FeatureStatisticsError("fitted feature statistics are not portable") from None


def _assert_statistics_compatible(
    sequence: PortableFeatureSequenceV1,
    statistics: FeatureStatisticsV1,
    plan: LandmarkFeaturePlanV1,
    plan_sha256: str,
) -> None:
    if sequence.statistics_sha256 is not None:
        raise FeatureStatisticsError("feature statistics have already been applied")
    if sequence.feature_plan_sha256 != plan_sha256 or statistics.feature_plan_sha256 != plan_sha256:
        raise FeatureStatisticsError("feature statistics use a different preprocessing plan")
    if (
        sequence.feature_names != plan.feature_order
        or statistics.feature_names != plan.feature_order
    ):
        raise FeatureStatisticsError("feature statistics use a different feature order")
    if (
        sequence.quantization_scale != plan.quantization_scale
        or statistics.quantization_scale != plan.quantization_scale
    ):
        raise FeatureStatisticsError("feature statistics use a different quantization scale")


def apply_feature_statistics(
    sequence: PortableFeatureSequenceV1,
    statistics: FeatureStatisticsV1,
    plan: LandmarkFeaturePlanV1,
) -> PortableFeatureSequenceV1:
    """Return a rehashed immutable sequence standardized by compatible statistics."""

    if not isinstance(sequence, PortableFeatureSequenceV1) or not isinstance(
        statistics, FeatureStatisticsV1
    ):
        raise FeatureStatisticsError("statistics application requires validated feature contracts")
    checked_plan, plan_sha256 = _validated_statistics_plan(plan)
    try:
        checked_sequence = PortableFeatureSequenceV1.model_validate(sequence, strict=True)
        checked_statistics = FeatureStatisticsV1.model_validate(statistics, strict=True)
    except ValueError:
        raise FeatureStatisticsError(
            "statistics application requires validated feature contracts"
        ) from None
    _assert_statistics_compatible(
        checked_sequence,
        checked_statistics,
        checked_plan,
        plan_sha256,
    )

    standardized_rows: list[list[int]] = []
    for frame_index, values in enumerate(checked_sequence.values_q):
        if checked_sequence.padding_mask[frame_index]:
            standardized_rows.append([0] * len(values))
            continue
        validity = checked_sequence.valid_mask[frame_index]
        standardized_rows.append(
            [
                round_ratio_half_away_from_zero(
                    (value - checked_statistics.mean_q[feature_index])
                    * checked_sequence.quantization_scale,
                    checked_statistics.standard_deviation_q[feature_index],
                )
                if validity[feature_index]
                else 0
                for feature_index, value in enumerate(values)
            ]
        )

    payload = cast(
        dict[str, object],
        checked_sequence.model_dump(mode="json", round_trip=True),
    )
    payload["values_q"] = standardized_rows
    payload["statistics_sha256"] = checked_statistics.statistics_sha256
    payload.pop("sequence_sha256", None)
    payload["sequence_sha256"] = portable_feature_sequence_digest(payload)
    try:
        return validate_portable_feature_sequence(canonical_json_bytes(payload))
    except (FeatureContractError, TypeError, ValueError):
        raise FeatureStatisticsError("standardized feature values are not portable") from None


__all__ = [
    "DEFAULT_STATISTICS_ID",
    "DEFAULT_STATISTICS_VERSION",
    "FeatureStatisticsError",
    "apply_feature_statistics",
    "fit_feature_statistics",
    "round_ratio_half_away_from_zero",
]
