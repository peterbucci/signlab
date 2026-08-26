"""Strict contracts for deterministic landmark-quality policy and reports."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Annotated, Final, Literal, Self, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    MAX_SAFE_INTEGER,
    NonNegativeSafeInteger,
    PositiveSafeInteger,
    SemanticVersion,
    StableId,
    StrictContractModel,
    contract_config,
)
from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    HAND_SLOT_IDS,
    BodyAnchorName,
    HandSlotId,
    LandmarkExtractionManifestV1,
    LandmarkFramesTableV1,
    landmark_frames_table_digest,
    validate_landmark_extraction_manifest,
    validate_landmark_frames_table,
)
from signlab.contracts.governance import RecordingId
from signlab.contracts.ingest import DatasetId
from signlab.contracts.taxonomy import Sha256Digest

PartsPerMillion = Annotated[NonNegativeSafeInteger, Field(le=1_000_000)]
ExpectedHandCount = Literal[1, 2]
QualityDisposition = Literal["pass", "warning", "quarantine", "reject"]
QualityFindingSeverity = Literal["warning", "quarantine", "reject"]
QualityMetricName = Literal[
    "expected_hand_coverage_ppm",
    "invalid_frame_fraction_ppm",
    "longest_unfilled_internal_hand_gap_us",
    "low_handedness_confidence_fraction_ppm",
    "minimum_pose_anchor_coverage_ppm",
    "suspected_hand_swap_count",
    "temporal_discontinuity_count",
    "timestamp_discontinuity_count",
]
QualityMetricDirection = Literal["higher_is_worse", "lower_is_worse"]
QualitySignalName = HandSlotId | BodyAnchorName
MissingIntervalBoundary = Literal["leading", "internal", "trailing", "entire_sequence"]
MissingIntervalDecision = Literal["interpolate_linear", "preserve_missing"]
MissingIntervalReason = Literal[
    "bridge_too_long",
    "eligible_short_internal_gap",
    "entire_sequence_missing",
    "identity_ambiguity",
    "invalid_frame",
    "leading_gap",
    "low_confidence",
    "timestamp_discontinuity",
    "too_many_missing_frames",
    "trailing_gap",
]
DatasetQualityStatus = Literal["ready", "ready_with_warnings", "blocked"]

QUALITY_METRIC_NAMES: Final[tuple[QualityMetricName, ...]] = (
    "expected_hand_coverage_ppm",
    "invalid_frame_fraction_ppm",
    "longest_unfilled_internal_hand_gap_us",
    "low_handedness_confidence_fraction_ppm",
    "minimum_pose_anchor_coverage_ppm",
    "suspected_hand_swap_count",
    "temporal_discontinuity_count",
    "timestamp_discontinuity_count",
)
QUALITY_METRIC_DIRECTIONS: Final[dict[QualityMetricName, QualityMetricDirection]] = {
    "expected_hand_coverage_ppm": "lower_is_worse",
    "invalid_frame_fraction_ppm": "higher_is_worse",
    "longest_unfilled_internal_hand_gap_us": "higher_is_worse",
    "low_handedness_confidence_fraction_ppm": "higher_is_worse",
    "minimum_pose_anchor_coverage_ppm": "lower_is_worse",
    "suspected_hand_swap_count": "higher_is_worse",
    "temporal_discontinuity_count": "higher_is_worse",
    "timestamp_discontinuity_count": "higher_is_worse",
}
QUALITY_DISPOSITION_ORDER: Final[dict[QualityDisposition, int]] = {
    "pass": 0,
    "warning": 1,
    "quarantine": 2,
    "reject": 3,
}
_FINDING_SEVERITY_ORDER: Final[dict[QualityFindingSeverity, int]] = {
    "warning": 1,
    "quarantine": 2,
    "reject": 3,
}
_QUALITY_SIGNALS: Final = (*HAND_SLOT_IDS, *BODY_ANCHOR_NAMES)
_QUALITY_SIGNAL_RANK: Final = {signal: index for index, signal in enumerate(_QUALITY_SIGNALS)}
_MAX_MATERIALIZED_GRID_POINTS: Final = 1_000_000


class QualityContractError(ValueError):
    """Raised when quality-policy content is invalid or incompatible."""


def ratio_ppm(numerator: int, denominator: int) -> int:
    """Return a deterministic round-half-up parts-per-million ratio."""

    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator < 0
        or denominator <= 0
        or numerator > denominator
    ):
        raise QualityContractError("quality ratio inputs are invalid")
    return (numerator * 1_000_000 + denominator // 2) // denominator


def _round_half_up(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise QualityContractError("elapsed-time grid inputs are invalid")
    return (2 * numerator + denominator) // (2 * denominator)


def _elapsed_time_grid_details(
    observed_span_us: int,
    target_rate_numerator: int,
    target_rate_denominator: int,
) -> tuple[int, int]:
    """Return complete-grid count and final nominal index without materializing it."""

    if (
        type(observed_span_us) is not int
        or type(target_rate_numerator) is not int
        or type(target_rate_denominator) is not int
        or observed_span_us < 0
        or target_rate_numerator <= 0
        or target_rate_denominator <= 0
        or observed_span_us > MAX_SAFE_INTEGER
        or target_rate_numerator > MAX_SAFE_INTEGER
        or target_rate_denominator > MAX_SAFE_INTEGER
    ):
        raise QualityContractError("elapsed-time grid inputs are invalid")
    interval_numerator = 1_000_000 * target_rate_denominator
    if target_rate_numerator > interval_numerator:
        raise QualityContractError("elapsed-time target rate exceeds microsecond precision")
    final_nominal_index = (
        2 * target_rate_numerator * observed_span_us + target_rate_numerator - 1
    ) // (2 * interval_numerator)
    last_nominal_timestamp_us = _round_half_up(
        final_nominal_index * interval_numerator,
        target_rate_numerator,
    )
    target_count = final_nominal_index + 1 + int(last_nominal_timestamp_us != observed_span_us)
    if target_count > MAX_SAFE_INTEGER:
        raise QualityContractError("elapsed-time grid count exceeds the safe-integer range")
    return target_count, final_nominal_index


def elapsed_time_grid_shape(
    observed_span_us: int,
    target_rate_numerator: int,
    target_rate_denominator: int,
) -> tuple[int, int]:
    """Return complete target count and final timestamp in constant time."""

    target_count, _final_nominal_index = _elapsed_time_grid_details(
        observed_span_us,
        target_rate_numerator,
        target_rate_denominator,
    )
    return target_count, observed_span_us


def elapsed_time_grid_commitment(
    observed_span_us: int,
    target_rate_numerator: int,
    target_rate_denominator: int,
) -> str:
    """Commit to the uniquely determined complete grid without materializing it."""

    elapsed_time_grid_shape(
        observed_span_us,
        target_rate_numerator,
        target_rate_denominator,
    )
    try:
        return canonical_sha256(
            {
                "grid_rule": "nominal_elapsed_time_append_final/1",
                "observed_span_us": observed_span_us,
                "target_rate_denominator": target_rate_denominator,
                "target_rate_numerator": target_rate_numerator,
            },
            domain="landmark-quality-resampling-grid-commitment/1",
        )
    except CanonicalizationError as error:
        raise QualityContractError("elapsed-time grid cannot be canonicalized") from error


def elapsed_time_grid_us(
    observed_span_us: int,
    target_rate_numerator: int,
    target_rate_denominator: int,
) -> tuple[int, ...]:
    """Materialize a bounded nominal grid with the exact observed endpoint."""

    target_count, final_nominal_index = _elapsed_time_grid_details(
        observed_span_us,
        target_rate_numerator,
        target_rate_denominator,
    )
    if target_count > _MAX_MATERIALIZED_GRID_POINTS:
        raise QualityContractError("elapsed-time grid is too large to materialize safely")
    interval_numerator = 1_000_000 * target_rate_denominator
    timestamps = tuple(
        _round_half_up(index * interval_numerator, target_rate_numerator)
        for index in range(final_nominal_index + 1)
    )
    if timestamps[-1] != observed_span_us:
        timestamps = (*timestamps, observed_span_us)
    if len(timestamps) != target_count or any(
        right <= left for left, right in pairwise(timestamps)
    ):
        raise QualityContractError("elapsed-time grid is not strictly ordered")
    return timestamps


class QualityThresholdRuleV1(StrictContractModel):
    """Optional warning/quarantine/reject limits for one registered integer metric."""

    schema_version: Literal["quality-threshold-rule/1"]
    rule_id: StableId
    metric: QualityMetricName
    direction: QualityMetricDirection
    warning: NonNegativeSafeInteger | None
    quarantine: NonNegativeSafeInteger | None
    reject: NonNegativeSafeInteger | None

    @model_validator(mode="after")
    def _require_registered_ordered_thresholds(self) -> Self:
        if self.rule_id != self.metric:
            raise ValueError("quality rule ID must equal its registered metric name")
        if self.direction != QUALITY_METRIC_DIRECTIONS[self.metric]:
            raise ValueError("quality rule direction does not match its registered metric")
        thresholds = tuple(
            value for value in (self.warning, self.quarantine, self.reject) if value is not None
        )
        if not thresholds:
            raise ValueError("a quality rule requires at least one severity threshold")
        if self.metric.endswith("_ppm") and any(value > 1_000_000 for value in thresholds):
            raise ValueError("parts-per-million quality thresholds cannot exceed one million")
        ordered = (
            tuple(sorted(thresholds))
            if self.direction == "higher_is_worse"
            else tuple(sorted(thresholds, reverse=True))
        )
        if thresholds != ordered:
            raise ValueError("quality thresholds must become weaker at higher severities")
        return self


class LandmarkQualityPolicyV1(StrictContractModel):
    """Fully resolved quality, gap, timing, and swap-diagnostic policy."""

    model_config = contract_config("landmark-quality-policy-1.schema.json")

    schema_version: Literal["landmark-quality-policy/1"]
    policy_id: StableId
    version: SemanticVersion
    expected_hand_cardinality_rule: Literal["recording_handedness_unknown_means_one"]
    target_rate_numerator: PositiveSafeInteger
    target_rate_denominator: PositiveSafeInteger
    resampling_rule: Literal["nominal_elapsed_time_append_final/1"]
    interpolation_method: Literal["linear_coordinates_only"]
    extrapolation_allowed: Literal[False]
    max_interpolated_missing_frames: NonNegativeSafeInteger
    max_interpolation_bridge_us: NonNegativeSafeInteger
    handedness_confidence_diagnostic_ppm: PartsPerMillion
    pose_visibility_diagnostic_ppm: PartsPerMillion
    pose_presence_diagnostic_ppm: PartsPerMillion
    timestamp_discontinuity_absolute_us: PositiveSafeInteger
    timestamp_discontinuity_median_multiplier_ppm: PositiveSafeInteger
    suspected_swap_cost_margin_ppm: NonNegativeSafeInteger
    max_palm_wrist_speed_units_per_second: PositiveSafeInteger
    threshold_rules: tuple[QualityThresholdRuleV1, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def _require_exact_registered_profile(self) -> Self:
        metrics = tuple(rule.metric for rule in self.threshold_rules)
        if metrics != QUALITY_METRIC_NAMES:
            raise ValueError("quality policy rules must cover the registered metrics in order")
        if (self.target_rate_numerator, self.target_rate_denominator) != (30, 1):
            raise ValueError("landmark quality policy v1 requires the registered 30 Hz target")
        rules = {rule.metric: rule for rule in self.threshold_rules}
        coverage_reject = rules["expected_hand_coverage_ppm"].reject
        invalid_reject = rules["invalid_frame_fraction_ppm"].reject
        if coverage_reject is None or coverage_reject <= 0:
            raise ValueError("quality policy must reject completely undetected hands")
        if invalid_reject is None or invalid_reject >= 1_000_000:
            raise ValueError("quality policy must reject a sequence with no valid frames")
        return self


class MissingIntervalV1(StrictContractModel):
    """One absent signal run and its explicit interpolation or preservation decision."""

    schema_version: Literal["missing-interval/1"]
    gap_id: StableId
    signal: QualitySignalName
    boundary: MissingIntervalBoundary
    first_missing_frame_index: NonNegativeSafeInteger
    last_missing_frame_index: NonNegativeSafeInteger
    first_missing_timestamp_us: NonNegativeSafeInteger
    last_missing_timestamp_us: NonNegativeSafeInteger
    missing_frame_count: PositiveSafeInteger
    duration_us: NonNegativeSafeInteger
    left_observed_frame_index: NonNegativeSafeInteger | None
    left_observed_timestamp_us: NonNegativeSafeInteger | None
    right_observed_frame_index: NonNegativeSafeInteger | None
    right_observed_timestamp_us: NonNegativeSafeInteger | None
    contains_invalid_frame: bool
    contains_identity_ambiguity: bool
    crosses_timestamp_discontinuity: bool
    crosses_suspected_hand_swap: bool
    decision: MissingIntervalDecision
    reasons: tuple[MissingIntervalReason, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_ordered_bounded_decision(self) -> Self:
        if self.last_missing_frame_index < self.first_missing_frame_index:
            raise ValueError("missing interval frame indexes must be ordered")
        if self.last_missing_timestamp_us < self.first_missing_timestamp_us:
            raise ValueError("missing interval timestamps must be ordered")
        if self.missing_frame_count != (
            self.last_missing_frame_index - self.first_missing_frame_index + 1
        ):
            raise ValueError("missing interval count must match its inclusive frame range")
        paired = (
            (self.left_observed_frame_index, self.left_observed_timestamp_us),
            (self.right_observed_frame_index, self.right_observed_timestamp_us),
        )
        if any((frame is None) != (timestamp is None) for frame, timestamp in paired):
            raise ValueError("observed gap bounds require both frame and timestamp")
        left_present = self.left_observed_frame_index is not None
        right_present = self.right_observed_frame_index is not None
        expected_bounds = {
            "leading": (False, True),
            "internal": (True, True),
            "trailing": (True, False),
            "entire_sequence": (False, False),
        }[self.boundary]
        if (left_present, right_present) != expected_bounds:
            raise ValueError("missing interval observed bounds do not match its boundary kind")
        if left_present:
            assert self.left_observed_frame_index is not None
            assert self.left_observed_timestamp_us is not None
            if (
                self.left_observed_frame_index != self.first_missing_frame_index - 1
                or self.left_observed_timestamp_us > self.first_missing_timestamp_us
            ):
                raise ValueError("left observation must immediately precede the missing interval")
        if right_present:
            assert self.right_observed_frame_index is not None
            assert self.right_observed_timestamp_us is not None
            if (
                self.right_observed_frame_index != self.last_missing_frame_index + 1
                or self.right_observed_timestamp_us < self.last_missing_timestamp_us
            ):
                raise ValueError("right observation must immediately follow the missing interval")
        if self.boundary == "leading":
            expected_duration = (
                cast(int, self.right_observed_timestamp_us) - self.first_missing_timestamp_us
            )
        elif self.boundary == "internal":
            expected_duration = cast(int, self.right_observed_timestamp_us) - cast(
                int, self.left_observed_timestamp_us
            )
        elif self.boundary == "trailing":
            expected_duration = self.last_missing_timestamp_us - cast(
                int, self.left_observed_timestamp_us
            )
        else:
            expected_duration = self.last_missing_timestamp_us - self.first_missing_timestamp_us
        if self.duration_us != expected_duration:
            raise ValueError("missing interval duration does not match its observed bounds")
        if self.reasons != tuple(sorted(set(self.reasons))):
            raise ValueError("missing interval reasons must be unique and sorted")
        boundary_reason: MissingIntervalReason | None = {
            "leading": "leading_gap",
            "internal": None,
            "trailing": "trailing_gap",
            "entire_sequence": "entire_sequence_missing",
        }[self.boundary]
        if boundary_reason is not None and boundary_reason not in self.reasons:
            raise ValueError("missing interval must retain its boundary reason")
        boundary_reasons = {
            "leading_gap",
            "trailing_gap",
            "entire_sequence_missing",
        }
        if set(self.reasons).intersection(boundary_reasons) != (
            set() if boundary_reason is None else {boundary_reason}
        ):
            raise ValueError("missing interval boundary reasons must match its boundary kind")
        reason_bindings = (
            (self.contains_invalid_frame, "invalid_frame"),
            (self.contains_identity_ambiguity, "identity_ambiguity"),
            (self.crosses_timestamp_discontinuity, "timestamp_discontinuity"),
        )
        if any(flag != (reason in self.reasons) for flag, reason in reason_bindings):
            raise ValueError("missing interval barriers must match their coded reasons")
        if self.crosses_suspected_hand_swap and not self.contains_identity_ambiguity:
            raise ValueError("a suspected-swap barrier requires identity ambiguity")
        eligible = "eligible_short_internal_gap" in self.reasons
        blocked = (
            self.contains_invalid_frame
            or self.contains_identity_ambiguity
            or self.crosses_timestamp_discontinuity
            or self.crosses_suspected_hand_swap
        )
        if self.decision == "interpolate_linear":
            if self.boundary != "internal" or not eligible or blocked:
                raise ValueError("only an eligible unblocked internal gap may interpolate")
            if self.reasons != ("eligible_short_internal_gap",):
                raise ValueError("an interpolated gap cannot retain a conflicting reason")
        elif eligible:
            raise ValueError("a preserved gap cannot claim interpolation eligibility")
        return self


class QualityFindingV1(StrictContractModel):
    """The highest configured threshold violation for one quality metric."""

    schema_version: Literal["quality-finding/1"]
    rule_id: StableId
    metric: QualityMetricName
    direction: QualityMetricDirection
    severity: QualityFindingSeverity
    observed_value: NonNegativeSafeInteger
    threshold: NonNegativeSafeInteger

    @model_validator(mode="after")
    def _require_registered_actual_violation(self) -> Self:
        if self.rule_id != self.metric:
            raise ValueError("quality finding rule ID must equal its metric")
        if self.direction != QUALITY_METRIC_DIRECTIONS[self.metric]:
            raise ValueError("quality finding direction does not match its metric")
        if self.metric.endswith("_ppm") and (
            self.observed_value > 1_000_000 or self.threshold > 1_000_000
        ):
            raise ValueError("parts-per-million findings cannot exceed one million")
        violated = (
            self.observed_value > self.threshold
            if self.direction == "higher_is_worse"
            else self.observed_value < self.threshold
        )
        if not violated:
            raise ValueError("an exact or passing quality threshold cannot create a finding")
        return self


class TemporalResamplingSummaryV1(StrictContractModel):
    """Compact evidence for a duration-preserving elapsed-time target grid."""

    schema_version: Literal["temporal-resampling-summary/1"]
    clock: Literal["relative_timestamp_us"]
    grid_rule: Literal["nominal_elapsed_time_append_final/1"]
    target_rate_numerator: PositiveSafeInteger
    target_rate_denominator: PositiveSafeInteger
    declared_duration_us: PositiveSafeInteger
    observed_span_us: NonNegativeSafeInteger
    declared_unobserved_tail_us: NonNegativeSafeInteger
    declared_unobserved_tail_decision: Literal["preserve_missing"]
    target_count: PositiveSafeInteger
    first_target_timestamp_us: Literal[0]
    last_target_timestamp_us: NonNegativeSafeInteger
    target_grid_commitment_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_exact_grid_summary(self) -> Self:
        if self.declared_unobserved_tail_us != max(
            self.declared_duration_us - self.observed_span_us,
            0,
        ):
            raise ValueError("declared unobserved tail does not match source duration evidence")
        target_count, last_target_timestamp_us = elapsed_time_grid_shape(
            self.observed_span_us,
            self.target_rate_numerator,
            self.target_rate_denominator,
        )
        if (
            self.target_count != target_count
            or self.last_target_timestamp_us != last_target_timestamp_us
            or self.target_grid_commitment_sha256
            != elapsed_time_grid_commitment(
                self.observed_span_us,
                self.target_rate_numerator,
                self.target_rate_denominator,
            )
        ):
            raise ValueError("temporal resampling summary does not match its exact grid")
        return self


class SequenceQualityMetricsV1(StrictContractModel):
    """Integer, denominator-carrying quality metrics for one recording sequence."""

    frame_count: PositiveSafeInteger
    valid_frame_count: NonNegativeSafeInteger
    source_invalid_frame_count: NonNegativeSafeInteger
    task_inference_failed_frame_count: NonNegativeSafeInteger
    invalid_frame_fraction_ppm: PartsPerMillion
    expected_hand_count: ExpectedHandCount
    expected_hand_observation_count: NonNegativeSafeInteger
    expected_hand_opportunity_count: PositiveSafeInteger
    expected_hand_coverage_ppm: PartsPerMillion
    handedness_confidence_observation_count: NonNegativeSafeInteger
    low_handedness_confidence_observation_count: NonNegativeSafeInteger
    low_handedness_confidence_fraction_ppm: PartsPerMillion | None
    pose_anchor_presence_counts: tuple[
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
        NonNegativeSafeInteger,
    ]
    pose_anchor_coverage_ppm: tuple[
        PartsPerMillion,
        PartsPerMillion,
        PartsPerMillion,
        PartsPerMillion,
        PartsPerMillion,
        PartsPerMillion,
    ]
    minimum_pose_anchor_coverage_ppm: PartsPerMillion
    pose_confidence_observation_count: NonNegativeSafeInteger
    low_pose_confidence_observation_count: NonNegativeSafeInteger
    interpolated_gap_count: NonNegativeSafeInteger
    preserved_gap_count: NonNegativeSafeInteger
    longest_unfilled_internal_hand_gap_us: NonNegativeSafeInteger
    timestamp_delta_count: NonNegativeSafeInteger
    median_timestamp_delta_us: NonNegativeSafeInteger | None
    maximum_timestamp_delta_us: NonNegativeSafeInteger | None
    timestamp_discontinuity_count: NonNegativeSafeInteger
    temporal_discontinuity_count: NonNegativeSafeInteger
    suspected_hand_swap_count: NonNegativeSafeInteger

    @model_validator(mode="after")
    def _require_reconciled_metrics(self) -> Self:
        invalid = self.source_invalid_frame_count + self.task_inference_failed_frame_count
        if self.valid_frame_count + invalid != self.frame_count:
            raise ValueError("valid and invalid counts must cover every quality frame")
        if self.invalid_frame_fraction_ppm != ratio_ppm(invalid, self.frame_count):
            raise ValueError("invalid frame fraction does not match its counts")
        expected_opportunities = self.frame_count * self.expected_hand_count
        if self.expected_hand_opportunity_count != expected_opportunities:
            raise ValueError("expected-hand opportunities must include every source frame")
        if self.expected_hand_observation_count > self.expected_hand_opportunity_count:
            raise ValueError("expected-hand observations exceed their opportunities")
        if self.expected_hand_coverage_ppm != ratio_ppm(
            self.expected_hand_observation_count,
            self.expected_hand_opportunity_count,
        ):
            raise ValueError("expected-hand coverage does not match its counts")
        if (
            self.low_handedness_confidence_observation_count
            > self.handedness_confidence_observation_count
        ):
            raise ValueError("low handedness-confidence count exceeds evaluated observations")
        expected_low_fraction = (
            None
            if self.handedness_confidence_observation_count == 0
            else ratio_ppm(
                self.low_handedness_confidence_observation_count,
                self.handedness_confidence_observation_count,
            )
        )
        if self.low_handedness_confidence_fraction_ppm != expected_low_fraction:
            raise ValueError("low handedness-confidence fraction does not match its counts")
        if any(count > self.frame_count for count in self.pose_anchor_presence_counts):
            raise ValueError("pose-anchor presence count exceeds the source frame count")
        expected_pose_coverage = tuple(
            ratio_ppm(count, self.frame_count) for count in self.pose_anchor_presence_counts
        )
        if self.pose_anchor_coverage_ppm != expected_pose_coverage:
            raise ValueError("pose-anchor coverage does not match its counts")
        if self.minimum_pose_anchor_coverage_ppm != min(expected_pose_coverage):
            raise ValueError("minimum pose-anchor coverage does not match its anchors")
        if self.low_pose_confidence_observation_count > self.pose_confidence_observation_count:
            raise ValueError("low pose-confidence count exceeds evaluated observations")
        expected_delta_count = self.frame_count - 1
        if self.timestamp_delta_count != expected_delta_count:
            raise ValueError("timestamp delta count must cover every adjacent frame pair")
        has_deltas = self.timestamp_delta_count > 0
        if has_deltas != (
            self.median_timestamp_delta_us is not None
            and self.maximum_timestamp_delta_us is not None
        ):
            raise ValueError("timestamp summaries must be present exactly when deltas exist")
        if (
            self.median_timestamp_delta_us is not None
            and self.maximum_timestamp_delta_us is not None
            and self.median_timestamp_delta_us > self.maximum_timestamp_delta_us
        ):
            raise ValueError("median timestamp delta cannot exceed the maximum")
        for count in (
            self.timestamp_discontinuity_count,
            self.temporal_discontinuity_count,
            self.suspected_hand_swap_count,
        ):
            if count > self.timestamp_delta_count:
                raise ValueError("transition diagnostic count exceeds adjacent frame opportunities")
        return self


class SequenceQualityReportV1(StrictContractModel):
    """One recording-level report without sample, split, feature, or path identity."""

    schema_version: Literal["sequence-quality-report/1"]
    source_recording_id: RecordingId
    source_sequence_content_sha256: Sha256Digest
    source_landmark_parquet_sha256: Sha256Digest
    policy_sha256: Sha256Digest
    metrics: SequenceQualityMetricsV1
    gaps: tuple[MissingIntervalV1, ...]
    resampling: TemporalResamplingSummaryV1
    findings: tuple[QualityFindingV1, ...]
    disposition: QualityDisposition
    report_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_canonical_report(self) -> Self:
        gap_order = tuple(
            (_QUALITY_SIGNAL_RANK[gap.signal], gap.first_missing_frame_index) for gap in self.gaps
        )
        if gap_order != tuple(sorted(set(gap_order))):
            raise ValueError(
                "quality gaps must be unique and sorted by registered signal and frame"
            )
        gap_ids = tuple(gap.gap_id for gap in self.gaps)
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("quality gap IDs must be unique")
        by_signal: dict[str, int] = {}
        for gap in self.gaps:
            if gap.boundary == "leading" and gap.first_missing_frame_index != 0:
                raise ValueError("a leading quality gap must start at the first frame")
            if (
                gap.boundary == "trailing"
                and gap.last_missing_frame_index != self.metrics.frame_count - 1
            ):
                raise ValueError("a trailing quality gap must end at the final frame")
            if gap.boundary == "entire_sequence" and (
                gap.first_missing_frame_index != 0
                or gap.last_missing_frame_index != self.metrics.frame_count - 1
            ):
                raise ValueError("an entire-sequence quality gap must cover every frame")
            frame_indexes = (
                gap.first_missing_frame_index,
                gap.last_missing_frame_index,
                gap.left_observed_frame_index,
                gap.right_observed_frame_index,
            )
            if any(
                index is not None and index >= self.metrics.frame_count for index in frame_indexes
            ):
                raise ValueError("quality gap frame evidence exceeds the sequence frame count")
            timestamps_us = (
                gap.first_missing_timestamp_us,
                gap.last_missing_timestamp_us,
                gap.left_observed_timestamp_us,
                gap.right_observed_timestamp_us,
            )
            if any(
                timestamp is not None and timestamp > self.resampling.observed_span_us
                for timestamp in timestamps_us
            ):
                raise ValueError("quality gap timestamp evidence exceeds the observed span")
            if gap.first_missing_frame_index == 0 and gap.first_missing_timestamp_us != 0:
                raise ValueError("first-frame gap evidence must start at timestamp zero")
            if (
                gap.last_missing_frame_index == self.metrics.frame_count - 1
                and gap.last_missing_timestamp_us != self.resampling.observed_span_us
            ):
                raise ValueError("final-frame gap evidence must end at the observed span")
            previous_end = by_signal.get(gap.signal)
            if previous_end is not None and gap.first_missing_frame_index <= previous_end + 1:
                raise ValueError("quality gaps for one signal cannot overlap or be adjacent")
            by_signal[gap.signal] = gap.last_missing_frame_index
        finding_metrics = tuple(finding.metric for finding in self.findings)
        if finding_metrics != tuple(sorted(set(finding_metrics))):
            raise ValueError("quality findings must have unique metrics in sorted order")
        expected_disposition: QualityDisposition = "pass"
        if self.findings:
            expected_disposition = cast(
                QualityDisposition,
                max(
                    self.findings,
                    key=lambda item: _FINDING_SEVERITY_ORDER[item.severity],
                ).severity,
            )
        if self.disposition != expected_disposition:
            raise ValueError("sequence disposition must equal its highest finding severity")
        if self.metrics.interpolated_gap_count != sum(
            gap.decision == "interpolate_linear" for gap in self.gaps
        ):
            raise ValueError("interpolated gap count does not match gap decisions")
        if self.metrics.preserved_gap_count != sum(
            gap.decision == "preserve_missing" for gap in self.gaps
        ):
            raise ValueError("preserved gap count does not match gap decisions")
        longest_unfilled_internal_hand_gap = max(
            (
                gap.duration_us
                for gap in self.gaps
                if gap.signal in HAND_SLOT_IDS
                and gap.boundary == "internal"
                and gap.decision == "preserve_missing"
            ),
            default=0,
        )
        if self.metrics.longest_unfilled_internal_hand_gap_us != longest_unfilled_internal_hand_gap:
            raise ValueError("longest unfilled hand gap does not match preserved gap evidence")
        if self.report_sha256 != sequence_quality_report_digest(self):
            raise ValueError("sequence quality report digest does not match canonical content")
        return self


class DatasetQualityReportV1(StrictContractModel):
    """Weighted aggregate over every sequence disposition and denominator."""

    schema_version: Literal["dataset-quality-report/1"]
    sequence_count: PositiveSafeInteger
    pass_count: NonNegativeSafeInteger
    warning_count: NonNegativeSafeInteger
    quarantine_count: NonNegativeSafeInteger
    reject_count: NonNegativeSafeInteger
    total_frame_count: PositiveSafeInteger
    total_invalid_frame_count: NonNegativeSafeInteger
    invalid_frame_fraction_ppm: PartsPerMillion
    total_expected_hand_observation_count: NonNegativeSafeInteger
    total_expected_hand_opportunity_count: PositiveSafeInteger
    expected_hand_coverage_ppm: PartsPerMillion
    minimum_pose_anchor_coverage_ppm: PartsPerMillion
    longest_unfilled_internal_hand_gap_us: NonNegativeSafeInteger
    timestamp_discontinuity_count: NonNegativeSafeInteger
    temporal_discontinuity_count: NonNegativeSafeInteger
    suspected_hand_swap_count: NonNegativeSafeInteger
    status: DatasetQualityStatus

    @model_validator(mode="after")
    def _require_reconciled_aggregate(self) -> Self:
        if (
            self.pass_count + self.warning_count + self.quarantine_count + self.reject_count
            != self.sequence_count
        ):
            raise ValueError("dataset disposition counts must cover every sequence")
        if self.total_invalid_frame_count > self.total_frame_count:
            raise ValueError("dataset invalid frames exceed total frames")
        if self.invalid_frame_fraction_ppm != ratio_ppm(
            self.total_invalid_frame_count,
            self.total_frame_count,
        ):
            raise ValueError("dataset invalid fraction does not match weighted counts")
        if self.total_expected_hand_observation_count > self.total_expected_hand_opportunity_count:
            raise ValueError("dataset expected-hand observations exceed opportunities")
        if self.expected_hand_coverage_ppm != ratio_ppm(
            self.total_expected_hand_observation_count,
            self.total_expected_hand_opportunity_count,
        ):
            raise ValueError("dataset expected-hand coverage does not match weighted counts")
        expected_status: DatasetQualityStatus = (
            "blocked"
            if self.quarantine_count or self.reject_count
            else "ready_with_warnings"
            if self.warning_count
            else "ready"
        )
        if self.status != expected_status:
            raise ValueError("dataset quality status does not match its disposition counts")
        return self


class LandmarkQualityManifestV1(StrictContractModel):
    """Self-digested quality handoff bound to extraction, raw data, and policy."""

    model_config = contract_config("landmark-quality-manifest-1.schema.json")

    schema_version: Literal["landmark-quality-manifest/1"]
    quality_id: StableId
    version: SemanticVersion
    raw_dataset_id: DatasetId
    raw_dataset_version: SemanticVersion
    raw_data_sha256: Sha256Digest
    raw_dataset_manifest_sha256: Sha256Digest
    extraction_id: StableId
    extraction_version: SemanticVersion
    extraction_manifest_sha256: Sha256Digest
    extraction_config_sha256: Sha256Digest
    policy: LandmarkQualityPolicyV1
    policy_sha256: Sha256Digest
    sequence_reports: tuple[SequenceQualityReportV1, ...] = Field(min_length=1)
    dataset_report: DatasetQualityReportV1
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_canonical_bound_reports(self) -> Self:
        recording_ids = tuple(report.source_recording_id for report in self.sequence_reports)
        if recording_ids != tuple(sorted(set(recording_ids))):
            raise ValueError("quality sequence reports must have unique recordings in order")
        report_digests = tuple(report.report_sha256 for report in self.sequence_reports)
        if len(report_digests) != len(set(report_digests)):
            raise ValueError("quality sequence report digests must be unique")
        if self.policy_sha256 != landmark_quality_policy_digest(self.policy):
            raise ValueError("policy_sha256 does not match the quality policy")
        if any(report.policy_sha256 != self.policy_sha256 for report in self.sequence_reports):
            raise ValueError("every quality sequence report must bind the exact policy")
        for report in self.sequence_reports:
            _assert_report_decisions_match_policy(report, self.policy)
            _assert_report_findings_match_policy(report, self.policy)
        _assert_dataset_report_matches_sequences(self.dataset_report, self.sequence_reports)
        if self.manifest_sha256 != landmark_quality_manifest_digest(self):
            raise ValueError("manifest_sha256 does not match canonical quality content")
        return self


type QualityInput = BaseModel | str | bytes | bytearray | Mapping[str, object]


def _validate_model[ModelT: BaseModel](
    document: QualityInput,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        if isinstance(document, BaseModel):
            payload = cast(
                Mapping[str, object],
                document.model_dump(mode="json", round_trip=True),
            )
        else:
            payload = cast(Mapping[str, object], parse_json_object(document))
        return model.model_validate_json(canonical_json_bytes(payload), strict=True)
    except (CanonicalizationError, ValidationError) as error:
        raise QualityContractError(f"invalid {label}") from error


def validate_landmark_quality_policy(document: QualityInput) -> LandmarkQualityPolicyV1:
    """Validate one exact quality policy without coercion or implicit migration."""

    return _validate_model(document, LandmarkQualityPolicyV1, "landmark quality policy")


def landmark_quality_policy_digest(document: QualityInput) -> str:
    """Return the stable identity of one fully resolved quality policy."""

    checked = validate_landmark_quality_policy(document)
    return canonical_sha256(checked, domain=checked.schema_version)


def sequence_quality_report_digest(
    document: SequenceQualityReportV1 | Mapping[str, object],
) -> str:
    """Hash a sequence report while excluding its self-referential digest."""

    payload = (
        cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
        if isinstance(document, BaseModel)
        else dict(document)
    )
    payload.pop("report_sha256", None)
    try:
        return canonical_sha256(payload, domain="sequence-quality-report/1")
    except CanonicalizationError as error:
        raise QualityContractError("sequence quality report cannot be canonicalized") from error


def landmark_quality_manifest_digest(
    document: LandmarkQualityManifestV1 | Mapping[str, object],
) -> str:
    """Hash a quality manifest while excluding its self-referential digest."""

    payload = (
        cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
        if isinstance(document, BaseModel)
        else dict(document)
    )
    payload.pop("manifest_sha256", None)
    try:
        return canonical_sha256(payload, domain="landmark-quality-manifest/1")
    except CanonicalizationError as error:
        raise QualityContractError("landmark quality manifest cannot be canonicalized") from error


def validate_landmark_quality_manifest(document: QualityInput) -> LandmarkQualityManifestV1:
    """Validate one self-digested quality handoff without coercion or migration."""

    return _validate_model(document, LandmarkQualityManifestV1, "landmark quality manifest")


def assert_sequence_quality_report_matches_table(
    report: SequenceQualityReportV1,
    table: LandmarkFramesTableV1 | QualityInput,
) -> None:
    """Bind one recording-level report to exact immutable source rows."""

    checked = (
        table if isinstance(table, LandmarkFramesTableV1) else validate_landmark_frames_table(table)
    )
    if report.source_recording_id != checked.rows[
        0
    ].source_recording_id or report.source_sequence_content_sha256 != landmark_frames_table_digest(
        checked
    ):
        raise QualityContractError("sequence quality report does not match source landmarks")


def assert_landmark_quality_bound_to_extraction(
    manifest: LandmarkQualityManifestV1 | QualityInput,
    extraction: LandmarkExtractionManifestV1 | QualityInput,
) -> None:
    """Prove that quality decisions consume one exact extraction handoff."""

    checked = (
        manifest
        if isinstance(manifest, LandmarkQualityManifestV1)
        else validate_landmark_quality_manifest(manifest)
    )
    source = (
        extraction
        if isinstance(extraction, LandmarkExtractionManifestV1)
        else validate_landmark_extraction_manifest(extraction)
    )
    expected_manifest = (
        source.raw_dataset_id,
        source.raw_dataset_version,
        source.raw_data_sha256,
        source.raw_dataset_manifest_sha256,
        source.extraction_id,
        source.version,
        source.manifest_sha256,
        source.config_sha256,
    )
    actual_manifest = (
        checked.raw_dataset_id,
        checked.raw_dataset_version,
        checked.raw_data_sha256,
        checked.raw_dataset_manifest_sha256,
        checked.extraction_id,
        checked.extraction_version,
        checked.extraction_manifest_sha256,
        checked.extraction_config_sha256,
    )
    if actual_manifest != expected_manifest:
        raise QualityContractError("quality manifest does not match extraction identity")
    expected_reports = tuple(
        (
            sequence.lineage.source_recording_id,
            sequence.content_sha256,
            sequence.lineage.artifact.sha256,
        )
        for sequence in source.sequences
    )
    actual_reports = tuple(
        (
            report.source_recording_id,
            report.source_sequence_content_sha256,
            report.source_landmark_parquet_sha256,
        )
        for report in checked.sequence_reports
    )
    if actual_reports != expected_reports:
        raise QualityContractError("quality reports do not match extraction sequences")


def _metric_value(
    metrics: SequenceQualityMetricsV1,
    metric: QualityMetricName,
) -> int | None:
    value = getattr(metrics, metric)
    if value is not None and type(value) is not int:
        raise ValueError("registered quality metric is not an integer")
    return value


def _violates_threshold(value: int, threshold: int, direction: QualityMetricDirection) -> bool:
    return value > threshold if direction == "higher_is_worse" else value < threshold


def _expected_finding(
    metrics: SequenceQualityMetricsV1,
    rule: QualityThresholdRuleV1,
) -> QualityFindingV1 | None:
    value = _metric_value(metrics, rule.metric)
    if value is None:
        return None
    selected: tuple[QualityFindingSeverity, int] | None = None
    severities: tuple[
        QualityFindingSeverity,
        QualityFindingSeverity,
        QualityFindingSeverity,
    ] = ("warning", "quarantine", "reject")
    for severity in severities:
        threshold = cast(int | None, getattr(rule, severity))
        if threshold is not None and _violates_threshold(value, threshold, rule.direction):
            selected = (severity, threshold)
    if selected is None:
        return None
    severity, threshold = selected
    return QualityFindingV1(
        schema_version="quality-finding/1",
        rule_id=rule.rule_id,
        metric=rule.metric,
        direction=rule.direction,
        severity=severity,
        observed_value=value,
        threshold=threshold,
    )


def _assert_report_findings_match_policy(
    report: SequenceQualityReportV1,
    policy: LandmarkQualityPolicyV1,
) -> None:
    expected = tuple(
        finding
        for rule in policy.threshold_rules
        for finding in (_expected_finding(report.metrics, rule),)
        if finding is not None
    )
    if report.findings != expected:
        raise ValueError("sequence quality findings do not match the embedded policy")


def _assert_report_decisions_match_policy(
    report: SequenceQualityReportV1,
    policy: LandmarkQualityPolicyV1,
) -> None:
    if (
        report.resampling.grid_rule != policy.resampling_rule
        or report.resampling.target_rate_numerator != policy.target_rate_numerator
        or report.resampling.target_rate_denominator != policy.target_rate_denominator
    ):
        raise ValueError("sequence resampling does not match the embedded policy")
    for gap in report.gaps:
        too_many = gap.missing_frame_count > policy.max_interpolated_missing_frames
        too_long = gap.duration_us > policy.max_interpolation_bridge_us
        if ("too_many_missing_frames" in gap.reasons) != too_many:
            raise ValueError("quality gap frame-limit decision does not match the policy")
        if ("bridge_too_long" in gap.reasons) != too_long:
            raise ValueError("quality gap duration-limit decision does not match the policy")
        blocked = (
            gap.contains_invalid_frame
            or gap.contains_identity_ambiguity
            or gap.crosses_timestamp_discontinuity
            or gap.crosses_suspected_hand_swap
            or "low_confidence" in gap.reasons
        )
        eligible = gap.boundary == "internal" and not too_many and not too_long and not blocked
        expected_decision: MissingIntervalDecision = (
            "interpolate_linear" if eligible else "preserve_missing"
        )
        if gap.decision != expected_decision:
            raise ValueError("quality gap interpolation decision does not match the policy")


def _assert_dataset_report_matches_sequences(
    dataset: DatasetQualityReportV1,
    reports: tuple[SequenceQualityReportV1, ...],
) -> None:
    dispositions = tuple(report.disposition for report in reports)
    total_frames = sum(report.metrics.frame_count for report in reports)
    total_invalid = sum(
        report.metrics.source_invalid_frame_count + report.metrics.task_inference_failed_frame_count
        for report in reports
    )
    total_hand_observations = sum(
        report.metrics.expected_hand_observation_count for report in reports
    )
    total_hand_opportunities = sum(
        report.metrics.expected_hand_opportunity_count for report in reports
    )
    expected = (
        len(reports),
        dispositions.count("pass"),
        dispositions.count("warning"),
        dispositions.count("quarantine"),
        dispositions.count("reject"),
        total_frames,
        total_invalid,
        ratio_ppm(total_invalid, total_frames),
        total_hand_observations,
        total_hand_opportunities,
        ratio_ppm(total_hand_observations, total_hand_opportunities),
        min(report.metrics.minimum_pose_anchor_coverage_ppm for report in reports),
        max(report.metrics.longest_unfilled_internal_hand_gap_us for report in reports),
        sum(report.metrics.timestamp_discontinuity_count for report in reports),
        sum(report.metrics.temporal_discontinuity_count for report in reports),
        sum(report.metrics.suspected_hand_swap_count for report in reports),
    )
    actual = (
        dataset.sequence_count,
        dataset.pass_count,
        dataset.warning_count,
        dataset.quarantine_count,
        dataset.reject_count,
        dataset.total_frame_count,
        dataset.total_invalid_frame_count,
        dataset.invalid_frame_fraction_ppm,
        dataset.total_expected_hand_observation_count,
        dataset.total_expected_hand_opportunity_count,
        dataset.expected_hand_coverage_ppm,
        dataset.minimum_pose_anchor_coverage_ppm,
        dataset.longest_unfilled_internal_hand_gap_us,
        dataset.timestamp_discontinuity_count,
        dataset.temporal_discontinuity_count,
        dataset.suspected_hand_swap_count,
    )
    if actual != expected:
        raise ValueError("dataset quality aggregate does not match sequence reports")


__all__ = [
    "QUALITY_DISPOSITION_ORDER",
    "QUALITY_METRIC_DIRECTIONS",
    "QUALITY_METRIC_NAMES",
    "DatasetQualityReportV1",
    "DatasetQualityStatus",
    "ExpectedHandCount",
    "LandmarkQualityManifestV1",
    "LandmarkQualityPolicyV1",
    "MissingIntervalBoundary",
    "MissingIntervalDecision",
    "MissingIntervalReason",
    "MissingIntervalV1",
    "PartsPerMillion",
    "QualityContractError",
    "QualityDisposition",
    "QualityFindingSeverity",
    "QualityFindingV1",
    "QualityInput",
    "QualityMetricDirection",
    "QualityMetricName",
    "QualitySignalName",
    "QualityThresholdRuleV1",
    "SequenceQualityMetricsV1",
    "SequenceQualityReportV1",
    "TemporalResamplingSummaryV1",
    "assert_landmark_quality_bound_to_extraction",
    "assert_sequence_quality_report_matches_table",
    "elapsed_time_grid_commitment",
    "elapsed_time_grid_shape",
    "elapsed_time_grid_us",
    "landmark_quality_manifest_digest",
    "landmark_quality_policy_digest",
    "ratio_ppm",
    "sequence_quality_report_digest",
    "validate_landmark_quality_manifest",
    "validate_landmark_quality_policy",
]
