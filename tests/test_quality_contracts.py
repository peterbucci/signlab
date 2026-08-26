from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from signlab.contracts.extraction import (
    LandmarkExtractionManifestV1,
    validate_landmark_extraction_manifest,
)
from signlab.contracts.quality import (
    QUALITY_METRIC_NAMES,
    DatasetQualityReportV1,
    LandmarkQualityManifestV1,
    LandmarkQualityPolicyV1,
    MissingIntervalV1,
    QualityContractError,
    QualityFindingV1,
    SequenceQualityMetricsV1,
    SequenceQualityReportV1,
    TemporalResamplingSummaryV1,
    assert_landmark_quality_bound_to_extraction,
    assert_sequence_quality_report_matches_table,
    elapsed_time_grid_commitment,
    elapsed_time_grid_shape,
    elapsed_time_grid_us,
    landmark_quality_manifest_digest,
    landmark_quality_policy_digest,
    ratio_ppm,
    sequence_quality_report_digest,
    validate_landmark_quality_manifest,
    validate_landmark_quality_policy,
)
from signlab.quality.resources import build_default_quality_policy
from test_extraction_contracts import (
    RECORDING_ID,
    ZERO_DIGEST,
    _json,
    _table,
)
from test_extraction_contracts import (
    _manifest_payload as _extraction_manifest_payload,
)


def _extraction_manifest() -> LandmarkExtractionManifestV1:
    return validate_landmark_extraction_manifest(_extraction_manifest_payload())


def _resampling() -> TemporalResamplingSummaryV1:
    target_count, last_target_timestamp_us = elapsed_time_grid_shape(33_333, 30, 1)
    return TemporalResamplingSummaryV1(
        schema_version="temporal-resampling-summary/1",
        clock="relative_timestamp_us",
        grid_rule="nominal_elapsed_time_append_final/1",
        target_rate_numerator=30,
        target_rate_denominator=1,
        declared_duration_us=66_667,
        observed_span_us=33_333,
        declared_unobserved_tail_us=33_334,
        declared_unobserved_tail_decision="preserve_missing",
        target_count=target_count,
        first_target_timestamp_us=0,
        last_target_timestamp_us=last_target_timestamp_us,
        target_grid_commitment_sha256=elapsed_time_grid_commitment(33_333, 30, 1),
    )


def _metrics() -> SequenceQualityMetricsV1:
    return SequenceQualityMetricsV1(
        frame_count=2,
        valid_frame_count=1,
        source_invalid_frame_count=0,
        task_inference_failed_frame_count=1,
        invalid_frame_fraction_ppm=500_000,
        expected_hand_count=1,
        expected_hand_observation_count=1,
        expected_hand_opportunity_count=2,
        expected_hand_coverage_ppm=500_000,
        handedness_confidence_observation_count=1,
        low_handedness_confidence_observation_count=0,
        low_handedness_confidence_fraction_ppm=0,
        pose_anchor_presence_counts=(1, 1, 1, 1, 1, 1),
        pose_anchor_coverage_ppm=(500_000, 500_000, 500_000, 500_000, 500_000, 500_000),
        minimum_pose_anchor_coverage_ppm=500_000,
        pose_confidence_observation_count=12,
        low_pose_confidence_observation_count=0,
        interpolated_gap_count=0,
        preserved_gap_count=0,
        longest_unfilled_internal_hand_gap_us=0,
        timestamp_delta_count=1,
        median_timestamp_delta_us=33_333,
        maximum_timestamp_delta_us=33_333,
        timestamp_discontinuity_count=0,
        temporal_discontinuity_count=0,
        suspected_hand_swap_count=0,
    )


def _finding() -> QualityFindingV1:
    return QualityFindingV1(
        schema_version="quality-finding/1",
        rule_id="invalid_frame_fraction_ppm",
        metric="invalid_frame_fraction_ppm",
        direction="higher_is_worse",
        severity="reject",
        observed_value=500_000,
        threshold=200_000,
    )


def _coverage_finding() -> QualityFindingV1:
    return QualityFindingV1(
        schema_version="quality-finding/1",
        rule_id="expected_hand_coverage_ppm",
        metric="expected_hand_coverage_ppm",
        direction="lower_is_worse",
        severity="quarantine",
        observed_value=500_000,
        threshold=800_000,
    )


def _pose_finding() -> QualityFindingV1:
    return QualityFindingV1(
        schema_version="quality-finding/1",
        rule_id="minimum_pose_anchor_coverage_ppm",
        metric="minimum_pose_anchor_coverage_ppm",
        direction="lower_is_worse",
        severity="quarantine",
        observed_value=500_000,
        threshold=700_000,
    )


def _sequence_report_payload() -> dict[str, Any]:
    extraction = _extraction_manifest()
    payload: dict[str, Any] = {
        "schema_version": "sequence-quality-report/1",
        "source_recording_id": RECORDING_ID,
        "source_sequence_content_sha256": extraction.sequences[0].content_sha256,
        "source_landmark_parquet_sha256": extraction.sequences[0].lineage.artifact.sha256,
        "policy_sha256": landmark_quality_policy_digest(build_default_quality_policy()),
        "metrics": _json(_metrics()),
        "gaps": [],
        "resampling": _json(_resampling()),
        "findings": [
            _json(_coverage_finding()),
            _json(_finding()),
            _json(_pose_finding()),
        ],
        "disposition": "reject",
        "report_sha256": ZERO_DIGEST,
    }
    payload["report_sha256"] = sequence_quality_report_digest(payload)
    return payload


def _sequence_report() -> SequenceQualityReportV1:
    return SequenceQualityReportV1.model_validate_json(
        json.dumps(_sequence_report_payload()),
        strict=True,
    )


def _dataset_report() -> DatasetQualityReportV1:
    metrics = _metrics()
    return DatasetQualityReportV1(
        schema_version="dataset-quality-report/1",
        sequence_count=1,
        pass_count=0,
        warning_count=0,
        quarantine_count=0,
        reject_count=1,
        total_frame_count=metrics.frame_count,
        total_invalid_frame_count=1,
        invalid_frame_fraction_ppm=metrics.invalid_frame_fraction_ppm,
        total_expected_hand_observation_count=metrics.expected_hand_observation_count,
        total_expected_hand_opportunity_count=metrics.expected_hand_opportunity_count,
        expected_hand_coverage_ppm=metrics.expected_hand_coverage_ppm,
        minimum_pose_anchor_coverage_ppm=metrics.minimum_pose_anchor_coverage_ppm,
        longest_unfilled_internal_hand_gap_us=metrics.longest_unfilled_internal_hand_gap_us,
        timestamp_discontinuity_count=metrics.timestamp_discontinuity_count,
        temporal_discontinuity_count=metrics.temporal_discontinuity_count,
        suspected_hand_swap_count=metrics.suspected_hand_swap_count,
        status="blocked",
    )


def _quality_manifest_payload() -> dict[str, Any]:
    extraction = _extraction_manifest()
    policy = build_default_quality_policy()
    payload: dict[str, Any] = {
        "schema_version": "landmark-quality-manifest/1",
        "quality_id": "landmark_quality_assessment",
        "version": "1.0.0",
        "raw_dataset_id": extraction.raw_dataset_id,
        "raw_dataset_version": extraction.raw_dataset_version,
        "raw_data_sha256": extraction.raw_data_sha256,
        "raw_dataset_manifest_sha256": extraction.raw_dataset_manifest_sha256,
        "extraction_id": extraction.extraction_id,
        "extraction_version": extraction.version,
        "extraction_manifest_sha256": extraction.manifest_sha256,
        "extraction_config_sha256": extraction.config_sha256,
        "policy": _json(policy),
        "policy_sha256": landmark_quality_policy_digest(policy),
        "sequence_reports": [_sequence_report_payload()],
        "dataset_report": _json(_dataset_report()),
        "manifest_sha256": ZERO_DIGEST,
    }
    payload["manifest_sha256"] = landmark_quality_manifest_digest(payload)
    return payload


def test_default_policy_has_exact_registered_metrics_directions_and_stable_digest() -> None:
    policy = build_default_quality_policy()
    assert tuple(rule.metric for rule in policy.threshold_rules) == QUALITY_METRIC_NAMES
    assert policy.target_rate_numerator == 30
    assert policy.target_rate_denominator == 1
    assert policy.max_interpolated_missing_frames == 2
    assert policy.max_interpolation_bridge_us == 100_000
    assert policy.handedness_confidence_diagnostic_ppm == 800_000
    assert policy.pose_visibility_diagnostic_ppm == 500_000
    assert policy.pose_presence_diagnostic_ppm == 500_000
    assert policy.timestamp_discontinuity_absolute_us == 100_000
    assert policy.timestamp_discontinuity_median_multiplier_ppm == 3_000_000
    assert policy.suspected_swap_cost_margin_ppm == 10_000
    assert policy.max_palm_wrist_speed_units_per_second == 12
    assert landmark_quality_policy_digest(policy) == landmark_quality_policy_digest(
        validate_landmark_quality_policy(policy.model_dump_json(round_trip=True))
    )


def test_policy_rules_reject_unknown_direction_order_and_missing_mandatory_rejection() -> None:
    payload = _json(build_default_quality_policy())
    rules = cast(list[dict[str, Any]], payload["threshold_rules"])

    wrong_direction = copy.deepcopy(payload)
    cast(list[dict[str, Any]], wrong_direction["threshold_rules"])[0]["direction"] = (
        "higher_is_worse"
    )
    with pytest.raises(QualityContractError, match="invalid landmark quality policy"):
        validate_landmark_quality_policy(wrong_direction)

    unordered = copy.deepcopy(payload)
    cast(list[dict[str, Any]], unordered["threshold_rules"])[1]["quarantine"] = 5_000
    with pytest.raises(QualityContractError, match="invalid landmark quality policy"):
        validate_landmark_quality_policy(unordered)

    no_coverage_reject = copy.deepcopy(payload)
    cast(list[dict[str, Any]], no_coverage_reject["threshold_rules"])[0]["reject"] = None
    with pytest.raises(QualityContractError, match="invalid landmark quality policy"):
        validate_landmark_quality_policy(no_coverage_reject)

    no_invalid_reject = copy.deepcopy(payload)
    cast(list[dict[str, Any]], no_invalid_reject["threshold_rules"])[1]["reject"] = 1_000_000
    with pytest.raises(QualityContractError, match="invalid landmark quality policy"):
        validate_landmark_quality_policy(no_invalid_reject)

    unknown = copy.deepcopy(payload)
    rules = cast(list[dict[str, Any]], unknown["threshold_rules"])
    rules[0]["metric"] = "made_up_metric"
    rules[0]["rule_id"] = "made_up_metric"
    with pytest.raises(QualityContractError, match="invalid landmark quality policy"):
        validate_landmark_quality_policy(unknown)


def test_ratio_and_nominal_elapsed_grid_are_integer_exact_and_duration_preserving() -> None:
    assert ratio_ppm(1, 3) == 333_333
    assert ratio_ppm(2, 3) == 666_667
    assert elapsed_time_grid_us(0, 30, 1) == (0,)
    assert elapsed_time_grid_us(90_000, 30, 1) == (0, 33_333, 66_667, 90_000)
    assert elapsed_time_grid_us(100_000, 30, 1) == (0, 33_333, 66_667, 100_000)
    assert elapsed_time_grid_shape(90_000, 30, 1) == (4, 90_000)
    assert elapsed_time_grid_commitment(90_000, 30, 1) != elapsed_time_grid_commitment(
        90_000, 60, 1
    )
    with pytest.raises(QualityContractError):
        ratio_ppm(2, 1)
    with pytest.raises(QualityContractError):
        elapsed_time_grid_us(-1, 30, 1)


def test_resampling_summary_preserves_declared_tail_without_extrapolation() -> None:
    summary = _resampling()
    assert summary.declared_unobserved_tail_us == 33_334
    assert summary.declared_unobserved_tail_decision == "preserve_missing"
    assert summary.last_target_timestamp_us == summary.observed_span_us

    payload = _json(summary)
    payload["declared_unobserved_tail_us"] = 0
    with pytest.raises(ValidationError):
        TemporalResamplingSummaryV1.model_validate(payload, strict=True)


def test_large_resampling_summary_validates_without_materializing_the_grid() -> None:
    observed_span_us = 1_000_000_000_000
    target_count, last_timestamp_us = elapsed_time_grid_shape(observed_span_us, 30, 1)

    summary = TemporalResamplingSummaryV1(
        schema_version="temporal-resampling-summary/1",
        clock="relative_timestamp_us",
        grid_rule="nominal_elapsed_time_append_final/1",
        target_rate_numerator=30,
        target_rate_denominator=1,
        declared_duration_us=observed_span_us,
        observed_span_us=observed_span_us,
        declared_unobserved_tail_us=0,
        declared_unobserved_tail_decision="preserve_missing",
        target_count=target_count,
        first_target_timestamp_us=0,
        last_target_timestamp_us=last_timestamp_us,
        target_grid_commitment_sha256=elapsed_time_grid_commitment(observed_span_us, 30, 1),
    )

    assert summary.target_count == 30_000_001
    with pytest.raises(QualityContractError, match="too large to materialize safely"):
        elapsed_time_grid_us(observed_span_us, 30, 1)
    payload = _json(summary)
    payload["target_count"] += 1
    with pytest.raises(ValidationError):
        TemporalResamplingSummaryV1.model_validate(payload, strict=True)


def test_gap_contract_allows_only_short_internal_unblocked_interpolation() -> None:
    gap = MissingIntervalV1(
        schema_version="missing-interval/1",
        gap_id="gap_hand_0_0001",
        signal="hand_0",
        boundary="internal",
        first_missing_frame_index=1,
        last_missing_frame_index=2,
        first_missing_timestamp_us=33_333,
        last_missing_timestamp_us=66_667,
        missing_frame_count=2,
        duration_us=100_000,
        left_observed_frame_index=0,
        left_observed_timestamp_us=0,
        right_observed_frame_index=3,
        right_observed_timestamp_us=100_000,
        contains_invalid_frame=False,
        contains_identity_ambiguity=False,
        crosses_timestamp_discontinuity=False,
        crosses_suspected_hand_swap=False,
        decision="interpolate_linear",
        reasons=("eligible_short_internal_gap",),
    )
    assert gap.decision == "interpolate_linear"

    blocked = _json(gap)
    blocked["contains_invalid_frame"] = True
    with pytest.raises(ValidationError):
        MissingIntervalV1.model_validate(blocked, strict=True)
    forged_duration = _json(gap)
    forged_duration["duration_us"] = 66_667
    with pytest.raises(ValidationError):
        MissingIntervalV1.model_validate(forged_duration, strict=True)
    nonadjacent_bound = _json(gap)
    nonadjacent_bound["right_observed_frame_index"] = 4
    with pytest.raises(ValidationError, match="immediately follow"):
        MissingIntervalV1.model_validate_json(json.dumps(nonadjacent_bound), strict=True)
    ambiguous = _json(gap)
    ambiguous.update(
        contains_identity_ambiguity=True,
        decision="preserve_missing",
        reasons=("identity_ambiguity",),
    )
    assert MissingIntervalV1.model_validate(ambiguous, strict=True).contains_identity_ambiguity
    suspected_without_ambiguity = copy.deepcopy(ambiguous)
    suspected_without_ambiguity["contains_identity_ambiguity"] = False
    suspected_without_ambiguity["crosses_suspected_hand_swap"] = True
    with pytest.raises(ValidationError):
        MissingIntervalV1.model_validate(suspected_without_ambiguity, strict=True)
    leading = _json(gap)
    leading.update(
        boundary="leading",
        duration_us=66_667,
        left_observed_frame_index=None,
        left_observed_timestamp_us=None,
        decision="preserve_missing",
        reasons=["leading_gap"],
    )
    assert (
        MissingIntervalV1.model_validate_json(json.dumps(leading), strict=True).boundary
        == "leading"
    )


def test_findings_require_a_strict_violation_and_accurate_metric_name() -> None:
    assert _finding().severity == "reject"
    exact = _json(_finding())
    exact["observed_value"] = exact["threshold"]
    with pytest.raises(ValidationError):
        QualityFindingV1.model_validate(exact, strict=True)
    wrong_direction = _json(_finding())
    wrong_direction["direction"] = "lower_is_worse"
    with pytest.raises(ValidationError):
        QualityFindingV1.model_validate(wrong_direction, strict=True)


def test_metrics_reconcile_weighted_denominators_nullable_confidence_and_timing() -> None:
    assert _metrics().invalid_frame_fraction_ppm == 500_000
    payload = _json(_metrics())
    payload["expected_hand_coverage_ppm"] = 1_000_000
    with pytest.raises(ValidationError):
        SequenceQualityMetricsV1.model_validate_json(json.dumps(payload), strict=True)
    payload = _json(_metrics())
    payload["handedness_confidence_observation_count"] = 0
    payload["low_handedness_confidence_fraction_ppm"] = 0
    with pytest.raises(ValidationError):
        SequenceQualityMetricsV1.model_validate_json(json.dumps(payload), strict=True)
    payload["low_handedness_confidence_fraction_ppm"] = None
    assert (
        SequenceQualityMetricsV1.model_validate_json(
            json.dumps(payload), strict=True
        ).low_handedness_confidence_fraction_ppm
        is None
    )


def test_sequence_report_is_recording_level_sorted_self_digested_and_source_bound() -> None:
    report = _sequence_report()
    assert report.disposition == "reject"
    assert report.report_sha256 == sequence_quality_report_digest(report)
    assert_sequence_quality_report_matches_table(report, _table())

    tampered = _sequence_report_payload()
    tampered["disposition"] = "warning"
    tampered["report_sha256"] = sequence_quality_report_digest(tampered)
    with pytest.raises(ValidationError):
        SequenceQualityReportV1.model_validate_json(json.dumps(tampered), strict=True)

    wrong_source = copy.deepcopy(_sequence_report_payload())
    wrong_source["source_sequence_content_sha256"] = "sha256:" + "f" * 64
    wrong_source["report_sha256"] = sequence_quality_report_digest(wrong_source)
    wrong_report = SequenceQualityReportV1.model_validate_json(
        json.dumps(wrong_source),
        strict=True,
    )
    with pytest.raises(QualityContractError, match="does not match source landmarks"):
        assert_sequence_quality_report_matches_table(wrong_report, _table())


def test_sequence_report_rejects_gap_evidence_outside_its_frame_and_time_envelope() -> None:
    outside_frames = _sequence_report_payload()
    cast(dict[str, Any], outside_frames["metrics"])["interpolated_gap_count"] = 1
    outside_frames["gaps"] = [
        {
            "schema_version": "missing-interval/1",
            "gap_id": "gap_hand_0_0010",
            "signal": "hand_0",
            "boundary": "internal",
            "first_missing_frame_index": 10,
            "last_missing_frame_index": 12,
            "first_missing_timestamp_us": 10_000,
            "last_missing_timestamp_us": 20_000,
            "missing_frame_count": 3,
            "duration_us": 33_333,
            "left_observed_frame_index": 9,
            "left_observed_timestamp_us": 0,
            "right_observed_frame_index": 13,
            "right_observed_timestamp_us": 33_333,
            "contains_invalid_frame": False,
            "contains_identity_ambiguity": False,
            "crosses_timestamp_discontinuity": False,
            "crosses_suspected_hand_swap": False,
            "decision": "interpolate_linear",
            "reasons": ["eligible_short_internal_gap"],
        }
    ]
    outside_frames["report_sha256"] = sequence_quality_report_digest(outside_frames)
    with pytest.raises(ValidationError, match="frame evidence exceeds"):
        SequenceQualityReportV1.model_validate_json(json.dumps(outside_frames), strict=True)

    outside_time = _sequence_report_payload()
    cast(dict[str, Any], outside_time["metrics"])["preserved_gap_count"] = 1
    outside_time["gaps"] = [
        {
            "schema_version": "missing-interval/1",
            "gap_id": "gap_hand_0_0000",
            "signal": "hand_0",
            "boundary": "leading",
            "first_missing_frame_index": 0,
            "last_missing_frame_index": 0,
            "first_missing_timestamp_us": 0,
            "last_missing_timestamp_us": 0,
            "missing_frame_count": 1,
            "duration_us": 100_000,
            "left_observed_frame_index": None,
            "left_observed_timestamp_us": None,
            "right_observed_frame_index": 1,
            "right_observed_timestamp_us": 100_000,
            "contains_invalid_frame": False,
            "contains_identity_ambiguity": False,
            "crosses_timestamp_discontinuity": False,
            "crosses_suspected_hand_swap": False,
            "decision": "preserve_missing",
            "reasons": ["leading_gap"],
        }
    ]
    outside_time["report_sha256"] = sequence_quality_report_digest(outside_time)
    with pytest.raises(ValidationError, match="timestamp evidence exceeds"):
        SequenceQualityReportV1.model_validate_json(json.dumps(outside_time), strict=True)

    leading_gap = copy.deepcopy(cast(list[dict[str, Any]], outside_time["gaps"])[0])
    leading_gap.update(
        first_missing_timestamp_us=1,
        last_missing_timestamp_us=1,
        duration_us=33_332,
        right_observed_timestamp_us=33_333,
    )
    wrong_endpoint = _sequence_report_payload()
    cast(dict[str, Any], wrong_endpoint["metrics"])["preserved_gap_count"] = 1
    wrong_endpoint["gaps"] = [leading_gap]
    wrong_endpoint["report_sha256"] = sequence_quality_report_digest(wrong_endpoint)
    with pytest.raises(ValidationError, match="timestamp zero"):
        SequenceQualityReportV1.model_validate_json(json.dumps(wrong_endpoint), strict=True)

    incomplete_entire_gap = copy.deepcopy(leading_gap)
    incomplete_entire_gap.update(
        gap_id="gap_hand_0_entire",
        boundary="entire_sequence",
        first_missing_timestamp_us=0,
        last_missing_timestamp_us=0,
        duration_us=0,
        right_observed_frame_index=None,
        right_observed_timestamp_us=None,
        reasons=["entire_sequence_missing"],
    )
    incomplete_entire = _sequence_report_payload()
    cast(dict[str, Any], incomplete_entire["metrics"])["preserved_gap_count"] = 1
    incomplete_entire["gaps"] = [incomplete_entire_gap]
    incomplete_entire["report_sha256"] = sequence_quality_report_digest(incomplete_entire)
    with pytest.raises(ValidationError, match="cover every frame"):
        SequenceQualityReportV1.model_validate_json(json.dumps(incomplete_entire), strict=True)

    leading_gap.update(
        first_missing_timestamp_us=0,
        last_missing_timestamp_us=0,
        duration_us=33_333,
    )
    trailing_gap = copy.deepcopy(leading_gap)
    trailing_gap.update(
        gap_id="gap_hand_0_0001",
        boundary="trailing",
        first_missing_frame_index=1,
        last_missing_frame_index=1,
        first_missing_timestamp_us=33_333,
        last_missing_timestamp_us=33_333,
        left_observed_frame_index=0,
        left_observed_timestamp_us=0,
        right_observed_frame_index=None,
        right_observed_timestamp_us=None,
        reasons=["trailing_gap"],
    )
    adjacent = _sequence_report_payload()
    cast(dict[str, Any], adjacent["metrics"])["preserved_gap_count"] = 2
    adjacent["gaps"] = [leading_gap, trailing_gap]
    adjacent["report_sha256"] = sequence_quality_report_digest(adjacent)
    with pytest.raises(ValidationError, match="overlap or be adjacent"):
        SequenceQualityReportV1.model_validate_json(json.dumps(adjacent), strict=True)


def test_manifest_binds_exact_extraction_policy_reports_aggregate_and_self_digest() -> None:
    extraction = _extraction_manifest()
    checked = validate_landmark_quality_manifest(_quality_manifest_payload())
    assert isinstance(checked, LandmarkQualityManifestV1)
    assert checked.manifest_sha256 == landmark_quality_manifest_digest(checked)
    assert_landmark_quality_bound_to_extraction(checked, extraction)

    for field in (
        "raw_data_sha256",
        "extraction_manifest_sha256",
        "extraction_config_sha256",
    ):
        tampered = _quality_manifest_payload()
        tampered[field] = "sha256:" + "f" * 64
        tampered["manifest_sha256"] = landmark_quality_manifest_digest(tampered)
        wrong = validate_landmark_quality_manifest(tampered)
        with pytest.raises(QualityContractError, match="does not match extraction identity"):
            assert_landmark_quality_bound_to_extraction(wrong, extraction)

    aggregate_tamper = _quality_manifest_payload()
    cast(dict[str, Any], aggregate_tamper["dataset_report"])["reject_count"] = 0
    cast(dict[str, Any], aggregate_tamper["dataset_report"])["pass_count"] = 1
    cast(dict[str, Any], aggregate_tamper["dataset_report"])["status"] = "ready"
    aggregate_tamper["manifest_sha256"] = landmark_quality_manifest_digest(aggregate_tamper)
    with pytest.raises(QualityContractError, match="invalid landmark quality manifest"):
        validate_landmark_quality_manifest(aggregate_tamper)

    finding_tamper = _quality_manifest_payload()
    sequence = cast(dict[str, Any], cast(list[object], finding_tamper["sequence_reports"])[0])
    sequence["findings"] = cast(list[object], sequence["findings"])[1:]
    sequence["report_sha256"] = sequence_quality_report_digest(sequence)
    finding_tamper["manifest_sha256"] = landmark_quality_manifest_digest(finding_tamper)
    with pytest.raises(QualityContractError, match="invalid landmark quality manifest"):
        validate_landmark_quality_manifest(finding_tamper)


def test_manifest_binds_resampling_rate_and_gap_decisions_to_embedded_policy() -> None:
    rate_tamper = _quality_manifest_payload()
    rate_report = cast(dict[str, Any], cast(list[object], rate_tamper["sequence_reports"])[0])
    resampling = cast(dict[str, Any], rate_report["resampling"])
    resampling["target_rate_numerator"] = 60
    target_count, last_timestamp_us = elapsed_time_grid_shape(33_333, 60, 1)
    resampling["target_count"] = target_count
    resampling["last_target_timestamp_us"] = last_timestamp_us
    resampling["target_grid_commitment_sha256"] = elapsed_time_grid_commitment(33_333, 60, 1)
    rate_report["report_sha256"] = sequence_quality_report_digest(rate_report)
    rate_tamper["manifest_sha256"] = landmark_quality_manifest_digest(rate_tamper)
    with pytest.raises(QualityContractError, match="invalid landmark quality manifest"):
        validate_landmark_quality_manifest(rate_tamper)

    decision_tamper = _quality_manifest_payload()
    decision_report = cast(
        dict[str, Any], cast(list[object], decision_tamper["sequence_reports"])[0]
    )
    metrics = cast(dict[str, Any], decision_report["metrics"])
    metrics.update(
        frame_count=3,
        valid_frame_count=2,
        source_invalid_frame_count=0,
        task_inference_failed_frame_count=1,
        invalid_frame_fraction_ppm=333_333,
        expected_hand_observation_count=1,
        expected_hand_opportunity_count=3,
        expected_hand_coverage_ppm=333_333,
        handedness_confidence_observation_count=1,
        pose_anchor_presence_counts=[1, 1, 1, 1, 1, 1],
        pose_anchor_coverage_ppm=[333_333, 333_333, 333_333, 333_333, 333_333, 333_333],
        minimum_pose_anchor_coverage_ppm=333_333,
        pose_confidence_observation_count=12,
        interpolated_gap_count=1,
        timestamp_delta_count=2,
        median_timestamp_delta_us=33_333,
        maximum_timestamp_delta_us=33_334,
    )
    decision_report["gaps"] = [
        {
            "schema_version": "missing-interval/1",
            "gap_id": "gap_hand_0_0001",
            "signal": "hand_0",
            "boundary": "internal",
            "first_missing_frame_index": 1,
            "last_missing_frame_index": 1,
            "first_missing_timestamp_us": 33_333,
            "last_missing_timestamp_us": 33_333,
            "missing_frame_count": 1,
            "duration_us": 66_667,
            "left_observed_frame_index": 0,
            "left_observed_timestamp_us": 0,
            "right_observed_frame_index": 2,
            "right_observed_timestamp_us": 66_667,
            "contains_invalid_frame": False,
            "contains_identity_ambiguity": False,
            "crosses_timestamp_discontinuity": False,
            "crosses_suspected_hand_swap": False,
            "decision": "interpolate_linear",
            "reasons": ["eligible_short_internal_gap"],
        }
    ]
    decision_resampling = cast(dict[str, Any], decision_report["resampling"])
    decision_resampling.update(
        observed_span_us=66_667,
        declared_unobserved_tail_us=0,
        target_count=3,
        last_target_timestamp_us=66_667,
        target_grid_commitment_sha256=elapsed_time_grid_commitment(66_667, 30, 1),
    )
    findings = cast(list[dict[str, Any]], decision_report["findings"])
    findings[0].update(severity="reject", observed_value=333_333, threshold=500_000)
    findings[1]["observed_value"] = 333_333
    findings[2]["observed_value"] = 333_333
    decision_report["report_sha256"] = sequence_quality_report_digest(decision_report)
    dataset = cast(dict[str, Any], decision_tamper["dataset_report"])
    dataset.update(
        total_frame_count=3,
        total_invalid_frame_count=1,
        invalid_frame_fraction_ppm=333_333,
        total_expected_hand_observation_count=1,
        total_expected_hand_opportunity_count=3,
        expected_hand_coverage_ppm=333_333,
        minimum_pose_anchor_coverage_ppm=333_333,
    )
    decision_tamper["manifest_sha256"] = landmark_quality_manifest_digest(decision_tamper)
    assert validate_landmark_quality_manifest(decision_tamper)

    policy_payload = cast(dict[str, Any], decision_tamper["policy"])
    policy_payload["max_interpolated_missing_frames"] = 0
    policy = validate_landmark_quality_policy(policy_payload)
    policy_sha256 = landmark_quality_policy_digest(policy)
    decision_tamper["policy"] = _json(policy)
    decision_tamper["policy_sha256"] = policy_sha256
    decision_report["policy_sha256"] = policy_sha256
    decision_report["report_sha256"] = sequence_quality_report_digest(decision_report)
    decision_tamper["manifest_sha256"] = landmark_quality_manifest_digest(decision_tamper)
    with pytest.raises(QualityContractError, match="invalid landmark quality manifest"):
        validate_landmark_quality_manifest(decision_tamper)

    policy_payload = _json(build_default_quality_policy())
    policy_payload["max_interpolation_bridge_us"] = 66_666
    policy = validate_landmark_quality_policy(policy_payload)
    policy_sha256 = landmark_quality_policy_digest(policy)
    decision_tamper["policy"] = _json(policy)
    decision_tamper["policy_sha256"] = policy_sha256
    decision_report["policy_sha256"] = policy_sha256
    decision_report["report_sha256"] = sequence_quality_report_digest(decision_report)
    decision_tamper["manifest_sha256"] = landmark_quality_manifest_digest(decision_tamper)
    with pytest.raises(QualityContractError, match="invalid landmark quality manifest"):
        validate_landmark_quality_manifest(decision_tamper)


def test_public_quality_readers_reject_extra_fields_nonobjects_and_bad_digests() -> None:
    policy = _json(build_default_quality_policy())
    policy["unexpected"] = True
    with pytest.raises(QualityContractError, match="invalid landmark quality policy"):
        validate_landmark_quality_policy(policy)
    with pytest.raises(QualityContractError, match="invalid landmark quality policy"):
        validate_landmark_quality_policy("[]")

    manifest = _quality_manifest_payload()
    manifest["manifest_sha256"] = "sha256:" + "f" * 64
    with pytest.raises(QualityContractError, match="invalid landmark quality manifest"):
        validate_landmark_quality_manifest(manifest)
    manifest = _quality_manifest_payload()
    manifest["unexpected"] = True
    with pytest.raises(QualityContractError, match="invalid landmark quality manifest"):
        validate_landmark_quality_manifest(json.dumps(manifest))


def test_policy_type_remains_a_closed_strict_contract() -> None:
    with pytest.raises(ValidationError):
        LandmarkQualityPolicyV1.model_validate(
            {**_json(build_default_quality_policy()), "target_rate_numerator": 30.5},
            strict=True,
        )
