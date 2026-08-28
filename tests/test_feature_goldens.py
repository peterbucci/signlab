from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from feature_fixtures import (
    EXTRACTION_CONFIG_SHA256,
    FeatureFixture,
    make_feature_fixture,
    make_feature_plan,
    make_hand_row,
)
from signlab.contracts.canonical import canonical_sha256
from signlab.contracts.features import (
    LandmarkFeaturePlanV1,
    PortableFeatureSequenceV1,
    landmark_feature_plan_digest,
)
from signlab.features.transforms import derive_feature_sequence

GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "public" / "features" / "portable-landmark-goldens-v1.json"
)


def _derive(
    fixture: FeatureFixture,
    plan: LandmarkFeaturePlanV1,
) -> PortableFeatureSequenceV1:
    return derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        plan,
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )


def _summary(
    plan: LandmarkFeaturePlanV1,
    sequence: PortableFeatureSequenceV1,
) -> dict[str, object]:
    return {
        "feature_plan_sha256": landmark_feature_plan_digest(plan),
        "sequence_sha256": sequence.sequence_sha256,
        "feature_order_sha256": canonical_sha256(
            {"feature_names": list(sequence.feature_names)},
            domain="feature-order/1",
        ),
        "feature_count": len(sequence.feature_names),
        "valid_cell_count": sum(sum(row) for row in sequence.valid_mask),
        "observed_cell_count": sum(sum(row) for row in sequence.observed_mask),
        "interpolated_cell_count": sum(sum(row) for row in sequence.interpolated_mask),
        "padding_frame_count": sum(sequence.padding_mask),
        "values_sha256": canonical_sha256(
            {"values_q": [list(row) for row in sequence.values_q]},
            domain="feature-values/1",
        ),
    }


def golden_document() -> dict[str, object]:
    """Return the complete deterministic public golden document."""

    cases = _golden_sequences()
    rendered_cases: dict[str, object] = {}
    for name, (plan, sequence) in cases.items():
        rendered_cases[name] = {
            **_summary(plan, sequence),
            "values_q": [list(row) for row in sequence.values_q],
        }
    return {
        "absolute_tolerance_q": 1,
        "cases": rendered_cases,
        "license": "MIT",
        "provenance": (
            "Project-authored synthetic landmarks; no person, camera, or human-derived coordinates."
        ),
        "quantization_scale": 1_000_000,
        "relative_tolerance": 0.0,
        "schema_version": "portable-landmark-goldens/1",
    }


def _golden_sequences() -> dict[str, tuple[LandmarkFeaturePlanV1, PortableFeatureSequenceV1]]:
    cases: dict[str, tuple[LandmarkFeaturePlanV1, PortableFeatureSequenceV1]] = {}
    base = make_feature_fixture()
    for representation in ("hand_local", "body_relative", "combined"):
        plan = make_feature_plan(representation, target_frame_count=4)
        cases[f"one_hand_{representation}"] = (plan, _derive(base, plan))

    combined = make_feature_plan("combined", target_frame_count=4)
    cases["two_hand_combined"] = (
        combined,
        _derive(make_feature_fixture(two_hands=True), combined),
    )
    cases["mirrored_combined"] = (
        combined,
        _derive(make_feature_fixture(mirrored=True), combined),
    )

    timestamps = (0, 33_333, 66_667)
    short_gap = make_feature_fixture(
        timestamps,
        hand_rows=(
            make_hand_row(timestamp_us=timestamps[0]),
            make_hand_row(timestamp_us=timestamps[1], first_present=False),
            make_hand_row(timestamp_us=timestamps[2]),
        ),
    )
    hand_local = make_feature_plan("hand_local", target_frame_count=4)
    cases["approved_short_gap"] = (hand_local, _derive(short_gap, hand_local))

    swap = make_feature_fixture(
        timestamps,
        two_hands=True,
        hand_rows=(
            make_hand_row(timestamp_us=timestamps[0], two_hands=True),
            make_hand_row(
                timestamp_us=timestamps[1],
                two_hands=True,
                first_present=False,
            ),
            make_hand_row(
                timestamp_us=timestamps[2],
                two_hands=True,
                first_anatomical_handedness="left",
                second_anatomical_handedness="right",
                first_center=(0.68, 0.58),
                second_center=(0.42, 0.58),
            ),
        ),
    )
    cases["swap_barrier"] = (hand_local, _derive(swap, hand_local))
    cases["absent_pose"] = (
        combined,
        _derive(make_feature_fixture(pose_present=False), combined),
    )

    kinematics = make_feature_plan(
        "combined",
        target_frame_count=4,
        include_velocity=True,
        include_acceleration=True,
        include_joint_angles=True,
        include_tip_distances=True,
    )
    irregular = make_feature_fixture(
        (0, 20_000, 70_000, 100_000),
        image_velocity_per_second=(0.2, 0.0),
    )
    cases["irregular_kinematics"] = (kinematics, _derive(irregular, kinematics))
    return cases


@pytest.mark.golden
def test_portable_landmark_golden_corpus_is_exact_and_synthetic() -> None:
    document = cast(dict[str, Any], json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))
    cases = _golden_sequences()

    assert document["schema_version"] == "portable-landmark-goldens/1"
    assert document["license"] == "MIT"
    assert document["quantization_scale"] == 1_000_000
    assert document["absolute_tolerance_q"] == 1
    assert document["relative_tolerance"] == 0.0
    assert "no person" in cast(str, document["provenance"]).casefold()
    expected_cases = cast(dict[str, dict[str, Any]], document["cases"])
    assert set(expected_cases) == set(cases)
    tolerance = cast(int, document["absolute_tolerance_q"])
    for name, (plan, sequence) in cases.items():
        expected = dict(expected_cases[name])
        expected_values = cast(list[list[int]], expected.pop("values_q"))
        assert expected == _summary(plan, sequence)
        assert len(expected_values) == len(sequence.values_q)
        assert all(
            abs(actual_value - expected_value) <= tolerance
            for actual_row, expected_row in zip(
                sequence.values_q,
                expected_values,
                strict=True,
            )
            for actual_value, expected_value in zip(actual_row, expected_row, strict=True)
        )

    baseline = cases["one_hand_combined"][1]
    mirrored = cases["mirrored_combined"][1]
    assert baseline.sequence_sha256 != mirrored.sequence_sha256
    assert baseline.valid_mask == mirrored.valid_mask
    assert all(
        abs(left - right) <= tolerance
        for left_row, right_row in zip(baseline.values_q, mirrored.values_q, strict=True)
        for left, right in zip(left_row, right_row, strict=True)
    )
