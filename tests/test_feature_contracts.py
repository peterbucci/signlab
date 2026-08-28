from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from feature_fixtures import EXTRACTION_CONFIG_SHA256, make_feature_fixture, make_feature_plan
from signlab.contracts.features import (
    FeatureContractError,
    landmark_feature_plan_digest,
    portable_feature_sequence_digest,
    validate_landmark_feature_plan,
    validate_portable_feature_sequence,
)
from signlab.features.transforms import derive_feature_sequence


def _json_payload(value: BaseModel) -> dict[str, object]:
    captured = json.loads(value.model_dump_json())
    assert isinstance(captured, dict)
    return captured


def test_registered_plan_freezes_complete_feature_order() -> None:
    plan = make_feature_plan(
        "combined",
        include_velocity=True,
        include_acceleration=True,
        include_joint_angles=True,
        include_tip_distances=True,
    )

    assert len(plan.feature_order) == 422
    assert plan.feature_order[0] == "hand_0.local.landmark_00.x"
    assert plan.feature_order[62] == "hand_0.local.landmark_20.z"
    assert plan.feature_order[63] == "hand_1.local.landmark_00.x"
    assert plan.feature_order[125] == "hand_1.local.landmark_20.z"
    assert plan.feature_order[126] == "hand_0.body.wrist.x"
    assert plan.feature_order[133] == "hand_1.body.palm.y"
    assert plan.feature_order[134] == "hand_0.geometry.thumb.angle"
    assert plan.feature_order[144] == "hand_0.geometry.thumb.tip_distance"
    assert plan.feature_order[154] == "hand_0.local.landmark_00.x.velocity"
    assert plan.feature_order[288] == "hand_0.local.landmark_00.x.acceleration"
    assert plan.feature_order[-1] == "hand_1.body.palm.y.acceleration"


def test_plan_validation_is_strict_and_digest_sensitive() -> None:
    plan = make_feature_plan("hand_local", target_frame_count=4)
    captured = _json_payload(plan)

    assert validate_landmark_feature_plan(captured) == plan
    baseline_digest = landmark_feature_plan_digest(plan)
    assert landmark_feature_plan_digest(captured) == baseline_digest
    assert baseline_digest != landmark_feature_plan_digest(
        make_feature_plan("hand_local", target_frame_count=5)
    )
    assert baseline_digest != landmark_feature_plan_digest(
        make_feature_plan("hand_local", target_frame_count=4, include_tip_distances=True)
    )

    reversed_order = _json_payload(plan)
    feature_order = reversed_order["feature_order"]
    assert isinstance(feature_order, list)
    feature_order[0], feature_order[1] = feature_order[1], feature_order[0]
    with pytest.raises(FeatureContractError, match="invalid landmark feature plan"):
        validate_landmark_feature_plan(reversed_order)
    with pytest.raises(FeatureContractError, match="invalid landmark feature plan"):
        landmark_feature_plan_digest(reversed_order)

    tampered_model = plan.model_copy(update={"feature_order": tuple(feature_order)})
    with pytest.raises(FeatureContractError, match="invalid landmark feature plan"):
        landmark_feature_plan_digest(tampered_model)

    unexpected = _json_payload(plan)
    unexpected["unregistered_rule"] = "silently_accept_nothing"
    with pytest.raises(FeatureContractError, match="invalid landmark feature plan"):
        validate_landmark_feature_plan(unexpected)


def test_plan_rejects_runtime_reordering_and_acceleration_without_velocity() -> None:
    plan = make_feature_plan("body_relative")
    runtime_reordered = _json_payload(plan)
    runtime_reordered["compatible_runtimes"] = ["typescript", "python"]
    with pytest.raises(FeatureContractError, match="invalid landmark feature plan"):
        validate_landmark_feature_plan(runtime_reordered)

    invalid_optional = _json_payload(plan)
    optional = invalid_optional["optional"]
    assert isinstance(optional, dict)
    optional["include_acceleration"] = True
    optional["include_velocity"] = False
    with pytest.raises(FeatureContractError, match="invalid landmark feature plan"):
        validate_landmark_feature_plan(invalid_optional)


@pytest.mark.parametrize(
    ("mask_name", "expected_error"),
    [
        ("hand_present_mask", "sampled hand observation"),
        ("body_available_mask", "body anchors"),
    ],
)
def test_sequence_rejects_availability_masks_that_contradict_valid_features(
    mask_name: str,
    expected_error: str,
) -> None:
    fixture = make_feature_fixture(two_hands=True)
    sequence = derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        make_feature_plan("combined", target_frame_count=3),
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )
    payload = _json_payload(sequence)
    if mask_name == "hand_present_mask":
        hand_mask = payload[mask_name]
        assert isinstance(hand_mask, list)
        assert isinstance(hand_mask[0], list)
        hand_mask[0][0] = False
    else:
        body_mask = payload[mask_name]
        assert isinstance(body_mask, list)
        body_mask[0] = False
    payload.pop("sequence_sha256")
    payload["sequence_sha256"] = portable_feature_sequence_digest(payload)

    with pytest.raises(FeatureContractError, match="invalid portable feature sequence") as captured:
        validate_portable_feature_sequence(payload)
    assert expected_error in str(captured.value.__cause__)


def test_sequence_rejects_missing_or_impossible_feature_evidence() -> None:
    fixture = make_feature_fixture()
    base = derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        make_feature_plan("combined", target_frame_count=3),
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )
    missing_evidence = _json_payload(base)
    observed = missing_evidence["observed_mask"]
    assert isinstance(observed, list)
    assert isinstance(observed[0], list)
    observed[0][0] = False
    missing_evidence.pop("sequence_sha256")
    missing_evidence["sequence_sha256"] = portable_feature_sequence_digest(missing_evidence)
    with pytest.raises(FeatureContractError) as captured:
        validate_portable_feature_sequence(missing_evidence)
    assert "non-derivative features require" in str(captured.value.__cause__)

    velocity_plan = make_feature_plan(
        "body_relative",
        target_frame_count=3,
        include_velocity=True,
    )
    velocity = derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        velocity_plan,
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )
    impossible_evidence = _json_payload(velocity)
    velocity_index = velocity.feature_names.index("hand_0.body.wrist.x.velocity")
    velocity_observed = impossible_evidence["observed_mask"]
    assert isinstance(velocity_observed, list)
    assert isinstance(velocity_observed[1], list)
    assert velocity.valid_mask[1][velocity_index]
    velocity_observed[1][velocity_index] = True
    impossible_evidence.pop("sequence_sha256")
    impossible_evidence["sequence_sha256"] = portable_feature_sequence_digest(impossible_evidence)
    with pytest.raises(FeatureContractError) as captured:
        validate_portable_feature_sequence(impossible_evidence)
    assert "derivative features cannot claim" in str(captured.value.__cause__)
