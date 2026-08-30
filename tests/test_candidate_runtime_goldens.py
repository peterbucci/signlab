"""Shared synthetic goldens for the candidate Python, TypeScript, and ONNX path."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
import pytest
from numpy.typing import NDArray
from onnx import TensorProto, helper

from feature_fixtures import (
    EXTRACTION_CONFIG_SHA256,
    FeatureFixture,
    make_feature_fixture,
    make_hand_row,
)
from signlab.candidate_events import candidate_event_config_digest, load_candidate_event_config
from signlab.contracts.features import LandmarkFeaturePlanV1, landmark_feature_plan_digest
from signlab.contracts.quality import (
    landmark_quality_policy_digest,
    validate_landmark_quality_policy,
)
from signlab.experiments.calibration import LABELS, _temperature_scale
from signlab.features.transforms import derive_feature_sequence
from signlab.model_bundle import DecisionPolicyV1, decision_policy_digest
from signlab.model_parity import _decisions
from signlab.quality.policy import elapsed_resampling_timestamps

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/public/parity/candidate-runtime-goldens-v1.json"
MODEL = ROOT / "tests/fixtures/public/parity/candidate-runtime-v1.onnx"
PLAN = ROOT / "src/signlab/resources/features/config/hand-local-64-1.default.json"
QUALITY_POLICY = (
    ROOT / "src/signlab/resources/quality/config/landmark-quality-policy-1.default.json"
)
DECISION_POLICY = ROOT / "docs/reports/popsign-constructed-calibration-policy-v1.json"
NATIVE_ONNX_REPORT = ROOT / "docs/reports/popsign-tcn-native-onnx-parity-v1.json"
SEGMENTER = ROOT / "configs/evaluation/candidate-event-detector-v1.json"


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _plan() -> LandmarkFeaturePlanV1:
    return LandmarkFeaturePlanV1.model_validate_json(PLAN.read_bytes(), strict=True)


def _cases() -> dict[str, FeatureFixture]:
    short = (0, 33_333, 66_667)

    def gap_transition(mirrored: bool) -> FeatureFixture:
        rows = (
            make_hand_row(timestamp_us=short[0], two_hands=True, mirrored=mirrored),
            make_hand_row(timestamp_us=short[1], first_present=False, mirrored=mirrored),
            make_hand_row(
                timestamp_us=short[2],
                two_hands=True,
                first_anatomical_handedness="left",
                mirrored=mirrored,
            ),
        )
        return make_feature_fixture(
            short,
            two_hands=True,
            pose_present=False,
            mirrored=mirrored,
            hand_rows=rows,
        )

    long_timestamps = elapsed_resampling_timestamps(
        2_133_333, rate_numerator=30, rate_denominator=1
    )
    return {
        "gap_unmirrored": gap_transition(False),
        "gap_mirrored": gap_transition(True),
        "long_endpoint_selection": make_feature_fixture(
            long_timestamps,
            pose_present=False,
            hand_rows=tuple(
                make_hand_row(
                    timestamp_us=timestamp,
                    first_present=False,
                )
                for timestamp in long_timestamps
            ),
        ),
    }


def _camel(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key.split("_")[0] + "".join(part.title() for part in key.split("_")[1:]): _camel(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_camel(item) for item in value]
    return value


def _frames(fixture: FeatureFixture) -> list[dict[str, object]]:
    return [
        cast(
            dict[str, object],
            _camel(
                {
                    "relative_timestamp_us": frame.relative_timestamp_us,
                    "valid": not frame.invalid,
                    "hands": [hand.model_dump(mode="json") for hand in frame.hands],
                    "body_anchors": [
                        anchor.model_dump(mode="json") for anchor in frame.body_anchors
                    ],
                }
            ),
        )
        for frame in fixture.table.rows
    ]


def _quality(fixture: FeatureFixture) -> dict[str, object]:
    approved = [gap for gap in fixture.quality.gaps if gap.decision == "interpolate_linear"]
    return {
        "timestampDiscontinuityCount": fixture.quality.metrics.timestamp_discontinuity_count,
        "gaps": [
            {
                "signal": gap.signal,
                "decision": gap.decision,
                "leftObservedFrameIndex": gap.left_observed_frame_index,
                "leftObservedTimestampUs": gap.left_observed_timestamp_us,
                "rightObservedFrameIndex": gap.right_observed_frame_index,
                "rightObservedTimestampUs": gap.right_observed_timestamp_us,
            }
            for gap in approved
            if gap.signal in {"hand_0", "hand_1", "left_shoulder", "right_shoulder"}
        ],
    }


def _quality_assertions(fixture: FeatureFixture) -> dict[str, object]:
    return {
        gap.decision: [
            {"signal": item.signal, "reasons": list(item.reasons)}
            for item in fixture.quality.gaps
            if item.decision == gap.decision
        ]
        for gap in fixture.quality.gaps
    }


def _expected(fixture: FeatureFixture, plan: LandmarkFeaturePlanV1) -> dict[str, object]:
    sequence = derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        plan,
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )
    count = len(sequence.selected_source_indices)
    tensor = np.asarray(sequence.values_q, dtype="<f4") / np.float32(1_000_000)
    expected: dict[str, object] = {
        "shape": [1, 64, 126],
        "sourceGridFrameCount": sequence.source_grid_frame_count,
        "selectedSourceIndices": list(sequence.selected_source_indices),
        "nonPaddingFrameCount": count,
        "timestampsUs": list(sequence.timestamps_us),
        "padding": {
            "side": "right",
            "frameCount": 64 - count,
            "valueQ": 0,
            "allMasksFalse": True,
            "timestampRule": "continue_nominal_grid",
        },
        "tensorSha256": _sha256(tensor.tobytes(order="C")),
        "portableSequenceSha256": sequence.sequence_sha256,
    }
    if any(any(row) for row in sequence.valid_mask[:count]):
        expected["rowEncoding"] = "explicit_nonpadding_rows"
        for name, rows in (
            ("valuesQ", sequence.values_q),
            ("validMask", sequence.valid_mask),
            ("observedMask", sequence.observed_mask),
            ("interpolatedMask", sequence.interpolated_mask),
            ("handPresentMask", sequence.hand_present_mask),
        ):
            expected[name] = [list(row) for row in rows[:count]]
        expected["bodyAvailableMask"] = list(sequence.body_available_mask[:count])
    else:
        expected["rowEncoding"] = "all_zero_nonpadding_rows"
    return expected


def _decision(
    active: bool, probabilities: list[float], policy: DecisionPolicyV1
) -> dict[str, object]:
    if not active:
        return {"kind": "inactive"}
    if (
        len(probabilities) != len(LABELS)
        or any(not math.isfinite(item) or item < 0 for item in probabilities)
        or not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-5, rel_tol=1e-5)
    ):
        return {"kind": "abstain"}
    matrix = np.asarray([probabilities], dtype=np.float64)
    label = _decisions(matrix, policy)[0]
    if label == "abstain":
        return {"kind": "abstain"}
    calibrated = _temperature_scale(matrix, cast(int, policy.temperature["temperature_milli"]))[0]
    selected = LABELS.index(label)
    kind = "other" if label == "other" else "target"
    return {
        "kind": kind,
        "label": label,
        "confidence": calibrated[selected],
        "calibratedProbabilities": calibrated.tolist(),
    }


def _decision_cases(policy: DecisionPolicyV1) -> list[dict[str, object]]:
    inputs = (
        ("target_temperature", True, [0.40, 0.30, 0.10, 0.10, 0.05, 0.05]),
        ("learned_other", True, [0.02, 0.02, 0.02, 0.02, 0.02, 0.90]),
        ("background_inactive", False, [0.02, 0.02, 0.02, 0.02, 0.02, 0.90]),
        ("malformed_sum", True, [0.20, 0.20, 0.20, 0.20, 0.20, 0.20]),
        ("tie_first_class_inclusive_zero", True, [0.50, 0.50, 0.0, 0.0, 0.0, 0.0]),
    )
    return [
        dict(
            id=alias,
            candidateActive=active,
            probabilities=probabilities,
            expected=_decision(active, probabilities, policy),
        )
        for alias, active, probabilities in inputs
    ]


def _resource(path: Path, raw: bytes | None = None, **facts: object) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "fileSha256": _sha256(path.read_bytes() if raw is None else raw),
        **facts,
    }


def _model_parameters() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    rows = np.arange(126, dtype=np.int32)[:, None]
    columns = np.arange(6, dtype=np.int32)[None, :]
    weights = (((rows + 1) * (columns + 3) % 17) - 8).astype(np.float32) / np.float32(500)
    bias = np.asarray([0.03, -0.02, 0.01, -0.01, 0.02, -0.03], dtype=np.float32)
    return weights, bias


def onnx_model_bytes() -> bytes:
    """Build the deterministic MIT test-only runtime probe."""

    weights, bias = _model_parameters()
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input", "mean_axes"], ["pooled"], keepdims=0),
            helper.make_node("Gemm", ["pooled", "weights", "bias"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "signlab_candidate_runtime_test_probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64, 126])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 6])],
        [
            helper.make_tensor("mean_axes", TensorProto.INT64, [1], [1]),
            helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.ravel()),
            helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="signlab-test-fixture",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.doc_string = "MIT test-only deterministic probe; not a trained candidate model."
    onnx.checker.check_model(model)
    return cast(bytes, model.SerializeToString(deterministic=True))


def _numpy_inference(tensor: NDArray[np.float32]) -> NDArray[np.float32]:
    weights, bias = _model_parameters()
    pooled = tensor.mean(axis=1)
    logits = pooled @ weights + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return cast(NDArray[np.float32], exponentials / exponentials.sum(axis=1, keepdims=True))


def _tensor(expected: dict[str, Any]) -> NDArray[np.float32]:
    count = cast(int, expected["nonPaddingFrameCount"])
    values = (
        [[0] * 126 for _ in range(count)]
        if expected["rowEncoding"] == "all_zero_nonpadding_rows"
        else cast(list[list[int]], expected["valuesQ"])
    )
    padded = values + [[0] * 126 for _ in range(64 - count)]
    return np.asarray(padded, dtype=np.float32)[None] / np.float32(1_000_000)


def golden_document() -> dict[str, object]:
    """Regenerate the complete cross-runtime fixture from production Python."""

    plan = _plan()
    policy = DecisionPolicyV1.model_validate_json(DECISION_POLICY.read_bytes(), strict=True)
    segmenter = load_candidate_event_config(SEGMENTER)
    report = cast(dict[str, Any], json.loads(NATIVE_ONNX_REPORT.read_text(encoding="utf-8")))
    cases = [
        {
            "id": alias,
            "sourceMirrorState": fixture.recording.mirror_state,
            "frames": _frames(fixture),
            "quality": _quality(fixture),
            "qualityAssertions": _quality_assertions(fixture),
            "expected": _expected(fixture, plan),
        }
        for alias, fixture in _cases().items()
    ]
    model_bytes = onnx_model_bytes()
    return {
        "format": "signlab-candidate-runtime-goldens/1",
        "metadata": {
            "license": "MIT",
            "provenance": (
                "Project-authored synthetic landmarks; no person, camera, public corpus, "
                "or human-derived coordinates."
            ),
            "digestEncoding": {
                "resources": "sha256 of exact file bytes, lowercase hex with sha256: prefix",
                "tensor": "little-endian row-major float32 after valuesQ / 1000000",
            },
            "limitations": [
                "The tiny ONNX model is a deterministic runtime probe, not the trained candidate.",
                "These synthetic cases prove contract parity, not recognition quality or safety.",
                "The native/ONNX report is bound as prior candidate evidence; this fixture does "
                "not redistribute the locally licensed candidate model.",
            ],
        },
        "resources": {
            "featurePlan": _resource(PLAN, semanticSha256=landmark_feature_plan_digest(plan)),
            "qualityPolicy": _resource(
                QUALITY_POLICY,
                semanticSha256=landmark_quality_policy_digest(
                    validate_landmark_quality_policy(QUALITY_POLICY.read_bytes())
                ),
            ),
            "decisionPolicy": _resource(
                DECISION_POLICY,
                semanticSha256=decision_policy_digest(policy),
                temperatureMilli=policy.temperature["temperature_milli"],
                thresholdPercent=policy.abstention["threshold_percent"],
                inclusive=policy.abstention["inclusive"],
            ),
            "nativeOnnxEvidence": _resource(
                NATIVE_ONNX_REPORT,
                status=report["status"],
                comparison=report["comparison"],
                identities=report["identities"],
            ),
            "segmenter": _resource(
                SEGMENTER,
                semanticSha256=candidate_event_config_digest(segmenter),
                scope="identity_only_behavior_tested_by_candidate_event_story",
            ),
            "testModel": _resource(
                MODEL, model_bytes, license="MIT", scope="test_only_runtime_probe"
            ),
            "syntheticExtractionConfigSha256": EXTRACTION_CONFIG_SHA256,
        },
        "labels": list(LABELS),
        "preprocessingCases": cases,
        "decisionCases": _decision_cases(policy),
        "onnx": {
            "input": {"name": "input", "shape": [1, 64, 126], "dtype": "float32"},
            "output": {
                "name": "probabilities",
                "shape": [1, 6],
                "dtype": "float32",
            },
            "tolerances": {"absolute": 1e-5, "relative": 1e-5},
            "cases": [
                {
                    "preprocessingCaseId": case["id"],
                    "probabilities": _numpy_inference(
                        _tensor(cast(dict[str, Any], case["expected"]))
                    )[0].tolist(),
                }
                for case in cases
            ],
        },
    }


@pytest.mark.golden
def test_candidate_runtime_golden_is_exact_and_bound() -> None:
    document = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert document == golden_document()
    by_id = {case["id"]: case for case in document["preprocessingCases"]}
    assert set(by_id) == {"gap_unmirrored", "gap_mirrored", "long_endpoint_selection"}
    assert (
        by_id["gap_unmirrored"]["expected"]["valuesQ"]
        == by_id["gap_mirrored"]["expected"]["valuesQ"]
    )
    gap = by_id["gap_mirrored"]
    assert gap["quality"]["gaps"][0]["signal"] == "hand_1"
    assert any(item["signal"] == "hand_0" for item in gap["qualityAssertions"]["preserve_missing"])
    assert not any(gap["expected"]["bodyAvailableMask"])
    long = by_id["long_endpoint_selection"]["expected"]
    assert long["nonPaddingFrameCount"] == 64
    assert long["selectedSourceIndices"][0] == 0
    assert long["selectedSourceIndices"][-1] == long["sourceGridFrameCount"] - 1
    assert long["rowEncoding"] == "all_zero_nonpadding_rows"
    resources = document["resources"]
    evidence = resources["nativeOnnxEvidence"]
    assert evidence["status"] == "pass"
    assert evidence["comparison"]["class_mismatches"] == 0
    assert evidence["comparison"]["decision_mismatches"] == 0
    assert (
        evidence["identities"]["feature_plan_sha256"] == resources["featurePlan"]["semanticSha256"]
    )
    assert (
        evidence["identities"]["decision_policy_sha256"]
        == resources["decisionPolicy"]["semanticSha256"]
    )
    policy = DecisionPolicyV1.model_validate_json(DECISION_POLICY.read_bytes(), strict=True)
    assert document["labels"] == [policy.class_map[str(index)] for index in range(6)]
    assert resources["segmenter"]["semanticSha256"] == candidate_event_config_digest(
        load_candidate_event_config(SEGMENTER)
    )


@pytest.mark.golden
def test_test_only_onnx_matches_numpy_reference() -> None:
    assert MODEL.read_bytes() == onnx_model_bytes()
    document = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    session = onnxruntime.InferenceSession(MODEL.read_bytes(), providers=["CPUExecutionProvider"])
    expected_by_id = {
        case["preprocessingCaseId"]: np.asarray(case["probabilities"], dtype=np.float32)
        for case in document["onnx"]["cases"]
    }
    for case in document["preprocessingCases"]:
        tensor = _tensor(case["expected"])
        numpy_probabilities = _numpy_inference(tensor)[0]
        runtime_probabilities = cast(
            NDArray[np.float32], session.run(["probabilities"], {"input": tensor})[0][0]
        )
        assert np.allclose(runtime_probabilities, numpy_probabilities, atol=1e-5, rtol=1e-5), case[
            "id"
        ]
        assert np.allclose(runtime_probabilities, expected_by_id[case["id"]], atol=1e-5, rtol=1e-5)
