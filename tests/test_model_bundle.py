import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256, parse_json_object
from signlab.contracts.extraction import validate_mediapipe_extraction_config
from signlab.model_bundle import (
    ModelBundleError,
    browser_bundle_json_schema,
    load_browser_bundle_manifest,
    validate_browser_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = ROOT / "src/signlab/resources/model_bundles"
EXAMPLE = RESOURCE_ROOT / "examples/browser-model-bundle-manifest.example.json"
SCHEMA = RESOURCE_ROOT / "schemas/browser-model-bundle-manifest-1.schema.json"
MODEL_BYTES = b"not-an-onnx-graph; structural fixture only\n"
GOLDEN_BYTES = b'{"format":"synthetic-golden-smoke/1","status":"structure_only"}\n'
EXPECTED_DIGEST = "sha256:27e6c5ef2fce3fed25933f5c0557f78b8d753eb647ceec9772169d8d4087c2ad"
SOURCES = {
    "decision-policy.json": "docs/reports/popsign-constructed-calibration-policy-v1.json",
    "feature-plan.json": "src/signlab/resources/features/config/hand-local-64-1.default.json",
    "landmarker.json": (
        "src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json"
    ),
    "model-card.md": "docs/cards/popsign-tcn-portable-export-candidate-v1.md",
    "quality-policy.json": (
        "src/signlab/resources/quality/config/landmark-quality-policy-1.default.json"
    ),
    "segmenter.json": "configs/evaluation/candidate-event-detector-v1.json",
}


def _payload(path: Path = EXAMPLE) -> dict[str, Any]:
    return cast(dict[str, Any], parse_json_object(path.read_bytes()))


def _refresh_asset(payload: dict[str, Any], role: str, raw: bytes) -> None:
    asset = next(row for row in payload["assets"] if row["role"] == role)
    asset["sha256"] = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    asset["size_bytes"] = len(raw)


def _expect_error(code: str, action: Callable[[], object], root: Path | None = None) -> None:
    with pytest.raises(ModelBundleError) as caught:
        action()
    assert caught.value.code == code
    assert str(caught.value) == code
    if root is not None:
        assert str(root) not in str(caught.value)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    for target, source in SOURCES.items():
        shutil.copyfile(ROOT / source, tmp_path / target)
    (tmp_path / "golden").mkdir()
    (tmp_path / "golden/smoke.json").write_bytes(GOLDEN_BYTES)
    (tmp_path / "model.onnx").write_bytes(MODEL_BYTES)
    shutil.copyfile(EXAMPLE, tmp_path / "manifest.json")
    return tmp_path


def test_schema_example_and_generated_contract_are_in_sync() -> None:
    schema = _payload(SCHEMA)
    payload = _payload()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert schema == browser_bundle_json_schema()
    assert load_browser_bundle_manifest(EXAMPLE.read_bytes()).bundle_id == payload["bundle_id"]


def test_complete_structure_only_bundle_validates(bundle: Path) -> None:
    _, digest = validate_browser_bundle(bundle)
    assert digest == EXPECTED_DIGEST
    landmarker = validate_mediapipe_extraction_config((bundle / "landmarker.json").read_bytes())
    assert (landmarker.hand_task_asset.sha256, landmarker.hand_task_asset.size_bytes) == (
        "sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        7_819_105,
    )
    assert (landmarker.pose_task_asset.sha256, landmarker.pose_task_asset.size_bytes) == (
        "sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
        5_777_746,
    )


@pytest.mark.parametrize(
    "case",
    [
        "unsupported",
        "noncanonical",
        "labels",
        "opset",
        "input",
        "output",
        "extra",
        "duplicate",
        "unsafe",
        "license",
    ],
)
def test_manifest_mutations_fail_closed(case: str) -> None:
    payload = _payload()
    if case == "unsupported":
        payload["format"] = "browser-model-bundle/2"
    elif case == "labels":
        payload["labels"] = list(reversed(payload["labels"]))
    elif case == "opset":
        payload["onnx"]["opset"] = 17
    elif case == "input":
        payload["onnx"]["input_shape"] = [1, 64, 127]
    elif case == "output":
        payload["onnx"]["output_semantics"] = "logits"
    elif case == "extra":
        payload["unexpected"] = True
    elif case == "duplicate":
        payload["assets"][1] = payload["assets"][0]
    elif case == "unsafe":
        payload["assets"][4]["locator"]["path"] = "../model.onnx"
    elif case == "license":
        payload["licenses"][3]["spdx"] = "MIT"
    raw = canonical_json_bytes(payload) + b"\n"
    if case == "noncanonical":
        raw = b" " + raw
    code = "unsupported_manifest_version" if case == "unsupported" else "invalid_manifest"
    if case == "noncanonical":
        code = "noncanonical_manifest"
    _expect_error(code, lambda: load_browser_bundle_manifest(raw))


@pytest.mark.parametrize("case", ["unavailable", "missing", "extra", "hash", "size", "symlink"])
def test_bundle_file_failures_are_stable_and_path_safe(bundle: Path, case: str) -> None:
    target = bundle / "model.onnx"
    root = bundle
    if case == "unavailable":
        root = bundle / "absent"
    elif case == "missing":
        target.unlink()
    elif case == "extra":
        (bundle / "extra.txt").write_text("extra", encoding="utf-8")
    elif case == "hash":
        target.write_bytes(b"x" * len(MODEL_BYTES))
    elif case == "size":
        target.write_bytes(MODEL_BYTES + b"x")
    elif case == "symlink":
        target.unlink()
        try:
            target.symlink_to(bundle / "model-card.md")
        except OSError:
            pytest.skip("symlinks are unavailable")
    code = (
        "bundle_unavailable"
        if case == "unavailable"
        else "bundle_inventory_mismatch"
        if case in {"missing", "extra"}
        else "bundle_asset_bytes_invalid"
    )
    _expect_error(code, lambda: validate_browser_bundle(root), bundle)


@pytest.mark.parametrize(
    "case",
    [
        "component",
        "checkpoint",
        "derivative",
        "policy",
        "inclusive_type",
        "threshold_type",
        "golden",
    ],
)
def test_cross_component_and_canonical_mutations_fail(bundle: Path, case: str) -> None:
    payload = _payload(bundle / "manifest.json")
    code = "bundle_component_identity_mismatch"
    if case == "component":
        payload["components"]["feature_plan_sha256"] = "sha256:" + "0" * 64
    elif case == "checkpoint":
        payload["candidate"]["research_checkpoint_sha256"] = "sha256:" + "1" * 64
    elif case in {"derivative", "policy", "inclusive_type", "threshold_type"}:
        policy_path = bundle / "decision-policy.json"
        policy = _payload(policy_path)
        if case == "derivative":
            policy["identities"]["derivative_set_sha256"] = "sha256:" + "2" * 64
        elif case == "policy":
            policy["temperature"]["temperature_milli"] = 51
        elif case == "inclusive_type":
            policy["abstention"]["inclusive"] = 1
        else:
            policy["abstention"]["threshold_percent"] = False
        raw = canonical_json_bytes(policy) + b"\n"
        policy_path.write_bytes(raw)
        _refresh_asset(payload, "decision_policy", raw)
        if case == "derivative":
            payload["components"]["decision_policy_sha256"] = canonical_sha256(
                policy, domain=policy["format"]
            )
        else:
            code = "invalid_decision_policy"
    else:
        role = "golden_smoke"
        path = bundle / "golden/smoke.json"
        raw = b" " + path.read_bytes()
        path.write_bytes(raw)
        _refresh_asset(payload, role, raw)
        code = f"noncanonical_{role}"
    (bundle / "manifest.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    _expect_error(code, lambda: validate_browser_bundle(bundle), bundle)
