"""Create one authorized-local exact-candidate ONNX Web parity receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256, parse_json_object
from signlab.model_bundle import (
    DecisionPolicyV1,
    validate_browser_bundle,
)
from signlab.model_parity import _decisions

type FloatArray = NDArray[np.float32]
type Case = tuple[str, FloatArray, str]

_ATOL = _RTOL = 1e-5
_RUNNER = Path(__file__).resolve().parents[1] / "apps/web/scripts/runCandidateOnnxWasm.mjs"

VerificationError = RuntimeError


def _sha(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _load(path: Path, *, canonical: bool = False) -> tuple[Any, bytes]:
    raw = path.resolve(strict=True).read_bytes()
    value = parse_json_object(raw)
    if canonical and raw != canonical_json_bytes(value) + b"\n":
        raise VerificationError("noncanonical_prior_report")
    return value, raw


def _fixture_cases(fixture: Any) -> list[Case]:
    if fixture.get("format") != "signlab-candidate-runtime-goldens/1":
        raise VerificationError("fixture_identity_mismatch")
    cases: list[Case] = []
    for item in fixture["preprocessingCases"]:
        expected, alias = item["expected"], item["id"]
        count, encoding = expected["nonPaddingFrameCount"], expected["rowEncoding"]
        if encoding == "all_zero_nonpadding_rows":
            rows = np.zeros((count, 126), dtype=np.float32)
        elif encoding == "explicit_nonpadding_rows":
            values = expected["valuesQ"]
            rows = cast(FloatArray, np.asarray(values, dtype=np.float32))
            rows /= np.float32(1_000_000)
        else:
            raise VerificationError("fixture_tensor_invalid")
        tensor = np.zeros((64, 126), dtype=np.float32)
        tensor[:count] = rows
        digest = _sha(tensor.astype("<f4", copy=False).tobytes(order="C"))
        if not isinstance(alias, str) or digest != expected["tensorSha256"]:
            raise VerificationError("fixture_tensor_identity_mismatch")
        cases.append((alias, tensor, digest))
    return cases


def _require_bindings(
    fixture: Any, prior: Any, prior_sha: str, manifest: Any, hashes: tuple[str, str]
) -> None:
    bundle_sha, model_sha = hashes
    resources, identities = fixture["resources"], prior["identities"]
    checks = (
        fixture["labels"] == list(manifest.labels),
        resources["featurePlan"]["semanticSha256"] == manifest.components.feature_plan_sha256,
        resources["decisionPolicy"]["semanticSha256"] == manifest.components.decision_policy_sha256,
        resources["segmenter"]["semanticSha256"] == manifest.components.segmenter_sha256,
        resources["nativeOnnxEvidence"]["fileSha256"] == prior_sha,
        prior["format"] == "signlab-native-onnx-parity-report/1",
        prior["status"] == "pass",
        identities["bundle_sha256"] == bundle_sha,
        identities["onnx_model_sha256"] == model_sha,
        identities["native_checkpoint_sha256"] == manifest.candidate.research_checkpoint_sha256,
        identities["feature_plan_sha256"] == manifest.components.feature_plan_sha256,
        identities["decision_policy_sha256"] == manifest.components.decision_policy_sha256,
    )
    if not all(checks):
        raise VerificationError("identity_mismatch")


def _checked(values: object, alias: str) -> FloatArray:
    result = np.asarray(values)
    if result.dtype != np.dtype("float32") or result.shape != (6,) or not np.isfinite(result).all():
        raise VerificationError(f"{alias}_probabilities_invalid")
    return cast(FloatArray, result)


def _python_infer(model: bytes, cases: list[Case]) -> list[FloatArray]:
    runtime = importlib.import_module("onnxruntime")
    session = runtime.InferenceSession(model, providers=["CPUExecutionProvider"])
    return [
        _checked(session.run(["probabilities"], {"input": tensor[None]})[0][0], alias)
        for alias, tensor, _digest in cases
    ]


def _web_infer(model: Path, cases: list[Case]) -> list[FloatArray]:
    request = {
        "modelPath": str(model),
        "inputName": "input",
        "outputName": "probabilities",
        "cases": [
            {"alias": alias, "values": tensor.reshape(-1).tolist()}
            for alias, tensor, _digest in cases
        ],
    }
    with tempfile.TemporaryDirectory(prefix="signlab-onnx-web-") as directory:
        source, target = Path(directory) / "request.json", Path(directory) / "output.json"
        source.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        run = subprocess.run(
            ("node", str(_RUNNER), str(source), str(target)),
            cwd=_RUNNER.parent.parent,
            check=False,
            capture_output=True,
            timeout=120,
        )
        if run.returncode:
            raise VerificationError("onnx_web_inference_failed")
        output = parse_json_object(target.read_bytes())
    rows = cast(Any, output["cases"])
    if (
        output.get("executionProvider") != "wasm"
        or output.get("wasmThreads") != 1
        or [row["alias"] for row in rows] != [case[0] for case in cases]
    ):
        raise VerificationError("onnx_web_contract_mismatch")
    return [
        _checked(np.asarray(row["probabilities"], dtype=np.float32), row["alias"]) for row in rows
    ]


def _report(manifest: Any, identities: Any, results: Any) -> dict[str, object]:
    identities |= {
        "feature_plan_sha256": manifest.components.feature_plan_sha256,
        "decision_policy_sha256": manifest.components.decision_policy_sha256,
        "segmenter_sha256": manifest.components.segmenter_sha256,
        "label_order_sha256": canonical_sha256(
            {"labels": list(manifest.labels)}, domain="signlab-candidate-label-order/1"
        ),
    }
    return {
        "format": "signlab-onnx-web-parity-report/1",
        "status": "pass",
        "identities": identities,
        "tolerances": {"absolute": _ATOL, "relative": _RTOL},
        "results": {
            "case_count": len(results),
            "probability_element_count": len(results) * 6,
            "maximum_absolute_difference": max(
                row["maximum_absolute_difference"] for row in results
            ),
            "python_provider": "CPUExecutionProvider",
            "onnx_web_provider": "wasm",
            "onnx_web_wasm_threads": 1,
            "cases": results,
        },
        "limitations": [
            "The local-evaluation model was read; no model bytes or paths are published.",
            "#37 remains the native-to-ONNX authority; no private inputs were reopened.",
            "Direct WASM only; no browser-worker integration or model-quality claim.",
        ],
    }


def verify(bundle: Path, fixture_path: Path, prior_path: Path, report_path: Path) -> None:
    try:
        root = bundle.resolve(strict=True)
        manifest, bundle_sha = validate_browser_bundle(root)
        model_path = root / "model.onnx"
        model, (fixture, fixture_raw), (prior, prior_raw) = (
            model_path.read_bytes(),
            _load(fixture_path),
            _load(prior_path, canonical=True),
        )
        model_sha = _sha(model)
        if model_sha != {asset.role: asset.sha256 for asset in manifest.assets}["model"]:
            raise VerificationError("bundle_model_identity_mismatch")
        policy = DecisionPolicyV1.model_validate_json(
            (root / "decision-policy.json").read_bytes(), strict=True
        )
        cases = _fixture_cases(fixture)
        _require_bindings(fixture, prior, _sha(prior_raw), manifest, (bundle_sha, model_sha))
        results: list[dict[str, Any]] = []
        for case, python, web in zip(
            cases, _python_infer(model, cases), _web_infer(model_path, cases), strict=True
        ):
            alias, _tensor, tensor_sha = case
            classes = (int(python.argmax()), int(web.argmax()))
            decisions = (
                _decisions(python[None].astype(np.float64), policy)[0],
                _decisions(web[None].astype(np.float64), policy)[0],
            )
            if not np.allclose(python, web, atol=_ATOL, rtol=_RTOL):
                raise VerificationError(f"{alias}_probability_mismatch")
            if classes[0] != classes[1] or decisions[0] != decisions[1]:
                raise VerificationError(f"{alias}_decision_mismatch")
            results.append(
                {
                    "alias": alias,
                    "input_tensor_sha256": tensor_sha,
                    "maximum_absolute_difference": float(
                        np.max(np.abs(python.astype(np.float64) - web.astype(np.float64)))
                    ),
                    "matched_argmax": manifest.labels[classes[0]],
                    "matched_decision": decisions[0],
                }
            )
        identities = {
            "bundle_sha256": bundle_sha,
            "model_sha256": model_sha,
            "fixture_sha256": _sha(fixture_raw),
            "prior_native_onnx_report_sha256": _sha(prior_raw),
        }
        if report_path.exists() or report_path.is_symlink():
            raise VerificationError("report_conflict")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as stream:
            stream.write(canonical_json_bytes(_report(manifest, identities, results)) + b"\n")
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError("verification_failed") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("bundle-root", "fixture", "prior-report", "output-report"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    values = parser.parse_args()
    try:
        verify(values.bundle_root, values.fixture, values.prior_report, values.output_report)
    except VerificationError as error:
        print(f"Candidate ONNX Web parity failed: {error}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
