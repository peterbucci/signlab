"""Development-only native Keras versus ONNX candidate parity gate."""

from __future__ import annotations

import hashlib
import importlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from signlab.contracts.canonical import canonical_json_bytes
from signlab.experiments.baselines import (
    BaselineExperimentError,
    _load_inputs,
    _load_samples,
    _partition,
)
from signlab.experiments.calibration import (
    LABELS,
    CalibrationConfig,
    CalibrationError,
    _hand_local_matrix,
    _temperature_scale,
    _transition_fragments,
    load_calibration_config,
)
from signlab.model_bundle import (
    BrowserModelBundleManifestV1,
    DecisionPolicyV1,
    ModelBundleError,
    validate_browser_bundle,
)

_ATOL: Final = 1e-5
_RTOL: Final = 1e-5
_ALIASES: Final = tuple(
    [f"validation_target_{index:03d}" for index in range(1, 16)]
    + [f"validation_other_{index:03d}" for index in range(1, 4)]
)


class ModelParityError(ValueError):
    """A stable, path-free native/ONNX parity failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ModelParityResult:
    report_path: Path
    row_count: int
    maximum_absolute_difference: float


@dataclass(frozen=True, slots=True)
class _Runtime:
    keras: Any
    onnxruntime: Any
    tensorflow: Any


def _runtime() -> _Runtime:
    try:
        return _Runtime(
            importlib.import_module("keras"),
            importlib.import_module("onnxruntime"),
            importlib.import_module("tensorflow"),
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise ModelParityError("parity_runtime_unavailable") from error


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _development_matrix(
    config: CalibrationConfig, corpus_root: Path, external_manifest_path: Path
) -> tuple[NDArray[np.float32], NDArray[np.int64], tuple[str, ...]]:
    try:
        split, external = _load_inputs(cast(Any, config), corpus_root, external_manifest_path)
        loaded = _load_samples(split, external, corpus_root, ("validation",))
        validation = tuple(sorted(_partition(loaded, "validation"), key=lambda row: row.sample_id))
        target_x, target_y = _hand_local_matrix(validation, config)
        other_x, _signers, _identity = _transition_fragments(
            validation, target_x, "validation", config.validation_derivatives
        )
    except (BaselineExperimentError, CalibrationError, OSError, TypeError, ValueError) as error:
        raise ModelParityError("parity_development_inputs_invalid") from error
    matrix = np.ascontiguousarray(np.concatenate((target_x, other_x)), dtype=np.float32)
    labels = np.concatenate((target_y, np.full(len(other_x), 5, dtype=np.int64)))
    if (
        matrix.shape != (18, 64, 126)
        or labels.shape != (18,)
        or Counter(target_y.tolist()) != Counter({index: 3 for index in range(5)})
        or len(other_x) != 3
    ):
        raise ModelParityError("parity_development_inputs_invalid")
    return matrix, labels, ("target",) * 15 + ("constructed_other",) * 3


def _bundle_assets(
    bundle_root: Path, manifest: BrowserModelBundleManifestV1
) -> tuple[bytes, DecisionPolicyV1]:
    try:
        raw = {
            "model": (bundle_root / "model.onnx").read_bytes(),
            "decision_policy": (bundle_root / "decision-policy.json").read_bytes(),
        }
        expected = {asset.role: asset.sha256 for asset in manifest.assets}
        if any(_sha256(value) != expected[role] for role, value in raw.items()):
            raise ModelParityError("parity_bundle_identity_mismatch")
        policy = DecisionPolicyV1.model_validate_json(raw["decision_policy"], strict=True)
        return raw["model"], policy
    except ModelParityError:
        raise
    except (KeyError, OSError, TypeError, ValidationError, ValueError) as error:
        raise ModelParityError("parity_bundle_identity_mismatch") from error


def _require_identities(
    manifest: BrowserModelBundleManifestV1,
    config: CalibrationConfig,
    config_bytes: bytes,
    checkpoint: bytes,
) -> None:
    candidate = manifest.candidate
    if (
        _sha256(config_bytes) != candidate.configuration_sha256
        or _sha256(checkpoint) != candidate.research_checkpoint_sha256
        or config.corpus_sha256 != candidate.corpus_sha256
        or config.external_dataset_sha256 != candidate.external_dataset_sha256
        or config.split_sha256 != candidate.split_sha256
        or config.feature_plan_sha256 != candidate.source_feature_plan_sha256
        or config.input_feature_plan_sha256 != candidate.input_feature_plan_sha256
        or config.taxonomy_sha256 != candidate.taxonomy_sha256
        or tuple(config.labels) != manifest.labels
        or candidate.test_status != "sealed_not_loaded"
    ):
        raise ModelParityError("parity_input_identity_mismatch")


def _infer(
    matrix: NDArray[np.float32], checkpoint: bytes, model_bytes: bytes, runtime: _Runtime
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    try:
        with TemporaryDirectory(prefix="signlab-parity-") as temporary:
            snapshot = Path(temporary) / "candidate.keras"
            snapshot.write_bytes(checkpoint)
            with runtime.tensorflow.device("/CPU:0"):
                model = runtime.keras.models.load_model(snapshot)
                native = np.concatenate(
                    [np.asarray(model.predict(row[None], verbose=0)) for row in matrix]
                )
        session = runtime.onnxruntime.InferenceSession(
            model_bytes, providers=["CPUExecutionProvider"]
        )
        portable = np.concatenate(
            [np.asarray(session.run(["probabilities"], {"input": row[None]})[0]) for row in matrix]
        )
        return cast(NDArray[np.float32], native), cast(NDArray[np.float32], portable)
    except Exception as error:
        raise ModelParityError("parity_inference_failed") from error


def _checked_probabilities(values: NDArray[np.float32], runtime: str) -> NDArray[np.float64]:
    if values.shape != (18, 6) or values.dtype != np.dtype("float32"):
        raise ModelParityError(f"{runtime}_probabilities_invalid")
    for alias, row in zip(_ALIASES, values, strict=True):
        if not np.isfinite(row).all() or (row < 0).any() or not np.isclose(row.sum(), 1, atol=1e-5):
            raise ModelParityError(f"{alias}_{runtime}_probabilities_invalid")
    return values.astype(np.float64)


def _decisions(probabilities: NDArray[np.float64], policy: DecisionPolicyV1) -> tuple[str, ...]:
    temperature = cast(int, policy.temperature["temperature_milli"])
    threshold = cast(int, policy.abstention["threshold_percent"]) / 100
    calibrated = _temperature_scale(probabilities, temperature)
    return tuple(
        LABELS[int(row.argmax())] if float(row.max()) >= threshold else "abstain"
        for row in calibrated
    )


def run_native_onnx_parity(
    bundle_root: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    corpus_root: str | Path,
    external_manifest_path: str | Path,
    report_path: str | Path,
) -> ModelParityResult:
    """Validate exact-candidate runtime parity without opening test data."""

    report_target = Path(report_path)
    if report_target.exists() or report_target.is_symlink():
        raise ModelParityError("parity_report_conflict")
    try:
        bundle = Path(bundle_root).resolve(strict=True)
        manifest, bundle_sha256 = validate_browser_bundle(bundle)
        model_bytes, policy = _bundle_assets(bundle, manifest)
        config, config_bytes = load_calibration_config(config_path)
        checkpoint = Path(checkpoint_path).read_bytes()
    except ModelParityError:
        raise
    except (CalibrationError, ModelBundleError, OSError, TypeError, ValueError) as error:
        raise ModelParityError("parity_inputs_invalid") from error
    _require_identities(manifest, config, config_bytes, checkpoint)
    matrix, truth, kinds = _development_matrix(
        config, Path(corpus_root), Path(external_manifest_path)
    )
    native_raw, portable_raw = _infer(matrix, checkpoint, model_bytes, _runtime())
    native = _checked_probabilities(native_raw, "native")
    portable = _checked_probabilities(portable_raw, "onnx")
    native_decisions = _decisions(native, policy)
    portable_decisions = _decisions(portable, policy)
    rows: list[dict[str, object]] = []
    for index, alias in enumerate(_ALIASES):
        difference = float(np.max(np.abs(native[index] - portable[index])))
        if not np.allclose(native[index], portable[index], atol=_ATOL, rtol=_RTOL):
            raise ModelParityError(f"{alias}_probability_mismatch")
        native_class, portable_class = int(native[index].argmax()), int(portable[index].argmax())
        if native_class != portable_class:
            raise ModelParityError(f"{alias}_class_mismatch")
        if native_decisions[index] != portable_decisions[index]:
            raise ModelParityError(f"{alias}_decision_mismatch")
        rows.append(
            {
                "alias": alias,
                "fixture_kind": kinds[index],
                "truth": LABELS[int(truth[index])],
                "native_class": LABELS[native_class],
                "onnx_class": LABELS[portable_class],
                "native_decision": native_decisions[index],
                "onnx_decision": portable_decisions[index],
                "maximum_absolute_difference": difference,
            }
        )
    maximum = max(cast(float, row["maximum_absolute_difference"]) for row in rows)
    payload = {
        "format": "signlab-native-onnx-parity-report/1",
        "status": "pass",
        "evidence_scope": "development_runtime_equivalence_only",
        "test_status": "sealed_not_loaded",
        "identities": {
            "bundle_sha256": bundle_sha256,
            "configuration_sha256": manifest.candidate.configuration_sha256,
            "corpus_sha256": manifest.candidate.corpus_sha256,
            "decision_policy_sha256": manifest.components.decision_policy_sha256,
            "derivative_set_sha256": manifest.candidate.derivative_set_sha256,
            "feature_plan_sha256": manifest.candidate.input_feature_plan_sha256,
            "native_checkpoint_sha256": manifest.candidate.research_checkpoint_sha256,
            "onnx_model_sha256": _sha256(model_bytes),
            "split_sha256": manifest.candidate.split_sha256,
        },
        "comparison": {
            "absolute_tolerance": _ATOL,
            "relative_tolerance": _RTOL,
            "row_count": 18,
            "probability_element_count": 108,
            "maximum_absolute_difference": maximum,
            "class_mismatches": 0,
            "decision_mismatches": 0,
            "native_abstentions": native_decisions.count("abstain"),
            "onnx_abstentions": portable_decisions.count("abstain"),
        },
        "rows": rows,
        "limits": [
            "The current zero-percent threshold yields no abstentions on these valid rows.",
            "This report proves runtime equivalence, not model quality or release readiness.",
        ],
    }
    try:
        report_target.parent.mkdir(parents=True, exist_ok=True)
        with report_target.open("xb") as stream:
            stream.write(canonical_json_bytes(payload) + b"\n")
    except (OSError, TypeError, ValueError) as error:
        raise ModelParityError("parity_report_unwritable") from error
    return ModelParityResult(report_target, 18, maximum)
