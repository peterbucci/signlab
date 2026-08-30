"""Deterministic local export of the single nominated browser candidate."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, cast

from pydantic import ValidationError

from signlab.candidate_events import CandidateEventConfigV1, candidate_event_config_digest
from signlab.candidate_nomination import (
    CandidateNominationError,
    DevelopmentCandidateDossier,
    build_candidate_nomination_report,
    load_candidate_dossier,
)
from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256
from signlab.contracts.extraction import mediapipe_extraction_config_digest
from signlab.contracts.features import landmark_feature_plan_digest, validate_landmark_feature_plan
from signlab.contracts.quality import landmark_quality_policy_digest
from signlab.model_bundle import (
    BrowserModelBundleManifestV1,
    DecisionPolicyV1,
    ModelBundleError,
    browser_bundle_digest,
    decision_policy_digest,
    load_browser_bundle_manifest,
    validate_browser_bundle,
)

_BUNDLE_ID: Final = "popsign_tcn_browser_candidate_v1"
_SOURCE_PATHS: Final = {
    "feature-plan.json": "src/signlab/resources/features/config/hand-local-64-1.default.json",
    "landmarker.json": (
        "src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json"
    ),
    "quality-policy.json": (
        "src/signlab/resources/quality/config/landmark-quality-policy-1.default.json"
    ),
    "segmenter.json": "configs/evaluation/candidate-event-detector-v1.json",
}
_ASSETS: Final = (
    ("decision_policy", "decision-policy.json", "application/json"),
    ("feature_plan", "feature-plan.json", "application/json"),
    ("golden_smoke", "golden/smoke.json", "application/json"),
    ("landmarker", "landmarker.json", "application/json"),
    ("model", "model.onnx", "application/onnx"),
    ("model_card", "model-card.md", "text/markdown"),
    ("quality_policy", "quality-policy.json", "application/json"),
    ("segmenter", "segmenter.json", "application/json"),
)
_ALLOWED_ONNX_OPS: Final = frozenset(
    {"Add", "Conv", "Gemm", "Pad", "ReduceMean"}
    | {"Relu", "Softmax", "Squeeze", "Transpose", "Unsqueeze"}
)


class ModelExportError(ValueError):
    """A stable, path-free candidate-export failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ModelExportResult:
    bundle_root: Path
    bundle_sha256: str
    model_sha256: str


@dataclass(frozen=True, slots=True)
class _Runtime:
    keras: Any
    numpy: Any
    onnx: Any
    onnxruntime: Any


def _runtime() -> _Runtime:
    try:
        return _Runtime(
            keras=importlib.import_module("keras"),
            numpy=importlib.import_module("numpy"),
            onnx=importlib.import_module("onnx"),
            onnxruntime=importlib.import_module("onnxruntime"),
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise ModelExportError("portable_export_runtime_unavailable") from error


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _read_repository_file(root: Path, path: str | Path) -> bytes:
    try:
        source = Path(path)
        resolved = (source if source.is_absolute() else root / source).resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise OSError
        return resolved.read_bytes()
    except (OSError, RuntimeError, ValueError) as error:
        raise ModelExportError("candidate_evidence_invalid") from error


def _verified_candidate(
    dossier_path: str | Path,
    nomination_report_path: str | Path,
    repository_root: Path,
    checkpoint_path: str | Path,
) -> tuple[DevelopmentCandidateDossier, bytes, bytes]:
    try:
        dossier = load_candidate_dossier(dossier_path)
        fresh = build_candidate_nomination_report(
            dossier,
            repository_root=repository_root,
            checkpoint_path=checkpoint_path,
        )
        checked = _read_repository_file(repository_root, nomination_report_path)
        if checked != canonical_json_bytes(fresh) + b"\n" or (
            fresh.get("candidate_status"),
            fresh.get("nomination_scope"),
            fresh.get("champion_status"),
            fresh.get("test_status"),
        ) != (
            "nominated_for_portable_export",
            "portable_export_only",
            "none_blocked",
            "sealed_not_loaded",
        ):
            raise ModelExportError("candidate_nomination_invalid")
        checkpoint = Path(checkpoint_path).read_bytes()
        if len(checkpoint) != dossier.checkpoint_size_bytes or _sha256(checkpoint) != (
            dossier.identities.research_model_sha256
        ):
            raise ModelExportError("candidate_nomination_invalid")
        return dossier, checked, checkpoint
    except ModelExportError:
        raise
    except (CandidateNominationError, OSError, TypeError, ValueError) as error:
        raise ModelExportError("candidate_nomination_invalid") from error


def _candidate_assets(root: Path, dossier: DevelopmentCandidateDossier) -> dict[str, bytes]:
    sources = {
        **_SOURCE_PATHS,
        "decision-policy.json": dossier.evidence.decision_policy.path,
        "model-card.md": dossier.evidence.model_card.path,
    }
    assets = {name: _read_repository_file(root, path) for name, path in sources.items()}
    if (
        _sha256(assets["decision-policy.json"]) != dossier.evidence.decision_policy.sha256
        or _sha256(assets["model-card.md"]) != dossier.evidence.model_card.sha256
    ):
        raise ModelExportError("candidate_evidence_invalid")
    return assets


def _normalize_onnx(model: Any, onnx: Any) -> bytes:
    try:
        graph = model.graph
        if (
            len(graph.input) != 1
            or graph.input[0].name != "input"
            or len(graph.output) != 1
            or graph.sparse_initializer
            or model.training_info
            or model.functions
        ):
            raise ModelExportError("onnx_interface_invalid")
        usage: dict[str, list[tuple[int, int]]] = {tensor.name: [] for tensor in graph.initializer}
        for node_index, node in enumerate(graph.node):
            if node.domain not in {"", "ai.onnx"} or node.op_type not in _ALLOWED_ONNX_OPS:
                raise ModelExportError("onnx_operator_unsupported")
            node.domain = ""
            for input_index, name in enumerate(node.input):
                if name in usage:
                    usage[name].append((node_index, input_index))
        renamed: dict[str, str] = {}
        for tensor in graph.initializer:
            if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
                raise ModelExportError("onnx_external_data_unsupported")
            copy = onnx.TensorProto()
            copy.CopyFrom(tensor)
            copy.name = ""
            identity = copy.SerializeToString(deterministic=True) + repr(
                tuple(usage[tensor.name])
            ).encode("ascii")
            renamed[tensor.name] = f"constant_{hashlib.sha256(identity).hexdigest()}"
        if len(set(renamed.values())) != len(renamed):
            raise ModelExportError("onnx_initializer_identity_conflict")
        for tensor in graph.initializer:
            tensor.name = renamed[tensor.name]
        output_name = graph.output[0].name
        values = {
            name: (
                "probabilities" if name == output_name else f"value_{node_index:03d}_{output_index}"
            )
            for node_index, node in enumerate(graph.node)
            for output_index, name in enumerate(node.output)
            if name
        }
        if output_name not in values or len(set(values.values())) != len(values):
            raise ModelExportError("onnx_interface_invalid")
        for node in graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = renamed.get(name, values.get(name, name))
            for index, name in enumerate(node.output):
                if name:
                    node.output[index] = values[name]
            node.name = ""
            node.doc_string = ""
        for value in graph.value_info:
            value.name = values.get(value.name, value.name)
        graph.output[0].name = "probabilities"
        initializers = sorted(graph.initializer, key=lambda item: item.name)
        del graph.initializer[:]
        graph.initializer.extend(initializers)
        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 18
        model.doc_string = ""
        graph.name = _BUNDLE_ID
        graph.doc_string = ""
        del model.metadata_props[:]
        onnx.checker.check_model(model, full_check=True)
        return cast(bytes, model.SerializeToString(deterministic=True))
    except ModelExportError:
        raise
    except Exception as error:
        raise ModelExportError("onnx_normalization_invalid") from error


def _export_onnx(checkpoint_path: str | Path, path: Path, runtime: _Runtime) -> bytes:
    try:
        model = runtime.keras.models.load_model(Path(checkpoint_path))
        if (
            tuple(model.input_shape) != (None, 64, 126)
            or tuple(model.output_shape) != (None, 6)
            or int(model.count_params()) != 29_094
            or tuple(model.output_names) != ("probabilities",)
        ):
            raise ModelExportError("candidate_model_invalid")
        model.export(
            path,
            format="onnx",
            input_signature=[
                runtime.keras.InputSpec(shape=(1, 64, 126), dtype="float32", name="input")
            ],
            opset_version=18,
            verbose=False,
        )
        exported = runtime.onnx.load(path, load_external_data=False)
        normalized = _normalize_onnx(exported, runtime.onnx)
        path.write_bytes(normalized)
        return normalized
    except ModelExportError:
        raise
    except Exception as error:
        raise ModelExportError("onnx_export_failed") from error


def _cpu_smoke(model_bytes: bytes, runtime: _Runtime) -> bytes:
    try:
        session = runtime.onnxruntime.InferenceSession(
            model_bytes, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (
            len(inputs) != 1
            or (inputs[0].name, list(inputs[0].shape), inputs[0].type)
            != ("input", [1, 64, 126], "tensor(float)")
            or len(outputs) != 1
            or (outputs[0].name, list(outputs[0].shape), outputs[0].type)
            != ("probabilities", [1, 6], "tensor(float)")
        ):
            raise ModelExportError("onnx_interface_invalid")
        matrix = runtime.numpy.zeros((1, 64, 126), dtype=runtime.numpy.float32)
        probabilities = runtime.numpy.asarray(session.run(["probabilities"], {"input": matrix})[0])
        if (
            probabilities.shape != (1, 6)
            or probabilities.dtype != runtime.numpy.dtype("float32")
            or not runtime.numpy.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not runtime.numpy.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        ):
            raise ModelExportError("onnx_smoke_invalid")
        return (
            canonical_json_bytes(
                {
                    "evidence_scope": "onnx_self_smoke_only",
                    "format": "signlab-onnx-cpu-smoke/1",
                    "input": {
                        "dtype": "float32",
                        "generator": "zeros",
                        "name": "input",
                        "shape": [1, 64, 126],
                        "sha256": _sha256(matrix.tobytes(order="C")),
                    },
                    "model_sha256": _sha256(model_bytes),
                    "output": {
                        "dtype": "float32",
                        "name": "probabilities",
                        "probabilities": [round(float(value), 4) for value in probabilities[0]],
                        "probabilities_scope": "display_only_quantized_4dp",
                        "shape": [1, 6],
                    },
                    "provider": "CPUExecutionProvider",
                }
            )
            + b"\n"
        )
    except ModelExportError:
        raise
    except Exception as error:
        raise ModelExportError("onnx_smoke_failed") from error


def _component_identities(assets: dict[str, bytes]) -> dict[str, str]:
    try:
        segmenter = CandidateEventConfigV1.model_validate_json(
            assets["segmenter.json"], strict=True
        )
        policy = DecisionPolicyV1.model_validate_json(assets["decision-policy.json"], strict=True)
        return {
            "decision_policy_sha256": decision_policy_digest(policy),
            "feature_plan_sha256": landmark_feature_plan_digest(
                validate_landmark_feature_plan(assets["feature-plan.json"])
            ),
            "landmarker_sha256": mediapipe_extraction_config_digest(assets["landmarker.json"]),
            "quality_policy_sha256": landmark_quality_policy_digest(assets["quality-policy.json"]),
            "segmenter_sha256": candidate_event_config_digest(segmenter),
        }
    except (KeyError, TypeError, ValidationError, ValueError) as error:
        raise ModelExportError("bundle_components_invalid") from error


def _manifest(
    dossier: DevelopmentCandidateDossier,
    nomination_report: bytes,
    assets: dict[str, bytes],
) -> tuple[BrowserModelBundleManifestV1, bytes]:
    payload = {
        "assets": [
            {
                "artifact_id": role,
                "locator": {"kind": "workspace_relative", "path": path},
                "media_type": media_type,
                "role": role,
                "schema_version": "artifact-reference/1",
                "sha256": _sha256(assets[path]),
                "size_bytes": len(assets[path]),
            }
            for role, path, media_type in _ASSETS
        ],
        "bundle_id": _BUNDLE_ID,
        "candidate": {
            "candidate_id": dossier.candidate_id,
            "candidate_version": dossier.version,
            "champion_status": "none_blocked",
            "configuration_sha256": dossier.identities.configuration_sha256,
            "corpus_sha256": dossier.identities.corpus_sha256,
            "derivative_set_sha256": dossier.identities.derivative_set_sha256,
            "dossier_sha256": canonical_sha256(dossier, domain=dossier.format),
            "external_dataset_sha256": dossier.identities.external_dataset_sha256,
            "input_feature_plan_sha256": dossier.identities.input_feature_plan_sha256,
            "metric_claim": "development_only",
            "nomination_report_sha256": _sha256(nomination_report),
            "nomination_scope": dossier.nomination_scope,
            "research_checkpoint_sha256": dossier.identities.research_model_sha256,
            "source_commit": dossier.source_commit,
            "source_feature_plan_sha256": dossier.identities.source_feature_plan_sha256,
            "source_run_id": dossier.source_run_id,
            "split_sha256": dossier.identities.split_sha256,
            "taxonomy_sha256": dossier.identities.taxonomy_sha256,
            "test_status": dossier.test_status,
        },
        "components": _component_identities(assets),
        "format": "browser-model-bundle/1",
        "labels": list(dossier.labels),
        "licenses": [
            {"distribution": "redistributable", "scope": "mediapipe", "spdx": "Apache-2.0"},
            {
                "distribution": "redistributable_with_attribution",
                "scope": "popsign_source_data",
                "spdx": "CC-BY-4.0",
            },
            {"distribution": "redistributable", "scope": "signlab_code", "spdx": "MIT"},
            {
                "distribution": "local_evaluation_only",
                "scope": "trained_model",
                "spdx": "NOASSERTION",
            },
        ],
        "onnx": {
            "format": "onnx",
            "input_dtype": "float32",
            "input_name": "input",
            "input_semantics": "hand_local_feature_sequence",
            "input_shape": [1, 64, 126],
            "opset": 18,
            "output_dtype": "float32",
            "output_name": "probabilities",
            "output_semantics": "uncalibrated_class_probabilities",
            "output_shape": [1, 6],
        },
        "version": dossier.version,
    }
    try:
        raw = canonical_json_bytes(payload) + b"\n"
        return load_browser_bundle_manifest(raw), raw
    except (ModelBundleError, TypeError, ValueError) as error:
        raise ModelExportError("bundle_manifest_invalid") from error


def export_browser_candidate_bundle(
    dossier_path: str | Path,
    nomination_report_path: str | Path,
    checkpoint_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> ModelExportResult:
    """Export, validate, and atomically publish one local-evaluation bundle."""

    try:
        root = Path(repository_root).resolve(strict=True)
        destination = Path(output_root)
        if destination.exists() or destination.is_symlink():
            raise ModelExportError("destination_conflict")
        dossier, nomination, checkpoint = _verified_candidate(
            dossier_path, nomination_report_path, root, checkpoint_path
        )
        source_assets = _candidate_assets(root, dossier)
        runtime = _runtime()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = destination.parent.resolve(strict=True) / destination.name
        with TemporaryDirectory(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        ) as staging_text:
            staging = Path(staging_text)
            snapshot = staging / "candidate.keras"
            snapshot.write_bytes(checkpoint)
            model_bytes = _export_onnx(snapshot, staging / "model.onnx", runtime)
            snapshot.unlink()
            assets = {
                **source_assets,
                "model.onnx": model_bytes,
                "golden/smoke.json": _cpu_smoke(model_bytes, runtime),
            }
            for _role, relative, _media_type in _ASSETS:
                target = staging.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(assets[relative])
            manifest, manifest_bytes = _manifest(dossier, nomination, assets)
            (staging / "manifest.json").write_bytes(manifest_bytes)
            checked, digest = validate_browser_bundle(staging)
            if checked != manifest or digest != browser_bundle_digest(manifest):
                raise ModelExportError("bundle_validation_failed")
            if destination.exists() or destination.is_symlink():
                raise ModelExportError("destination_conflict")
            staging.rename(destination)
        return ModelExportResult(destination, digest, _sha256(model_bytes))
    except ModelExportError:
        raise
    except (ModelBundleError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ModelExportError("bundle_publication_failed") from error
