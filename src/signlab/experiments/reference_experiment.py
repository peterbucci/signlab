"""One bounded, no-person reference experiment for reproducibility checks."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Final, Literal, cast

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
from numpy.typing import NDArray
from onnx import TensorProto, helper
from pydantic import model_validator
from sklearn.metrics import (  # type: ignore[import-untyped]
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from signlab import __version__
from signlab.contracts import core as cc
from signlab.contracts import dataset as dc
from signlab.contracts import extraction as ec
from signlab.contracts import pipeline as pc
from signlab.contracts import taxonomy as tax
from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.contracts.features import landmark_feature_plan_digest
from signlab.contracts.quality import landmark_quality_policy_digest
from signlab.experiments import tracking
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import derive_feature_source
from signlab.governance.resources import build_governance_policy
from signlab.quality.policy import assess_landmark_source
from signlab.quality.resources import build_default_quality_policy

# The contract constructors below stay deliberately compact so this bounded demo does not
# grow into a second orchestration framework.
# fmt: off
FORMAT: Final = "signlab-reference-experiment/1"
SUMMARY_FORMAT: Final = "signlab-reference-experiment-summary/1"
EXTRACTION_SHA: Final = "sha256:" + "e" * 64
LABELS: Final = tax.EXPECTED_CLASS_IDS
PARTITIONS: Final = ("train", "validation", "test")


class ReferenceExperimentError(ValueError):
    """A stable failure from the bounded reference-experiment boundary."""


class _Recipe(cc.StrictContractModel):
    format: Literal["signlab-reference-experiment/1"]
    run_name: cc.StableId
    random_seed: cc.SafeInteger
    labels: tuple[str, ...]
    signer_groups: Literal[6]
    sessions_per_signer: Literal[2]
    classifier_samples: Literal[36]
    inactive_cases: Literal[2]
    learned_other_kind: Literal["oov_gesture"]
    source_kind: Literal["project_authored_synthetic"]
    license_spdx: Literal["MIT"]
    contains_person_data: Literal[False]
    feature_plan_id: Literal["hand_local_64_frames"]
    feature_plan_sha256: str
    quality_policy_sha256: str
    taxonomy_sha256: str
    parity_tolerance: float
    expected_summary_path: Literal[
        "configs/experiments/signlab-reference-experiment-expected-v1.json"
    ]
    max_elapsed_seconds: Literal[180]
    max_peak_memory_mib: Literal[2048]
    max_pack_bytes: Literal[5242880]

    @model_validator(mode="after")
    def _fixed(self) -> _Recipe:
        if self.labels != LABELS or self.parity_tolerance != 1e-5:
            raise ValueError("reference recipe vocabulary or tolerance drifted")
        return self


@dataclass(frozen=True, slots=True)
class ReferenceExperimentResult:
    output_root: Path
    summary_path: Path
    sample_count: int
    label_count: int
    parity_maximum_absolute_difference: float
    tracking: tracking.ReferenceRunReceipt


@dataclass(frozen=True, slots=True)
class _Sample:
    sample_id: str
    participant_id: str
    session_id: str
    recording_id: str
    partition: str
    label: str
    tensor: NDArray[np.float32]
    feature_sha256: str
    feature_size: int


def _sha(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _json(value: cc.StrictContractModel | dict[str, object]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _uri(kind: str, name: str) -> cc.ArtifactUriLocatorV1:
    return cc.ArtifactUriLocatorV1(
        kind="artifact_uri", uri=f"signlab://reference/{kind}/{name}"
    )


def _load_recipe(path: Path) -> tuple[_Recipe, bytes]:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > 128 * 1024:
            raise ValueError
        recipe = _Recipe.model_validate_json(raw, strict=True)
    except (OSError, ValueError) as error:
        raise ReferenceExperimentError("reference recipe is unavailable or invalid") from error
    actual = (
        landmark_feature_plan_digest(load_packaged_default_feature_plan("hand_local")),
        landmark_quality_policy_digest(build_default_quality_policy()),
        tax.taxonomy_reference(tax.load_builtin_taxonomy()).sha256,
    )
    expected = (
        recipe.feature_plan_sha256,
        recipe.quality_policy_sha256,
        recipe.taxonomy_sha256,
    )
    if actual != expected:
        raise ReferenceExperimentError("reference recipe resource identities drifted")
    return recipe, raw


def _git_identity(anchor: Path) -> tuple[Path, str]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(anchor), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        root = Path(run("rev-parse", "--show-toplevel")).resolve()
        commit = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError) as error:
        raise ReferenceExperimentError("reference source identity is unavailable") from error
    if dirty or len(commit) != 40:
        raise ReferenceExperimentError("reference experiment requires a clean committed checkout")
    return root, commit


def _peak_memory_mib() -> float:
    if os.name == "nt":
        info = import_module("psutil").Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 2**20
    resource = import_module("resource")
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (2**20 if sys.platform == "darwin" else 2**10)


def _feature(
    recording: str, label_index: int | None, signer: int, session: int
) -> tuple[NDArray[np.float32], str, int, bool]:
    absent = ec.HandSlotV1(
        slot_id="hand_1", present=False, detector_index=None, tracking_id=None,
        handedness=None, handedness_confidence=None, image_landmarks=None,
        world_landmarks=None,
    )
    anchors = cast(Any, tuple(
        ec.BodyAnchorV1(name=name, present=False, image_point=None, world_point=None)
        for name in ec.BODY_ANCHOR_NAMES
    ))
    rows: list[ec.LandmarkFrameV1] = []
    for frame, timestamp in enumerate((0, 33_333, 66_667)):
        if label_index is None:
            first = absent.model_copy(update={"slot_id": "hand_0"})
        else:
            world: list[ec.Point3V1] = []
            for index in range(21):
                x = (index % 5) * 0.08 + signer * 0.0007 * (index % 3)
                y = (index // 5) * 0.11 + session * 0.0005 * (index % 2)
                z = ((index * 3) % 7) * 0.01 + frame * 0.0002 * (index % 4)
                if index == label_index + 1:
                    x += 0.26 + label_index * 0.025
                    y += label_index * 0.018
                world.append(ec.Point3V1(
                    x=x, y=y, z=z, visibility=None, presence=None
                ))
            image = tuple(ec.Point3V1(
                x=0.45 + point.x * 0.1, y=0.55 - point.y * 0.1,
                z=point.z * 0.1, visibility=None, presence=None
            ) for point in world)
            first = ec.HandSlotV1(
                slot_id="hand_0", present=True, detector_index=0, tracking_id="hand_0",
                handedness="left", handedness_confidence=0.99,
                image_landmarks=image, world_landmarks=tuple(world),
            )
        rows.append(ec.LandmarkFrameV1(
            schema_version="landmark-frame/1", source_recording_id=recording,
            frame_index=frame, source_pts=1_000_000 + timestamp,
            source_time_base_numerator=1, source_time_base_denominator=1_000_000,
            relative_timestamp_us=timestamp, task_timestamp_ms=timestamp // 1_000,
            invalid=False, invalid_reason=None, hands=cast(Any, (first, absent)),
            body_anchors=anchors, observed_hand_count=int(label_index is not None),
            observed_body_anchor_count=0,
        ))
    table = ec.LandmarkFramesTableV1(
        schema_version="landmark-frames-table/1", rows=tuple(rows)
    )
    source_sha = ec.landmark_frames_table_digest(table)
    parquet_sha = _sha(f"parquet:{recording}".encode())
    quality = assess_landmark_source(
        table, build_default_quality_policy(), source_recording_id=recording,
        source_sequence_content_sha256=source_sha,
        source_landmark_parquet_sha256=parquet_sha,
        declared_duration_us=100_000, expected_hand_count=1,
    )
    sequence = derive_feature_source(
        table, quality, load_packaged_default_feature_plan("hand_local"),
        source_recording_id=recording,
        source_media_sha256=_sha(f"media:{recording}".encode()),
        source_landmarks_sha256=source_sha,
        source_landmark_parquet_sha256=parquet_sha,
        source_mirror_state="not_mirrored", extraction_config_sha256=EXTRACTION_SHA,
    )
    raw = canonical_json_bytes(sequence)
    tensor = np.asarray(sequence.values_q, dtype=np.float32)
    tensor /= np.float32(sequence.quantization_scale)
    active = any(any(row) for row in sequence.hand_present_mask)
    return tensor, _sha(raw), len(raw), active


def _samples() -> tuple[_Sample, ...]:
    result: list[_Sample] = []
    for signer in range(6):
        participant = f"participant_{signer + 1:032x}"
        for label_index, label in enumerate(LABELS):
            number, session = signer * 6 + label_index + 1, label_index // 3
            sample, recording = f"sample_{number:032x}", f"recording_{number:032x}"
            tensor, digest, size, active = _feature(recording, label_index, signer, session)
            if not active or tensor.shape != (64, 126):
                raise ReferenceExperimentError("synthetic classifier feature shape drifted")
            result.append(_Sample(
                sample, participant, f"session_{signer * 2 + session + 1:032x}",
                recording, PARTITIONS[signer // 2], label, tensor, digest, size,
            ))
    return tuple(result)


def _artifact(name: str, role: str, media: str, raw: bytes) -> cc.ArtifactRefV1:
    return cc.ArtifactRefV1(
        schema_version="artifact-reference/1", artifact_id=name, role=role,
        media_type=media, sha256=_sha(raw), size_bytes=len(raw),
        locator=_uri("artifacts", name),
    )


def _component(
    role: Literal["model", "optimizer", "trainer", "evaluator"],
    implementation: str,
    parameter: tuple[str, int],
) -> cc.ComponentSpecV1:
    return cc.ComponentSpecV1(
        schema_version="component-spec/1", role=role,
        implementation_id=implementation, implementation_version="1.0.0",
        parameters=(cc.ParameterV1(name=parameter[0], value=parameter[1]),),
    )


def _core(
    samples: tuple[_Sample, ...], seed: int
) -> tuple[dc.DatasetManifestV1, pc.SplitManifestV1, pc.PreprocessingPlanV1,
           pc.ResolvedConfigurationV1]:
    taxonomy = tax.taxonomy_reference(tax.load_builtin_taxonomy())
    content = dc.DatasetContentV1(
        schema_version="dataset-content/1", taxonomy=taxonomy,
        governance_policy=build_governance_policy().policy_document,
        lineage_inventory_sha256=_sha(b"signlab-synthetic-no-person-lineage-v1"),
        sample_schema_version="portable-feature-sequence/1",
        samples=tuple(dc.DatasetSampleIdentityV1(
            sample_id=item.sample_id, participant_id=item.participant_id,
            session_id=item.session_id, source_recording_id=item.recording_id,
            label_id=cast(Any, item.label),
            artifact=cc.ArtifactRefV1(
                schema_version="artifact-reference/1", artifact_id=item.sample_id,
                role="sample_data", media_type="application/json",
                sha256=item.feature_sha256, size_bytes=item.feature_size,
                locator=_uri("samples", item.sample_id),
            ),
        ) for item in samples),
    )
    dataset = dc.DatasetManifestV1(
        schema_version="dataset-manifest/1", dataset_id="synthetic_reference_dataset",
        version="1.0.0", content=content, data_sha256=pc.dataset_content_digest(content),
    )
    dataset_ref = pc.contract_reference(dataset, _uri("contracts", "dataset"))
    partitions = tuple(pc.SplitPartitionV1(
        name=cast(Any, name),
        sample_ids=tuple(item.sample_id for item in samples if item.partition == name),
        participant_ids=tuple(sorted(
            {item.participant_id for item in samples if item.partition == name}
        )),
        session_ids=tuple(sorted(
            {item.session_id for item in samples if item.partition == name}
        )),
        source_recording_ids=tuple(sorted(
            {item.recording_id for item in samples if item.partition == name}
        )),
    ) for name in PARTITIONS)
    split = pc.SplitManifestV1(
        schema_version="split-manifest/1", split_id="synthetic_signer_session_split",
        version="1.0.0", dataset=dataset_ref, dataset_data_sha256=dataset.data_sha256,
        strategy="participant-and-session-grouped", random_seed=seed,
        partitions=cast(Any, partitions),
    )
    preprocessing = pc.PreprocessingPlanV1(
        schema_version="preprocessing-plan/1", plan_id="hand_local_reference_input",
        version="1.0.0", taxonomy=taxonomy,
        input_schema_version="portable-feature-sequence/1",
        output_schema_version="model-input/1", compatible_runtimes=("python", "typescript"),
        steps=(pc.PreprocessingStepV1(
            index=0, operation_id="dequantize_hand_local", implementation_version="1.0.0",
            input_schema_version="portable-feature-sequence/1",
            output_schema_version="model-input/1",
            parameters=(cc.ParameterV1(name="frames", value=64),
                        cc.ParameterV1(name="width", value=126)),
        ),),
    )
    configuration = pc.ResolvedConfigurationV1(
        schema_version="resolved-configuration/1", config_id="reference_experiment_v1",
        version="1.0.0", taxonomy=taxonomy, dataset=dataset_ref,
        dataset_data_sha256=dataset.data_sha256,
        split=pc.contract_reference(split, _uri("contracts", "split")),
        preprocessing=pc.contract_reference(preprocessing, _uri("contracts", "preprocessing")),
        random_seed=seed, deterministic_algorithms=True,
        model=_component("model", "nearest_centroid_linear_scores", ("classes", 6)),
        optimizer=_component("optimizer", "closed_form_class_centroids", ("passes", 1)),
        trainer=_component("trainer", "single_thread_fixed_split", ("threads", 1)),
        evaluator=_component("evaluator", "fixed_partition_metrics", ("test_accesses", 1)),
    )
    return dataset, split, preprocessing, configuration


def _fit(samples: tuple[_Sample, ...]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    train = tuple(item for item in samples if item.partition == "train")
    pooled = np.stack([item.tensor.mean(axis=0) for item in train]).astype(np.float32)
    labels = np.asarray([LABELS.index(item.label) for item in train], dtype=np.int64)
    with threadpool_limits(limits=1):
        centroids = np.stack([
            pooled[labels == index].mean(axis=0) for index in range(6)
        ]).astype(np.float32)
    return 2 * centroids.T, -np.sum(centroids * centroids, axis=1)


def _predict(
    samples: tuple[_Sample, ...], weights: NDArray[np.float32], bias: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    pooled = np.stack([item.tensor.mean(axis=0) for item in samples]).astype(np.float32)
    with threadpool_limits(limits=1):
        logits = pooled @ weights + bias
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True)).astype(np.float32)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return probabilities, probabilities.argmax(axis=1).astype(np.int64)


def _onnx(weights: NDArray[np.float32], bias: NDArray[np.float32]) -> bytes:
    nodes = [
        helper.make_node("ReduceMean", ["input", "axes"], ["pooled"], keepdims=0),
        helper.make_node("Gemm", ["pooled", "weights", "bias"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "signlab_reference_nearest_centroid",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64, 126])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
         helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.ravel()),
         helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias)],
    )
    model = helper.make_model(
        graph, producer_name="signlab-reference-experiment",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.doc_string = "MIT no-person mechanics-only nearest-centroid reference model."
    onnx.checker.check_model(model)
    return cast(bytes, model.SerializeToString(deterministic=True))


def _evaluate(
    samples: tuple[_Sample, ...], weights: NDArray[np.float32], bias: NDArray[np.float32]
) -> tuple[dict[str, Any], bytes, bytes]:
    results: dict[str, Any] = {}
    matrices: dict[str, list[list[int]]] = {}
    rows: list[tuple[object, ...]] = []
    for partition in ("validation", "test"):
        selected = tuple(item for item in samples if item.partition == partition)
        probabilities, predicted = _predict(selected, weights, bias)
        expected = np.asarray([LABELS.index(item.label) for item in selected])
        matrices[partition] = confusion_matrix(
            expected, predicted, labels=range(6)
        ).astype(int).tolist()
        results[partition] = {
            "macro_f1": float(f1_score(expected, predicted, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        }
        for item, actual, guess, probability in zip(
            selected, expected, predicted, probabilities, strict=True
        ):
            rows.append((
                partition, item.sample_id, LABELS[int(actual)], LABELS[int(guess)],
                str(actual == guess).lower(), *map(float, probability),
            ))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("partition", "sample_id", "actual", "predicted", "correct",
                     *(f"p_{label}" for label in LABELS)))
    writer.writerows(rows)
    matrix_raw = _json({
        "format": "signlab-reference-confusion/1", "labels": list(LABELS),
        "partitions": matrices,
    })
    return results, matrix_raw, stream.getvalue().encode()


def _parity(
    samples: tuple[_Sample, ...], model: bytes, native: NDArray[np.float32], tolerance: float
) -> dict[str, object]:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(
        model, sess_options=options, providers=["CPUExecutionProvider"]
    )
    observed = np.vstack([
        session.run(["probabilities"], {"input": item.tensor[None]})[0]
        for item in samples
    ]).astype(np.float32)
    difference = np.abs(observed - native)
    if not np.allclose(observed, native, atol=tolerance, rtol=tolerance):
        raise ReferenceExperimentError("native and ONNX probabilities exceed parity tolerance")
    return {
        "format": "signlab-reference-onnx-parity/1", "status": "pass",
        "providers": ["CPUExecutionProvider"],
        "cases": len(samples), "absolute_tolerance": tolerance,
        "relative_tolerance": tolerance,
        "maximum_absolute_difference": float(difference.max(initial=0.0)),
        "maximum_relative_difference": float(
            (difference / np.maximum(np.abs(native), 1e-12)).max(initial=0.0)
        ),
        "class_mismatches": int(np.count_nonzero(observed.argmax(1) != native.argmax(1))),
    }


def _structure(
    recipe: bytes, chain: tuple[dc.DatasetManifestV1, pc.SplitManifestV1,
                                pc.PreprocessingPlanV1, pc.ResolvedConfigurationV1]
) -> dict[str, object]:
    dataset, split, preprocessing, configuration = chain
    return {
        "format": "signlab-reference-experiment-expected/1",
        "recipe_sha256": _sha(recipe),
        "contract_schemas": [dataset.schema_version, split.schema_version,
                             preprocessing.schema_version, configuration.schema_version,
                             "run-record/1", "model-manifest/1"],
        "identities": {
            "taxonomy_sha256": dataset.content.taxonomy.sha256,
            "dataset_data_sha256": dataset.data_sha256,
            "dataset_sha256": pc.contract_digest(dataset),
            "split_sha256": pc.contract_digest(split),
            "preprocessing_sha256": pc.contract_digest(preprocessing),
            "configuration_sha256": pc.contract_digest(configuration),
        },
        "labels": list(LABELS),
        "partitions": {item.name: list(item.sample_ids) for item in split.partitions},
        "metric_keys": ["test.balanced_accuracy", "test.macro_f1",
                        "validation.balanced_accuracy", "validation.macro_f1"],
        "metric_expectations": {
            "absolute_tolerance": 0.05,
            "values": {"test.balanced_accuracy": 1.0, "test.macro_f1": 1.0,
                       "validation.balanced_accuracy": 1.0,
                       "validation.macro_f1": 1.0},
        },
        "sample_counts": {"classifier": 36, "inactive": 2, "per_partition": 12,
                          "signer_groups": 6, "sessions_per_signer": 2},
    }


def _write(root: Path, name: str, raw: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _metric_values(evaluation: dict[str, Any]) -> dict[str, float]:
    return {
        f"{partition}.{name}": float(value)
        for partition, values in evaluation.items()
        for name, value in values.items()
    }


def run_reference_experiment(
    config_path: str | Path,
    *,
    output_root: str | Path,
    tracking_uri: str | None = None,
) -> ReferenceExperimentResult:
    """Run the single deterministic synthetic experiment and atomically publish its pack."""

    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    started = perf_counter()
    config_file = Path(config_path).resolve()
    recipe, recipe_raw = _load_recipe(config_file)
    repository, commit = _git_identity(config_file.parent)
    destination = Path(output_root).resolve()
    if destination.exists():
        raise ReferenceExperimentError("reference output must not already exist")
    samples = _samples()
    chain = _core(samples, recipe.random_seed)
    dataset, split, preprocessing, configuration = chain
    weights, bias = _fit(samples)
    probabilities, _ = _predict(samples, weights, bias)
    evaluation, confusion_raw, predictions_raw = _evaluate(samples, weights, bias)
    inactive = [
        _feature(f"recording_{1000 + index:032x}", None, index, index)[3]
        for index in range(recipe.inactive_cases)
    ]
    if any(inactive):
        raise ReferenceExperimentError("inactive examples entered the classifier path")
    model_raw = _onnx(weights, bias)
    parity = _parity(samples, model_raw, probabilities, recipe.parity_tolerance)
    parameters_raw = _json({
        "format": "signlab-reference-native-parameters/1",
        "weights": weights.tolist(), "bias": bias.tolist(),
    })
    evaluation_raw = _json({
        "format": "signlab-reference-evaluation/1",
        "claim_scope": "synthetic_mechanics_only", "labels": list(LABELS),
        "data": {"classifier_samples": 36, "signer_groups": 6,
                 "sessions_per_signer": 2,
                 "learned_other": {"kind": recipe.learned_other_kind, "samples": 6},
                 "inactive": {"cases": 2, "handled_as_inactive": 2,
                              "included_in_classifier": False}},
        "partitions": evaluation,
        "limitations": ["Synthetic scores do not estimate recognition quality.",
                        "This model is not the PopSign browser candidate."],
    })
    parity_raw = _json(parity)
    specs = {
        "evaluation.json": ("reference_evaluation", "evaluation", "application/json",
                            evaluation_raw),
        "model.onnx": ("reference_model_onnx", "model", "application/octet-stream", model_raw),
        "native-parameters.json": ("reference_native_parameters", "parameters",
                                   "application/json", parameters_raw),
        "parity.json": ("reference_onnx_parity", "parity", "application/json", parity_raw),
        "predictions.csv": ("reference_predictions", "predictions", "text/csv",
                            predictions_raw),
    }
    artifacts = {name: _artifact(*spec) for name, spec in specs.items()}
    finished_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    system = platform.system().casefold()
    run = pc.RunRecordV1(
        schema_version="run-record/1", run_id="reference_experiment_v1", version="1.0.0",
        status="succeeded", started_at=started_at, finished_at=finished_at,
        source=pc.SourceIdentityV1(
            repository="https://github.com/peterbucci/signlab", git_commit=commit,
            lockfile_sha256=_sha((repository / "uv.lock").read_bytes()),
            working_tree_clean=True),
        runtime=pc.RuntimeIdentityV1(
            signlab_version=__version__, python_version=platform.python_version(),
            os_family=cast(Any, "macos" if system == "darwin" else system),
            accelerator="cpu", deterministic_algorithms=True),
        resolved_configuration=pc.contract_reference(
            configuration, _uri("contracts", "configuration")
        ),
        dataset=pc.contract_reference(dataset, _uri("contracts", "dataset")),
        dataset_data_sha256=dataset.data_sha256,
        split=pc.contract_reference(split, _uri("contracts", "split")),
        preprocessing=pc.contract_reference(preprocessing, _uri("contracts", "preprocessing")),
        metrics=tuple(pc.MetricRecordV1(
            name=cast(Any, name), partition=cast(Any, partition),
            value=evaluation[partition][name], unit="ratio"
        ) for partition in ("test", "validation")
          for name in ("balanced_accuracy", "macro_f1")),
        outputs=tuple(sorted(artifacts.values(), key=lambda item: (item.role, item.artifact_id))),
        failure=None,
    )
    model = pc.ModelManifestV1(
        schema_version="model-manifest/1", model_id="synthetic_reference_model",
        version="1.0.0", model_format="onnx", format_version="1.22.0",
        taxonomy=dataset.content.taxonomy, label_order=LABELS,
        input_schema_version="model-input/1", output_schema_version="class-probabilities/1",
        training_run=pc.contract_reference(run, _uri("contracts", "run")),
        resolved_configuration=run.resolved_configuration, dataset=run.dataset,
        dataset_data_sha256=dataset.data_sha256, split=run.split,
        preprocessing=run.preprocessing, artifact=artifacts["model.onnx"],
    )
    structure = _structure(recipe_raw, chain)
    try:
        expected = parse_json_object((repository / recipe.expected_summary_path).read_bytes())
    except OSError as error:
        raise ReferenceExperimentError("expected reference structure is unavailable") from error
    observed_metrics = _metric_values(evaluation)
    metric_rule = cast(dict[str, Any], expected.get("metric_expectations"))
    expected_metrics = cast(dict[str, float], metric_rule.get("values"))
    tolerance = metric_rule.get("absolute_tolerance")
    if (
        structure != expected
        or observed_metrics.keys() != expected_metrics.keys()
        or not isinstance(tolerance, (int, float))
        or any(
            abs(observed_metrics[key] - expected_metrics[key]) > tolerance
            for key in observed_metrics
        )
    ):
        raise ReferenceExperimentError("reference structure or expected metrics drifted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "pack"
        stage.mkdir()
        paths = {
            "configuration_path": _write(
                stage, "configuration.json", pc.canonical_contract_json(configuration) + b"\n"
            ),
            "report_path": _write(stage, "evaluation.json", evaluation_raw),
            "confusion_matrix_path": _write(stage, "confusion-matrix.json", confusion_raw),
            "predictions_path": _write(stage, "predictions.csv", predictions_raw),
        }
        payloads = {
            "recipe.json": recipe_raw,
            "dataset.json": pc.canonical_contract_json(dataset) + b"\n",
            "split.json": pc.canonical_contract_json(split) + b"\n",
            "preprocessing.json": pc.canonical_contract_json(preprocessing) + b"\n",
            "run.json": pc.canonical_contract_json(run) + b"\n",
            "model.json": pc.canonical_contract_json(model) + b"\n",
            "model.onnx": model_raw, "native-parameters.json": parameters_raw,
            "parity.json": parity_raw,
            "dataset-card.md": (
                b"# Synthetic reference dataset\n\nMIT, project-authored, no-person "
                b"landmarks; mechanics only.\n"
            ),
            "model-card.md": (
                b"# Synthetic reference model\n\nNearest-centroid linear scores; "
                b"no quality claim and not a deployable candidate.\n"
            ),
        }
        for name, raw in payloads.items():
            _write(stage, name, raw)
        for name, reference in artifacts.items():
            raw = (stage / name).read_bytes()
            if (_sha(raw), len(raw)) != (reference.sha256, reference.size_bytes):
                raise ReferenceExperimentError("reference pack artifact verification failed")
        validators = {
            "dataset.json": pc.validate_dataset_manifest_v1,
            "split.json": pc.validate_split_manifest,
            "preprocessing.json": pc.validate_preprocessing_plan,
            "configuration.json": pc.validate_resolved_configuration,
            "run.json": pc.validate_run_record,
            "model.json": pc.validate_model_manifest,
        }
        for name, validator in validators.items():
            validator((stage / name).read_bytes())
        pc.assert_model_compatible(dataset, split, preprocessing, configuration, run, model)
        receipt = tracking.log_reference_run(
            tracking.ReferenceRunInput(
                run_name=recipe.run_name, git_commit=commit, git_dirty=False,
                corpus_sha256=dataset.data_sha256,
                split_sha256=pc.contract_digest(split),
                feature_plan_sha256=recipe.feature_plan_sha256, seed=recipe.random_seed,
                parameters={"classifier_samples": 36, "model": "nearest_centroid",
                            "threads": 1},
                metrics=observed_metrics, **paths,
            ),
            tracking_uri=tracking_uri,
        )
        if tracking.verify_reference_run(receipt.run_id, tracking_uri=tracking_uri) != receipt:
            raise ReferenceExperimentError("reference MLflow verification drifted")
        elapsed, peak = perf_counter() - started, _peak_memory_mib()
        summary: dict[str, Any] = {
            "format": SUMMARY_FORMAT, "status": "pass", "structure": structure,
            "metrics": evaluation, "parity": parity,
            "tracking": {"run_id": receipt.run_id, "verified": True},
            "budgets": {"elapsed_seconds": round(elapsed, 3),
                        "peak_process_mib": round(peak, 3), "pack_bytes": 0,
                        "max_elapsed_seconds": 180, "max_peak_memory_mib": 2048,
                        "max_pack_bytes": 5_242_880},
            "limitations": ["Synthetic mechanics evidence only; no quality claim.",
                            "Browser runtime parity remains evidence from Story #39."],
        }
        for _ in range(3):
            _write(stage, "summary.json", _json(summary))
            size = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
            summary["budgets"]["pack_bytes"] = size
        _write(stage, "summary.json", _json(summary))
        size = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
        if (
            perf_counter() - started > recipe.max_elapsed_seconds
            or peak > recipe.max_peak_memory_mib
            or size > recipe.max_pack_bytes
        ):
            raise ReferenceExperimentError("reference experiment exceeded its resource budget")
        stage.replace(destination)
    return ReferenceExperimentResult(
        destination, destination / "summary.json", len(samples), len(LABELS),
        cast(float, parity["maximum_absolute_difference"]), receipt,
    )

# fmt: on

__all__ = [
    "FORMAT",
    "SUMMARY_FORMAT",
    "ReferenceExperimentError",
    "ReferenceExperimentResult",
    "run_reference_experiment",
]
