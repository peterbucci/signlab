"""One bounded GRU-versus-TCN engineering-feasibility experiment."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import math
import os
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Annotated, Any, Final, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, model_validator

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.contracts.core import (
    FiniteFloat,
    StableId,
    StrictContractModel,
)
from signlab.contracts.external_dataset import ExternalTargetLabel
from signlab.contracts.taxonomy import Sha256Digest
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments.baselines import (
    BaselineExperimentError,
    _git_identity,
    _load_inputs,
    _load_samples,
    _partition,
)
from signlab.experiments.tracking import (
    ReferenceRunInput,
    ReferenceRunReceipt,
    log_reference_run,
    verify_reference_run,
)
from signlab.features.resources import load_packaged_default_feature_plan

type ModelName = Literal["gru", "tcn"]

TARGET_LABELS: Final[tuple[ExternalTargetLabel, ...]] = (
    "hello",
    "no",
    "please",
    "thank_you",
    "yes",
)
MODEL_NAMES: Final[tuple[ModelName, ...]] = ("gru", "tcn")
CONFIG_FORMAT: Final = "signlab-sequence-baselines/1"
REPORT_FORMAT: Final = "signlab-sequence-baseline-report/1"
CONFUSION_FORMAT: Final = "signlab-sequence-confusion-matrices/1"
_EXPECTED_PARAMETERS: Final[dict[ModelName, int]] = {"gru": 26_741, "tcn": 29_317}
_SPLIT_QUOTAS: Final = {
    "train": Counter({label: 10 for label in TARGET_LABELS}),
    "validation": Counter({label: 3 for label in TARGET_LABELS}),
}


class SequenceBaselineError(ValueError):
    """A stable, path-free failure from this optional experiment boundary."""


class SequenceBaselineConfig(StrictContractModel):
    """The single reviewed choice set for the two-model feasibility run."""

    format: Literal["signlab-sequence-baselines/1"]
    run_name: StableId
    corpus_sha256: Sha256Digest
    external_dataset_sha256: Sha256Digest
    split_sha256: Sha256Digest
    feature_plan_id: Literal["combined_64_frames"]
    feature_plan_sha256: Sha256Digest
    taxonomy_id: Literal["signlab-five"]
    taxonomy_version: Literal["1.0.0"]
    taxonomy_sha256: Sha256Digest
    labels: Annotated[tuple[ExternalTargetLabel, ...], Field(min_length=5, max_length=5)]
    models: Annotated[tuple[ModelName, ...], Field(min_length=2, max_length=2)]
    vectorizer: Literal["frame_major_dequantized_values_v1"]
    input_frames: Literal[64]
    input_width: Literal[134]
    seed: Literal[20260828]
    optimizer: Literal["adam"]
    learning_rate: Annotated[FiniteFloat, Field(gt=0)]
    batch_size: Literal[32]
    maximum_epochs: Literal[30]
    early_stopping_metric: Literal["val_loss"]
    early_stopping_mode: Literal["min"]
    early_stopping_min_delta: Annotated[FiniteFloat, Field(ge=0)]
    early_stopping_patience: Literal[8]
    restore_best_weights: Literal[False]
    shuffle: Literal[True]
    evaluation_id: Literal["fixed_label_validation_metrics_v1"]
    split_id: Literal["popsign-five-isolated-smoke-v1"]
    gru_units: Literal[48]
    tcn_channels: Literal[32]
    tcn_kernel_size: Literal[3]
    tcn_dilations: Annotated[tuple[int, ...], Field(min_length=4, max_length=4)]
    latency_warmup_runs: Literal[10]
    latency_measurement_runs: Literal[100]

    @model_validator(mode="after")
    def _require_reviewed_design(self) -> Self:
        if self.labels != TARGET_LABELS or self.models != MODEL_NAMES:
            raise ValueError("sequence-baseline labels or model order drifted")
        if (
            self.learning_rate != 0.001
            or self.early_stopping_min_delta != 0.0
            or self.tcn_dilations != (1, 2, 4, 8)
        ):
            raise ValueError("sequence-baseline reviewed protocol drifted")
        return self


@dataclass(frozen=True, slots=True)
class SequenceBaselineResult:
    """Small receipt returned after artifacts and lineage are verified."""

    output_root: Path
    public_report_path: Path
    tracking: ReferenceRunReceipt


@dataclass(frozen=True, slots=True)
class _Runtime:
    keras: Any
    tensorflow: Any


@dataclass(frozen=True, slots=True)
class _Evaluation:
    predictions: tuple[ExternalTargetLabel, ...]
    probabilities: NDArray[np.float32]
    metrics: dict[str, Any]
    confusion: list[list[int]]


@dataclass(frozen=True, slots=True)
class _ModelRun:
    report: dict[str, Any]
    best_bytes: bytes
    last_bytes: bytes
    evaluation: _Evaluation


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def load_sequence_baseline_config(
    path: str | Path,
) -> tuple[SequenceBaselineConfig, bytes]:
    """Load only the canonical, LF-terminated reviewed configuration."""

    try:
        raw = Path(path).read_bytes()
        payload = parse_json_object(raw)
        config = SequenceBaselineConfig.model_validate_json(
            canonical_json_bytes(payload), strict=True
        )
        if raw != canonical_json_bytes(config) + b"\n":
            raise ValueError
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise SequenceBaselineError("sequence-baseline configuration is invalid") from error
    return config, raw


def _vectorize(
    samples: Sequence[PublicCorpusSample], config: SequenceBaselineConfig
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    plan = load_packaged_default_feature_plan("combined")
    expected_shape = (config.input_frames, config.input_width)
    label_index = {label: index for index, label in enumerate(config.labels)}
    values: list[tuple[tuple[int, ...], ...]] = []
    labels: list[int] = []
    for sample in samples:
        feature = sample.feature
        if (
            feature.feature_plan_sha256 != config.feature_plan_sha256
            or feature.feature_names != plan.feature_order
            or feature.quantization_scale != plan.quantization_scale
            or feature.statistics_sha256 is not None
            or (len(feature.values_q), len(feature.values_q[0])) != expected_shape
        ):
            raise SequenceBaselineError("portable sequence feature shape or identity drifted")
        values.append(feature.values_q)
        labels.append(label_index[sample.target_label_id])
    if not values:
        raise SequenceBaselineError("sequence-baseline partition is empty")
    matrix = np.asarray(values, dtype=np.float32)
    matrix /= np.float32(plan.quantization_scale)
    return np.ascontiguousarray(matrix), np.asarray(labels, dtype=np.int64)


def _runtime() -> _Runtime:
    if os.environ.setdefault("KERAS_BACKEND", "tensorflow") != "tensorflow":
        raise SequenceBaselineError("Keras must use the TensorFlow backend")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    try:
        runtime = _Runtime(
            keras=importlib.import_module("keras"),
            tensorflow=importlib.import_module("tensorflow"),
        )
        runtime.tensorflow.config.set_visible_devices([], "GPU")
    except (ImportError, ModuleNotFoundError) as error:
        raise SequenceBaselineError("install the SignLab experiments extra") from error
    except Exception as error:
        raise SequenceBaselineError("the CPU-only Keras runtime is unavailable") from error
    return runtime


def _seed_runtime(runtime: _Runtime, seed: int) -> None:
    try:
        runtime.keras.utils.set_random_seed(seed)
        runtime.tensorflow.config.experimental.enable_op_determinism()
    except Exception as error:
        raise SequenceBaselineError("deterministic Keras execution is unavailable") from error


def _compile(model: Any, config: SequenceBaselineConfig, runtime: _Runtime) -> Any:
    model.compile(
        optimizer=runtime.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


def _build_graph(
    name: ModelName,
    input_width: int,
    config: SequenceBaselineConfig,
    runtime: _Runtime,
) -> Any:
    """Build the reviewed GRU or TCN graph for one registered input width."""

    try:
        layers = runtime.keras.layers
        inputs = runtime.keras.Input(
            shape=(config.input_frames, input_width), dtype="float32", name="input"
        )
        if name == "gru":
            values = layers.GRU(
                config.gru_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=0.0,
                recurrent_dropout=0.0,
                return_sequences=True,
                reset_after=True,
                use_cudnn=False,
                name="gru",
            )(inputs)
        elif name == "tcn":
            values = layers.Conv1D(config.tcn_channels, 1, padding="same", name="input_projection")(
                inputs
            )
            for dilation in config.tcn_dilations:
                residual = values
                values = layers.Conv1D(
                    config.tcn_channels,
                    config.tcn_kernel_size,
                    padding="causal",
                    dilation_rate=dilation,
                    activation="relu",
                    name=f"d{dilation}_conv_1",
                )(values)
                values = layers.Conv1D(
                    config.tcn_channels,
                    config.tcn_kernel_size,
                    padding="causal",
                    dilation_rate=dilation,
                    name=f"d{dilation}_conv_2",
                )(values)
                values = layers.Add(name=f"d{dilation}_residual")([residual, values])
                values = layers.ReLU(name=f"d{dilation}_relu")(values)
        else:  # pragma: no cover - the strict configuration makes this unreachable.
            raise SequenceBaselineError("unknown sequence-baseline model")
        pooled = layers.GlobalAveragePooling1D(name="temporal_average")(values)
        probabilities = layers.Dense(
            len(config.labels), activation="softmax", name="probabilities"
        )(pooled)
        model = runtime.keras.Model(inputs, probabilities, name=f"sequence_{name}")
        _compile(model, config, runtime)
    except SequenceBaselineError:
        raise
    except Exception as error:
        raise SequenceBaselineError(f"the {name} graph could not be built") from error
    return model


def _build_model(name: ModelName, config: SequenceBaselineConfig, runtime: _Runtime) -> Any:
    model = _build_graph(name, config.input_width, config, runtime)
    if int(model.count_params()) != _EXPECTED_PARAMETERS[name]:
        raise SequenceBaselineError(f"the {name} parameter count drifted")
    return model


def _predict_probabilities(model: Any, matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    try:
        values = np.asarray(model.predict(matrix, batch_size=1, verbose=0), dtype=np.float32)
    except Exception as error:
        raise SequenceBaselineError("Keras validation prediction failed") from error
    if (
        values.shape != (len(matrix), len(TARGET_LABELS))
        or not np.isfinite(values).all()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise SequenceBaselineError("Keras validation output is invalid")
    return values


def _verify_checkpoint(
    name: ModelName, model: Any, matrix: NDArray[np.float32]
) -> NDArray[np.float32]:
    if (
        tuple(model.input_shape) != (None, 64, 134)
        or tuple(model.output_shape) != (None, len(TARGET_LABELS))
        or int(model.count_params()) != _EXPECTED_PARAMETERS[name]
    ):
        raise SequenceBaselineError(f"the reloaded {name} checkpoint contract drifted")
    return _predict_probabilities(model, matrix)


def _evaluate(expected: NDArray[np.int64], probabilities: NDArray[np.float32]) -> _Evaluation:
    predicted = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=np.int64)
    for actual, guess in zip(expected.tolist(), predicted.tolist(), strict=True):
        confusion[actual, guess] += 1
    per_class: dict[str, dict[str, int]] = {}
    f1_values: list[float] = []
    for index, label in enumerate(TARGET_LABELS):
        correct = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / support if support else 0.0
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[label] = {
            "support": support,
            "predicted": predicted_count,
            "correct": correct,
            "errors": support - correct,
        }
    clipped = np.clip(probabilities[np.arange(len(expected)), expected], 1e-7, 1.0)
    return _Evaluation(
        predictions=tuple(TARGET_LABELS[index] for index in predicted.tolist()),
        probabilities=probabilities,
        metrics={
            "loss": float(-np.mean(np.log(clipped))),
            "accuracy": float(np.mean(predicted == expected)),
            "macro_f1": float(np.mean(f1_values)),
            "per_class": per_class,
        },
        confusion=confusion.tolist(),
    )


def _nearest_rank(values: Sequence[float], proportion: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]


def _measure_latency(
    model: Any,
    matrix: NDArray[np.float32],
    *,
    warmup_runs: int,
    measurement_runs: int,
) -> dict[str, float | int | str]:
    try:
        for index in range(warmup_runs):
            model(matrix[index % len(matrix) : index % len(matrix) + 1], training=False)
        durations: list[float] = []
        for index in range(measurement_runs):
            row = matrix[index % len(matrix) : index % len(matrix) + 1]
            started = perf_counter_ns()
            model(row, training=False)
            durations.append((perf_counter_ns() - started) / 1_000_000)
    except Exception as error:
        raise SequenceBaselineError("CPU validation latency measurement failed") from error
    return {
        "scope": "prevectorized_batch_1_forward_only",
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "p50_ms": _nearest_rank(durations, 0.50),
        "p95_ms": _nearest_rank(durations, 0.95),
    }


def _train_one(
    name: ModelName,
    runtime: _Runtime,
    config: SequenceBaselineConfig,
    train_x: NDArray[np.float32],
    train_y: NDArray[np.int64],
    validation_x: NDArray[np.float32],
    validation_y: NDArray[np.int64],
    staging: Path,
) -> _ModelRun:
    _seed_runtime(runtime, config.seed)
    model = _build_model(name, config, runtime)
    model_root = staging / name
    model_root.mkdir(parents=True)
    best_path = model_root / "best.keras"
    last_path = model_root / "last.keras"
    try:
        callbacks = [
            runtime.keras.callbacks.ModelCheckpoint(
                best_path,
                monitor=config.early_stopping_metric,
                mode=config.early_stopping_mode,
                save_best_only=True,
                verbose=0,
            ),
            runtime.keras.callbacks.EarlyStopping(
                monitor=config.early_stopping_metric,
                mode=config.early_stopping_mode,
                min_delta=config.early_stopping_min_delta,
                patience=config.early_stopping_patience,
                restore_best_weights=config.restore_best_weights,
            ),
        ]
        _seed_runtime(runtime, config.seed)
        started = perf_counter_ns()
        history = model.fit(
            train_x,
            train_y,
            validation_data=(validation_x, validation_y),
            batch_size=config.batch_size,
            epochs=config.maximum_epochs,
            callbacks=callbacks,
            shuffle=config.shuffle,
            verbose=0,
        ).history
        fit_seconds = (perf_counter_ns() - started) / 1_000_000_000
        model.save(last_path)
        validation_losses = [float(value) for value in history["val_loss"]]
        if (
            not validation_losses
            or len(validation_losses) > config.maximum_epochs
            or not all(math.isfinite(value) for value in validation_losses)
        ):
            raise ValueError
        best_model = runtime.keras.models.load_model(best_path)
        last_model = runtime.keras.models.load_model(last_path)
        best_probabilities = _verify_checkpoint(name, best_model, validation_x)
        _verify_checkpoint(name, last_model, validation_x)
        best_bytes = best_path.read_bytes()
        last_bytes = last_path.read_bytes()
    except Exception as error:
        raise SequenceBaselineError(f"the bounded {name} fit or reload failed") from error
    evaluation = _evaluate(validation_y, best_probabilities)
    latency = _measure_latency(
        best_model,
        validation_x,
        warmup_runs=config.latency_warmup_runs,
        measurement_runs=config.latency_measurement_runs,
    )
    best_index = int(np.argmin(np.asarray(validation_losses)))
    report = {
        "parameter_count": int(model.count_params()),
        "training": {
            "epochs_completed": len(validation_losses),
            "best_epoch": best_index + 1,
            "best_recorded_validation_loss": validation_losses[best_index],
            "fit_seconds": fit_seconds,
        },
        "checkpoints": {
            "best": {
                "sha256": _sha256(best_bytes),
                "bytes": len(best_bytes),
                "reload_verified": True,
            },
            "last": {
                "sha256": _sha256(last_bytes),
                "bytes": len(last_bytes),
                "reload_verified": True,
            },
        },
        "validation": evaluation.metrics,
        "latency": latency,
    }
    return _ModelRun(report, best_bytes, last_bytes, evaluation)


def _aliases(
    samples: Sequence[PublicCorpusSample],
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    sample_aliases = tuple(f"sample_{index:03d}" for index in range(1, len(samples) + 1))
    signer_aliases = {
        signer: f"signer_{index:03d}"
        for index, signer in enumerate(
            sorted({sample.source_signer_id for sample in samples}), start=1
        )
    }
    return sample_aliases, signer_aliases


def _failure_cases(
    samples: Sequence[PublicCorpusSample],
    evaluation: _Evaluation,
    sample_aliases: Sequence[str],
    signer_aliases: Mapping[str, str],
) -> dict[str, Any]:
    examples = []
    for sample, alias, predicted, probabilities in zip(
        samples,
        sample_aliases,
        evaluation.predictions,
        evaluation.probabilities,
        strict=True,
    ):
        if predicted != sample.target_label_id:
            examples.append(
                {
                    "sample_alias": alias,
                    "signer_alias": signer_aliases[sample.source_signer_id],
                    "quality": sample.quality_disposition,
                    "actual": sample.target_label_id,
                    "predicted": predicted,
                    "uncalibrated_max_probability": float(np.max(probabilities)),
                }
            )
    return {"observed_error_count": len(examples), "examples": examples}


def _predictions_csv(
    samples: Sequence[PublicCorpusSample],
    runs: Mapping[ModelName, _ModelRun],
    sample_aliases: Sequence[str],
    signer_aliases: Mapping[str, str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "model",
            "partition",
            "sample_alias",
            "signer_alias",
            "quality",
            "actual",
            "predicted",
            "uncalibrated_max_probability",
        )
    )
    for name in MODEL_NAMES:
        evaluation = runs[name].evaluation
        for sample, alias, predicted, probabilities in zip(
            samples,
            sample_aliases,
            evaluation.predictions,
            evaluation.probabilities,
            strict=True,
        ):
            writer.writerow(
                (
                    name,
                    "validation",
                    alias,
                    signer_aliases[sample.source_signer_id],
                    sample.quality_disposition,
                    sample.target_label_id,
                    predicted,
                    f"{float(np.max(probabilities)):.9f}",
                )
            )
    return stream.getvalue().encode("utf-8")


def _public_markdown(report: Mapping[str, Any]) -> bytes:
    models = cast(dict[str, Any], report["models"])
    lines = [
        "# PopSign GRU/TCN feasibility v1",
        "",
        (
            "> Engineering feasibility only. Fifteen validation clips selected the best "
            "epochs and supplied these descriptive results; neither model is a winner."
        ),
        "",
        "## Reproducibility identity",
        "",
        (
            f"- Source commit: `{report['source']['git_commit']}` "
            f"(dirty: `{str(report['source']['git_dirty']).lower()}`)"
        ),
        f"- Configuration: `{report['identities']['configuration_sha256']}`",
        f"- Frozen split: `{report['identities']['split_sha256']}`",
        f"- Feature plan: `{report['identities']['feature_plan_sha256']}`",
        f"- Seed: `{report['protocol']['seed']}`",
        "- Opened features: 50 train and 15 validation; test features were never requested.",
        "",
        "## Side-by-side observations",
        "",
        (
            "| Model | Params | Epochs / best | Validation loss | Accuracy | "
            "Macro-F1 | CPU p50 / p95 ms | Errors |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in MODEL_NAMES:
        model = models[name]
        validation = model["validation"]
        training = model["training"]
        latency = model["latency"]
        failures = model["failure_analysis"]
        lines.append(
            f"| {name.upper()} | {model['parameter_count']:,} | "
            f"{training['epochs_completed']} / {training['best_epoch']} | "
            f"{validation['loss']:.3f} | {validation['accuracy']:.3f} | "
            f"{validation['macro_f1']:.3f} | {latency['p50_ms']:.3f} / "
            f"{latency['p95_ms']:.3f} | {failures['observed_error_count']} |"
        )
    lines.extend(
        [
            "",
            "## Training and checkpoint evidence",
            "",
            "| Model | Fit seconds | Checkpoint | Bytes | SHA-256 | Reloaded |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for name in MODEL_NAMES:
        model = models[name]
        for checkpoint_name in ("best", "last"):
            checkpoint = model["checkpoints"][checkpoint_name]
            lines.append(
                f"| {name.upper()} | {model['training']['fit_seconds']:.3f} | "
                f"{checkpoint_name} | {checkpoint['bytes']:,} | "
                f"`{checkpoint['sha256']}` | yes |"
            )
    lines.extend(
        [
            "",
            "## Per-class validation counts",
            "",
            "| Model | Class | Support | Predicted | Correct | Errors |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in MODEL_NAMES:
        for label in TARGET_LABELS:
            counts = models[name]["validation"]["per_class"][label]
            lines.append(
                f"| {name.upper()} | {label} | {counts['support']} | "
                f"{counts['predicted']} | {counts['correct']} | {counts['errors']} |"
            )
    lines.extend(["", "## Concrete validation failures", ""])
    for name in MODEL_NAMES:
        if lines[-1]:
            lines.append("")
        failures = models[name]["failure_analysis"]
        if failures["observed_error_count"] == 0:
            lines.append(f"- {name.upper()}: zero observed errors.")
            continue
        lines.extend(
            [
                f"### {name.upper()}",
                "",
                "| Sample | Actual | Predicted | Quality | Uncalibrated max score |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for example in failures["examples"]:
            lines.append(
                f"| {example['sample_alias']} | {example['actual']} | "
                f"{example['predicted']} | {example['quality']} | "
                f"{example['uncalibrated_max_probability']:.3f} |"
            )
    lines.extend(
        [
            "",
            (
                "Each model produced one best and one actual-last `.keras` checkpoint; "
                "all four were reloaded and exercised."
            ),
            "Checkpoint byte sizes include training state and are not deployment bundle sizes.",
            "",
            "## What this does and does not show",
            "",
            (
                "- Both fixed graphs can train, checkpoint, reload, and run on the "
                "current 64 x 134 feature contract."
            ),
            (
                "- Validation was used twice--early-stopping selection and reporting--"
                "so the numbers are optimistic and descriptive."
            ),
            (
                "- Both models average all 64 steps, including neutral zero padding; "
                "this experiment does not test mask-aware inputs."
            ),
            (
                "- This does not establish generalization, production or real-time "
                "performance, calibration, robustness, fairness, continuous signing, "
                "or broader sign-language recognition."
            ),
            (
                "- Architecture evidence across seeds or folds belongs to the next "
                "evaluation story, not this run."
            ),
            "",
            "## Reproduce",
            "",
            "```shell",
            (
                "uv run --locked --extra experiments signlab train sequence-baselines "
                "configs/experiments/popsign-sequence-baselines-v1.json "
                "--corpus-root <frozen-split-root> "
                "--external-manifest <external-dataset-manifest.json> "
                "--output-root runs/popsign-sequence-baselines-v1"
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _new_path(path: str | Path, *, label: str) -> Path:
    target = Path(path).resolve()
    if target.exists() or target.is_symlink():
        raise SequenceBaselineError(f"{label} must not already exist")
    return target


def _write_new(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise SequenceBaselineError("sequence-baseline artifacts could not be written") from error


def run_sequence_baselines(
    config_path: str | Path,
    *,
    corpus_root: str | Path,
    external_manifest_path: str | Path,
    output_root: str | Path,
    public_report_path: str | Path | None = None,
    tracking_uri: str | None = None,
) -> SequenceBaselineResult:
    """Run the two fixed models on train/validation and never request test features."""

    config, config_bytes = load_sequence_baseline_config(config_path)
    destination = _new_path(output_root, label="sequence-baseline output")
    human_report = (
        _new_path(public_report_path, label="public sequence-baseline report")
        if public_report_path is not None
        else destination / "public-report.md"
    )
    config_file = Path(config_path).resolve(strict=True)
    try:
        _repository, git_commit, git_dirty = _git_identity(config_file.parent)
        split_bytes, manifest_bytes = _load_inputs(
            cast(Any, config), Path(corpus_root), Path(external_manifest_path)
        )
        development = _load_samples(
            split_bytes,
            manifest_bytes,
            Path(corpus_root),
            ("train", "validation"),
        )
    except BaselineExperimentError as error:
        raise SequenceBaselineError("frozen sequence-baseline inputs are invalid") from error
    train_samples = _partition(development, "train")
    validation_samples = _partition(development, "validation")
    if (
        Counter(sample.target_label_id for sample in train_samples) != _SPLIT_QUOTAS["train"]
        or Counter(sample.target_label_id for sample in validation_samples)
        != _SPLIT_QUOTAS["validation"]
    ):
        raise SequenceBaselineError("sequence-baseline partition quotas drifted")
    train_x, train_y = _vectorize(train_samples, config)
    validation_x, validation_y = _vectorize(validation_samples, config)
    runtime = _runtime()
    with TemporaryDirectory(prefix="signlab-sequence-baselines-") as temporary_text:
        staging = Path(temporary_text)
        runs = {
            name: _train_one(
                name,
                runtime,
                config,
                train_x,
                train_y,
                validation_x,
                validation_y,
                staging,
            )
            for name in config.models
        }

    sample_aliases, signer_aliases = _aliases(validation_samples)
    model_reports: dict[str, Any] = {}
    for name in MODEL_NAMES:
        model_reports[name] = {
            **runs[name].report,
            "failure_analysis": _failure_cases(
                validation_samples,
                runs[name].evaluation,
                sample_aliases,
                signer_aliases,
            ),
        }
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "run_name": config.run_name,
        "claim_scope": "engineering_feasibility_only_no_winner",
        "source": {"git_commit": git_commit, "git_dirty": git_dirty},
        "identities": {
            "configuration_sha256": _sha256(config_bytes),
            "corpus_sha256": config.corpus_sha256,
            "external_dataset_sha256": config.external_dataset_sha256,
            "split_sha256": config.split_sha256,
            "split_id": config.split_id,
            "feature_plan_id": config.feature_plan_id,
            "feature_plan_sha256": config.feature_plan_sha256,
            "taxonomy_id": config.taxonomy_id,
            "taxonomy_version": config.taxonomy_version,
            "taxonomy_sha256": config.taxonomy_sha256,
        },
        "data": {
            "labels": list(config.labels),
            "opened_partition_counts": {
                "train": len(train_samples),
                "validation": len(validation_samples),
            },
            "test_status": "sealed_not_opened",
            "padding_policy": "all_64_steps_pooled_including_neutral_zero_padding",
        },
        "protocol": {
            "models": list(config.models),
            "input_shape": [config.input_frames, config.input_width],
            "seed": config.seed,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "maximum_epochs": config.maximum_epochs,
            "early_stopping": {
                "monitor": config.early_stopping_metric,
                "mode": config.early_stopping_mode,
                "min_delta": config.early_stopping_min_delta,
                "patience": config.early_stopping_patience,
                "restore_best_weights": config.restore_best_weights,
            },
            "shuffle": config.shuffle,
            "fit_calls": 2,
            "checkpoint_files": 4,
            "evaluation_id": config.evaluation_id,
            "validation_role": "checkpoint_selection_and_descriptive_reporting",
        },
        "models": model_reports,
        "environment": {
            "architecture": platform.machine().casefold() or "unknown",
            "keras": version("keras"),
            "numpy": version("numpy"),
            "os": platform.system().casefold() or "unknown",
            "python": platform.python_version(),
            "tensorflow": version("tensorflow"),
            "cpu_only": not bool(runtime.tensorflow.config.get_visible_devices("GPU")),
            "inter_op_threads": (
                runtime.tensorflow.config.threading.get_inter_op_parallelism_threads()
            ),
            "intra_op_threads": (
                runtime.tensorflow.config.threading.get_intra_op_parallelism_threads()
            ),
            "model_order": list(config.models),
        },
        "limitations": [
            "Only 50 training and 15 validation clips were used, with one split and one seed.",
            (
                "Validation selected checkpoints and supplied reported metrics; "
                "results are optimistic and descriptive."
            ),
            "The test feature partition was never requested.",
            "Neither model is selected or declared a quality winner.",
            (
                "Both models pool all 64 steps, including neutral zero padding; "
                "no mask-aware contract was tested."
            ),
            "Checkpoint byte sizes include training state and are not deployment bundle sizes.",
            "Latency is machine-specific prevectorized batch-one CPU forward time only.",
            (
                "No generalization, production, continuous-signing, calibration, "
                "robustness, fairness, or broader sign-language claim is supported."
            ),
        ],
    }
    confusion_bytes = (
        canonical_json_bytes(
            {
                "format": CONFUSION_FORMAT,
                "labels": list(config.labels),
                "orientation": {"rows": "actual", "columns": "predicted"},
                "validation": {name: runs[name].evaluation.confusion for name in MODEL_NAMES},
            }
        )
        + b"\n"
    )
    report_bytes = canonical_json_bytes(report) + b"\n"
    predictions_bytes = _predictions_csv(validation_samples, runs, sample_aliases, signer_aliases)
    public_bytes = _public_markdown(report)
    try:
        destination.mkdir(parents=True)
    except OSError as error:
        raise SequenceBaselineError("sequence-baseline output could not be created") from error
    paths = {
        "configuration_path": destination / "configuration.json",
        "report_path": destination / "report.json",
        "confusion_matrix_path": destination / "validation-confusion-matrices.json",
        "predictions_path": destination / "validation-predictions.csv",
    }
    for path, payload in (
        (paths["configuration_path"], config_bytes),
        (paths["report_path"], report_bytes),
        (paths["confusion_matrix_path"], confusion_bytes),
        (paths["predictions_path"], predictions_bytes),
        (destination / "gru" / "best.keras", runs["gru"].best_bytes),
        (destination / "gru" / "last.keras", runs["gru"].last_bytes),
        (destination / "tcn" / "best.keras", runs["tcn"].best_bytes),
        (destination / "tcn" / "last.keras", runs["tcn"].last_bytes),
    ):
        _write_new(path, payload)
    run_input = ReferenceRunInput(
        run_name=config.run_name,
        git_commit=git_commit,
        git_dirty=git_dirty,
        corpus_sha256=config.corpus_sha256,
        split_sha256=config.split_sha256,
        feature_plan_sha256=config.feature_plan_sha256,
        seed=config.seed,
        parameters={
            "batch_size": config.batch_size,
            "checkpoint_files": 4,
            "fit_calls": 2,
            "input_frames": config.input_frames,
            "input_width": config.input_width,
            "maximum_epochs": config.maximum_epochs,
            "models": "gru-tcn",
            "optimizer": config.optimizer,
            "evaluation_id": config.evaluation_id,
            "test_status": "sealed_not_opened",
        },
        metrics={
            **{
                f"validation.{name}.{metric}": float(runs[name].report["validation"][metric])
                for name in MODEL_NAMES
                for metric in ("loss", "accuracy", "macro_f1")
            },
            **{
                f"validation.{name}.{metric}": float(runs[name].report["latency"][metric])
                for name in MODEL_NAMES
                for metric in ("p50_ms", "p95_ms")
            },
        },
        **paths,
    )
    tracking = log_reference_run(run_input, tracking_uri=tracking_uri)
    if verify_reference_run(tracking.run_id, tracking_uri=tracking_uri) != tracking:
        raise SequenceBaselineError("sequence-baseline ledger verification drifted")
    _write_new(human_report, public_bytes)
    return SequenceBaselineResult(destination, human_report, tracking)


__all__ = [
    "CONFIG_FORMAT",
    "CONFUSION_FORMAT",
    "MODEL_NAMES",
    "REPORT_FORMAT",
    "TARGET_LABELS",
    "SequenceBaselineConfig",
    "SequenceBaselineError",
    "SequenceBaselineResult",
    "load_sequence_baseline_config",
    "run_sequence_baselines",
]
