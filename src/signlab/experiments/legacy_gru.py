"""One bounded Keras-to-ONNX compatibility run for the recovered legacy GRU."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import os
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.contracts.core import (
    FiniteFloat,
    PositiveSafeInteger,
    SafeInteger,
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

CONFIG_FORMAT: Final = "signlab-legacy-gru-compatibility/1"
REPORT_FORMAT: Final = "signlab-legacy-gru-compatibility-report/1"
CONFUSION_FORMAT: Final = "signlab-legacy-gru-validation-confusion/1"
TARGET_LABELS: Final[tuple[ExternalTargetLabel, ...]] = (
    "hello",
    "no",
    "please",
    "thank_you",
    "yes",
)
_SPLIT_QUOTAS: Final = {
    "train": Counter({label: 10 for label in TARGET_LABELS}),
    "validation": Counter({label: 3 for label in TARGET_LABELS}),
}
_CONFIG_SHA256: Final = "sha256:2358c943f7ec74d0f214864c934c62e1e83ecd736b63f793abad04e86b40928b"


class LegacyGruCompatibilityError(ValueError):
    """A stable, path-free failure from the optional compatibility boundary."""


class LegacyGruCompatibilityConfig(StrictContractModel):
    """The exact one-run choice set approved for Story #26."""

    format: Literal["signlab-legacy-gru-compatibility/1"]
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
    seed: SafeInteger
    legacy_run_id: Literal["20251222_154233_gru_phase_3_run_001"]
    legacy_model_sha256: Sha256Digest
    legacy_label_map_sha256: Sha256Digest
    legacy_config_sha256: Sha256Digest
    legacy_input_frames: PositiveSafeInteger
    legacy_declared_input_width: PositiveSafeInteger
    legacy_effective_input_width: PositiveSafeInteger
    input_frames: PositiveSafeInteger
    input_width: PositiveSafeInteger
    gru_layers: PositiveSafeInteger
    gru_units: PositiveSafeInteger
    dropout: FiniteFloat
    attention_pooling: bool
    parameter_count: PositiveSafeInteger
    optimizer: Literal["adam"]
    learning_rate: Annotated[FiniteFloat, Field(gt=0)]
    weight_decay: None
    batch_size: PositiveSafeInteger
    maximum_epochs: PositiveSafeInteger
    early_stopping_metric: Literal["val_accuracy"]
    early_stopping_patience: PositiveSafeInteger
    restore_best_weights: bool
    shuffle: bool
    onnx_opset: PositiveSafeInteger
    parity_absolute_tolerance: Annotated[FiniteFloat, Field(gt=0)]
    parity_relative_tolerance: Annotated[FiniteFloat, Field(gt=0)]


@dataclass(frozen=True, slots=True)
class LegacyGruCompatibilityResult:
    """Small receipt returned after the one run and its ledger are verified."""

    validation_macro_f1: float
    epochs_completed: int
    output_root: Path
    public_report_path: Path
    tracking: ReferenceRunReceipt


@dataclass(frozen=True, slots=True)
class _Runtime:
    keras: Any
    tensorflow: Any
    onnx: Any
    onnxruntime: Any


@dataclass(frozen=True, slots=True)
class _Training:
    epochs_completed: int
    best_epoch: int
    best_validation_accuracy: float
    best_validation_loss: float


@dataclass(frozen=True, slots=True)
class _Evaluation:
    predictions: tuple[ExternalTargetLabel, ...]
    probabilities: NDArray[np.float32]
    metrics: dict[str, Any]
    confusion: list[list[int]]


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def load_legacy_gru_compatibility_config(
    path: str | Path,
) -> tuple[LegacyGruCompatibilityConfig, bytes]:
    """Load only the canonical, LF-terminated reviewed configuration."""

    try:
        raw = Path(path).read_bytes()
        payload = parse_json_object(raw)
        config = LegacyGruCompatibilityConfig.model_validate_json(
            canonical_json_bytes(payload), strict=True
        )
        if raw != canonical_json_bytes(config) + b"\n" or _sha256(raw) != _CONFIG_SHA256:
            raise ValueError
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise LegacyGruCompatibilityError(
            "legacy GRU compatibility configuration is invalid"
        ) from error
    return config, raw


def _runtime() -> _Runtime:
    if os.environ.setdefault("KERAS_BACKEND", "tensorflow") != "tensorflow":
        raise LegacyGruCompatibilityError("Keras must use the TensorFlow backend")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    try:
        return _Runtime(
            keras=importlib.import_module("keras"),
            tensorflow=importlib.import_module("tensorflow"),
            onnx=importlib.import_module("onnx"),
            onnxruntime=importlib.import_module("onnxruntime"),
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise LegacyGruCompatibilityError(
            "install the SignLab legacy-compatibility extra"
        ) from error


def _seed_runtime(runtime: _Runtime, seed: int) -> None:
    try:
        runtime.keras.utils.set_random_seed(seed)
        runtime.tensorflow.config.experimental.enable_op_determinism()
    except Exception as error:
        raise LegacyGruCompatibilityError(
            "the compatibility runtime could not enable deterministic execution"
        ) from error


def _build_model(config: LegacyGruCompatibilityConfig, runtime: _Runtime) -> Any:
    try:
        layers = runtime.keras.layers
        inputs = runtime.keras.Input(
            shape=(config.input_frames, config.input_width), dtype="float32", name="input"
        )
        values = inputs
        for index in range(config.gru_layers):
            values = layers.GRU(
                config.gru_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=config.dropout,
                recurrent_dropout=0.0,
                return_sequences=True,
                stateful=False,
                reset_after=True,
                use_cudnn=False,
                name=f"causal_gru_{index}",
            )(values)
        scores = layers.Dense(1, activation="tanh", name="attn_scores")(values)
        weights = layers.Softmax(axis=1, name="attn_softmax")(scores)
        context = layers.Dot(axes=(1, 1), name="attn_reduce")([weights, values])
        features = layers.Reshape((config.gru_units,), name="features")(context)
        probabilities = layers.Dense(
            len(config.labels), activation="softmax", name="probabilities"
        )(features)
        model = runtime.keras.Model(inputs=inputs, outputs=probabilities, name="legacy_causal_gru")
        model.compile(
            optimizer=runtime.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=False,
        )
    except Exception as error:
        raise LegacyGruCompatibilityError("the legacy GRU graph could not be built") from error
    if int(model.count_params()) != config.parameter_count:
        raise LegacyGruCompatibilityError("the adapted legacy GRU parameter count drifted")
    return model


def _vectorize(
    samples: Sequence[PublicCorpusSample], config: LegacyGruCompatibilityConfig
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    plan = load_packaged_default_feature_plan("combined")
    expected_shape = (config.input_frames, config.input_width)
    label_index = {label: index for index, label in enumerate(config.labels)}
    values: list[tuple[tuple[int, ...], ...]] = []
    labels: list[int] = []
    for sample in samples:
        feature = sample.feature
        if (
            plan.plan_id != config.feature_plan_id
            or feature.feature_plan_sha256 != config.feature_plan_sha256
            or feature.feature_names != plan.feature_order
            or feature.quantization_scale != plan.quantization_scale
            or feature.statistics_sha256 is not None
            or (len(feature.values_q), len(feature.values_q[0])) != expected_shape
        ):
            raise LegacyGruCompatibilityError(
                "portable compatibility feature shape or identity drifted"
            )
        values.append(feature.values_q)
        labels.append(label_index[sample.target_label_id])
    if not values:
        raise LegacyGruCompatibilityError("compatibility partition is empty")
    matrix = np.asarray(values, dtype=np.float32)
    matrix /= np.float32(plan.quantization_scale)
    return np.ascontiguousarray(matrix), np.asarray(labels, dtype=np.int64)


def _train_model(
    model: Any,
    train_x: NDArray[np.float32],
    train_y: NDArray[np.int64],
    validation_x: NDArray[np.float32],
    validation_y: NDArray[np.int64],
    config: LegacyGruCompatibilityConfig,
    runtime: _Runtime,
) -> _Training:
    try:
        early_stopping = runtime.keras.callbacks.EarlyStopping(
            monitor=config.early_stopping_metric,
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=config.restore_best_weights,
        )
        history = model.fit(
            train_x,
            train_y,
            validation_data=(validation_x, validation_y),
            batch_size=config.batch_size,
            epochs=config.maximum_epochs,
            callbacks=[early_stopping],
            shuffle=config.shuffle,
            verbose=0,
        ).history
        validation_accuracy = [float(value) for value in history["val_accuracy"]]
        validation_loss = [float(value) for value in history["val_loss"]]
    except Exception as error:
        raise LegacyGruCompatibilityError("the one compatibility fit failed") from error
    if (
        not validation_accuracy
        or len(validation_accuracy) != len(validation_loss)
        or len(validation_accuracy) > config.maximum_epochs
    ):
        raise LegacyGruCompatibilityError("the compatibility training history is invalid")
    best_index = int(np.argmax(np.asarray(validation_accuracy)))
    return _Training(
        epochs_completed=len(validation_accuracy),
        best_epoch=best_index + 1,
        best_validation_accuracy=validation_accuracy[best_index],
        best_validation_loss=validation_loss[best_index],
    )


def _predict_probabilities(model: Any, matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    try:
        values = np.asarray(model.predict(matrix, batch_size=1, verbose=0), dtype=np.float32)
    except Exception as error:
        raise LegacyGruCompatibilityError("Keras validation prediction failed") from error
    if values.shape != (len(matrix), len(TARGET_LABELS)) or not np.isfinite(values).all():
        raise LegacyGruCompatibilityError("Keras validation output is invalid")
    return values


def _save_model(model: Any, path: Path) -> bytes:
    try:
        model.save(path)
        return path.read_bytes()
    except Exception as error:
        raise LegacyGruCompatibilityError(
            "the Keras compatibility artifact could not be saved"
        ) from error


def _export_and_compare(
    model: Any,
    validation_x: NDArray[np.float32],
    keras_probabilities: NDArray[np.float32],
    path: Path,
    config: LegacyGruCompatibilityConfig,
    runtime: _Runtime,
) -> tuple[bytes, dict[str, Any]]:
    try:
        signature = [
            runtime.keras.InputSpec(
                shape=(1, config.input_frames, config.input_width),
                dtype="float32",
                name="input",
            )
        ]
        model.export(
            path,
            format="onnx",
            input_signature=signature,
            opset_version=config.onnx_opset,
            verbose=False,
        )
        runtime.onnx.checker.check_model(str(path), full_check=True)
        session = runtime.onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        input_metadata = session.get_inputs()
        if len(input_metadata) != 1 or list(input_metadata[0].shape) != [
            1,
            config.input_frames,
            config.input_width,
        ]:
            raise LegacyGruCompatibilityError("the ONNX input signature drifted")
        converted = np.concatenate(
            [
                np.asarray(
                    session.run(None, {input_metadata[0].name: row[None, ...]})[0],
                    dtype=np.float32,
                )
                for row in validation_x
            ],
            axis=0,
        )
    except LegacyGruCompatibilityError:
        raise
    except Exception as error:
        raise LegacyGruCompatibilityError("ONNX export or CPU validation failed") from error
    if converted.shape != keras_probabilities.shape or not np.isfinite(converted).all():
        raise LegacyGruCompatibilityError("ONNX validation output is invalid")
    maximum_absolute_difference = float(np.max(np.abs(keras_probabilities - converted)))
    labels_match = bool(
        np.array_equal(np.argmax(keras_probabilities, axis=1), np.argmax(converted, axis=1))
    )
    values_match = bool(
        np.allclose(
            keras_probabilities,
            converted,
            atol=config.parity_absolute_tolerance,
            rtol=config.parity_relative_tolerance,
        )
    )
    if not values_match or not labels_match:
        raise LegacyGruCompatibilityError("Keras and ONNX validation outputs do not match")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise LegacyGruCompatibilityError("the ONNX artifact could not be read") from error
    return payload, {
        "all_outputs_within_tolerance": True,
        "compared_examples": len(validation_x),
        "identical_predicted_labels": True,
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def _evaluate(expected: NDArray[np.int64], probabilities: NDArray[np.float32]) -> _Evaluation:
    predicted = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=np.int64)
    for actual, guess in zip(expected.tolist(), predicted.tolist(), strict=True):
        confusion[actual, guess] += 1
    per_class: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, label in enumerate(TARGET_LABELS):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "support": support}
        f1_values.append(f1)
        recalls.append(recall)
    return _Evaluation(
        predictions=tuple(TARGET_LABELS[index] for index in predicted.tolist()),
        probabilities=probabilities,
        metrics={
            "accuracy": float(np.mean(predicted == expected)),
            "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1_values)),
            "per_class": per_class,
        },
        confusion=confusion.tolist(),
    )


def _predictions_csv(samples: Sequence[PublicCorpusSample], evaluation: _Evaluation) -> bytes:
    signer_aliases = {
        signer: f"signer_{index:03d}"
        for index, signer in enumerate(
            sorted({sample.source_signer_id for sample in samples}), start=1
        )
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("partition", "sample_alias", "signer_alias", "quality", "actual", "predicted"))
    for index, (sample, predicted) in enumerate(
        zip(samples, evaluation.predictions, strict=True), start=1
    ):
        writer.writerow(
            (
                "validation",
                f"sample_{index:03d}",
                signer_aliases[sample.source_signer_id],
                sample.quality_disposition,
                sample.target_label_id,
                predicted,
            )
        )
    return stream.getvalue().encode("utf-8")


def _public_markdown(report: Mapping[str, Any]) -> bytes:
    metrics = cast(dict[str, Any], report["validation"])["metrics"]
    parity = cast(dict[str, Any], report["onnx"])["parity"]
    lines = [
        "# PopSign legacy GRU compatibility v1",
        "",
        (
            "> Compatibility/export smoke only. This 50-train, 15-validation run is not "
            "model-quality, product-performance, or sign-language evidence."
        ),
        "",
        "## What was reproduced",
        "",
        f"- Recovered legacy run: `{report['legacy']['run_id']}`",
        f"- Recovered model: `{report['legacy']['model_sha256']}`",
        (
            "- Same functional architecture: two forward GRU layers with 128 units, "
            "attention pooling, and a five-class softmax."
        ),
        (
            "- Required input adaptation: legacy `30 x 126` effective tensors became "
            "current-contract `64 x 134` tensors."
        ),
        "- The legacy config's `input_dim=63` did not match its saved model's width of 126.",
        (
            "- Attention reduction uses an exporter-friendly dot product equivalent "
            "to the legacy weighted sum."
        ),
        (
            "- The legacy Adam run did not actually apply its declared weight decay; "
            "this run does not add it."
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
        f"- Seed: `{report['training']['seed']}`",
        "- Opened data: 50 training clips and 15 validation clips; final test stayed sealed.",
        "",
        "## Result",
        "",
        f"- Epochs completed: `{report['training']['epochs_completed']}`",
        f"- Best epoch: `{report['training']['best_epoch']}`",
        f"- Validation accuracy: `{metrics['accuracy']:.3f}`",
        f"- Validation loss at the best epoch: `{report['training']['best_validation_loss']:.3f}`",
        f"- Validation macro-F1: `{metrics['macro_f1']:.3f}`",
        f"- Keras artifact: `{report['keras']['sha256']}`",
        f"- ONNX artifact: `{report['onnx']['sha256']}`",
        (
            f"- ONNX parity: all {parity['compared_examples']} validation outputs passed "
            "at `1e-5`; maximum absolute difference "
            f"`{parity['maximum_absolute_difference']:.3g}`; "
            "predicted labels were identical."
        ),
        "",
        "## Decision",
        "",
        (
            "The outcome and exporter risk are recorded in "
            "[ADR 0001](../decisions/0001-training-framework.md)."
        ),
        "",
        "## Reproduce",
        "",
        "```shell",
        (
            "uv run --locked --extra legacy-compatibility signlab train "
            "legacy-gru-compatibility configs/experiments/popsign-legacy-gru-compatibility-v1.json "
            "--corpus-root <frozen-split-root> "
            "--external-manifest <external-dataset-manifest.json> "
            "--output-root runs/popsign-legacy-gru-compatibility-v1"
        ),
        "```",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _new_path(path: str | Path, *, label: str) -> Path:
    target = Path(path).resolve()
    if target.exists() or target.is_symlink():
        raise LegacyGruCompatibilityError(f"{label} must not already exist")
    return target


def _write_new(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise LegacyGruCompatibilityError(
            "legacy GRU compatibility artifacts could not be written"
        ) from error


def run_legacy_gru_compatibility(
    config_path: str | Path,
    *,
    corpus_root: str | Path,
    external_manifest_path: str | Path,
    output_root: str | Path,
    public_report_path: str | Path | None = None,
    tracking_uri: str | None = None,
) -> LegacyGruCompatibilityResult:
    """Train once on train, compare Keras/ONNX on validation, and never open test."""

    config, config_bytes = load_legacy_gru_compatibility_config(config_path)
    destination = _new_path(output_root, label="compatibility output")
    human_report = (
        _new_path(public_report_path, label="public compatibility report")
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
        raise LegacyGruCompatibilityError("frozen compatibility inputs are invalid") from error
    train_samples = _partition(development, "train")
    validation_samples = _partition(development, "validation")
    if (
        Counter(sample.target_label_id for sample in train_samples) != _SPLIT_QUOTAS["train"]
        or Counter(sample.target_label_id for sample in validation_samples)
        != _SPLIT_QUOTAS["validation"]
    ):
        raise LegacyGruCompatibilityError("compatibility partition quotas drifted")
    train_x, train_y = _vectorize(train_samples, config)
    validation_x, validation_y = _vectorize(validation_samples, config)

    runtime = _runtime()
    _seed_runtime(runtime, config.seed)
    model = _build_model(config, runtime)
    training = _train_model(model, train_x, train_y, validation_x, validation_y, config, runtime)
    keras_probabilities = _predict_probabilities(model, validation_x)
    evaluation = _evaluate(validation_y, keras_probabilities)
    with TemporaryDirectory(prefix="signlab-legacy-gru-") as temporary_text:
        temporary = Path(temporary_text)
        keras_bytes = _save_model(model, temporary / "model.keras")
        onnx_bytes, parity = _export_and_compare(
            model,
            validation_x,
            keras_probabilities,
            temporary / "model.onnx",
            config,
            runtime,
        )

    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "run_name": config.run_name,
        "claim_scope": "compatibility_and_export_smoke_only",
        "source": {"git_commit": git_commit, "git_dirty": git_dirty},
        "identities": {
            "configuration_sha256": _sha256(config_bytes),
            "corpus_sha256": config.corpus_sha256,
            "external_dataset_sha256": config.external_dataset_sha256,
            "split_sha256": config.split_sha256,
            "feature_plan_id": config.feature_plan_id,
            "feature_plan_sha256": config.feature_plan_sha256,
            "taxonomy_id": config.taxonomy_id,
            "taxonomy_version": config.taxonomy_version,
            "taxonomy_sha256": config.taxonomy_sha256,
        },
        "legacy": {
            "run_id": config.legacy_run_id,
            "model_sha256": config.legacy_model_sha256,
            "label_map_sha256": config.legacy_label_map_sha256,
            "config_sha256": config.legacy_config_sha256,
            "input_shape": [config.legacy_input_frames, config.legacy_effective_input_width],
            "declared_input_width": config.legacy_declared_input_width,
            "parameter_count": 198_150,
            "effective_weight_decay": None,
        },
        "adaptation": {
            "input_shape": [config.input_frames, config.input_width],
            "parameter_count": config.parameter_count,
            "label_identifier_normalization": {"thank you": "thank_you"},
        },
        "data": {
            "labels": list(config.labels),
            "opened_partition_counts": {
                "train": len(train_samples),
                "validation": len(validation_samples),
            },
            "test_status": "sealed_not_opened",
        },
        "training": {
            "seed": config.seed,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "maximum_epochs": config.maximum_epochs,
            "epochs_completed": training.epochs_completed,
            "best_epoch": training.best_epoch,
            "best_recorded_validation_accuracy": training.best_validation_accuracy,
            "best_validation_loss": training.best_validation_loss,
            "early_stopping_patience": config.early_stopping_patience,
            "restore_best_weights": config.restore_best_weights,
            "shuffle": config.shuffle,
            "fit_calls": 1,
        },
        "validation": {
            "example_count": len(validation_samples),
            "metrics": evaluation.metrics,
        },
        "keras": {"sha256": _sha256(keras_bytes), "bytes": len(keras_bytes)},
        "onnx": {
            "sha256": _sha256(onnx_bytes),
            "bytes": len(onnx_bytes),
            "opset": config.onnx_opset,
            "fixed_input_shape": [1, config.input_frames, config.input_width],
            "checker_full_check": True,
            "provider": "CPUExecutionProvider",
            "parity": parity,
        },
        "environment": {
            "architecture": platform.machine().casefold() or "unknown",
            "keras": version("keras"),
            "numpy": version("numpy"),
            "onnx": version("onnx"),
            "onnxruntime": version("onnxruntime"),
            "os": platform.system().casefold() or "unknown",
            "python": platform.python_version(),
            "tensorflow": version("tensorflow"),
            "tf2onnx": version("tf2onnx"),
        },
        "limitations": [
            "Only 50 training and 15 validation clips were used, with one split and one seed.",
            "The final test partition was deliberately not opened.",
            "The run checks Keras training and ONNX export compatibility, not model quality.",
            "The TensorFlow ONNX export path depends on tf2onnx and remains a maintenance risk.",
        ],
    }
    confusion_bytes = (
        canonical_json_bytes(
            {
                "format": CONFUSION_FORMAT,
                "labels": list(config.labels),
                "orientation": {"rows": "actual", "columns": "predicted"},
                "validation": evaluation.confusion,
            }
        )
        + b"\n"
    )
    report_bytes = canonical_json_bytes(report) + b"\n"
    predictions_bytes = _predictions_csv(validation_samples, evaluation)
    public_bytes = _public_markdown(report)

    try:
        destination.mkdir(parents=True)
    except OSError as error:
        raise LegacyGruCompatibilityError("compatibility output could not be created") from error
    paths = {
        "configuration_path": destination / "configuration.json",
        "report_path": destination / "report.json",
        "confusion_matrix_path": destination / "validation-confusion-matrix.json",
        "predictions_path": destination / "validation-predictions.csv",
    }
    for path, payload in (
        (paths["configuration_path"], config_bytes),
        (paths["report_path"], report_bytes),
        (paths["confusion_matrix_path"], confusion_bytes),
        (paths["predictions_path"], predictions_bytes),
        (destination / "model.keras", keras_bytes),
        (destination / "model.onnx", onnx_bytes),
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
            "fit_calls": 1,
            "gru_layers": config.gru_layers,
            "gru_units": config.gru_units,
            "input_frames": config.input_frames,
            "input_width": config.input_width,
            "maximum_epochs": config.maximum_epochs,
            "onnx_opset": config.onnx_opset,
            "test_status": "sealed_not_opened",
        },
        metrics={
            "validation.accuracy": float(evaluation.metrics["accuracy"]),
            "validation.balanced_accuracy": float(evaluation.metrics["balanced_accuracy"]),
            "validation.loss": training.best_validation_loss,
            "validation.macro_f1": float(evaluation.metrics["macro_f1"]),
            "validation.onnx_maximum_absolute_difference": float(
                parity["maximum_absolute_difference"]
            ),
        },
        **paths,
    )
    tracking = log_reference_run(run_input, tracking_uri=tracking_uri)
    if verify_reference_run(tracking.run_id, tracking_uri=tracking_uri) != tracking:
        raise LegacyGruCompatibilityError("compatibility ledger verification drifted")
    _write_new(human_report, public_bytes)
    return LegacyGruCompatibilityResult(
        validation_macro_f1=float(evaluation.metrics["macro_f1"]),
        epochs_completed=training.epochs_completed,
        output_root=destination,
        public_report_path=human_report,
        tracking=tracking,
    )


__all__ = [
    "CONFIG_FORMAT",
    "CONFUSION_FORMAT",
    "REPORT_FORMAT",
    "TARGET_LABELS",
    "LegacyGruCompatibilityConfig",
    "LegacyGruCompatibilityError",
    "LegacyGruCompatibilityResult",
    "load_legacy_gru_compatibility_config",
    "run_legacy_gru_compatibility",
]
