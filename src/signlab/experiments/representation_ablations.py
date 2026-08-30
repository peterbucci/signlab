"""One bounded grouped-development ablation for registered feature views."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import platform
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, model_validator
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from signlab.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    FiniteFloat,
    PositiveSafeInteger,
    SafeInteger,
    StableId,
    StrictContractModel,
)
from signlab.contracts.external_dataset import ExternalTargetLabel
from signlab.contracts.features import FeatureRepresentation, landmark_feature_plan_digest
from signlab.contracts.taxonomy import Sha256Digest
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments.baselines import (
    BaselineExperimentError,
    _git_identity,
    _load_inputs,
    _load_samples,
)
from signlab.experiments.baselines import _measure_latency as _measure_logistic_latency
from signlab.experiments.sequence_baselines import (
    SequenceBaselineError,
    _aliases,
    _build_graph,
    _evaluate,
    _Evaluation,
    _predict_probabilities,
    _Runtime,
    _runtime,
    _seed_runtime,
)
from signlab.experiments.sequence_baselines import _measure_latency as _measure_deep_latency
from signlab.experiments.tracking import (
    ReferenceRunInput,
    ReferenceRunReceipt,
    log_reference_run,
    verify_reference_run,
)
from signlab.features.resources import load_packaged_default_feature_plan

type AblationModel = Literal["logistic", "gru", "tcn"]
type ComparisonName = Literal["body_context", "architecture"]

TARGET_LABELS: Final[tuple[ExternalTargetLabel, ...]] = (
    "hello",
    "no",
    "please",
    "thank_you",
    "yes",
)
REPRESENTATIONS: Final[tuple[FeatureRepresentation, ...]] = (
    "hand_local",
    "body_relative",
    "combined",
)
EXPECTED_CONDITIONS: Final[tuple[tuple[AblationModel, FeatureRepresentation], ...]] = (
    ("logistic", "hand_local"),
    ("logistic", "body_relative"),
    ("logistic", "combined"),
    ("gru", "combined"),
    ("tcn", "hand_local"),
    ("tcn", "combined"),
)
EXPECTED_COMPARISONS: Final[dict[ComparisonName, tuple[StableId, StableId]]] = {
    "body_context": ("tcn_hand_local", "tcn_combined"),
    "architecture": ("gru_combined", "tcn_combined"),
}
CONFIG_FORMAT: Final = "signlab-representation-ablations/1"
REPORT_FORMAT: Final = "signlab-representation-ablation-report/1"
CONFUSION_FORMAT: Final = "signlab-representation-ablation-confusions/1"
_FOLD_DOMAIN: Final = "signlab-representation-ablation-folds/1"


class RepresentationAblationError(ValueError):
    """A stable, path-free ablation failure."""


class FeatureViewConfig(StrictContractModel):
    plan_id: StableId
    plan_sha256: Sha256Digest
    input_width: PositiveSafeInteger


class AblationCondition(StrictContractModel):
    model: AblationModel
    representation: FeatureRepresentation


class RepresentationAblationConfig(StrictContractModel):
    """The frozen Story #30 matrix and budget."""

    format: Literal["signlab-representation-ablations/1"]
    run_name: StableId
    corpus_sha256: Sha256Digest
    external_dataset_sha256: Sha256Digest
    split_id: Literal["popsign-five-isolated-smoke-v1"]
    split_sha256: Sha256Digest
    source_feature_plan_id: Literal["combined_64_frames"]
    source_feature_plan_sha256: Sha256Digest
    taxonomy_id: Literal["signlab-five"]
    taxonomy_version: Literal["1.0.0"]
    taxonomy_sha256: Sha256Digest
    labels: Annotated[tuple[ExternalTargetLabel, ...], Field(min_length=5, max_length=5)]
    feature_views: dict[FeatureRepresentation, FeatureViewConfig]
    conditions: Annotated[tuple[AblationCondition, ...], Field(min_length=6, max_length=6)]
    comparisons: dict[ComparisonName, tuple[StableId, StableId]]
    fold_count: Literal[3]
    fold_seed: SafeInteger
    fold_strategy: Literal["stratified_group_kfold_by_signer_v1"]
    vectorizer: Literal["registered_channel_projection_v1"]
    input_frames: Literal[64]
    logistic_c: FiniteFloat
    logistic_solver: Literal["lbfgs"]
    logistic_tolerance: Annotated[FiniteFloat, Field(gt=0)]
    logistic_max_iterations: Annotated[PositiveSafeInteger, Field(le=10000)]
    model_seed: SafeInteger
    optimizer: Literal["adam"]
    learning_rate: Annotated[FiniteFloat, Field(gt=0)]
    batch_size: PositiveSafeInteger
    maximum_epochs: PositiveSafeInteger
    fixed_epoch_rule: Literal["train_exact_epochs_report_final_weights"]
    shuffle: Literal[True]
    gru_units: PositiveSafeInteger
    tcn_channels: PositiveSafeInteger
    tcn_dilations: Annotated[tuple[PositiveSafeInteger, ...], Field(min_length=4, max_length=4)]
    tcn_kernel_size: PositiveSafeInteger
    latency_warmup_runs: PositiveSafeInteger
    latency_measurement_runs: PositiveSafeInteger
    practical_delta_threshold: Annotated[FiniteFloat, Field(gt=0, lt=1)]
    test_partition_policy: Literal["sealed_not_loaded"]

    @model_validator(mode="after")
    def _freeze_design(self) -> Self:
        rows = tuple((row.model, row.representation) for row in self.conditions)
        views = {
            "hand_local": ("hand_local_64_frames", 126),
            "body_relative": ("body_relative_64_frames", 8),
            "combined": ("combined_64_frames", 134),
        }
        if (
            self.labels != TARGET_LABELS
            or rows != EXPECTED_CONDITIONS
            or self.comparisons != EXPECTED_COMPARISONS
        ):
            raise ValueError("ablation matrix drifted")
        if set(self.feature_views) != set(REPRESENTATIONS) or any(
            (self.feature_views[name].plan_id, self.feature_views[name].input_width) != views[name]
            for name in REPRESENTATIONS
        ):
            raise ValueError("ablation views drifted")
        frozen = (
            self.fold_seed,
            self.logistic_c,
            self.model_seed,
            self.learning_rate,
            self.maximum_epochs,
            self.gru_units,
            self.tcn_channels,
            self.tcn_dilations,
            self.tcn_kernel_size,
            self.practical_delta_threshold,
        )
        if frozen != (20260830, 0.1, 20260830, 0.001, 30, 48, 32, (1, 2, 4, 8), 3, 0.05):
            raise ValueError("ablation protocol drifted")
        return self

    @property
    def feature_plan_id(self) -> str:
        return self.source_feature_plan_id

    @property
    def feature_plan_sha256(self) -> str:
        return self.source_feature_plan_sha256


@dataclass(frozen=True, slots=True)
class RepresentationAblationResult:
    output_root: Path
    public_report_path: Path
    tracking: ReferenceRunReceipt


def _ablation_runtime() -> _Runtime:
    try:
        return _runtime()
    except SequenceBaselineError as error:
        raise RepresentationAblationError("neural ablation runtime is unavailable") from error


@dataclass(frozen=True, slots=True)
class _Fold:
    number: int
    fit: tuple[int, ...]
    evaluate: tuple[int, ...]
    fit_signers: int
    evaluation_signers: int


@dataclass(frozen=True, slots=True)
class _Cell:
    evaluation: _Evaluation
    parameters: int
    p50_ms: float
    p95_ms: float


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _name(model: AblationModel, representation: FeatureRepresentation) -> str:
    return f"{model}_{representation}"


def load_representation_ablation_config(
    path: str | Path,
) -> tuple[RepresentationAblationConfig, bytes]:
    try:
        raw = Path(path).read_bytes()
        config = RepresentationAblationConfig.model_validate_json(
            canonical_json_bytes(parse_json_object(raw)), strict=True
        )
        if raw != canonical_json_bytes(config) + b"\n":
            raise ValueError
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise RepresentationAblationError("representation-ablation config is invalid") from error
    return config, raw


def _views(
    samples: Sequence[PublicCorpusSample], config: RepresentationAblationConfig
) -> tuple[dict[FeatureRepresentation, NDArray[np.float32]], NDArray[np.int64]]:
    source = load_packaged_default_feature_plan("combined")
    if (
        source.plan_id != config.source_feature_plan_id
        or landmark_feature_plan_digest(source) != config.source_feature_plan_sha256
    ):
        raise RepresentationAblationError("source feature identity drifted")
    indexes: dict[FeatureRepresentation, tuple[int, ...]] = {}
    source_index = {name: index for index, name in enumerate(source.feature_order)}
    for name in REPRESENTATIONS:
        plan = load_packaged_default_feature_plan(name)
        declared = config.feature_views[name]
        if (
            plan.plan_id != declared.plan_id
            or landmark_feature_plan_digest(plan) != declared.plan_sha256
            or len(plan.feature_order) != declared.input_width
        ):
            raise RepresentationAblationError("registered feature-view identity drifted")
        indexes[name] = tuple(source_index[feature] for feature in plan.feature_order)
    rows: list[tuple[tuple[int, ...], ...]] = []
    labels: list[int] = []
    label_index = {label: index for index, label in enumerate(TARGET_LABELS)}
    for sample in samples:
        feature = sample.feature
        if (
            feature.feature_plan_sha256 != config.source_feature_plan_sha256
            or feature.feature_names != source.feature_order
            or feature.quantization_scale != source.quantization_scale
            or feature.statistics_sha256 is not None
            or (len(feature.values_q), len(feature.values_q[0])) != (64, 134)
        ):
            raise RepresentationAblationError("portable feature identity drifted")
        rows.append(feature.values_q)
        labels.append(label_index[sample.target_label_id])
    combined = np.asarray(rows, dtype=np.float32) / np.float32(source.quantization_scale)
    return (
        {name: np.ascontiguousarray(combined[:, :, indexes[name]]) for name in REPRESENTATIONS},
        np.asarray(labels, dtype=np.int64),
    )


def _folds(
    samples: Sequence[PublicCorpusSample],
    labels: NDArray[np.int64],
    config: RepresentationAblationConfig,
) -> tuple[tuple[_Fold, ...], str]:
    groups = np.asarray([sample.source_signer_id for sample in samples], dtype=np.str_)
    splitter = StratifiedGroupKFold(
        n_splits=config.fold_count, shuffle=True, random_state=config.fold_seed
    )
    folds: list[_Fold] = []
    assigned: dict[int, int] = {}
    for number, (fit_raw, evaluation_raw) in enumerate(
        splitter.split(np.zeros(len(samples)), labels, groups), start=1
    ):
        fit = tuple(int(value) for value in fit_raw)
        evaluate = tuple(int(value) for value in evaluation_raw)
        fit_groups = set(groups[list(fit)])
        evaluation_groups = set(groups[list(evaluate)])
        fit_labels = set(labels[list(fit)])
        evaluation_labels = set(labels[list(evaluate)])
        expected_labels = set(range(len(TARGET_LABELS)))
        if (
            fit_groups & evaluation_groups
            or fit_labels != expected_labels
            or evaluation_labels != expected_labels
        ):
            raise RepresentationAblationError("grouped fold is invalid")
        for index in evaluate:
            if index in assigned:
                raise RepresentationAblationError("grouped folds overlap")
            assigned[index] = number
        folds.append(_Fold(number, fit, evaluate, len(fit_groups), len(evaluation_groups)))
    if set(assigned) != set(range(len(samples))):
        raise RepresentationAblationError("grouped folds are incomplete")
    digest = canonical_sha256(
        {
            "strategy": config.fold_strategy,
            "seed": config.fold_seed,
            "assignment": [
                {
                    "sample": sample.sample_id,
                    "signer": sample.source_signer_id,
                    "fold": assigned[index],
                }
                for index, sample in enumerate(samples)
            ],
        },
        domain=_FOLD_DOMAIN,
    )
    return tuple(folds), digest


def _fit_cell(
    model_name: AblationModel,
    train_x: NDArray[np.float32],
    train_y: NDArray[np.int64],
    evaluation_x: NDArray[np.float32],
    evaluation_y: NDArray[np.int64],
    config: RepresentationAblationConfig,
    runtime: _Runtime,
) -> _Cell:
    if model_name == "logistic":
        fit_x = np.ascontiguousarray(train_x.reshape(len(train_x), -1), dtype=np.float64)
        score_x = np.ascontiguousarray(
            evaluation_x.reshape(len(evaluation_x), -1), dtype=np.float64
        )
        model = LogisticRegression(
            C=config.logistic_c,
            solver=config.logistic_solver,
            tol=config.logistic_tolerance,
            max_iter=config.logistic_max_iterations,
            random_state=config.model_seed,
        )
        try:
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(fit_x, train_y)
                probabilities = np.asarray(model.predict_proba(score_x), dtype=np.float32)
            parameters = int(model.coef_.size + model.intercept_.size)
            latency = _measure_logistic_latency(
                model,
                score_x,
                warmup_runs=config.latency_warmup_runs,
                measurement_runs=config.latency_measurement_runs,
            )
        except (ConvergenceWarning, FloatingPointError, ValueError) as error:
            raise RepresentationAblationError("fixed logistic fit failed") from error
    else:
        try:
            runtime.keras.backend.clear_session()
            _seed_runtime(runtime, config.model_seed)
            model = _build_graph(
                cast(Any, model_name), train_x.shape[2], cast(Any, config), runtime
            )
            history = model.fit(
                train_x,
                train_y,
                batch_size=config.batch_size,
                epochs=config.maximum_epochs,
                shuffle=config.shuffle,
                verbose=0,
            ).history
            losses = [float(value) for value in history["loss"]]
            if len(losses) != config.maximum_epochs or not all(map(math.isfinite, losses)):
                raise ValueError
            probabilities = _predict_probabilities(model, evaluation_x)
            parameters = int(model.count_params())
            latency = _measure_deep_latency(
                model,
                evaluation_x,
                warmup_runs=config.latency_warmup_runs,
                measurement_runs=config.latency_measurement_runs,
            )
        except Exception as error:
            raise RepresentationAblationError("fixed exact-epoch neural fit failed") from error
    return _Cell(
        _evaluate(evaluation_y, probabilities),
        parameters,
        float(latency["p50_ms"]),
        float(latency["p95_ms"]),
    )


def _balanced(evaluation: _Evaluation) -> float:
    recalls = [
        row[index] / sum(row) if sum(row) else 0.0 for index, row in enumerate(evaluation.confusion)
    ]
    return float(np.mean(recalls))


def _metrics(evaluation: _Evaluation) -> dict[str, Any]:
    return {**evaluation.metrics, "balanced_accuracy": _balanced(evaluation)}


def _summarize(
    model: AblationModel,
    representation: FeatureRepresentation,
    samples: Sequence[PublicCorpusSample],
    labels: NDArray[np.int64],
    folds: Sequence[_Fold],
    cells: Sequence[_Cell],
    signer_aliases: Mapping[str, str],
) -> tuple[dict[str, Any], _Evaluation]:
    probabilities = np.zeros((len(samples), len(TARGET_LABELS)), dtype=np.float32)
    for fold, cell in zip(folds, cells, strict=True):
        probabilities[list(fold.evaluate)] = cell.evaluation.probabilities
    pooled = _evaluate(labels, probabilities)
    per_signer: dict[str, dict[str, int]] = {}
    for sample, predicted in zip(samples, pooled.predictions, strict=True):
        alias = signer_aliases[sample.source_signer_id]
        row = per_signer.setdefault(alias, {"support": 0, "correct": 0, "errors": 0})
        row["support"] += 1
        row["correct" if predicted == sample.target_label_id else "errors"] += 1
    parameters = {cell.parameters for cell in cells}
    if len(parameters) != 1:
        raise RepresentationAblationError("parameter count changed between folds")
    report = {
        "model": model,
        "representation": representation,
        "parameter_count": parameters.pop(),
        "out_of_fold": _metrics(pooled),
        "folds": [
            {
                "fold": fold.number,
                "fit_samples": len(fold.fit),
                "evaluation_samples": len(fold.evaluate),
                "fit_signers": fold.fit_signers,
                "evaluation_signers": fold.evaluation_signers,
                "metrics": _metrics(cell.evaluation),
            }
            for fold, cell in zip(folds, cells, strict=True)
        ],
        "latency": {
            "scope": "median_of_three_prevectorized_fold_models_batch_1_cpu",
            "p50_ms": float(np.median([cell.p50_ms for cell in cells])),
            "p95_ms": float(np.median([cell.p95_ms for cell in cells])),
        },
        "per_signer": dict(sorted(per_signer.items())),
    }
    return report, pooled


def _compare(
    reports: Mapping[str, dict[str, Any]],
    reference: str,
    candidate: str,
    threshold: float,
) -> dict[str, Any]:
    fold_deltas = [
        float(candidate_fold["metrics"]["macro_f1"]) - float(reference_fold["metrics"]["macro_f1"])
        for reference_fold, candidate_fold in zip(
            reports[reference]["folds"], reports[candidate]["folds"], strict=True
        )
    ]
    pooled = float(reports[candidate]["out_of_fold"]["macro_f1"]) - float(
        reports[reference]["out_of_fold"]["macro_f1"]
    )
    if all(delta > 0 for delta in fold_deltas):
        decision = "supported_for_carry_forward" if pooled >= threshold else "unsupported"
    elif all(delta < 0 for delta in fold_deltas):
        decision = "unsupported"
    else:
        decision = "inconclusive"
    return {
        "reference": reference,
        "candidate": candidate,
        "pooled_macro_f1_delta": pooled,
        "fold_macro_f1_deltas": fold_deltas,
        "decision": decision,
    }


def _predictions(
    samples: Sequence[PublicCorpusSample],
    folds: Sequence[_Fold],
    evaluations: Mapping[str, _Evaluation],
    sample_aliases: Sequence[str],
    signer_aliases: Mapping[str, str],
) -> bytes:
    fold_for = {index: fold.number for fold in folds for index in fold.evaluate}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "condition",
            "fold",
            "sample_alias",
            "signer_alias",
            "actual",
            "predicted",
            "correct",
        )
    )
    for name, evaluation in evaluations.items():
        for index, (sample, alias, predicted) in enumerate(
            zip(samples, sample_aliases, evaluation.predictions, strict=True)
        ):
            writer.writerow(
                (
                    name,
                    fold_for[index],
                    alias,
                    signer_aliases[sample.source_signer_id],
                    sample.target_label_id,
                    predicted,
                    str(predicted == sample.target_label_id).lower(),
                )
            )
    return stream.getvalue().encode()


def _markdown(report: Mapping[str, Any], run_id: str) -> bytes:
    lines = [
        "# PopSign grouped development ablation v1",
        "",
        "> Development evidence only; no model winner or test result.",
        "",
        f"- Development-fold identity: `{report['identities']['development_fold_sha256']}`",
        f"- Verified local MLflow run: `{run_id}`",
        f"- Test status: `{report['data']['test_status']}`",
        "",
        (
            "| Model | View | OOF macro-F1 | Balanced accuracy | Fold macro-F1 | "
            "Params | CPU p50/p95 ms |"
        ),
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for condition in report["conditions"].values():
        folds = ", ".join(f"{row['metrics']['macro_f1']:.3f}" for row in condition["folds"])
        lines.append(
            f"| {condition['model'].upper()} | {condition['representation'].replace('_', ' ')} | "
            f"{condition['out_of_fold']['macro_f1']:.3f} | "
            f"{condition['out_of_fold']['balanced_accuracy']:.3f} | {folds} | "
            f"{condition['parameter_count']:,} | {condition['latency']['p50_ms']:.3f}/"
            f"{condition['latency']['p95_ms']:.3f} |"
        )
    lines.extend(["", "## Pre-registered comparisons", ""])
    for name, comparison in report["comparisons"].items():
        deltas = ", ".join(f"{value:+.3f}" for value in comparison["fold_macro_f1_deltas"])
        lines.append(
            f"- **{name.replace('_', ' ').title()}:** pooled delta "
            f"{comparison['pooled_macro_f1_delta']:+.3f}; folds {deltas}; "
            f"{comparison['decision'].replace('_', ' ')}."
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {text}" for text in report["limitations"])
    lines.extend(
        [
            "",
            (
                "Sanitized per-signer counts and concrete out-of-fold errors are in "
                "the evidence artifacts."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode()


def _write(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise RepresentationAblationError("ablation artifacts could not be written") from error


def run_representation_ablations(
    config_path: str | Path,
    *,
    corpus_root: str | Path,
    external_manifest_path: str | Path,
    output_root: str | Path,
    public_report_path: str | Path | None = None,
    tracking_uri: str | None = None,
) -> RepresentationAblationResult:
    """Run exactly six rows across three signer-grouped development folds."""

    config, config_bytes = load_representation_ablation_config(config_path)
    destination = Path(output_root).resolve()
    public_report = (
        Path(public_report_path).resolve()
        if public_report_path is not None
        else destination / "public-report.md"
    )
    if destination.exists() or public_report.exists():
        raise RepresentationAblationError("ablation output must not already exist")
    try:
        _, commit, dirty = _git_identity(Path(config_path).resolve(strict=True).parent)
        split, manifest = _load_inputs(
            cast(Any, config), Path(corpus_root), Path(external_manifest_path)
        )
        loaded = _load_samples(split, manifest, Path(corpus_root), ("train", "validation"))
    except BaselineExperimentError as error:
        raise RepresentationAblationError("frozen ablation inputs are invalid") from error
    samples = tuple(sorted(loaded, key=lambda sample: sample.sample_id))
    if (
        Counter(sample.partition for sample in samples) != Counter(train=50, validation=15)
        or Counter(sample.target_label_id for sample in samples)
        != Counter({label: 13 for label in TARGET_LABELS})
        or len({sample.source_signer_id for sample in samples}) != 33
    ):
        raise RepresentationAblationError("development pool drifted")
    views, labels = _views(samples, config)
    folds, fold_sha256 = _folds(samples, labels, config)
    runtime = _ablation_runtime()
    reports: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, _Evaluation] = {}
    sample_aliases, signer_aliases = _aliases(samples)
    for row in config.conditions:
        name = _name(row.model, row.representation)
        matrix = views[row.representation]
        cells = [
            _fit_cell(
                row.model,
                matrix[list(fold.fit)],
                labels[list(fold.fit)],
                matrix[list(fold.evaluate)],
                labels[list(fold.evaluate)],
                config,
                runtime,
            )
            for fold in folds
        ]
        reports[name], evaluations[name] = _summarize(
            row.model,
            row.representation,
            samples,
            labels,
            folds,
            cells,
            signer_aliases,
        )
    comparisons = {
        name: _compare(reports, reference, candidate, config.practical_delta_threshold)
        for name, (reference, candidate) in config.comparisons.items()
    }
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "claim_scope": "grouped_development_only_no_winner",
        "source": {"git_commit": commit, "git_dirty": dirty},
        "identities": {
            "configuration_sha256": _sha256(config_bytes),
            "corpus_sha256": config.corpus_sha256,
            "external_dataset_sha256": config.external_dataset_sha256,
            "split_sha256": config.split_sha256,
            "source_feature_plan_sha256": config.source_feature_plan_sha256,
            "feature_view_sha256": {
                name: config.feature_views[name].plan_sha256 for name in REPRESENTATIONS
            },
            "development_fold_sha256": fold_sha256,
            "taxonomy_sha256": config.taxonomy_sha256,
        },
        "data": {
            "development_samples": len(samples),
            "development_signers": len(signer_aliases),
            "opened_partitions": {"train": 50, "validation": 15},
            "test_status": config.test_partition_policy,
        },
        "protocol": {
            "matrix_rows": 6,
            "folds": 3,
            "fit_calls": 18,
            "neural_fit_calls": 9,
            "epochs": config.maximum_epochs,
            "early_stopping": False,
            "checkpoint_files": 0,
        },
        "conditions": reports,
        "comparisons": comparisons,
        "environment": {
            "os": platform.system().casefold(),
            "architecture": platform.machine().casefold(),
            "python": platform.python_version(),
            "keras": version("keras"),
            "tensorflow": version("tensorflow"),
            "scikit_learn": version("scikit-learn"),
        },
        "limitations": [
            "This five-gesture smoke corpus does not estimate population performance.",
            "Three grouped folds are too few for a formal confidence interval.",
            "Story #28 informed this development experiment; it is not confirmatory.",
            "Test features were never requested.",
            "Carry-forward decisions are not model selection or promotion.",
            "Latency is machine-specific prevectorized CPU time.",
        ],
    }
    confusion = (
        canonical_json_bytes(
            {
                "format": CONFUSION_FORMAT,
                "labels": list(TARGET_LABELS),
                "conditions": {name: value.confusion for name, value in evaluations.items()},
            }
        )
        + b"\n"
    )
    payloads = {
        "configuration_path": ("configuration.json", config_bytes),
        "report_path": ("report.json", canonical_json_bytes(report) + b"\n"),
        "confusion_matrix_path": ("out-of-fold-confusions.json", confusion),
        "predictions_path": (
            "out-of-fold-predictions.csv",
            _predictions(samples, folds, evaluations, sample_aliases, signer_aliases),
        ),
    }
    try:
        destination.mkdir(parents=True)
    except OSError as error:
        raise RepresentationAblationError("ablation output could not be created") from error
    paths: dict[str, Path] = {}
    for field, (filename, payload) in payloads.items():
        paths[field] = destination / filename
        _write(paths[field], payload)
    run = ReferenceRunInput(
        run_name=config.run_name,
        git_commit=commit,
        git_dirty=dirty,
        corpus_sha256=config.corpus_sha256,
        split_sha256=config.split_sha256,
        feature_plan_sha256=config.source_feature_plan_sha256,
        seed=config.fold_seed,
        parameters={
            "fit_calls": 18,
            "fold_count": 3,
            "matrix_rows": 6,
            "neural_fit_calls": 9,
            "test_status": config.test_partition_policy,
        },
        metrics={
            **{
                f"oof.{name}.macro_f1": float(value["out_of_fold"]["macro_f1"])
                for name, value in reports.items()
            },
            **{
                f"comparison.{name}.macro_f1_delta": float(value["pooled_macro_f1_delta"])
                for name, value in comparisons.items()
            },
        },
        **paths,
    )
    receipt = log_reference_run(run, tracking_uri=tracking_uri)
    if verify_reference_run(receipt.run_id, tracking_uri=tracking_uri) != receipt:
        raise RepresentationAblationError("ablation ledger verification drifted")
    _write(public_report, _markdown(report, receipt.run_id))
    return RepresentationAblationResult(destination, public_report, receipt)


__all__ = [
    "CONFIG_FORMAT",
    "EXPECTED_CONDITIONS",
    "REPORT_FORMAT",
    "REPRESENTATIONS",
    "RepresentationAblationConfig",
    "RepresentationAblationError",
    "RepresentationAblationResult",
    "load_representation_ablation_config",
    "run_representation_ablations",
]
