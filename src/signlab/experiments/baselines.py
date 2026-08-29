"""Three deliberately small, reproducible recognition baselines."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import platform
import subprocess
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from typing import Annotated, Any, Final, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, model_validator
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.contracts.core import (
    FiniteFloat,
    PositiveSafeInteger,
    SafeInteger,
    StableId,
    StrictContractModel,
)
from signlab.contracts.external_dataset import (
    ExternalTargetLabel,
    validate_external_dataset_manifest,
)
from signlab.contracts.features import landmark_feature_plan_digest
from signlab.contracts.taxonomy import Sha256Digest
from signlab.datasets.public_split import (
    PublicCorpusSample,
    PublicCorpusSplitError,
    PublicPartition,
    reconcile_public_corpus_split,
    validate_public_corpus_split,
)
from signlab.experiments.tracking import (
    ReferenceRunInput,
    ReferenceRunReceipt,
    log_reference_run,
    verify_reference_run,
)
from signlab.features.resources import load_packaged_default_feature_plan

type BaselineName = Literal["majority", "stratified_random", "logistic_regression"]

TARGET_LABELS: Final[tuple[ExternalTargetLabel, ...]] = (
    "hello",
    "no",
    "please",
    "thank_you",
    "yes",
)
BASELINE_NAMES: Final[tuple[BaselineName, ...]] = (
    "majority",
    "stratified_random",
    "logistic_regression",
)
CONFIG_FORMAT: Final = "signlab-reference-baselines/1"
REPORT_FORMAT: Final = "signlab-reference-baseline-report/1"
CONFUSION_FORMAT: Final = "signlab-reference-confusion-matrices/1"
VECTORIZER_ID: Final = "frame_major_dequantized_values_v1"
_SPLIT_FILENAME: Final = "public-corpus-split.json"


class BaselineExperimentError(ValueError):
    """A stable, path-free failure from the reference-baseline boundary."""


class ReferenceBaselineConfig(StrictContractModel):
    """The one reviewed choice set for the frozen public smoke benchmark."""

    format: Literal["signlab-reference-baselines/1"]
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
    vectorizer: Literal["frame_major_dequantized_values_v1"]
    logistic_c_values: Annotated[tuple[FiniteFloat, ...], Field(min_length=2, max_length=2)]
    logistic_solver: Literal["lbfgs"]
    logistic_l1_ratio: FiniteFloat
    logistic_tolerance: Annotated[FiniteFloat, Field(gt=0)]
    logistic_max_iterations: Annotated[PositiveSafeInteger, Field(le=10000)]
    selection_metric: Literal["validation_macro_f1"]
    selection_tie_break: Literal["smaller_c"]
    latency_warmup_runs: Annotated[PositiveSafeInteger, Field(le=1000)]
    latency_measurement_runs: Annotated[PositiveSafeInteger, Field(le=10000)]

    @model_validator(mode="after")
    def _require_the_reviewed_experiment(self) -> Self:
        if self.labels != TARGET_LABELS:
            raise ValueError("reference baseline labels are not in the published target order")
        if self.logistic_c_values != (0.1, 1.0) or self.logistic_l1_ratio != 0.0:
            raise ValueError("reference baseline logistic choices have drifted")
        return self


@dataclass(frozen=True, slots=True)
class ReferenceBaselineResult:
    """Small receipt returned after local artifacts and ledger evidence are verified."""

    selected_c: float
    output_root: Path
    public_report_path: Path
    tracking: ReferenceRunReceipt


@dataclass(frozen=True, slots=True)
class _Evaluation:
    predictions: tuple[str, ...]
    metrics: dict[str, Any]
    confusion: list[list[int]]


@dataclass(frozen=True, slots=True)
class _LogisticSelection:
    model: LogisticRegression
    selected_c: float
    candidate_scores: tuple[dict[str, float], ...]


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def load_reference_baseline_config(
    path: str | Path,
) -> tuple[ReferenceBaselineConfig, bytes]:
    """Load only canonical, LF-terminated configuration bytes."""

    try:
        raw = Path(path).read_bytes()
        payload = parse_json_object(raw)
        config = ReferenceBaselineConfig.model_validate_json(
            canonical_json_bytes(payload), strict=True
        )
        if raw != canonical_json_bytes(config) + b"\n":
            raise ValueError
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise BaselineExperimentError("reference baseline configuration is invalid") from error
    return config, raw


def _git_identity(anchor: Path) -> tuple[Path, str, bool]:
    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(anchor), *arguments),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise BaselineExperimentError("Git source identity is unavailable") from error
        if completed.returncode != 0:
            raise BaselineExperimentError("Git source identity is unavailable")
        return completed.stdout.strip()

    try:
        repository = Path(run("rev-parse", "--show-toplevel")).resolve(strict=True)
    except OSError as error:
        raise BaselineExperimentError("Git source identity is unavailable") from error
    commit = run("rev-parse", "--verify", "HEAD^{commit}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise BaselineExperimentError("Git source identity is unavailable")
    dirty = bool(run("status", "--porcelain=v1", "--untracked-files=all"))
    return repository, commit, dirty


def _load_inputs(
    config: ReferenceBaselineConfig,
    corpus_root: Path,
    external_manifest_path: Path,
) -> tuple[bytes, bytes]:
    try:
        split_bytes = (corpus_root / _SPLIT_FILENAME).read_bytes()
        manifest_bytes = external_manifest_path.read_bytes()
        split = validate_public_corpus_split(split_bytes)
        manifest = validate_external_dataset_manifest(manifest_bytes)
        plan = load_packaged_default_feature_plan("combined")
    except (OSError, TypeError, ValueError) as error:
        raise BaselineExperimentError(
            "frozen baseline inputs are unavailable or invalid"
        ) from error
    if (
        split.source_corpus_sha256 != config.corpus_sha256
        or split.external_dataset_sha256 != config.external_dataset_sha256
        or split.split_sha256 != config.split_sha256
        or split.feature_plan_id != config.feature_plan_id
        or split.feature_plan_sha256 != config.feature_plan_sha256
        or manifest.content_sha256 != config.external_dataset_sha256
        or manifest.taxonomy.id != config.taxonomy_id
        or manifest.taxonomy.version != config.taxonomy_version
        or manifest.taxonomy.sha256 != config.taxonomy_sha256
        or plan.plan_id != config.feature_plan_id
        or landmark_feature_plan_digest(plan) != config.feature_plan_sha256
        or plan.learned_statistics.mode != "none"
    ):
        raise BaselineExperimentError("frozen baseline identities do not match configuration")
    return split_bytes, manifest_bytes


def _load_samples(
    split_bytes: bytes,
    manifest_bytes: bytes,
    corpus_root: Path,
    partitions: Sequence[PublicPartition],
) -> tuple[PublicCorpusSample, ...]:
    try:
        return reconcile_public_corpus_split(
            split_bytes,
            external_manifest_document=manifest_bytes,
            corpus_root=corpus_root,
            partitions=partitions,
        )
    except PublicCorpusSplitError as error:
        raise BaselineExperimentError("frozen public split could not be reconciled") from error


def _partition(
    samples: Sequence[PublicCorpusSample], partition: PublicPartition
) -> tuple[PublicCorpusSample, ...]:
    return tuple(sample for sample in samples if sample.partition == partition)


def _vectorize(
    samples: Sequence[PublicCorpusSample],
    config: ReferenceBaselineConfig,
) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    plan = load_packaged_default_feature_plan("combined")
    expected_shape = (plan.padding.target_frame_count, len(plan.feature_order))
    values: list[tuple[tuple[int, ...], ...]] = []
    labels: list[str] = []
    for sample in samples:
        feature = sample.feature
        if (
            feature.feature_plan_sha256 != config.feature_plan_sha256
            or feature.feature_names != plan.feature_order
            or feature.quantization_scale != plan.quantization_scale
            or feature.statistics_sha256 is not None
            or (len(feature.values_q), len(feature.values_q[0])) != expected_shape
        ):
            raise BaselineExperimentError("portable baseline feature shape or identity drifted")
        values.append(feature.values_q)
        labels.append(sample.target_label_id)
    if not values:
        raise BaselineExperimentError("reference baseline partition is empty")
    matrix = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    matrix /= float(plan.quantization_scale)
    return np.ascontiguousarray(matrix), np.asarray(labels, dtype=np.str_)


def _predict(model: Any, matrix: NDArray[np.float64]) -> tuple[str, ...]:
    predicted = cast(NDArray[Any], model.predict(matrix))
    values = tuple(str(value) for value in predicted.tolist())
    if any(value not in TARGET_LABELS for value in values):
        raise BaselineExperimentError("reference baseline produced an unknown target")
    return values


def _fit_logistic(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.str_],
    validation_x: NDArray[np.float64],
    validation_y: NDArray[np.str_],
    config: ReferenceBaselineConfig,
) -> _LogisticSelection:
    candidates: list[tuple[float, float, LogisticRegression]] = []
    for regularization in config.logistic_c_values:
        model = LogisticRegression(
            C=regularization,
            solver=config.logistic_solver,
            l1_ratio=config.logistic_l1_ratio,
            tol=config.logistic_tolerance,
            max_iter=config.logistic_max_iterations,
            random_state=config.seed,
        )
        try:
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train_x, train_y)
        except (ConvergenceWarning, ValueError) as error:
            raise BaselineExperimentError(
                "logistic baseline could not be fit reproducibly"
            ) from error
        predictions = _predict(model, validation_x)
        score = float(
            f1_score(
                validation_y,
                predictions,
                labels=TARGET_LABELS,
                average="macro",
                zero_division=0,
            )
        )
        candidates.append((score, regularization, model))
    selected_score, selected_c, selected_model = min(
        candidates, key=lambda candidate: (-candidate[0], candidate[1])
    )
    scores = tuple(
        {"c": regularization, "validation_macro_f1": score}
        for score, regularization, _model in candidates
    )
    if not math.isfinite(selected_score):
        raise BaselineExperimentError("logistic validation score is invalid")
    return _LogisticSelection(selected_model, selected_c, scores)


def _fit_models(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.str_],
    validation_x: NDArray[np.float64],
    validation_y: NDArray[np.str_],
    config: ReferenceBaselineConfig,
) -> tuple[dict[BaselineName, Any], _LogisticSelection]:
    majority = DummyClassifier(strategy="most_frequent")
    random = DummyClassifier(strategy="stratified", random_state=config.seed)
    with threadpool_limits(limits=1):
        majority.fit(train_x, train_y)
        random.fit(train_x, train_y)
    if _predict(majority, train_x[:1]) != (config.labels[0],):
        raise BaselineExperimentError("majority baseline tie-break drifted")
    logistic = _fit_logistic(train_x, train_y, validation_x, validation_y, config)
    return (
        {
            "majority": majority,
            "stratified_random": random,
            "logistic_regression": logistic.model,
        },
        logistic,
    )


def _evaluate(
    model: Any,
    matrix: NDArray[np.float64],
    expected: NDArray[np.str_],
    *,
    random_state: int | None = None,
) -> _Evaluation:
    if random_state is not None:
        model.set_params(random_state=random_state)
    predicted = _predict(model, matrix)
    precision, recall, _f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=TARGET_LABELS,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=TARGET_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(TARGET_LABELS)
        },
    }
    matrix_values = cast(
        list[list[int]],
        confusion_matrix(expected, predicted, labels=TARGET_LABELS).astype(int).tolist(),
    )
    return _Evaluation(predicted, metrics, matrix_values)


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _measure_latency(
    model: Any,
    matrix: NDArray[np.float64],
    *,
    warmup_runs: int,
    measurement_runs: int,
) -> dict[str, int | float | str]:
    with threadpool_limits(limits=1):
        for index in range(warmup_runs):
            model.predict(matrix[index % len(matrix) : index % len(matrix) + 1])
        durations: list[int] = []
        for index in range(measurement_runs):
            row = matrix[index % len(matrix) : index % len(matrix) + 1]
            started = perf_counter_ns()
            model.predict(row)
            durations.append(perf_counter_ns() - started)
    return {
        "batch_size": 1,
        "measurement_runs": measurement_runs,
        "p50_ms": _percentile(durations, 0.50) / 1_000_000,
        "p95_ms": _percentile(durations, 0.95) / 1_000_000,
        "scope": "prevectorized_model_predict_cpu_single_thread",
        "warmup_runs": warmup_runs,
    }


def _failure_analysis(
    samples: Sequence[PublicCorpusSample], predictions: Sequence[str]
) -> dict[str, Any]:
    signer_aliases = {
        signer: f"signer_{index:03d}"
        for index, signer in enumerate(
            sorted({sample.source_signer_id for sample in samples}), start=1
        )
    }
    class_rows = {label: {"support": 0, "errors": 0} for label in TARGET_LABELS}
    quality_rows = {disposition: {"support": 0, "errors": 0} for disposition in ("pass", "warning")}
    signer_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"support": 0, "errors": 0})
    for sample, predicted in zip(samples, predictions, strict=True):
        error = int(predicted != sample.target_label_id)
        class_rows[sample.target_label_id]["support"] += 1
        class_rows[sample.target_label_id]["errors"] += error
        quality_rows[sample.quality_disposition]["support"] += 1
        quality_rows[sample.quality_disposition]["errors"] += error
        alias = signer_aliases[sample.source_signer_id]
        signer_rows[alias]["support"] += 1
        signer_rows[alias]["errors"] += error
    return {
        "scope": "selected_logistic_regression_test_only",
        "by_class": class_rows,
        "by_quality_disposition": quality_rows,
        "by_signer": [
            {"signer_alias": alias, **counts} for alias, counts in sorted(signer_rows.items())
        ],
    }


def _prediction_csv(
    samples_by_partition: Mapping[str, Sequence[PublicCorpusSample]],
    evaluations: Mapping[str, Mapping[BaselineName, _Evaluation]],
) -> bytes:
    all_samples = tuple(
        sample for partition in ("validation", "test") for sample in samples_by_partition[partition]
    )
    sample_aliases = {
        sample.sample_id: f"sample_{index:03d}" for index, sample in enumerate(all_samples, start=1)
    }
    signer_aliases = {
        signer: f"signer_{index:03d}"
        for index, signer in enumerate(
            sorted({sample.source_signer_id for sample in all_samples}), start=1
        )
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ("partition", "model", "sample_alias", "signer_alias", "quality", "actual", "predicted")
    )
    for partition in ("validation", "test"):
        samples = samples_by_partition[partition]
        for model_name in BASELINE_NAMES:
            predictions = evaluations[partition][model_name].predictions
            for sample, predicted in zip(samples, predictions, strict=True):
                writer.writerow(
                    (
                        partition,
                        model_name,
                        sample_aliases[sample.sample_id],
                        signer_aliases[sample.source_signer_id],
                        sample.quality_disposition,
                        sample.target_label_id,
                        predicted,
                    )
                )
    return stream.getvalue().encode("utf-8")


def _limitations(parameter_count: int) -> list[str]:
    return [
        (
            "This is a 50-train, 15-validation, 15-test smoke benchmark with three "
            "test examples per class."
        ),
        (
            f"Logistic regression fits {parameter_count:,} coefficients and intercepts "
            "from only 50 training examples."
        ),
        (
            "Results use one signer-disjoint split and one seed; they do not estimate "
            "uncertainty across splits or seeds."
        ),
        (
            "Only two test examples have a warning disposition, and each test signer "
            "contributes few examples."
        ),
        (
            "The benchmark covers five isolated targets only: it provides no evidence "
            "for other, inactive, abstention, continuous events, or sign-language "
            "translation."
        ),
        (
            "Latency measures pre-vectorized single-example CPU prediction and varies "
            "by machine; extraction and JSON loading are excluded."
        ),
    ]


def _public_markdown(report: Mapping[str, Any]) -> bytes:
    models = cast(dict[str, Any], report["models"])
    failure = cast(dict[str, Any], report["failure_analysis"])
    by_signer = cast(list[dict[str, Any]], failure["by_signer"])
    lines = [
        "# PopSign reference baselines v1",
        "",
        (
            "> Exploratory smoke benchmark only. These numbers are not "
            "product-performance or sign-language claims."
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
        f"- Source corpus: `{report['identities']['corpus_sha256']}`",
        f"- Feature plan: `{report['identities']['feature_plan_sha256']}`",
        f"- Seed: `{report['seed']}`",
        "",
        (
            "The signer-disjoint split contains 50 training, 15 validation, and 15 "
            "final-test clips across five targets."
        ),
        (
            "Test features were opened only after the single logistic-regression choice "
            "was fixed on validation macro-F1."
        ),
        "",
        "## Results",
        "",
        (
            "| Model | Validation macro-F1 | Test macro-F1 | Test balanced accuracy | "
            "Parameters | CPU p50 / p95 ms |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in BASELINE_NAMES:
        model = models[model_name]
        lines.append(
            "| "
            + model_name.replace("_", " ").title()
            + f" | {model['validation']['macro_f1']:.3f}"
            + f" | {model['test']['macro_f1']:.3f}"
            + f" | {model['test']['balanced_accuracy']:.3f}"
            + f" | {model['parameter_count']:,}"
            + f" | {model['latency']['p50_ms']:.3f} / {model['latency']['p95_ms']:.3f} |"
        )
    selection = report["selection"]
    lines.extend(
        [
            "",
            "The predeclared logistic choice was "
            f"`C={selection['selected_c']}` from `{selection['candidate_c_values']}`; "
            "ties resolve to the smaller value and the selected train-only fit was not refit.",
            "",
            "## Selected-model test errors",
            "",
            "| Class | Support | Errors |",
            "| --- | ---: | ---: |",
        ]
    )
    for label in TARGET_LABELS:
        values = failure["by_class"][label]
        lines.append(f"| {label} | {values['support']} | {values['errors']} |")
    lines.extend(
        [
            "",
            "| Quality disposition | Support | Errors |",
            "| --- | ---: | ---: |",
        ]
    )
    for disposition in ("pass", "warning"):
        values = failure["by_quality_disposition"][disposition]
        lines.append(f"| {disposition} | {values['support']} | {values['errors']} |")
    signers_with_errors = sum(int(row["errors"] > 0) for row in by_signer)
    maximum_errors = max((int(row["errors"]) for row in by_signer), default=0)
    lines.extend(
        [
            "",
            (
                f"Signer grouping: {len(by_signer)} held-out signers; "
                f"{signers_with_errors} had at least one error; the maximum was "
                f"{maximum_errors} errors for one signer. No source identifier is "
                "published."
            ),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
            "## Reproduce",
            "",
            "```shell",
            (
                "uv run --locked --extra experiments signlab train reference-baselines "
                "configs/experiments/popsign-reference-baselines-v1.json "
                "--corpus-root <frozen-split-root> "
                "--external-manifest <external-dataset-manifest.json> "
                "--output-root runs/popsign-reference-baselines-v1"
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _new_path(path: str | Path, *, label: str) -> Path:
    target = Path(path).resolve()
    if target.exists() or target.is_symlink():
        raise BaselineExperimentError(f"{label} must not already exist")
    return target


def _write_new(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise BaselineExperimentError(
            "reference baseline artifacts could not be written"
        ) from error


def run_reference_baselines(
    config_path: str | Path,
    *,
    corpus_root: str | Path,
    external_manifest_path: str | Path,
    output_root: str | Path,
    public_report_path: str | Path | None = None,
    tracking_uri: str | None = None,
) -> ReferenceBaselineResult:
    """Run, report, log, and verify the three frozen reference baselines."""

    config, config_bytes = load_reference_baseline_config(config_path)
    destination = _new_path(output_root, label="reference baseline output")
    human_report = (
        _new_path(public_report_path, label="public baseline report")
        if public_report_path is not None
        else destination / "public-report.md"
    )
    config_file = Path(config_path).resolve(strict=True)
    _repository, git_commit, git_dirty = _git_identity(config_file.parent)
    split_bytes, manifest_bytes = _load_inputs(
        config, Path(corpus_root), Path(external_manifest_path)
    )

    development = _load_samples(
        split_bytes,
        manifest_bytes,
        Path(corpus_root),
        ("train", "validation"),
    )
    train_samples = _partition(development, "train")
    validation_samples = _partition(development, "validation")
    if Counter(sample.target_label_id for sample in train_samples) != Counter(
        {label: 10 for label in TARGET_LABELS}
    ) or Counter(sample.target_label_id for sample in validation_samples) != Counter(
        {label: 3 for label in TARGET_LABELS}
    ):
        raise BaselineExperimentError("reference baseline partition quotas drifted")
    train_x, train_y = _vectorize(train_samples, config)
    validation_x, validation_y = _vectorize(validation_samples, config)
    models, selection = _fit_models(train_x, train_y, validation_x, validation_y, config)

    # The final partition is intentionally not reconciled until model choice is sealed.
    test_samples = _load_samples(
        split_bytes,
        manifest_bytes,
        Path(corpus_root),
        ("test",),
    )
    if Counter(sample.target_label_id for sample in test_samples) != Counter(
        {label: 3 for label in TARGET_LABELS}
    ):
        raise BaselineExperimentError("reference baseline test quota drifted")
    test_x, test_y = _vectorize(test_samples, config)

    partition_inputs = {
        "validation": (validation_x, validation_y, config.seed + 1),
        "test": (test_x, test_y, config.seed + 2),
    }
    evaluations: dict[str, dict[BaselineName, _Evaluation]] = {}
    for partition_name, (matrix, expected, random_seed) in partition_inputs.items():
        evaluations[partition_name] = {
            name: _evaluate(
                model,
                matrix,
                expected,
                random_state=random_seed if name == "stratified_random" else None,
            )
            for name, model in models.items()
        }
    parameter_counts: dict[BaselineName, int] = {
        "majority": 0,
        "stratified_random": 0,
        "logistic_regression": int(selection.model.coef_.size + selection.model.intercept_.size),
    }
    latency = {
        name: _measure_latency(
            model,
            test_x,
            warmup_runs=config.latency_warmup_runs,
            measurement_runs=config.latency_measurement_runs,
        )
        for name, model in models.items()
    }
    failure = _failure_analysis(
        test_samples, evaluations["test"]["logistic_regression"].predictions
    )
    train_counts = Counter(str(label) for label in train_y.tolist())
    model_results: dict[str, Any] = {
        name: {
            "parameter_count": parameter_counts[name],
            "validation": evaluations["validation"][name].metrics,
            "test": evaluations["test"][name].metrics,
            "latency": latency[name],
        }
        for name in BASELINE_NAMES
    }
    model_results["majority"]["state"] = {
        "selected_label": config.labels[0],
        "tie_break": "first_label_in_config_order",
        "training_class_counts": {label: train_counts[label] for label in TARGET_LABELS},
    }
    model_results["stratified_random"]["state"] = {
        "class_priors": {
            label: train_counts[label] / len(train_samples) for label in TARGET_LABELS
        },
        "seed": config.seed,
        "partition_seeds": {
            "validation": config.seed + 1,
            "test": config.seed + 2,
        },
    }
    model_results["logistic_regression"]["state"] = {
        "coefficient_shape": list(selection.model.coef_.shape),
        "intercept_shape": list(selection.model.intercept_.shape),
        "iterations": [int(value) for value in selection.model.n_iter_.tolist()],
    }
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "run_name": config.run_name,
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
        "seed": config.seed,
        "data": {
            "labels": list(config.labels),
            "partition_counts": {"train": 50, "validation": 15, "test": 15},
            "vectorizer": config.vectorizer,
            "feature_width": int(train_x.shape[1]),
        },
        "selection": {
            "candidate_c_values": list(config.logistic_c_values),
            "candidate_scores": list(selection.candidate_scores),
            "fit_partition": "train",
            "metric": config.selection_metric,
            "selected_c": selection.selected_c,
            "test_access": "after_selection_sealed",
            "tie_break": config.selection_tie_break,
            "refit_after_selection": False,
        },
        "models": model_results,
        "failure_analysis": failure,
        "limitations": _limitations(parameter_counts["logistic_regression"]),
        "environment": {
            "architecture": platform.machine().casefold() or "unknown",
            "numpy": version("numpy"),
            "os": platform.system().casefold() or "unknown",
            "python": platform.python_version(),
            "scikit_learn": version("scikit-learn"),
            "scipy": version("scipy"),
        },
        "reproducibility": {
            "deterministic_artifacts": [
                "configuration.json",
                "confusion-matrix.json",
                "predictions.csv",
            ],
            "observational_fields": [
                "models.*.latency",
                "environment",
                "source.git_commit",
                "source.git_dirty",
            ],
        },
    }
    confusion_payload: dict[str, Any] = {
        "format": CONFUSION_FORMAT,
        "labels": list(config.labels),
        "orientation": {"rows": "actual", "columns": "predicted"},
        "validation": {name: evaluations["validation"][name].confusion for name in BASELINE_NAMES},
        "test": {name: evaluations["test"][name].confusion for name in BASELINE_NAMES},
    }
    predictions_bytes = _prediction_csv(
        {"validation": validation_samples, "test": test_samples}, evaluations
    )
    report_bytes = canonical_json_bytes(report) + b"\n"
    confusion_bytes = canonical_json_bytes(confusion_payload) + b"\n"
    public_bytes = _public_markdown(report)

    try:
        destination.mkdir(parents=True)
    except OSError as error:
        raise BaselineExperimentError("reference baseline output could not be created") from error
    paths = {
        "configuration_path": destination / "configuration.json",
        "report_path": destination / "report.json",
        "confusion_matrix_path": destination / "confusion-matrix.json",
        "predictions_path": destination / "predictions.csv",
    }
    for path, payload in (
        (paths["configuration_path"], config_bytes),
        (paths["report_path"], report_bytes),
        (paths["confusion_matrix_path"], confusion_bytes),
        (paths["predictions_path"], predictions_bytes),
    ):
        _write_new(path, payload)

    logged_metrics = {
        f"{partition}.{name}.{metric}": float(evaluations[partition][name].metrics[metric])
        for partition in ("validation", "test")
        for name in BASELINE_NAMES
        for metric in ("macro_f1", "balanced_accuracy")
    }
    logged_metrics.update(
        {f"test.{name}.latency_p50_ms": float(latency[name]["p50_ms"]) for name in BASELINE_NAMES}
    )
    run_input = ReferenceRunInput(
        run_name=config.run_name,
        git_commit=git_commit,
        git_dirty=git_dirty,
        corpus_sha256=config.corpus_sha256,
        split_sha256=config.split_sha256,
        feature_plan_sha256=config.feature_plan_sha256,
        seed=config.seed,
        parameters={
            "baseline_count": 3,
            "feature_width": int(train_x.shape[1]),
            "logistic.candidate_count": len(config.logistic_c_values),
            "logistic.selected_c": selection.selected_c,
            "logistic.solver": config.logistic_solver,
            "random.partition_seed_rule": "base_plus_partition_offset_v1",
            "test_access_after_selection": True,
            "vectorizer": config.vectorizer,
        },
        metrics=logged_metrics,
        **paths,
    )
    if (
        report["source"] != {"git_commit": run_input.git_commit, "git_dirty": run_input.git_dirty}
        or report["identities"]["configuration_sha256"] != _sha256(config_bytes)
        or report["identities"]["corpus_sha256"] != run_input.corpus_sha256
        or report["identities"]["split_sha256"] != run_input.split_sha256
        or report["identities"]["feature_plan_sha256"] != run_input.feature_plan_sha256
        or report["seed"] != run_input.seed
    ):
        raise BaselineExperimentError("reference baseline report lineage is inconsistent")
    tracking = log_reference_run(run_input, tracking_uri=tracking_uri)
    if verify_reference_run(tracking.run_id, tracking_uri=tracking_uri) != tracking:
        raise BaselineExperimentError("reference baseline ledger verification drifted")
    _write_new(human_report, public_bytes)
    return ReferenceBaselineResult(selection.selected_c, destination, human_report, tracking)


__all__ = [
    "BASELINE_NAMES",
    "CONFIG_FORMAT",
    "CONFUSION_FORMAT",
    "REPORT_FORMAT",
    "TARGET_LABELS",
    "VECTORIZER_ID",
    "BaselineExperimentError",
    "ReferenceBaselineConfig",
    "ReferenceBaselineResult",
    "load_reference_baseline_config",
    "run_reference_baselines",
]
