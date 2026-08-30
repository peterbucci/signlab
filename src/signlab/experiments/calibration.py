"""One bounded six-class calibration run over constructed development fragments."""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Final, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, model_validator

from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256, parse_json_object
from signlab.contracts.core import FiniteFloat, StableId, StrictContractModel
from signlab.contracts.features import landmark_feature_plan_digest
from signlab.contracts.taxonomy import Sha256Digest
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments.baselines import (
    BaselineExperimentError,
    _git_identity,
    _load_inputs,
    _load_samples,
    _partition,
    _sha256,
)
from signlab.experiments.sequence_baselines import (
    _aliases,
    _build_graph,
    _runtime,
    _seed_runtime,
)
from signlab.experiments.tracking import (
    ReferenceRunInput,
    ReferenceRunReceipt,
    log_reference_run,
    verify_reference_run,
)
from signlab.features.resources import load_packaged_default_feature_plan

LABELS: Final = ("hello", "no", "please", "thank_you", "yes", "other")
TARGET_LABELS: Final = LABELS[:-1]
CONFIG_FORMAT: Final = "signlab-constructed-calibration/1"
REPORT_FORMAT: Final = "signlab-constructed-calibration-report/1"
POLICY_FORMAT: Final = "signlab-decision-policy/1"
CONFUSION_FORMAT: Final = "signlab-calibration-confusion/1"
DERIVATIVE_FORMAT: Final = "signlab-constructed-transition-derivative-set/1"
SELECTION_RULE: Final = "eligible_same_signer_lexicographic_opaque_identity/1"
RECIPE_ID: Final = "hand_local_latter_half_a_first_half_b_pad_64/1"
OBJECTIVE: Final = "maximize_target_coverage_zero_observed_accepted_errors/1"
EVIDENCE_KIND: Final = "constructed_transition_calibration_conformance"
METRIC_CLAIM: Final = "development_mechanics_only"
_EXPECTED_PARAMETERS: Final = 29_094


class CalibrationError(ValueError):
    """A path-free failure from the bounded calibration experiment."""


class CalibrationConfig(StrictContractModel):
    format: Literal["signlab-constructed-calibration/1"]
    run_name: StableId
    corpus_sha256: Sha256Digest
    external_dataset_sha256: Sha256Digest
    split_id: Literal["popsign-five-isolated-smoke-v1"]
    split_sha256: Sha256Digest
    feature_plan_id: Literal["combined_64_frames"]
    feature_plan_sha256: Sha256Digest
    input_feature_plan_id: Literal["hand_local_64_frames"]
    input_feature_plan_sha256: Sha256Digest
    taxonomy_id: Literal["signlab-five"]
    taxonomy_version: Literal["1.0.0"]
    taxonomy_sha256: Sha256Digest
    labels: Annotated[tuple[str, ...], Field(min_length=6, max_length=6)]
    model: Literal["tcn"]
    optimizer: Literal["adam"]
    input_frames: Literal[64]
    input_width: Literal[126]
    seed: Literal[20260830]
    learning_rate: FiniteFloat
    batch_size: Literal[32]
    maximum_epochs: Literal[30]
    tcn_channels: Literal[32]
    tcn_dilations: tuple[Literal[1], Literal[2], Literal[4], Literal[8]]
    tcn_kernel_size: Literal[3]
    derivative_selection_rule: Literal["eligible_same_signer_lexicographic_opaque_identity/1"]
    derivative_recipe: Literal["hand_local_latter_half_a_first_half_b_pad_64/1"]
    train_derivatives: Literal[10]
    validation_derivatives: Literal[3]
    temperature_min_milli: Literal[50]
    temperature_max_milli: Literal[10000]
    threshold_step_percent: Literal[1]
    threshold_objective: Literal["maximize_target_coverage_zero_observed_accepted_errors/1"]
    reliability_bins: Literal[3]
    test_partition_policy: Literal["sealed_not_loaded"]

    @model_validator(mode="after")
    def _freeze_labels(self) -> Self:
        if self.labels != LABELS or self.learning_rate != 0.001:
            raise ValueError("calibration protocol drifted")
        return self


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    output_root: Path
    public_report_path: Path
    public_policy_path: Path
    tracking: ReferenceRunReceipt


def load_calibration_config(path: str | Path) -> tuple[CalibrationConfig, bytes]:
    """Load only the canonical, LF-terminated frozen protocol."""

    try:
        raw = Path(path).read_bytes()
        config = CalibrationConfig.model_validate_json(
            canonical_json_bytes(parse_json_object(raw)), strict=True
        )
        if raw != canonical_json_bytes(config) + b"\n":
            raise ValueError
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise CalibrationError("calibration configuration is invalid") from error
    return config, raw


def _hand_local_matrix(
    samples: Sequence[PublicCorpusSample], config: CalibrationConfig
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    source = load_packaged_default_feature_plan("combined")
    hand = load_packaged_default_feature_plan("hand_local")
    if (
        source.plan_id != config.feature_plan_id
        or landmark_feature_plan_digest(source) != config.feature_plan_sha256
        or hand.plan_id != config.input_feature_plan_id
        or landmark_feature_plan_digest(hand) != config.input_feature_plan_sha256
    ):
        raise CalibrationError("calibration feature identity drifted")
    if not samples:
        raise CalibrationError("calibration partition is empty")
    source_indexes = {name: index for index, name in enumerate(source.feature_order)}
    indexes = tuple(source_indexes[name] for name in hand.feature_order)
    label_index = {label: index for index, label in enumerate(TARGET_LABELS)}
    values = np.asarray([sample.feature.values_q for sample in samples], dtype=np.float32)
    if values.shape != (len(samples), 64, 134):
        raise CalibrationError("calibration feature input drifted")
    values /= np.float32(source.quantization_scale)
    labels = np.asarray([label_index[sample.target_label_id] for sample in samples], dtype=np.int64)
    return np.ascontiguousarray(values[:, :, indexes]), labels


def _transition_fragments(
    samples: Sequence[PublicCorpusSample],
    matrix: NDArray[np.float32],
    partition: str,
    count: int,
) -> tuple[NDArray[np.float32], tuple[str, ...], str]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.partition != partition:
            raise CalibrationError("constructed transition partition drifted")
        grouped[sample.source_signer_id].append(index)
    eligible = sorted(
        signer
        for signer, indexes in grouped.items()
        if len({samples[index].target_label_id for index in indexes}) >= 2
    )
    if len(eligible) < count:
        raise CalibrationError("constructed transition quota is unavailable")
    rows: list[NDArray[np.float32]] = []
    signers: list[str] = []
    lineage: list[str] = []
    for signer in eligible[:count]:
        indexes = sorted(grouped[signer], key=lambda index: samples[index].sample_id)
        left = indexes[0]
        right = next(
            index
            for index in indexes[1:]
            if samples[index].target_label_id != samples[left].target_label_id
        )
        left_count = sum(not value for value in samples[left].feature.padding_mask)
        right_count = sum(not value for value in samples[right].feature.padding_mask)
        values = np.concatenate(
            (matrix[left, left_count // 2 : left_count], matrix[right, : right_count // 2])
        )
        fragment = np.zeros((64, matrix.shape[2]), dtype=np.float32)
        fragment[: len(values)] = values
        derived_sha256 = _sha256(fragment.tobytes(order="C"))
        rows.append(fragment)
        signers.append(signer)
        lineage.append(
            canonical_sha256(
                {
                    "partition": partition,
                    "recipe_id": RECIPE_ID,
                    "signer": signer,
                    "parents": [
                        [
                            samples[index].sample_id,
                            samples[index].target_label_id,
                            samples[index].feature.sequence_sha256,
                        ]
                        for index in (left, right)
                    ],
                    "derived_feature_sha256": derived_sha256,
                },
                domain=DERIVATIVE_FORMAT,
            )
        )
    identity = canonical_sha256(
        {
            "partition": partition,
            "selection_rule_id": SELECTION_RULE,
            "recipe_id": RECIPE_ID,
            "derivative_lineage_sha256s": sorted(lineage),
        },
        domain=DERIVATIVE_FORMAT,
    )
    return np.stack(rows), tuple(signers), identity


def _temperature_scale(
    probabilities: NDArray[np.float64], temperature_milli: int
) -> NDArray[np.float64]:
    scaled = np.log(np.clip(probabilities, 1e-7, 1.0)) / (temperature_milli / 1000)
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def _nll(probabilities: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def _brier(probabilities: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    truth = np.eye(len(LABELS), dtype=np.float64)[labels]
    return float(np.square(probabilities - truth).sum(axis=1).mean())


def _fit_temperature(
    probabilities: NDArray[np.float64], labels: NDArray[np.int64], config: CalibrationConfig
) -> tuple[int, NDArray[np.float64]]:
    candidates = (
        (milli, _temperature_scale(probabilities, milli))
        for milli in range(config.temperature_min_milli, config.temperature_max_milli + 1)
    )
    return min(
        candidates,
        key=lambda row: (_nll(row[1], labels), abs(row[0] - 1000), row[0]),
    )


def _policy_rows(
    probabilities: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    target_truth = labels < len(TARGET_LABELS)
    fragment_truth = ~target_truth
    fragment_count = int(fragment_truth.sum())
    rows: list[dict[str, Any]] = []
    for percent in range(101):
        accepted = confidence >= percent / 100
        correct = accepted & (predicted == labels)
        errors = accepted & (predicted != labels)
        accepted_targets = accepted & target_truth
        correct_targets = correct & target_truth
        false_target = accepted & fragment_truth & (predicted < len(TARGET_LABELS))
        row = {
            "threshold_percent": percent,
            "coverage": float(accepted.mean()),
            "accepted_risk": float(errors.sum() / accepted.sum()) if accepted.any() else None,
            "target_acceptance_coverage": float(accepted_targets.sum() / target_truth.sum()),
            "correct_target_coverage": float(correct_targets.sum() / target_truth.sum()),
            "accepted_target_accuracy": (
                float(correct_targets.sum() / accepted_targets.sum())
                if accepted_targets.any()
                else None
            ),
            "accepted_target_errors": int((errors & target_truth).sum()),
            "constructed_abstention_rate": (
                float((fragment_truth & ~accepted).sum() / fragment_count)
                if fragment_count
                else None
            ),
            "constructed_non_target_decision_rate": (
                float(1 - false_target.sum() / fragment_count) if fragment_count else None
            ),
            "constructed_to_target_false_acceptance": (
                float(false_target.sum() / fragment_count) if fragment_count else None
            ),
            "accepted_errors": int(errors.sum()),
        }
        rows.append(row)
    viable = [
        row for row in rows if row["accepted_errors"] == 0 and row["correct_target_coverage"] > 0
    ]
    if not viable:
        return {"status": "no_viable_operating_point", "threshold_percent": None}, rows
    selected = min(
        viable,
        key=lambda row: (
            -row["correct_target_coverage"],
            -row["coverage"],
            row["threshold_percent"],
        ),
    )
    return {"status": "selected", **selected}, rows


def _reliability(
    probabilities: NDArray[np.float64], labels: NDArray[np.int64]
) -> list[dict[str, Any]]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    rows: list[dict[str, Any]] = []
    for index in range(3):
        low, high = index / 3, (index + 1) / 3
        selected = (confidence >= low) & (confidence <= high if index == 2 else confidence < high)
        mean_confidence = float(confidence[selected].mean()) if selected.any() else None
        accuracy = float(correct[selected].mean()) if selected.any() else None
        rows.append(
            {
                "lower": low,
                "upper": high,
                "count": int(selected.sum()),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": (
                    abs(mean_confidence - accuracy)
                    if mean_confidence is not None and accuracy is not None
                    else None
                ),
            }
        )
    return rows


def _predictions_csv(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    row_signers: Sequence[str],
    sample_aliases: Sequence[str],
    signer_aliases: Mapping[str, str],
    selected: Mapping[str, Any],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "example_alias",
            "signer_alias",
            "evidence_kind",
            "actual",
            "predicted",
            "confidence",
            "decision",
        )
    )
    threshold = selected["threshold_percent"]
    for index, (truth, values, signer) in enumerate(
        zip(labels, probabilities, row_signers, strict=True)
    ):
        predicted = int(values.argmax())
        accepted = selected["status"] == "selected" and float(values.max()) >= threshold / 100
        writer.writerow(
            (
                sample_aliases[index],
                signer_aliases[signer],
                "target" if truth < len(TARGET_LABELS) else "transition_fragment",
                LABELS[int(truth)],
                LABELS[predicted],
                f"{float(values.max()):.9f}",
                LABELS[predicted] if accepted else "abstain",
            )
        )
    return stream.getvalue().encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise CalibrationError("calibration artifact could not be written") from error


def _markdown(report: Mapping[str, Any], run_id: str) -> bytes:
    metrics = report["metrics"]
    policy = report["policy"]
    threshold = policy["threshold_percent"]
    lines = [
        "# Constructed transition calibration check",
        "",
        "This is a development-only mechanics check, not a model-quality or deployment claim.",
        "",
        f"- Evidence: `{report['evidence_kind']}`",
        f"- Test partition: `{report['test_status']}`",
        f"- Validation rows: {report['data']['validation_rows']} (15 source + 3 constructed)",
        f"- Temperature: {policy['temperature_milli'] / 1000:.3f}",
        (
            f"- Policy: `{policy['status']}`; threshold: "
            f"{threshold if threshold is not None else 'none'}%"
        ),
        (
            f"- NLL before/after: {metrics['uncalibrated']['nll']:.6f} / "
            f"{metrics['calibrated']['nll']:.6f}"
        ),
        (
            f"- Brier before/after: {metrics['uncalibrated']['brier']:.6f} / "
            f"{metrics['calibrated']['brier']:.6f}"
        ),
        f"- Verified ledger run: `{run_id}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {value}" for value in report["limitations"])
    return ("\n".join(lines) + "\n").encode()


def run_calibration(
    config_path: str | Path,
    *,
    corpus_root: str | Path,
    external_manifest_path: str | Path,
    output_root: str | Path,
    public_report_path: str | Path | None = None,
    public_policy_path: str | Path | None = None,
    tracking_uri: str | None = None,
) -> CalibrationResult:
    """Fit once on development inputs, calibrate validation, and keep test sealed."""

    config, config_bytes = load_calibration_config(config_path)
    destination = Path(output_root).resolve()
    public_report = (
        Path(public_report_path).resolve()
        if public_report_path is not None
        else destination / "public-report.md"
    )
    public_policy = (
        Path(public_policy_path).resolve()
        if public_policy_path is not None
        else destination / "public-policy.json"
    )
    if (
        destination.exists()
        or public_report.exists()
        or public_policy.exists()
        or public_report == public_policy
    ):
        raise CalibrationError("calibration output must not already exist")
    try:
        _, commit, dirty = _git_identity(Path(config_path).resolve(strict=True).parent)
        split_bytes, manifest_bytes = _load_inputs(
            cast(Any, config), Path(corpus_root), Path(external_manifest_path)
        )
        loaded = _load_samples(
            split_bytes, manifest_bytes, Path(corpus_root), ("train", "validation")
        )
    except BaselineExperimentError as error:
        raise CalibrationError("frozen calibration inputs are invalid") from error
    train = tuple(sorted(_partition(loaded, "train"), key=lambda row: row.sample_id))
    validation = tuple(sorted(_partition(loaded, "validation"), key=lambda row: row.sample_id))
    if Counter(row.target_label_id for row in train) != Counter(
        {label: 10 for label in TARGET_LABELS}
    ) or Counter(row.target_label_id for row in validation) != Counter(
        {label: 3 for label in TARGET_LABELS}
    ):
        raise CalibrationError("calibration development quotas drifted")
    train_x, train_y = _hand_local_matrix(train, config)
    validation_x, validation_y = _hand_local_matrix(validation, config)
    train_other, train_other_signers, train_derivative = _transition_fragments(
        train, train_x, "train", config.train_derivatives
    )
    validation_other, validation_other_signers, validation_derivative = _transition_fragments(
        validation, validation_x, "validation", config.validation_derivatives
    )
    fit_x = np.concatenate((train_x, train_other))
    fit_y = np.concatenate((train_y, np.full(len(train_other), 5, dtype=np.int64)))
    evaluate_x = np.concatenate((validation_x, validation_other))
    evaluate_y = np.concatenate((validation_y, np.full(len(validation_other), 5, dtype=np.int64)))
    derivative_sha256 = canonical_sha256(
        {
            "format": DERIVATIVE_FORMAT,
            "source_split_sha256": config.split_sha256,
            "taxonomy_sha256": config.taxonomy_sha256,
            "feature_plan_sha256": config.feature_plan_sha256,
            "selection_rule_id": SELECTION_RULE,
            "recipe_id": RECIPE_ID,
            "test_status": config.test_partition_policy,
            "partitions": [
                {"partition": "train", "count": len(train_other), "sha256": train_derivative},
                {
                    "partition": "validation",
                    "count": len(validation_other),
                    "sha256": validation_derivative,
                },
            ],
        },
        domain=DERIVATIVE_FORMAT,
    )
    try:
        runtime = _runtime()
        _seed_runtime(runtime, config.seed)
        model = _build_graph("tcn", config.input_width, cast(Any, config), runtime)
        if (
            tuple(model.input_shape) != (None, 64, 126)
            or tuple(model.output_shape) != (None, 6)
            or model.count_params() != _EXPECTED_PARAMETERS
        ):
            raise CalibrationError("six-class calibration graph drifted")
        model.fit(
            fit_x,
            fit_y,
            batch_size=config.batch_size,
            epochs=config.maximum_epochs,
            shuffle=True,
            verbose=0,
        )
        probabilities = np.asarray(
            model.predict(evaluate_x, batch_size=config.batch_size, verbose=0), dtype=np.float64
        )
        if (
            probabilities.shape != (18, 6)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        ):
            raise CalibrationError("calibration probabilities are invalid")
        with TemporaryDirectory(prefix="signlab-calibration-") as temporary:
            model_path = Path(temporary) / "model.keras"
            model.save(model_path)
            model_bytes = model_path.read_bytes()
    except CalibrationError:
        raise
    except Exception as error:
        raise CalibrationError("six-class calibration run failed") from error
    temperature_milli, calibrated = _fit_temperature(probabilities, evaluate_y, config)
    selected, risk_coverage = _policy_rows(calibrated, evaluate_y)
    model_sha256 = _sha256(model_bytes)
    policy = {
        "format": POLICY_FORMAT,
        "status": selected["status"],
        "class_map": {str(index): label for index, label in enumerate(LABELS)},
        "temperature": {
            "method": "softmax_log_probability_scalar_temperature/1",
            "temperature_milli": temperature_milli,
        },
        "abstention": {
            "threshold_percent": selected["threshold_percent"],
            "inclusive": True,
            "objective": OBJECTIVE,
        },
        "decision_precedence": [
            "no_candidate_to_inactive",
            "invalid_policy_or_probabilities_to_abstain",
            "below_threshold_to_abstain",
            "accepted_argmax_to_target_or_other",
        ],
        "identities": {
            "configuration_sha256": _sha256(config_bytes),
            "corpus_sha256": config.corpus_sha256,
            "split_sha256": config.split_sha256,
            "taxonomy_sha256": config.taxonomy_sha256,
            "source_feature_plan_sha256": config.feature_plan_sha256,
            "input_feature_plan_sha256": config.input_feature_plan_sha256,
            "derivative_set_sha256": derivative_sha256,
            "model_sha256": model_sha256,
        },
        "evidence_kind": EVIDENCE_KIND,
        "metric_claim": METRIC_CLAIM,
        "test_status": config.test_partition_policy,
    }
    policy_bytes = canonical_json_bytes(policy) + b"\n"
    predicted = calibrated.argmax(axis=1)
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (evaluate_y, predicted), 1)
    sample_aliases, signer_aliases = _aliases(validation)
    all_aliases = (*sample_aliases, *(f"fragment_{index:03d}" for index in range(1, 4)))
    row_signers = (*[row.source_signer_id for row in validation], *validation_other_signers)
    class_strata = [
        {
            "label": label,
            "count": int((evaluate_y == index).sum()),
            "argmax_correct": int(((evaluate_y == index) & (predicted == evaluate_y)).sum()),
        }
        for index, label in enumerate(LABELS)
    ]
    signer_strata = []
    for signer, alias in sorted(signer_aliases.items(), key=lambda row: row[1]):
        indexes = [index for index, value in enumerate(row_signers) if value == signer]
        signer_strata.append(
            {
                "signer_alias": alias,
                "rows": len(indexes),
                "target_rows": sum(index < 15 for index in indexes),
                "constructed_rows": sum(index >= 15 for index in indexes),
                "argmax_correct": int(
                    sum(predicted[index] == evaluate_y[index] for index in indexes)
                ),
            }
        )
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "run_name": config.run_name,
        "source": {"git_commit": commit, "git_dirty": dirty},
        "evidence_kind": EVIDENCE_KIND,
        "metric_claim": METRIC_CLAIM,
        "test_status": config.test_partition_policy,
        "identities": {
            **policy["identities"],
            "external_dataset_sha256": config.external_dataset_sha256,
            "policy_sha256": _sha256(policy_bytes),
        },
        "data": {
            "train_target_rows": len(train),
            "train_constructed_rows": len(train_other),
            "train_constructed_signers": len(set(train_other_signers)),
            "validation_target_rows": len(validation),
            "validation_constructed_rows": len(validation_other),
            "validation_constructed_signers": len(set(validation_other_signers)),
            "validation_rows": len(evaluate_y),
            "validation_unique_source_clips": len(validation),
        },
        "protocol": {
            "fit_calls": 1,
            "epochs": config.maximum_epochs,
            "seed": config.seed,
            "model": config.model,
            "model_parameters": _EXPECTED_PARAMETERS,
            "input_shape": [64, 126],
            "source_target_truth": "preserved",
            "constructed_examples": "separate_transition_fragment_rows",
        },
        "metrics": {
            "uncalibrated": {
                "nll": _nll(probabilities, evaluate_y),
                "brier": _brier(probabilities, evaluate_y),
                "reliability": _reliability(probabilities, evaluate_y),
            },
            "calibrated": {
                "nll": _nll(calibrated, evaluate_y),
                "brier": _brier(calibrated, evaluate_y),
                "reliability": _reliability(calibrated, evaluate_y),
            },
            "risk_coverage": risk_coverage,
        },
        "policy": {
            "status": selected["status"],
            "temperature_milli": temperature_milli,
            "threshold_percent": selected["threshold_percent"],
        },
        "strata": {
            "class": class_strata,
            "sanitized_signer": signer_strata,
            "unavailable": ["session", "capture_condition"],
        },
        "limitations": [
            "The 18 validation rows contain only 15 unique source clips.",
            (
                "The three other examples are deterministic transition fragments, "
                "not natural out-of-vocabulary signs."
            ),
            (
                "Validation fit the temperature and selected the threshold, so all "
                "metrics are development diagnostics."
            ),
            (
                "No session, capture-condition, continuous-signing, false-activations-per-hour, "
                "event-recall, promotion, or deployment claim is supported."
            ),
            "Test features were never requested.",
        ],
    }
    payloads = {
        "configuration_path": ("configuration.json", config_bytes),
        "report_path": ("report.json", canonical_json_bytes(report) + b"\n"),
        "confusion_matrix_path": (
            "validation-confusion.json",
            canonical_json_bytes(
                {
                    "format": CONFUSION_FORMAT,
                    "labels": list(LABELS),
                    "orientation": {"rows": "actual", "columns": "predicted"},
                    "calibrated_argmax": confusion.tolist(),
                }
            )
            + b"\n",
        ),
        "predictions_path": (
            "validation-predictions.csv",
            _predictions_csv(
                evaluate_y, calibrated, row_signers, all_aliases, signer_aliases, selected
            ),
        ),
    }
    try:
        destination.mkdir(parents=True)
    except OSError as error:
        raise CalibrationError("calibration output could not be created") from error
    paths: dict[str, Path] = {}
    for field, (filename, payload) in payloads.items():
        paths[field] = destination / filename
        _write(paths[field], payload)
    _write(destination / "model.keras", model_bytes)
    _write(destination / "decision-policy.json", policy_bytes)
    run = ReferenceRunInput(
        run_name=config.run_name,
        git_commit=commit,
        git_dirty=dirty,
        corpus_sha256=config.corpus_sha256,
        split_sha256=config.split_sha256,
        feature_plan_sha256=config.feature_plan_sha256,
        seed=config.seed,
        parameters={
            "fit_calls": 1,
            "model_parameters": _EXPECTED_PARAMETERS,
            "temperature_milli": temperature_milli,
            "threshold_percent": selected["threshold_percent"]
            if selected["threshold_percent"] is not None
            else -1,
            "policy_status": selected["status"],
            "test_status": config.test_partition_policy,
        },
        metrics={
            "validation.pre.nll": float(report["metrics"]["uncalibrated"]["nll"]),
            "validation.pre.brier": float(report["metrics"]["uncalibrated"]["brier"]),
            "validation.post.nll": float(report["metrics"]["calibrated"]["nll"]),
            "validation.post.brier": float(report["metrics"]["calibrated"]["brier"]),
        },
        **paths,
    )
    receipt = log_reference_run(run, tracking_uri=tracking_uri)
    if verify_reference_run(receipt.run_id, tracking_uri=tracking_uri) != receipt:
        raise CalibrationError("calibration ledger verification drifted")
    _write(public_policy, policy_bytes)
    _write(public_report, _markdown(report, receipt.run_id))
    return CalibrationResult(destination, public_report, public_policy, receipt)


__all__ = [
    "CONFIG_FORMAT",
    "EVIDENCE_KIND",
    "LABELS",
    "METRIC_CLAIM",
    "POLICY_FORMAT",
    "REPORT_FORMAT",
    "CalibrationConfig",
    "CalibrationError",
    "CalibrationResult",
    "load_calibration_config",
    "run_calibration",
]
