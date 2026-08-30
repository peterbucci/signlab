from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from feature_fixtures import EXTRACTION_CONFIG_SHA256, make_feature_fixture
from signlab import cli
from signlab.contracts.features import FeatureRepresentation
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments import representation_ablations as ablations
from signlab.experiments.sequence_baselines import (
    SequenceBaselineError,
    _evaluate,
    _Runtime,
)
from signlab.experiments.tracking import ReferenceRunInput, ReferenceRunReceipt
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import derive_feature_sequence

_CONFIG = Path("configs/experiments/popsign-representation-ablations-v1.json")


def _config() -> ablations.RepresentationAblationConfig:
    return ablations.load_representation_ablation_config(_CONFIG)[0]


def test_frozen_config_and_lossless_registered_view_projection(tmp_path: Path) -> None:
    config, raw = ablations.load_representation_ablation_config(_CONFIG)
    fixture = make_feature_fixture()
    features = {
        name: derive_feature_sequence(
            fixture.table,
            fixture.sequence,
            fixture.quality,
            load_packaged_default_feature_plan(name),
            extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
        )
        for name in ablations.REPRESENTATIONS
    }
    combined = features["combined"]
    sample = PublicCorpusSample(
        partition="train",
        target_label_id="hello",
        sample_id="sample_" + "1" * 32,
        source_recording_id=combined.source_recording_id,
        source_signer_id="participant_" + "2" * 32,
        quality_disposition="pass",
        feature=combined,
    )

    views, labels = ablations._views((sample,), config)

    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\r\n")
    assert tuple((row.model, row.representation) for row in config.conditions) == (
        ablations.EXPECTED_CONDITIONS
    )
    assert config.comparisons == ablations.EXPECTED_COMPARISONS
    assert (config.fold_count, config.fold_seed, config.maximum_epochs) == (3, 20260830, 30)
    assert config.feature_plan_id == "combined_64_frames"
    assert config.feature_plan_sha256 == config.source_feature_plan_sha256
    assert labels.tolist() == [0]
    view_widths: tuple[tuple[FeatureRepresentation, int], ...] = (
        ("hand_local", 126),
        ("body_relative", 8),
        ("combined", 134),
    )
    for name, width in view_widths:
        expected = np.asarray(cast(Any, features[name].values_q), dtype=np.float32) / np.float32(
            1_000_000
        )
        assert views[name].shape == (1, 64, width)
        assert views[name].flags.c_contiguous
        np.testing.assert_array_equal(views[name][0], expected)

    payload = json.loads(raw)
    payload["maximum_epochs"] = 29
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ablations.RepresentationAblationError, match="config is invalid"):
        ablations.load_representation_ablation_config(drifted)

    payload = json.loads(raw)
    payload["comparisons"]["architecture"] = ["gru_combined", "tcn_hand_local"]
    drifted_comparison = tmp_path / "drifted-comparison.json"
    drifted_comparison.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ablations.RepresentationAblationError, match="config is invalid"):
        ablations.load_representation_ablation_config(drifted_comparison)

    payload = json.loads(raw)
    payload["feature_views"]["hand_local"]["input_width"] = 125
    drifted_view = tmp_path / "drifted-view.json"
    drifted_view.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ablations.RepresentationAblationError, match="config is invalid"):
        ablations.load_representation_ablation_config(drifted_view)


def _grouped_samples() -> tuple[tuple[Any, ...], NDArray[np.int64]]:
    samples: list[Any] = []
    labels: list[int] = []
    for signer in range(13):
        for label_index, label in enumerate(ablations.TARGET_LABELS):
            samples.append(
                SimpleNamespace(
                    sample_id=f"sample_{signer:02d}_{label}",
                    source_signer_id=f"signer_{signer:02d}",
                    target_label_id=label,
                )
            )
            labels.append(label_index)
    return tuple(samples), np.asarray(labels, dtype=np.int64)


def test_three_folds_are_reproducible_group_isolated_and_label_complete() -> None:
    samples, labels = _grouped_samples()
    first, first_digest = ablations._folds(samples, labels, _config())
    second, second_digest = ablations._folds(samples, labels, _config())

    assert first == second
    assert first_digest == second_digest
    assert sorted(index for fold in first for index in fold.evaluate) == list(range(65))
    for fold in first:
        fit_signers = {samples[index].source_signer_id for index in fold.fit}
        evaluation_signers = {samples[index].source_signer_id for index in fold.evaluate}
        assert fit_signers.isdisjoint(evaluation_signers)
        assert {labels[index] for index in fold.fit} == set(range(5))
        assert {labels[index] for index in fold.evaluate} == set(range(5))


def test_neural_cell_uses_exact_final_epoch_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class Model:
        def fit(self, *_args: object, **kwargs: object) -> Any:
            calls["fit"] = kwargs
            return SimpleNamespace(history={"loss": [1.0] * 30})

        def count_params(self) -> int:
            return 29_061

    model = Model()
    runtime = _Runtime(
        keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: calls.setdefault("clear", 0))
        ),
        tensorflow=object(),
    )
    monkeypatch.setattr(ablations, "_seed_runtime", lambda *_args: calls.setdefault("seed", 0))
    monkeypatch.setattr(
        ablations,
        "_build_graph",
        lambda name, width, *_args: calls.update(model=name, width=width) or model,
    )
    probabilities = np.eye(5, dtype=np.float32)
    monkeypatch.setattr(ablations, "_predict_probabilities", lambda *_args: probabilities)
    monkeypatch.setattr(
        ablations,
        "_measure_deep_latency",
        lambda *_args, **_kwargs: {"p50_ms": 1.0, "p95_ms": 2.0},
    )
    train_x = np.zeros((10, 64, 126), dtype=np.float32)
    train_y = np.repeat(np.arange(5, dtype=np.int64), 2)
    evaluation_x = np.zeros((5, 64, 126), dtype=np.float32)
    evaluation_y = np.arange(5, dtype=np.int64)

    cell = ablations._fit_cell(
        "tcn", train_x, train_y, evaluation_x, evaluation_y, _config(), runtime
    )

    assert (calls["model"], calls["width"]) == ("tcn", 126)
    assert calls["fit"] == {
        "batch_size": 32,
        "epochs": 30,
        "shuffle": True,
        "verbose": 0,
    }
    assert cell.evaluation.predictions == ablations.TARGET_LABELS
    assert cell.parameters == 29_061


def test_logistic_cell_uses_the_fixed_registered_view() -> None:
    train_y = np.tile(np.arange(5, dtype=np.int64), 2)
    train_x = np.zeros((10, 64, 8), dtype=np.float32)
    train_x[np.arange(10), :, train_y] = 1.0
    evaluation_y = np.arange(5, dtype=np.int64)
    evaluation_x = np.zeros((5, 64, 8), dtype=np.float32)
    evaluation_x[np.arange(5), :, evaluation_y] = 1.0

    cell = ablations._fit_cell(
        "logistic",
        train_x,
        train_y,
        evaluation_x,
        evaluation_y,
        _config(),
        cast(_Runtime, SimpleNamespace()),
    )

    assert cell.evaluation.predictions == ablations.TARGET_LABELS
    assert cell.parameters == 2_565
    assert 0 <= cell.p50_ms <= cell.p95_ms


def test_runtime_failure_is_translated_without_private_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> _Runtime:
        raise SequenceBaselineError("runtime failed under C" + ":" + r"\Users\private\experiments")

    monkeypatch.setattr(ablations, "_runtime", fail)

    with pytest.raises(
        ablations.RepresentationAblationError,
        match=r"^neural ablation runtime is unavailable$",
    ) as caught:
        ablations._ablation_runtime()

    assert "private" not in str(caught.value).casefold()


def _cell(expected: NDArray[np.int64], predicted: NDArray[np.int64]) -> ablations._Cell:
    probabilities = np.full((len(expected), 5), 0.025, dtype=np.float32)
    probabilities[np.arange(len(expected)), predicted] = 0.9
    return ablations._Cell(_evaluate(expected, probabilities), 100, 1.0, 2.0)


def test_pooled_oof_metrics_and_comparison_decisions_are_exact() -> None:
    labels = np.tile(np.arange(5, dtype=np.int64), 3)
    samples = tuple(
        SimpleNamespace(
            target_label_id=ablations.TARGET_LABELS[int(label)],
            source_signer_id=f"private_signer_{index:02d}",
        )
        for index, label in enumerate(labels)
    )
    folds = (
        ablations._Fold(1, tuple(range(5, 15)), tuple(range(0, 5)), 10, 5),
        ablations._Fold(2, (*range(0, 5), *range(10, 15)), tuple(range(5, 10)), 10, 5),
        ablations._Fold(3, tuple(range(0, 10)), tuple(range(10, 15)), 10, 5),
    )
    cells = (
        _cell(labels[:5], labels[:5]),
        _cell(labels[5:10], np.zeros(5, dtype=np.int64)),
        _cell(labels[10:], labels[10:]),
    )
    signer_aliases = {
        sample.source_signer_id: f"signer_{index:03d}"
        for index, sample in enumerate(samples, start=1)
    }

    report, _evaluation = ablations._summarize(
        "tcn",
        "combined",
        cast(Sequence[PublicCorpusSample], samples),
        labels,
        folds,
        cells,
        signer_aliases,
    )

    fold_mean = np.mean([fold["metrics"]["macro_f1"] for fold in report["folds"]])
    assert report["out_of_fold"]["macro_f1"] == pytest.approx(0.76)
    assert report["out_of_fold"]["macro_f1"] != pytest.approx(fold_mean)
    assert sum(row["support"] for row in report["per_signer"].values()) == 15

    def reports(pooled: float, deltas: Sequence[float]) -> dict[str, dict[str, Any]]:
        return {
            "reference": {
                "out_of_fold": {"macro_f1": 0.5},
                "folds": [{"metrics": {"macro_f1": 0.5}} for _ in deltas],
            },
            "candidate": {
                "out_of_fold": {"macro_f1": 0.5 + pooled},
                "folds": [{"metrics": {"macro_f1": 0.5 + delta}} for delta in deltas],
            },
        }

    assert (
        ablations._compare(reports(0.06, (0.01, 0.06, 0.1)), "reference", "candidate", 0.05)[
            "decision"
        ]
        == "supported_for_carry_forward"
    )
    assert (
        ablations._compare(reports(0.02, (0.01, 0.02, 0.03)), "reference", "candidate", 0.05)[
            "decision"
        ]
        == "unsupported"
    )
    assert (
        ablations._compare(reports(0.06, (0.1, -0.01, 0.09)), "reference", "candidate", 0.05)[
            "decision"
        ]
        == "inconclusive"
    )


def _development_samples() -> tuple[Any, ...]:
    rows: list[Any] = []
    index = 0
    for label in ablations.TARGET_LABELS:
        for position in range(13):
            rows.append(
                SimpleNamespace(
                    partition="train" if position < 10 else "validation",
                    target_label_id=label,
                    sample_id=f"private_sample_{index:03d}",
                    source_signer_id=f"private_signer_{index % 33:03d}",
                )
            )
            index += 1
    return tuple(rows)


def test_one_run_executes_18_development_fits_and_verifies_sanitized_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _development_samples()
    labels = np.asarray(
        [ablations.TARGET_LABELS.index(sample.target_label_id) for sample in samples],
        dtype=np.int64,
    )
    folds = (
        ablations._Fold(1, tuple(range(22, 65)), tuple(range(0, 22)), 22, 11),
        ablations._Fold(2, (*range(0, 22), *range(44, 65)), tuple(range(22, 44)), 22, 11),
        ablations._Fold(3, tuple(range(0, 44)), tuple(range(44, 65)), 22, 11),
    )
    requested: list[tuple[str, ...]] = []
    fits: list[tuple[str, int, int]] = []
    captured: list[ReferenceRunInput] = []
    verified: list[str] = []
    monkeypatch.setattr(ablations, "_git_identity", lambda *_args: (Path.cwd(), "d" * 40, False))
    monkeypatch.setattr(ablations, "_load_inputs", lambda *_args: (b"split", b"manifest"))

    def load_samples(*_args: object, **_kwargs: object) -> tuple[Any, ...]:
        partitions = cast(Sequence[str], _args[-1])
        requested.append(tuple(partitions))
        return samples

    monkeypatch.setattr(ablations, "_load_samples", load_samples)
    monkeypatch.setattr(
        ablations,
        "_views",
        lambda *_args: (
            {
                "hand_local": np.zeros((65, 64, 126), dtype=np.float32),
                "body_relative": np.zeros((65, 64, 8), dtype=np.float32),
                "combined": np.zeros((65, 64, 134), dtype=np.float32),
            },
            labels,
        ),
    )
    monkeypatch.setattr(ablations, "_folds", lambda *_args: (folds, "sha256:" + "f" * 64))
    monkeypatch.setattr(ablations, "_runtime", lambda: SimpleNamespace())

    def fit_cell(
        model: str,
        train_x: NDArray[np.float32],
        _train_y: NDArray[np.int64],
        evaluation_x: NDArray[np.float32],
        evaluation_y: NDArray[np.int64],
        *_args: object,
    ) -> ablations._Cell:
        fits.append((model, train_x.shape[2], len(evaluation_x)))
        return _cell(evaluation_y, evaluation_y)

    monkeypatch.setattr(ablations, "_fit_cell", fit_cell)
    receipt = ReferenceRunReceipt(
        run_id="e" * 32,
        experiment_id="1",
        artifact_sha256={
            "configuration.json": "sha256:" + "1" * 64,
            "report.json": "sha256:" + "2" * 64,
            "confusion-matrix.json": "sha256:" + "3" * 64,
            "predictions.csv": "sha256:" + "4" * 64,
        },
    )

    def log_run(run: ReferenceRunInput, **_kwargs: object) -> ReferenceRunReceipt:
        captured.append(run)
        assert all(
            path.is_file()
            for path in (
                run.configuration_path,
                run.report_path,
                run.confusion_matrix_path,
                run.predictions_path,
            )
        )
        return receipt

    monkeypatch.setattr(ablations, "log_reference_run", log_run)

    def verify(run_id: str, **_kwargs: object) -> ReferenceRunReceipt:
        verified.append(run_id)
        return receipt

    monkeypatch.setattr(ablations, "verify_reference_run", verify)
    output = tmp_path / "private-output"
    public_report = tmp_path / "public.md"

    result = ablations.run_representation_ablations(
        _CONFIG,
        corpus_root=tmp_path / "private-corpus",
        external_manifest_path=tmp_path / "private-manifest.json",
        output_root=output,
        public_report_path=public_report,
    )

    assert result.tracking == receipt
    assert requested == [("train", "validation")]
    assert len(fits) == 18
    assert sum(model != "logistic" for model, _width, _count in fits) == 9
    assert len(captured) == 1
    assert verified == [receipt.run_id]
    assert captured[0].parameters["fit_calls"] == 18
    assert captured[0].parameters["test_status"] == "sealed_not_loaded"
    assert not tuple(output.rglob("*.keras"))
    evidence = b"\n".join(path.read_bytes() for path in (*output.iterdir(), public_report))
    assert b"private" not in evidence.lower()
    assert b"signer_" in evidence


def test_cli_reports_one_verified_ablation_without_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ablations,
        "run_representation_ablations",
        lambda *_args, **_kwargs: SimpleNamespace(tracking=SimpleNamespace(run_id="e" * 32)),
    )
    result = CliRunner(env={"NO_COLOR": "1"}).invoke(
        cli.app,
        [
            "train",
            "representation-ablations",
            str(_CONFIG),
            "--corpus-root",
            str(tmp_path / "private-corpus"),
            "--external-manifest",
            str(tmp_path / "private-manifest.json"),
            "--output-root",
            str(tmp_path / "private-output"),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == (
        f"Representation ablations verified: 18 fits across three folds; ledger run {'e' * 32}."
    )
    assert "private" not in result.output.casefold()
