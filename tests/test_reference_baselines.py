from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from feature_fixtures import EXTRACTION_CONFIG_SHA256, make_feature_fixture
from signlab import cli
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments import baselines
from signlab.experiments.tracking import ReferenceRunInput, ReferenceRunReceipt
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import derive_feature_sequence

_CONFIG_PATH = Path("configs/experiments/popsign-reference-baselines-v1.json")


def _config() -> baselines.ReferenceBaselineConfig:
    return baselines.load_reference_baseline_config(_CONFIG_PATH)[0]


def test_checked_in_config_is_canonical_and_vectorizes_the_portable_sequence() -> None:
    config, raw = baselines.load_reference_baseline_config(_CONFIG_PATH)
    fixture = make_feature_fixture()
    feature = derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        load_packaged_default_feature_plan("combined"),
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )
    sample = PublicCorpusSample(
        partition="train",
        target_label_id="hello",
        sample_id="sample_" + "1" * 32,
        source_recording_id=feature.source_recording_id,
        source_signer_id="participant_" + "2" * 32,
        quality_disposition="pass",
        feature=feature,
    )

    matrix, labels = baselines._vectorize((sample,), config)

    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\r\n")
    assert matrix.shape == (1, 64 * 134)
    assert labels.tolist() == ["hello"]
    assert matrix[0].tolist() == pytest.approx(
        [value / 1_000_000 for row in feature.values_q for value in row]
    )


def test_source_identity_and_configured_input_identities_are_captured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    repository, commit, _dirty = baselines._git_identity(Path.cwd())
    assert repository == Path.cwd().resolve()
    assert len(commit) == 40
    assert set(commit) <= set("0123456789abcdef")

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "public-corpus-split.json").write_bytes(b"split")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"manifest")
    split = SimpleNamespace(
        source_corpus_sha256=config.corpus_sha256,
        external_dataset_sha256=config.external_dataset_sha256,
        split_sha256=config.split_sha256,
        feature_plan_id=config.feature_plan_id,
        feature_plan_sha256=config.feature_plan_sha256,
    )
    manifest = SimpleNamespace(
        content_sha256=config.external_dataset_sha256,
        taxonomy=SimpleNamespace(
            id=config.taxonomy_id,
            version=config.taxonomy_version,
            sha256=config.taxonomy_sha256,
        ),
    )
    plan = SimpleNamespace(
        plan_id=config.feature_plan_id,
        learned_statistics=SimpleNamespace(mode="none"),
    )
    monkeypatch.setattr(baselines, "validate_public_corpus_split", lambda _value: split)
    monkeypatch.setattr(baselines, "validate_external_dataset_manifest", lambda _value: manifest)
    monkeypatch.setattr(baselines, "load_packaged_default_feature_plan", lambda _value: plan)
    monkeypatch.setattr(
        baselines, "landmark_feature_plan_digest", lambda _value: config.feature_plan_sha256
    )

    assert baselines._load_inputs(config, corpus_root, manifest_path) == (b"split", b"manifest")


def test_latency_uses_warmups_and_nearest_rank_percentiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((100, 200, 300, 500, 800, 1100))
    calls = 0

    class Predictor:
        def predict(self, _matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            nonlocal calls
            calls += 1
            return np.asarray(["hello"])

    monkeypatch.setattr(baselines, "perf_counter_ns", lambda: next(ticks))
    result = baselines._measure_latency(
        Predictor(),
        np.zeros((2, 1)),
        warmup_runs=1,
        measurement_runs=3,
    )

    assert calls == 4
    assert result["p50_ms"] == pytest.approx(0.0002)
    assert result["p95_ms"] == pytest.approx(0.0003)


def test_three_models_are_seeded_train_only_and_use_one_validation_tie_break() -> None:
    config = _config()
    train_y = np.asarray(
        [label for label in baselines.TARGET_LABELS for _ in range(4)], dtype=np.str_
    )
    train_x = np.asarray(
        [
            [float(index == baselines.TARGET_LABELS.index(label)) for index in range(5)]
            for label in train_y.tolist()
        ]
    )
    validation_y = np.asarray(baselines.TARGET_LABELS, dtype=np.str_)
    validation_x = np.eye(5, dtype=np.float64)

    models, selection = baselines._fit_models(train_x, train_y, validation_x, validation_y, config)

    assert tuple(models) == baselines.BASELINE_NAMES
    assert baselines._predict(models["majority"], validation_x) == ("hello",) * 5
    random_predictions = baselines._predict(models["stratified_random"], validation_x)
    assert random_predictions == baselines._predict(models["stratified_random"], validation_x)
    assert random_predictions == ("hello", "please", "hello", "no", "thank_you")
    validation_random = baselines._evaluate(
        models["stratified_random"],
        validation_x,
        validation_y,
        random_state=config.seed + 1,
    ).predictions
    test_random = baselines._evaluate(
        models["stratified_random"],
        validation_x,
        validation_y,
        random_state=config.seed + 2,
    ).predictions
    assert (
        validation_random
        == baselines._evaluate(
            models["stratified_random"],
            validation_x,
            validation_y,
            random_state=config.seed + 1,
        ).predictions
    )
    assert test_random != validation_random
    assert selection.selected_c == 0.1
    assert [row["c"] for row in selection.candidate_scores] == [0.1, 1.0]
    assert selection.model.coef_.shape == (5, 5)

    evaluation = baselines._evaluate(
        SimpleNamespace(
            predict=lambda _matrix: np.asarray(
                ("hello", "hello", "please", "yes", "yes"), dtype=np.str_
            )
        ),
        validation_x,
        validation_y,
    )
    assert evaluation.metrics["macro_f1"] == pytest.approx(7 / 15)
    assert evaluation.metrics["balanced_accuracy"] == pytest.approx(3 / 5)
    assert evaluation.metrics["per_class"]["no"] == {
        "precision": 0.0,
        "recall": 0.0,
        "support": 1,
    }
    assert evaluation.confusion == [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]


def _samples(partition: str, per_class: int) -> tuple[Any, ...]:
    rows = []
    index = 0
    for label in baselines.TARGET_LABELS:
        for position in range(per_class):
            rows.append(
                SimpleNamespace(
                    partition=partition,
                    target_label_id=label,
                    sample_id=f"private_sample_{index:03d}",
                    source_recording_id=f"private_recording_{index:03d}",
                    source_signer_id=f"private_signer_{position % 2:03d}",
                    quality_disposition=(
                        "warning" if partition == "test" and index in {8, 9} else "pass"
                    ),
                )
            )
            index += 1
    return tuple(rows)


class _PerfectPredictor:
    def set_params(self, **_parameters: object) -> _PerfectPredictor:
        return self

    def predict(self, matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray([baselines.TARGET_LABELS[int(row[0])] for row in matrix], dtype=np.str_)


def test_full_command_seals_selection_before_test_and_hands_one_run_to_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    captured: list[ReferenceRunInput] = []
    development = (*_samples("train", 10), *_samples("validation", 3))
    test = _samples("test", 3)

    monkeypatch.setattr(baselines, "_git_identity", lambda _path: (Path.cwd(), "d" * 40, False))
    monkeypatch.setattr(baselines, "_load_inputs", lambda *_args: (b"split", b"manifest"))

    def load_samples(
        _split: bytes,
        _manifest: bytes,
        _root: Path,
        partitions: tuple[str, ...],
    ) -> tuple[Any, ...]:
        events.append("load:" + ",".join(partitions))
        return development if "train" in partitions else test

    def vectorize(
        samples: tuple[Any, ...], _config: baselines.ReferenceBaselineConfig
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        matrix = np.zeros((len(samples), 8576), dtype=np.float64)
        labels = np.asarray([sample.target_label_id for sample in samples], dtype=np.str_)
        matrix[:, 0] = [baselines.TARGET_LABELS.index(label) for label in labels.tolist()]
        return matrix, labels

    predictor = _PerfectPredictor()

    def fit_models(*_args: object) -> tuple[dict[str, Any], Any]:
        events.append("fit")
        logistic = SimpleNamespace(
            predict=predictor.predict,
            coef_=np.zeros((5, 8576)),
            intercept_=np.zeros(5),
            n_iter_=np.asarray([7]),
        )
        return (
            {
                name: predictor if name != "logistic_regression" else logistic
                for name in baselines.BASELINE_NAMES
            },
            baselines._LogisticSelection(
                logistic,
                0.1,
                (
                    {"c": 0.1, "validation_macro_f1": 1.0},
                    {"c": 1.0, "validation_macro_f1": 1.0},
                ),
            ),
        )

    monkeypatch.setattr(baselines, "_load_samples", load_samples)
    monkeypatch.setattr(baselines, "_vectorize", vectorize)
    monkeypatch.setattr(baselines, "_fit_models", fit_models)
    monkeypatch.setattr(
        baselines,
        "_measure_latency",
        lambda *_args, **_kwargs: {
            "batch_size": 1,
            "measurement_runs": 100,
            "p50_ms": 0.1,
            "p95_ms": 0.2,
            "scope": "prevectorized_model_predict_cpu_single_thread",
            "warmup_runs": 10,
        },
    )
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

    def log_reference_run(run: ReferenceRunInput, **_kwargs: object) -> ReferenceRunReceipt:
        captured.append(run)
        return receipt

    monkeypatch.setattr(baselines, "log_reference_run", log_reference_run)
    monkeypatch.setattr(baselines, "verify_reference_run", lambda *_args, **_kwargs: receipt)
    output = tmp_path / "run"
    public_report = tmp_path / "public.md"

    result = baselines.run_reference_baselines(
        _CONFIG_PATH,
        corpus_root=tmp_path / "licensed",
        external_manifest_path=tmp_path / "external.json",
        output_root=output,
        public_report_path=public_report,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.sqlite').as_posix()}",
    )

    assert events == ["load:train,validation", "fit", "load:test"]
    assert result.tracking == receipt
    assert result.selected_c == 0.1
    assert len(captured) == 1
    assert captured[0].seed == 20260828
    assert captured[0].parameters["test_access_after_selection"] is True
    assert all(
        path.is_file() for path in captured[0].model_dump().values() if isinstance(path, Path)
    )
    report = json.loads((output / "report.json").read_bytes())
    assert report["selection"]["test_access"] == "after_selection_sealed"
    assert report["selection"]["refit_after_selection"] is False
    assert report["models"]["logistic_regression"]["parameter_count"] == 42_885
    assert b"private_" not in (output / "predictions.csv").read_bytes()
    assert result.public_report_path == public_report.resolve()
    assert public_report.is_file()
    assert not (output / "public-report.md").exists()


def test_failure_analysis_groups_repeated_signers_without_source_identifiers() -> None:
    samples = _samples("test", 1)
    predictions = ("no", "no", "please", "yes", "yes")

    result = baselines._failure_analysis(samples, predictions)

    assert result["by_class"]["hello"] == {"support": 1, "errors": 1}
    assert result["by_quality_disposition"]["pass"] == {"support": 5, "errors": 2}
    assert result["by_signer"] == [{"signer_alias": "signer_001", "support": 5, "errors": 2}]
    assert "private" not in json.dumps(result)


def test_cli_runs_the_one_reference_command_without_loading_paths_into_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        baselines,
        "run_reference_baselines",
        lambda *_args, **_kwargs: SimpleNamespace(
            selected_c=0.1,
            tracking=SimpleNamespace(run_id="e" * 32),
        ),
    )
    result = CliRunner(env={"NO_COLOR": "1"}).invoke(
        cli.app,
        [
            "train",
            "reference-baselines",
            str(_CONFIG_PATH),
            "--corpus-root",
            str(tmp_path / "private-corpus"),
            "--external-manifest",
            str(tmp_path / "private-manifest.json"),
            "--output-root",
            str(tmp_path / "run"),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == (
        f"Reference baselines verified: selected C=0.1; ledger run {'e' * 32}."
    )
    assert "private" not in result.output
