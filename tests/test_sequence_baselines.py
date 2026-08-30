from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from feature_fixtures import EXTRACTION_CONFIG_SHA256, make_feature_fixture
from signlab import cli
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments import sequence_baselines
from signlab.experiments.tracking import ReferenceRunInput, ReferenceRunReceipt
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import derive_feature_sequence

_CONFIG_PATH = Path("configs/experiments/popsign-sequence-baselines-v1.json")


def _config() -> sequence_baselines.SequenceBaselineConfig:
    return sequence_baselines.load_sequence_baseline_config(_CONFIG_PATH)[0]


def test_checked_in_config_is_canonical_frozen_and_vectorizes_one_sequence(
    tmp_path: Path,
) -> None:
    config, raw = sequence_baselines.load_sequence_baseline_config(_CONFIG_PATH)
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

    matrix, labels = sequence_baselines._vectorize((sample,), config)
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\r\n")
    assert config.models == ("gru", "tcn")
    assert config.labels == sequence_baselines.TARGET_LABELS
    assert (config.input_frames, config.input_width, config.seed) == (64, 134, 20260828)
    assert (config.maximum_epochs, config.early_stopping_patience) == (30, 8)
    assert (config.gru_units, config.tcn_channels, config.tcn_dilations) == (
        48,
        32,
        (1, 2, 4, 8),
    )
    assert matrix.shape == (1, 64, 134)
    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous
    assert labels.tolist() == [0]
    np.testing.assert_allclose(
        matrix[0],
        np.asarray(feature.values_q, dtype=np.float32) / np.float32(1_000_000),
    )

    payload = json.loads(raw)
    payload["models"] = ["tcn", "gru"]
    drifted = tmp_path / "drifted.json"
    drifted.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        sequence_baselines.SequenceBaselineError,
        match="configuration is invalid",
    ):
        sequence_baselines.load_sequence_baseline_config(drifted)
    with pytest.raises(ValueError, match="reviewed protocol drifted"):
        sequence_baselines.SequenceBaselineConfig.model_validate(
            {**config.model_dump(), "learning_rate": 0.002}
        )


def test_real_gru_and_tcn_have_fixed_repeatable_topologies() -> None:
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    config = _config()
    runtime = sequence_baselines._runtime()

    expected_layers = {
        "gru": ["input", "gru", "temporal_average", "probabilities"],
        "tcn": [
            "input",
            "input_projection",
            *[
                layer
                for dilation in config.tcn_dilations
                for layer in (
                    f"d{dilation}_conv_1",
                    f"d{dilation}_conv_2",
                    f"d{dilation}_residual",
                    f"d{dilation}_relu",
                )
            ],
            "temporal_average",
            "probabilities",
        ],
    }
    for name in sequence_baselines.MODEL_NAMES:
        sequence_baselines._seed_runtime(runtime, config.seed)
        first = sequence_baselines._build_model(name, config, runtime)
        first_weights = tuple(np.asarray(weight).copy() for weight in first.get_weights())
        sequence_baselines._seed_runtime(runtime, config.seed)
        second = sequence_baselines._build_model(name, config, runtime)

        assert first.input_shape == (None, 64, 134)
        assert first.output_shape == (None, 5)
        assert first.count_params() == {"gru": 26_741, "tcn": 29_317}[name]
        assert [layer.name for layer in first.layers] == expected_layers[name]
        assert all(
            np.array_equal(expected, actual)
            for expected, actual in zip(first_weights, second.get_weights(), strict=True)
        )
        if name == "gru":
            gru = first.get_layer("gru")
            assert gru.units == 48
            assert gru.return_sequences is True
            assert gru.use_cudnn is False
        else:
            assert [
                first.get_layer(f"d{dilation}_conv_1").dilation_rate
                for dilation in config.tcn_dilations
            ] == [(1,), (2,), (4,), (8,)]


def test_both_real_models_complete_one_fit_and_reload_best_and_last(tmp_path: Path) -> None:
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    config = _config().model_copy(
        update={
            "maximum_epochs": 1,
            "early_stopping_patience": 1,
            "latency_warmup_runs": 1,
            "latency_measurement_runs": 2,
        }
    )
    runtime = sequence_baselines._runtime()
    random = np.random.default_rng(20260828)
    train_x = random.normal(size=(10, 64, 134)).astype(np.float32)
    validation_x = random.normal(size=(5, 64, 134)).astype(np.float32)
    train_y = np.repeat(np.arange(5, dtype=np.int64), 2)
    validation_y = np.arange(5, dtype=np.int64)

    for name in sequence_baselines.MODEL_NAMES:
        run = sequence_baselines._train_one(
            name,
            runtime,
            config,
            train_x,
            train_y,
            validation_x,
            validation_y,
            tmp_path,
        )

        assert run.report["training"]["epochs_completed"] == 1
        assert run.report["training"]["best_epoch"] == 1
        assert run.report["parameter_count"] == {"gru": 26_741, "tcn": 29_317}[name]
        assert run.report["checkpoints"]["best"]["reload_verified"] is True
        assert run.report["checkpoints"]["last"]["reload_verified"] is True
        assert run.report["checkpoints"]["best"]["sha256"] == sequence_baselines._sha256(
            run.best_bytes
        )
        assert run.report["checkpoints"]["last"]["sha256"] == sequence_baselines._sha256(
            run.last_bytes
        )
        assert run.evaluation.probabilities.shape == (5, 5)
        assert np.isfinite(run.evaluation.probabilities).all()


def test_evaluation_failure_aliases_and_latency_are_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = np.arange(5, dtype=np.int64)
    predicted = np.asarray((0, 0, 2, 4, 4), dtype=np.int64)
    probabilities = np.full((5, 5), 0.025, dtype=np.float32)
    probabilities[np.arange(5), predicted] = 0.9
    evaluation = sequence_baselines._evaluate(expected, probabilities)
    samples: Sequence[Any] = tuple(
        SimpleNamespace(
            target_label_id=label,
            source_signer_id=f"private_signer_{index % 2}",
            quality_disposition="pass",
        )
        for index, label in enumerate(sequence_baselines.TARGET_LABELS)
    )
    sample_aliases, signer_aliases = sequence_baselines._aliases(samples)
    failures = sequence_baselines._failure_cases(
        samples,
        evaluation,
        sample_aliases,
        signer_aliases,
    )

    assert evaluation.metrics["accuracy"] == pytest.approx(3 / 5)
    assert evaluation.metrics["macro_f1"] == pytest.approx(7 / 15)
    assert evaluation.confusion == [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
    assert failures["observed_error_count"] == 2
    assert "private" not in json.dumps(failures)

    ticks = iter((100, 200, 300, 500, 800, 1100))
    calls = 0

    class Model:
        def __call__(self, _row: NDArray[np.float32], *, training: bool) -> None:
            nonlocal calls
            assert training is False
            calls += 1

    monkeypatch.setattr(sequence_baselines, "perf_counter_ns", lambda: next(ticks))
    latency = sequence_baselines._measure_latency(
        Model(),
        np.zeros((2, 64, 134), dtype=np.float32),
        warmup_runs=1,
        measurement_runs=3,
    )

    assert calls == 4
    assert latency["p50_ms"] == pytest.approx(0.0002)
    assert latency["p95_ms"] == pytest.approx(0.0003)


def test_checkpoint_runtime_and_output_boundaries_fail_closed(
    tmp_path: Path,
) -> None:
    class DriftedCheckpoint:
        input_shape = (None, 63, 134)
        output_shape = (None, 5)

        def count_params(self) -> int:
            return 26_741

    with pytest.raises(
        sequence_baselines.SequenceBaselineError,
        match="checkpoint contract drifted",
    ):
        sequence_baselines._verify_checkpoint(
            "gru",
            DriftedCheckpoint(),
            np.zeros((1, 64, 134), dtype=np.float32),
        )

    class InvalidPredictor:
        def predict(self, *_args: object, **_kwargs: object) -> NDArray[np.float32]:
            return np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(
        sequence_baselines.SequenceBaselineError,
        match="validation output is invalid",
    ):
        sequence_baselines._predict_probabilities(
            InvalidPredictor(), np.zeros((1, 64, 134), dtype=np.float32)
        )

    class FailingUtils:
        def set_random_seed(self, _seed: int) -> None:
            raise RuntimeError("private runtime detail")

    runtime = sequence_baselines._Runtime(
        keras=SimpleNamespace(utils=FailingUtils()),
        tensorflow=SimpleNamespace(),
    )
    with pytest.raises(
        sequence_baselines.SequenceBaselineError,
        match="deterministic Keras execution is unavailable",
    ) as failure:
        sequence_baselines._seed_runtime(runtime, 20260828)
    assert "private runtime detail" not in str(failure.value)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(
        sequence_baselines.SequenceBaselineError,
        match="must not already exist",
    ):
        sequence_baselines._new_path(existing, label="sequence-baseline output")


def _samples(partition: str, per_class: int) -> tuple[Any, ...]:
    return tuple(
        SimpleNamespace(
            partition=partition,
            target_label_id=label,
            sample_id=f"private_sample_{label}_{position}",
            source_recording_id=f"private_recording_{label}_{position}",
            source_signer_id=f"private_signer_{position % 2}",
            quality_disposition="pass",
        )
        for label in sequence_baselines.TARGET_LABELS
        for position in range(per_class)
    )


def test_one_run_uses_only_development_data_and_logs_four_sanitized_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    captured: list[ReferenceRunInput] = []
    requested_partitions: list[tuple[str, ...]] = []
    development = (*_samples("train", 10), *_samples("validation", 3))
    corpus_root = tmp_path / "private-corpus"
    manifest_path = tmp_path / "private-manifest.json"
    monkeypatch.setattr(
        sequence_baselines,
        "_git_identity",
        lambda _path: (Path.cwd(), "d" * 40, False),
    )
    monkeypatch.setattr(sequence_baselines, "_load_inputs", lambda *_args: (b"split", b"manifest"))

    def load_samples(
        _split: bytes,
        external_manifest_document: bytes,
        supplied_corpus_root: Path,
        partitions: tuple[str, ...],
    ) -> tuple[Any, ...]:
        assert external_manifest_document == b"manifest"
        assert supplied_corpus_root == corpus_root
        requested_partitions.append(partitions)
        events.append("load:" + ",".join(partitions))
        return development

    monkeypatch.setattr(sequence_baselines, "_load_samples", load_samples)

    def vectorize(
        samples: Sequence[Any],
        _config_value: sequence_baselines.SequenceBaselineConfig,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        events.append("vectorize:" + samples[0].partition)
        labels = np.asarray(
            [sequence_baselines.TARGET_LABELS.index(sample.target_label_id) for sample in samples],
            dtype=np.int64,
        )
        return np.zeros((len(samples), 64, 134), dtype=np.float32), labels

    monkeypatch.setattr(sequence_baselines, "_vectorize", vectorize)
    runtime = sequence_baselines._Runtime(
        keras=object(),
        tensorflow=SimpleNamespace(
            config=SimpleNamespace(
                get_visible_devices=lambda _kind: [],
                threading=SimpleNamespace(
                    get_inter_op_parallelism_threads=lambda: 1,
                    get_intra_op_parallelism_threads=lambda: 1,
                ),
            )
        ),
    )
    monkeypatch.setattr(sequence_baselines, "_runtime", lambda: runtime)
    monkeypatch.setattr(sequence_baselines, "version", lambda _name: "fixture")

    def train_one(
        name: sequence_baselines.ModelName,
        _runtime: object,
        _config_value: sequence_baselines.SequenceBaselineConfig,
        train_x: NDArray[np.float32],
        _train_y: NDArray[np.int64],
        validation_x: NDArray[np.float32],
        validation_y: NDArray[np.int64],
        _staging: Path,
    ) -> sequence_baselines._ModelRun:
        events.append("fit:" + name)
        assert train_x.shape == (50, 64, 134)
        assert validation_x.shape == (15, 64, 134)
        guessed = validation_y.copy()
        if name == "tcn":
            guessed[0] = 1
        probabilities = np.full((15, 5), 0.025, dtype=np.float32)
        probabilities[np.arange(15), guessed] = 0.9
        evaluation = sequence_baselines._evaluate(validation_y, probabilities)
        best = f"{name}-best-checkpoint".encode()
        last = f"{name}-last-checkpoint".encode()
        report = {
            "parameter_count": {"gru": 26_741, "tcn": 29_317}[name],
            "training": {
                "epochs_completed": 2,
                "best_epoch": 1,
                "best_recorded_validation_loss": evaluation.metrics["loss"],
                "fit_seconds": 0.1,
            },
            "checkpoints": {
                "best": {
                    "sha256": sequence_baselines._sha256(best),
                    "bytes": len(best),
                    "reload_verified": True,
                },
                "last": {
                    "sha256": sequence_baselines._sha256(last),
                    "bytes": len(last),
                    "reload_verified": True,
                },
            },
            "validation": evaluation.metrics,
            "latency": {
                "scope": "prevectorized_batch_1_forward_only",
                "warmup_runs": 10,
                "measurement_runs": 100,
                "p50_ms": 0.1,
                "p95_ms": 0.2,
            },
        }
        return sequence_baselines._ModelRun(report, best, last, evaluation)

    monkeypatch.setattr(sequence_baselines, "_train_one", train_one)
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
        events.append("track")
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

    monkeypatch.setattr(sequence_baselines, "log_reference_run", log_run)

    def verify_run(_run_id: str, **_kwargs: object) -> ReferenceRunReceipt:
        events.append("verify")
        return receipt

    monkeypatch.setattr(sequence_baselines, "verify_reference_run", verify_run)
    output = tmp_path / "run"
    public_report = tmp_path / "public.md"

    result = sequence_baselines.run_sequence_baselines(
        _CONFIG_PATH,
        corpus_root=corpus_root,
        external_manifest_path=manifest_path,
        output_root=output,
        public_report_path=public_report,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.sqlite').as_posix()}",
    )

    assert requested_partitions == [("train", "validation")]
    assert events == [
        "load:train,validation",
        "vectorize:train",
        "vectorize:validation",
        "fit:gru",
        "fit:tcn",
        "track",
        "verify",
    ]
    assert result.tracking == receipt
    assert len(captured) == 1
    assert captured[0].parameters["fit_calls"] == 2
    assert captured[0].parameters["checkpoint_files"] == 4
    assert captured[0].parameters["test_status"] == "sealed_not_opened"
    checkpoints = sorted(output.glob("*/*.keras"))
    assert [path.relative_to(output).as_posix() for path in checkpoints] == [
        "gru/best.keras",
        "gru/last.keras",
        "tcn/best.keras",
        "tcn/last.keras",
    ]
    report = json.loads((output / "report.json").read_bytes())
    assert report["claim_scope"] == "engineering_feasibility_only_no_winner"
    assert report["data"]["opened_partition_counts"] == {"train": 50, "validation": 15}
    assert report["data"]["test_status"] == "sealed_not_opened"
    assert report["protocol"]["fit_calls"] == 2
    assert report["protocol"]["checkpoint_files"] == 4
    assert report["models"]["tcn"]["failure_analysis"]["observed_error_count"] == 1
    public_evidence = b"\n".join(
        path.read_bytes()
        for path in (*output.rglob("*.json"), *output.rglob("*.csv"), public_report)
    )
    assert b"private" not in public_evidence.lower()


def test_cli_reports_one_verified_feasibility_run_without_private_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sequence_baselines,
        "run_sequence_baselines",
        lambda *_args, **_kwargs: SimpleNamespace(
            tracking=SimpleNamespace(run_id="e" * 32),
        ),
    )

    result = CliRunner(env={"NO_COLOR": "1"}).invoke(
        cli.app,
        [
            "train",
            "sequence-baselines",
            str(_CONFIG_PATH),
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
        f"GRU/TCN feasibility verified: four checkpoints reloaded; ledger run {'e' * 32}."
    )
    assert "private" not in result.output.casefold()
