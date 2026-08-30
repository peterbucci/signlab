from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from feature_fixtures import EXTRACTION_CONFIG_SHA256, make_feature_fixture
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments import calibration
from signlab.experiments.tracking import ReferenceRunReceipt
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import derive_feature_sequence

_CONFIG = Path("configs/experiments/popsign-constructed-calibration-v1.json")


def _config() -> calibration.CalibrationConfig:
    return calibration.load_calibration_config(_CONFIG)[0]


def _samples(partition: str, per_class: int) -> tuple[Any, ...]:
    return tuple(
        SimpleNamespace(
            partition=partition,
            target_label_id=label,
            sample_id=f"sample_{number:032x}",
            source_signer_id=f"signer_{number % 13:032x}",
        )
        for label_index, label in enumerate(calibration.TARGET_LABELS)
        for index in range(per_class)
        for number in (label_index * per_class + index,)
    )


def _fragment_inputs(count: int) -> tuple[tuple[Any, ...], np.ndarray[Any, np.dtype[np.float32]]]:
    rows = []
    matrix = np.zeros((count * 2, 64, 126), dtype=np.float32)
    for signer in range(count):
        for side, label in enumerate(("hello", "no")):
            index = signer * 2 + side
            matrix[index, :10] = index + 1
            rows.append(
                SimpleNamespace(
                    partition="train",
                    source_signer_id=f"signer_{signer:032x}",
                    target_label_id=label,
                    sample_id=f"sample_{index:032x}",
                    feature=SimpleNamespace(
                        padding_mask=(False,) * 10 + (True,) * 54,
                        sequence_sha256=f"sha256:{index:064x}",
                    ),
                )
            )
    return tuple(rows), matrix


def test_checked_in_config_is_canonical_and_frozen(tmp_path: Path) -> None:
    config, raw = calibration.load_calibration_config(_CONFIG)
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\r\n")
    assert config.labels == calibration.LABELS
    assert (config.model, config.input_width, config.maximum_epochs) == ("tcn", 126, 30)
    assert (config.train_derivatives, config.validation_derivatives) == (10, 3)
    assert config.test_partition_policy == "sealed_not_loaded"

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
        target_label_id="no",
        sample_id="sample_" + "1" * 32,
        source_recording_id=feature.source_recording_id,
        source_signer_id="participant_" + "2" * 32,
        quality_disposition="pass",
        feature=feature,
    )
    matrix, labels = calibration._hand_local_matrix((sample,), config)
    assert matrix.shape == (1, 64, 126)
    assert matrix.dtype == np.float32
    assert labels.tolist() == [1]
    with pytest.raises(calibration.CalibrationError, match="partition is empty"):
        calibration._hand_local_matrix((), config)

    payload = json.loads(raw)
    payload["labels"] = list(reversed(payload["labels"]))
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(calibration.CalibrationError, match="configuration is invalid"):
        calibration.load_calibration_config(drifted)


def test_transition_fragments_are_deterministic_same_signer_and_neutrally_padded() -> None:
    samples, matrix = _fragment_inputs(10)
    source = matrix.copy()
    fragments, signers, identity = calibration._transition_fragments(
        tuple(reversed(samples)), matrix[::-1].copy(), "train", 10
    )
    repeated = calibration._transition_fragments(samples, matrix, "train", 10)

    assert fragments.shape == (10, 64, 126)
    assert len(set(signers)) == 10
    assert identity == repeated[2]
    assert identity == "sha256:f8e1363f14c252b240610f7981eaeb53dec8f21dfe72a5238c642baa34df82da"
    np.testing.assert_array_equal(fragments, repeated[0])
    np.testing.assert_array_equal(repeated[0][0, :5], np.ones((5, 126), dtype=np.float32))
    np.testing.assert_array_equal(repeated[0][0, 5:10], np.full((5, 126), 2, dtype=np.float32))
    assert not repeated[0][:, 10:].any()
    np.testing.assert_array_equal(matrix, source)

    same_label = SimpleNamespace(**{**vars(samples[0]), "sample_id": f"sample_{1:032x}"})
    different = SimpleNamespace(**{**vars(samples[1]), "sample_id": f"sample_{2:032x}"})
    three_rows = np.stack((matrix[0], np.full_like(matrix[0], 9), matrix[1]))
    cross_label = calibration._transition_fragments(
        cast(Any, (samples[0], same_label, different)), three_rows, "train", 1
    )[0]
    assert not (cross_label == 9).any()
    with pytest.raises(calibration.CalibrationError, match="partition drifted"):
        calibration._transition_fragments(samples, matrix, "validation", 1)
    with pytest.raises(calibration.CalibrationError, match="quota is unavailable"):
        calibration._transition_fragments(samples, matrix, "train", 11)


def test_temperature_metrics_and_reliability_are_exact() -> None:
    uniform = np.full((6, 6), 1 / 6, dtype=np.float64)
    labels = np.arange(6, dtype=np.int64)
    temperature, scaled = calibration._fit_temperature(uniform, labels, _config())
    assert temperature == 1000
    np.testing.assert_allclose(scaled.sum(axis=1), 1)

    probabilities = np.asarray(
        [
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.04, 0.04, 0.04, 0.04, 0.04, 0.8],
        ],
        dtype=np.float64,
    )
    truth = np.asarray([0, 1, 5], dtype=np.int64)
    assert calibration._nll(probabilities, truth) == pytest.approx(
        -np.mean(np.log([0.2, 0.1, 0.8]))
    )
    assert calibration._brier(probabilities, truth) == pytest.approx((0.78 + 1.10 + 0.048) / 3)
    reliability = calibration._reliability(probabilities, truth)
    assert [row["count"] for row in reliability] == [1, 1, 1]
    assert reliability[0]["accuracy"] == 1
    assert reliability[1]["accuracy"] == 0
    assert reliability[2]["absolute_gap"] == pytest.approx(0.2)


def test_threshold_selection_is_inclusive_deterministic_and_fails_closed() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
            [0.06, 0.06, 0.7, 0.06, 0.06, 0.06],
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.6],
        ],
        dtype=np.float64,
    )
    selected, rows = calibration._policy_rows(probabilities, np.asarray([0, 1, 5], dtype=np.int64))
    assert selected["status"] == "selected"
    assert selected["threshold_percent"] == 71
    assert rows[70]["accepted_errors"] == 1
    assert rows[71]["accepted_errors"] == 0
    assert rows[71]["target_acceptance_coverage"] == 0.5
    assert rows[71]["constructed_abstention_rate"] == 1

    wrong = np.asarray([[0.02, 0.02, 0.02, 0.02, 0.02, 0.9]], dtype=np.float64)
    closed, _ = calibration._policy_rows(wrong, np.asarray([0], dtype=np.int64))
    assert closed == {"status": "no_viable_operating_point", "threshold_percent": None}


def test_one_run_keeps_test_sealed_tracks_and_sanitizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = []
    train, validation = _samples("train", 10), _samples("validation", 3)
    monkeypatch.setattr(calibration, "_git_identity", lambda _path: (tmp_path, "a" * 40, False))
    monkeypatch.setattr(calibration, "_load_inputs", lambda *_args: (b"split", b"manifest"))

    def load_samples(*_args: object, **_kwargs: object) -> tuple[Any, ...]:
        partitions = cast(tuple[str, ...], _args[-1])
        requested.append(partitions)
        return (*train, *validation)

    monkeypatch.setattr(calibration, "_load_samples", load_samples)

    def matrix(samples: tuple[Any, ...], _config: object) -> tuple[np.ndarray, np.ndarray]:
        labels = {label: index for index, label in enumerate(calibration.TARGET_LABELS)}
        return np.zeros((len(samples), 64, 126), dtype=np.float32), np.asarray(
            [labels[row.target_label_id] for row in samples], dtype=np.int64
        )

    monkeypatch.setattr(calibration, "_hand_local_matrix", matrix)

    def fragments(
        samples: tuple[Any, ...], _matrix: np.ndarray, partition: str, count: int
    ) -> tuple[np.ndarray, tuple[str, ...], str]:
        return (
            np.zeros((count, 64, 126), dtype=np.float32),
            tuple(row.source_signer_id for row in samples[:count]),
            "sha256:" + ("1" if partition == "train" else "2") * 64,
        )

    monkeypatch.setattr(calibration, "_transition_fragments", fragments)
    events: list[Any] = []

    class Model:
        input_shape = (None, 64, 126)
        output_shape = (None, 6)

        def count_params(self) -> int:
            return 29_094

        def fit(self, values: np.ndarray, _labels: np.ndarray, **kwargs: object) -> None:
            events.append(("fit", values.shape, kwargs))

        def predict(self, values: np.ndarray, **_kwargs: object) -> np.ndarray:
            events.append(("predict", values.shape))
            return np.full((18, 6), 1 / 6, dtype=np.float64)

        def save(self, path: Path) -> None:
            path.write_bytes(b"bounded-model")

    monkeypatch.setattr(calibration, "_runtime", lambda: SimpleNamespace())
    monkeypatch.setattr(calibration, "_seed_runtime", lambda *_args: None)
    monkeypatch.setattr(calibration, "_build_graph", lambda *_args: Model())
    receipt = ReferenceRunReceipt(
        run_id="e" * 32,
        experiment_id="1",
        artifact_sha256={
            name: "sha256:" + str(index) * 64
            for index, name in enumerate(
                ("configuration.json", "report.json", "confusion-matrix.json", "predictions.csv"),
                1,
            )
        },
    )
    monkeypatch.setattr(calibration, "log_reference_run", lambda *_args, **_kwargs: receipt)
    monkeypatch.setattr(calibration, "verify_reference_run", lambda *_args, **_kwargs: receipt)
    output, report, policy = tmp_path / "run", tmp_path / "report.md", tmp_path / "policy.json"
    result = calibration.run_calibration(
        _CONFIG,
        corpus_root=tmp_path / "private-corpus",
        external_manifest_path=tmp_path / "private-manifest.json",
        output_root=output,
        public_report_path=report,
        public_policy_path=policy,
    )

    assert requested == [("train", "validation")]
    assert events[0][0:2] == ("fit", (60, 64, 126))
    assert events[0][2]["epochs"] == 30
    assert events[1] == ("predict", (18, 64, 126))
    assert result.tracking == receipt
    evidence = b"\n".join(
        path.read_bytes()
        for path in (*output.glob("*.json"), *output.glob("*.csv"), report, policy)
    )
    assert b"private" not in evidence.lower()
    assert train[0].sample_id.encode() not in evidence
    assert train[0].source_signer_id.encode() not in evidence
    local_report = json.loads((output / "report.json").read_bytes())
    assert local_report["test_status"] == "sealed_not_loaded"
    assert local_report["evidence_kind"] == calibration.EVIDENCE_KIND
    assert local_report["metric_claim"] == calibration.METRIC_CLAIM
    assert local_report["strata"]["unavailable"] == ["session", "capture_condition"]
