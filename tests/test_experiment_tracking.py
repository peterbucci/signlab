from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from signlab.experiments import tracking
from signlab.experiments.tracking import (
    ExperimentTrackingError,
    ReferenceRunInput,
    log_reference_run,
    verify_reference_run,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None,
    reason="the experiments extra is not installed",
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _reference_input(root: Path) -> ReferenceRunInput:
    artifacts = {
        "configuration_path": (
            '{"format":"signlab-baseline-config/1","seed":20260828,'
            f'"corpus_sha256":"{_sha("a")}","split_sha256":"{_sha("b")}",'
            f'"feature_plan_sha256":"{_sha("c")}"}}\n'
        ).encode(),
        "report_path": b'{"format":"signlab-baseline-report/1","macro_f1":0.5}\n',
        "confusion_matrix_path": b'{"labels":["hello","other"],"matrix":[[1,0],[0,1]]}\n',
        "predictions_path": b"sample_id,actual,predicted\nsample_1,hello,hello\n",
    }
    paths: dict[str, Path] = {}
    for field_name, payload in artifacts.items():
        path = root / f"{field_name}.fixture"
        path.write_bytes(payload)
        paths[field_name] = path
    return ReferenceRunInput(
        run_name="reference_smoke",
        git_commit="d" * 40,
        git_dirty=True,
        corpus_sha256=_sha("a"),
        split_sha256=_sha("b"),
        feature_plan_sha256=_sha("c"),
        seed=20260828,
        parameters={"baseline_count": 3, "solver": "lbfgs"},
        metrics={"test.macro_f1": 0.5, "validation.macro_f1": 0.625},
        **paths,
    )


@pytest.mark.integration
def test_one_reference_run_logs_queries_verifies_and_detects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _reference_input(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.sqlite').resolve().as_posix()}"
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")

    logged = log_reference_run(run, tracking_uri=tracking_uri)
    verified = verify_reference_run(logged.run_id, tracking_uri=tracking_uri)

    assert verified == logged
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    mlflow = __import__("mlflow")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    matches = client.search_runs(
        [logged.experiment_id],
        filter_string=f"tags.`signlab.corpus_sha256` = '{run.corpus_sha256}'",
    )
    assert [match.info.run_id for match in matches] == [logged.run_id]
    tracked = matches[0]
    assert tracked.data.params == {
        "baseline_count": "3",
        "seed": "20260828",
        "solver": "lbfgs",
    }
    assert tracked.data.metrics == run.metrics
    assert tracked.data.tags["signlab.git_commit"] == run.git_commit
    assert tracked.data.tags["signlab.git_dirty"] == "true"
    assert tracked.data.tags["signlab.feature_plan_sha256"] == run.feature_plan_sha256
    assert tracked.data.tags["signlab.environment.python_version"]
    assert tracked.data.tags["signlab.hardware.architecture"]
    assert tracked.data.tags["signlab.lineage_sha256"].startswith("sha256:")

    client.set_tag(logged.run_id, "signlab.split_sha256", _sha("f"))
    client.log_metric(logged.run_id, "test.macro_f1", 0.0)
    with pytest.raises(ExperimentTrackingError, match="tracked reference run is unavailable"):
        verify_reference_run(logged.run_id, tracking_uri=tracking_uri)
    client.set_tag(logged.run_id, "signlab.split_sha256", run.split_sha256)
    client.log_metric(logged.run_id, "test.macro_f1", run.metrics["test.macro_f1"])

    evidence = tmp_path / "mlflow-artifacts" / logged.run_id / "artifacts" / "evidence"
    stored_lineage = evidence / "lineage.json"
    original_lineage = stored_lineage.read_bytes()
    stored_lineage.write_bytes(original_lineage.replace(b'"git_dirty":true', b'"git_dirty":false'))
    with pytest.raises(
        ExperimentTrackingError,
        match="reference-run artifact verification failed",
    ):
        verify_reference_run(logged.run_id, tracking_uri=tracking_uri)
    stored_lineage.write_bytes(original_lineage)

    stored_report = evidence / "report.json"
    stored_report.write_bytes(b"corrupted\n")
    with pytest.raises(
        ExperimentTrackingError,
        match="reference-run artifact verification failed",
    ):
        verify_reference_run(logged.run_id, tracking_uri=tracking_uri)


@pytest.mark.parametrize(
    "tracking_uri",
    ["file:///runs/mlflow", "sqlite:///:memory:", "sqlite:///runs/mlflow.sqlite?mode=ro"],
)
def test_non_persistent_or_non_local_tracking_uri_is_rejected(tracking_uri: str) -> None:
    with pytest.raises(ExperimentTrackingError, match="persistent local SQLite"):
        tracking._local_store(tracking_uri)


def test_relative_tracking_uri_resolves_beneath_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    tracking_uri, artifact_uri = tracking._local_store("sqlite:///tracking.sqlite")

    assert tracking_uri == f"sqlite:///{(tmp_path / 'tracking.sqlite').as_posix()}"
    assert artifact_uri == (tmp_path / "mlflow-artifacts").as_uri()


def test_missing_experiment_dependency_has_a_stable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_name: str) -> None:
        raise ModuleNotFoundError

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ExperimentTrackingError, match="experiments extra"):
        tracking._mlflow_client("sqlite:///unused.sqlite")


def test_missing_artifact_fails_before_a_run_is_created(tmp_path: Path) -> None:
    run = _reference_input(tmp_path)
    run.report_path.unlink()

    with pytest.raises(ExperimentTrackingError, match="artifacts are unavailable"):
        log_reference_run(
            run,
            tracking_uri=f"sqlite:///{(tmp_path / 'tracking.sqlite').resolve().as_posix()}",
        )


def test_nul_containing_artifact_is_rejected(tmp_path: Path) -> None:
    run = _reference_input(tmp_path)
    run.predictions_path.write_bytes(b"invalid\0predictions")

    with pytest.raises(ExperimentTrackingError, match="artifacts are unavailable"):
        tracking._artifact_payloads(run)


def test_seed_is_reserved_for_the_dedicated_lineage_field(tmp_path: Path) -> None:
    values = _reference_input(tmp_path).model_dump()
    values["parameters"] = {"seed": 1}

    with pytest.raises(ValueError, match="dedicated reference-run field"):
        ReferenceRunInput.model_validate(values)


def test_existing_experiment_must_use_the_derived_artifact_store() -> None:
    expected = SimpleNamespace(experiment_id="1", artifact_location="file:///expected-store")
    assert (
        tracking._experiment_id(
            SimpleNamespace(get_experiment_by_name=lambda _name: expected),
            "file:///expected-store",
        )
        == "1"
    )
    client = SimpleNamespace(
        get_experiment_by_name=lambda _name: SimpleNamespace(
            experiment_id="1",
            artifact_location="file:///wrong-store",
        )
    )

    with pytest.raises(ExperimentTrackingError, match="ledger operation failed"):
        tracking._experiment_id(client, "file:///expected-store")


def test_invalid_or_unknown_run_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ExperimentTrackingError, match="tracked reference run is unavailable"):
        verify_reference_run("not-a-run-id")
    with pytest.raises(ExperimentTrackingError, match="tracked reference run is unavailable"):
        verify_reference_run(
            "f" * 32,
            tracking_uri=f"sqlite:///{(tmp_path / 'tracking.sqlite').resolve().as_posix()}",
        )


def test_corrupt_tracking_database_has_a_sanitized_failure(tmp_path: Path) -> None:
    database = tmp_path / "corrupt.sqlite"
    database.write_bytes(b"not a SQLite database")

    with pytest.raises(ExperimentTrackingError, match="ledger operation failed"):
        log_reference_run(
            _reference_input(tmp_path),
            tracking_uri=f"sqlite:///{database.resolve().as_posix()}",
        )


def test_failed_logging_marks_the_created_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses: list[str] = []

    class FailingClient:
        def get_experiment_by_name(self, _name: str) -> None:
            return None

        def create_experiment(self, _name: str, *, artifact_location: str) -> str:
            assert artifact_location.startswith("file:")
            return "1"

        def create_run(self, _experiment_id: str, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(info=SimpleNamespace(run_id="e" * 32))

        def log_param(self, _run_id: str, _name: str, _value: str) -> None:
            raise RuntimeError

        def set_terminated(self, _run_id: str, *, status: str) -> None:
            statuses.append(status)

    monkeypatch.setattr(tracking, "_mlflow_client", lambda _uri: FailingClient())
    with pytest.raises(ExperimentTrackingError, match="ledger operation failed"):
        log_reference_run(
            _reference_input(tmp_path),
            tracking_uri=f"sqlite:///{(tmp_path / 'tracking.sqlite').resolve().as_posix()}",
        )
    assert statuses == ["FAILED"]


def test_tracking_module_import_does_not_load_mlflow() -> None:
    probe = (
        "import sys; import signlab.experiments.tracking; "
        "raise SystemExit('mlflow imported' if 'mlflow' in sys.modules else 0)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
