"""One-run local MLflow ledger used by the first baseline story."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, Self, cast

from pydantic import Field, model_validator

from signlab import __version__
from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.contracts.core import (
    DottedId,
    FiniteFloat,
    GitCommit,
    ParameterScalar,
    SafeInteger,
    StableId,
    StrictContractModel,
)
from signlab.contracts.taxonomy import Sha256Digest

DEFAULT_TRACKING_URI: Final = "sqlite:///runs/mlflow.sqlite"
TRACKING_URI_ENV: Final = "SIGNLAB_MLFLOW_TRACKING_URI"
EXPERIMENT_NAME: Final = "signlab-reference-baselines"
LINEAGE_FORMAT: Final = "signlab-baseline-lineage/1"
_ARTIFACT_PATHS: Final = {
    "configuration.json": "configuration_path",
    "report.json": "report_path",
    "confusion-matrix.json": "confusion_matrix_path",
    "predictions.csv": "predictions_path",
}
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_ARTIFACT_ERROR = "reference-run artifacts are unavailable or invalid"
_DEPENDENCY_ERROR = "install the SignLab experiments extra to use tracking"
_LEDGER_ERROR = "the local experiment ledger operation failed"
_RUN_ERROR = "the tracked reference run is unavailable or invalid"
_URI_ERROR = "tracking URI must identify a persistent local SQLite file"
_VERIFICATION_ERROR = "reference-run artifact verification failed"


class ExperimentTrackingError(ValueError):
    """A stable, path-free failure from the optional experiment boundary."""


class ReferenceRunInput(StrictContractModel):
    """The exact small input that Story #24 supplies after evaluating its baselines."""

    run_name: StableId
    git_commit: GitCommit
    git_dirty: bool
    corpus_sha256: Sha256Digest
    split_sha256: Sha256Digest
    feature_plan_sha256: Sha256Digest
    seed: SafeInteger
    parameters: dict[DottedId, ParameterScalar] = Field(min_length=1, max_length=256)
    metrics: dict[DottedId, FiniteFloat] = Field(min_length=1, max_length=256)
    configuration_path: Path
    report_path: Path
    confusion_matrix_path: Path
    predictions_path: Path

    @model_validator(mode="after")
    def _reserve_seed_parameter(self) -> Self:
        if "seed" in self.parameters:
            raise ValueError("seed is a dedicated reference-run field")
        return self


class ReferenceRunReceipt(StrictContractModel):
    """Small tracker-neutral result returned after logging or verification."""

    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    experiment_id: str = Field(min_length=1, max_length=64)
    artifact_sha256: dict[str, Sha256Digest]


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _local_store(raw_uri: str) -> tuple[str, str]:
    prefix = "sqlite:///"
    if not raw_uri.startswith(prefix):
        raise ExperimentTrackingError(_URI_ERROR)
    database_text = raw_uri.removeprefix(prefix)
    if (
        not database_text
        or database_text == ":memory:"
        or any(character in database_text for character in ("?", "#", "%", "\\"))
    ):
        raise ExperimentTrackingError(_URI_ERROR)
    database = Path(database_text)
    if not database.is_absolute():
        database = Path.cwd() / database
    try:
        database = database.resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        artifact_root = database.parent / "mlflow-artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ExperimentTrackingError(_URI_ERROR) from error
    return f"{prefix}{database.as_posix()}", artifact_root.resolve().as_uri()


def _mlflow_client(tracking_uri: str) -> Any:
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    try:
        mlflow = importlib.import_module("mlflow")
    except (ImportError, ModuleNotFoundError) as error:
        raise ExperimentTrackingError(_DEPENDENCY_ERROR) from error
    try:
        return mlflow.MlflowClient(tracking_uri=tracking_uri)
    except Exception as error:
        raise ExperimentTrackingError(_LEDGER_ERROR) from error


def _artifact_payloads(run: ReferenceRunInput) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    try:
        for logical_name, field_name in _ARTIFACT_PATHS.items():
            path = getattr(run, field_name)
            if not path.is_file():
                raise OSError
            payload = path.read_bytes()
            if not payload or b"\0" in payload:
                raise OSError
            payloads[logical_name] = payload
    except OSError as error:
        raise ExperimentTrackingError(_ARTIFACT_ERROR) from error
    return payloads


def _parameter_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _runtime_identity() -> tuple[dict[str, str], dict[str, str | int]]:
    os_family = {"darwin": "macos"}.get(platform.system().casefold(), platform.system().casefold())
    environment = {
        "signlab_version": __version__,
        "python_version": platform.python_version(),
        "os_family": os_family or "unknown",
    }
    hardware: dict[str, str | int] = {
        "architecture": platform.machine().casefold() or "unknown",
        "logical_cpu_count": os.cpu_count() or 0,
    }
    return environment, hardware


def _lineage_document(
    run: ReferenceRunInput,
    payloads: dict[str, bytes],
) -> tuple[dict[str, object], dict[str, str]]:
    environment, hardware = _runtime_identity()
    artifact_sha256 = {name: _sha256(payload) for name, payload in sorted(payloads.items())}
    document: dict[str, object] = {
        "format": LINEAGE_FORMAT,
        "run_name": run.run_name,
        "source": {"git_commit": run.git_commit, "git_dirty": run.git_dirty},
        "identities": {
            "configuration_sha256": artifact_sha256["configuration.json"],
            "corpus_sha256": run.corpus_sha256,
            "feature_plan_sha256": run.feature_plan_sha256,
            "split_sha256": run.split_sha256,
        },
        "seed": run.seed,
        "environment": environment,
        "hardware": hardware,
        "parameters": dict(sorted(run.parameters.items())),
        "metrics": dict(sorted(run.metrics.items())),
        "artifact_sha256": artifact_sha256,
    }
    return document, artifact_sha256


def _tracking_values(
    lineage: Mapping[str, object],
    lineage_sha256: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    source = lineage.get("source")
    identities = lineage.get("identities")
    environment = lineage.get("environment")
    hardware = lineage.get("hardware")
    parameters = lineage.get("parameters")
    metrics = lineage.get("metrics")
    artifact_sha256 = lineage.get("artifact_sha256")
    if not all(
        isinstance(value, dict)
        for value in (
            source,
            identities,
            environment,
            hardware,
            parameters,
            metrics,
            artifact_sha256,
        )
    ):
        raise ExperimentTrackingError(_RUN_ERROR)
    source = cast(dict[str, object], source)
    identities = cast(dict[str, object], identities)
    environment = cast(dict[str, object], environment)
    hardware = cast(dict[str, object], hardware)
    parameters = cast(dict[str, object], parameters)
    metrics = cast(dict[str, object], metrics)
    artifact_sha256 = cast(dict[str, object], artifact_sha256)
    seed = lineage.get("seed")
    if type(seed) is not int or not all(
        type(value) in (bool, int, float, str) for value in parameters.values()
    ):
        raise ExperimentTrackingError(_RUN_ERROR)
    if not all(type(value) in (int, float) for value in metrics.values()):
        raise ExperimentTrackingError(_RUN_ERROR)
    tags = {
        "signlab.format": LINEAGE_FORMAT,
        "signlab.git_commit": str(source.get("git_commit")),
        "signlab.git_dirty": str(source.get("git_dirty")).casefold(),
        **{f"signlab.{key}": str(value) for key, value in identities.items()},
        **{f"signlab.environment.{key}": str(value) for key, value in environment.items()},
        **{f"signlab.hardware.{key}": str(value) for key, value in hardware.items()},
        **{
            f"signlab.artifact_sha256.{name}": str(digest)
            for name, digest in artifact_sha256.items()
        },
        "signlab.lineage_sha256": lineage_sha256,
    }
    logged_parameters = {
        name: _parameter_text(value) for name, value in {**parameters, "seed": seed}.items()
    }
    logged_metrics = {name: float(cast(int | float, value)) for name, value in metrics.items()}
    return tags, logged_parameters, logged_metrics


def _experiment_id(client: Any, artifact_location: str) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        if str(experiment.artifact_location).rstrip("/") != artifact_location.rstrip("/"):
            raise ExperimentTrackingError(_LEDGER_ERROR)
        return str(experiment.experiment_id)
    return str(client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location))


def log_reference_run(
    run: ReferenceRunInput,
    *,
    tracking_uri: str | None = None,
) -> ReferenceRunReceipt:
    """Log one completed baseline run to the local ledger and return its identity."""

    checked = ReferenceRunInput.model_validate(run)
    raw_uri = tracking_uri or os.environ.get(TRACKING_URI_ENV, DEFAULT_TRACKING_URI)
    local_uri, artifact_location = _local_store(raw_uri)
    payloads = _artifact_payloads(checked)
    lineage, artifact_sha256 = _lineage_document(checked, payloads)
    lineage_bytes = canonical_json_bytes(lineage) + b"\n"
    tags, parameters, metrics = _tracking_values(lineage, _sha256(lineage_bytes))
    client = _mlflow_client(local_uri)
    active_run_id: str | None = None
    try:
        experiment_id = _experiment_id(client, artifact_location)
        active = client.create_run(experiment_id, tags=tags, run_name=checked.run_name)
        active_run_id = str(active.info.run_id)
        for name, parameter_value in sorted(parameters.items()):
            client.log_param(active_run_id, name, parameter_value)
        for name, metric_value in sorted(metrics.items()):
            client.log_metric(active_run_id, name, metric_value)
        with TemporaryDirectory(prefix="signlab-mlflow-artifacts-") as staging_text:
            staging = Path(staging_text)
            for name, payload in payloads.items():
                (staging / name).write_bytes(payload)
            (staging / "lineage.json").write_bytes(lineage_bytes)
            client.log_artifacts(active_run_id, str(staging), artifact_path="evidence")
        client.set_terminated(active_run_id, status="FINISHED")
    except Exception as error:
        if active_run_id is not None:
            with suppress(Exception):
                client.set_terminated(active_run_id, status="FAILED")
        raise ExperimentTrackingError(_LEDGER_ERROR) from error
    return ReferenceRunReceipt(
        run_id=active_run_id,
        experiment_id=experiment_id,
        artifact_sha256=artifact_sha256,
    )


def verify_reference_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
) -> ReferenceRunReceipt:
    """Query one completed run and re-hash every artifact named by its lineage file."""

    if _RUN_ID.fullmatch(run_id) is None:
        raise ExperimentTrackingError(_RUN_ERROR)
    raw_uri = tracking_uri or os.environ.get(TRACKING_URI_ENV, DEFAULT_TRACKING_URI)
    local_uri, artifact_location = _local_store(raw_uri)
    client = _mlflow_client(local_uri)
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None or str(experiment.artifact_location).rstrip(
            "/"
        ) != artifact_location.rstrip("/"):
            raise ExperimentTrackingError(_RUN_ERROR)
        matches = client.search_runs(
            experiment_ids=[str(experiment.experiment_id)],
            filter_string=f"attributes.run_id = '{run_id}'",
        )
        if len(matches) != 1 or matches[0].info.status != "FINISHED":
            raise ExperimentTrackingError(_RUN_ERROR)
        tracked = matches[0]
        if tracked.data.tags.get("signlab.format") != LINEAGE_FORMAT:
            raise ExperimentTrackingError(_RUN_ERROR)
        with TemporaryDirectory(prefix="signlab-mlflow-verification-") as destination:
            lineage_path = Path(
                client.download_artifacts(run_id, "evidence/lineage.json", dst_path=destination)
            )
            lineage_bytes = lineage_path.read_bytes()
            if _sha256(lineage_bytes) != tracked.data.tags.get("signlab.lineage_sha256"):
                raise ExperimentTrackingError(_VERIFICATION_ERROR)
            lineage = parse_json_object(lineage_bytes)
            tags, parameters, metrics = _tracking_values(lineage, _sha256(lineage_bytes))
            if (
                tracked.info.run_name != lineage.get("run_name")
                or tracked.data.params != parameters
                or tracked.data.metrics != metrics
                or any(tracked.data.tags.get(key) != value for key, value in tags.items())
            ):
                raise ExperimentTrackingError(_RUN_ERROR)
            artifact_sha256 = lineage.get("artifact_sha256")
            if lineage.get("format") != LINEAGE_FORMAT or not isinstance(artifact_sha256, dict):
                raise ExperimentTrackingError(_RUN_ERROR)
            if set(artifact_sha256) != set(_ARTIFACT_PATHS):
                raise ExperimentTrackingError(_RUN_ERROR)
            for name, expected_sha256 in artifact_sha256.items():
                if tracked.data.tags.get(f"signlab.artifact_sha256.{name}") != expected_sha256:
                    raise ExperimentTrackingError(_VERIFICATION_ERROR)
                artifact_path = Path(
                    client.download_artifacts(run_id, f"evidence/{name}", dst_path=destination)
                )
                if _sha256(artifact_path.read_bytes()) != expected_sha256:
                    raise ExperimentTrackingError(_VERIFICATION_ERROR)
    except ExperimentTrackingError:
        raise
    except Exception as error:
        raise ExperimentTrackingError(_RUN_ERROR) from error
    return ReferenceRunReceipt(
        run_id=run_id,
        experiment_id=str(tracked.info.experiment_id),
        artifact_sha256=cast(dict[str, str], artifact_sha256),
    )


__all__ = [
    "DEFAULT_TRACKING_URI",
    "EXPERIMENT_NAME",
    "LINEAGE_FORMAT",
    "TRACKING_URI_ENV",
    "ExperimentTrackingError",
    "ReferenceRunInput",
    "ReferenceRunReceipt",
    "log_reference_run",
    "verify_reference_run",
]
