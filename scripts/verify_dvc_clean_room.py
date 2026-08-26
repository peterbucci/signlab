"""Prove the public DVC fixture through an isolated producer and consumer."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, cast

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.reproducibility import DVC_VERSION
from signlab.reproducibility.provenance import build_dvc_snapshot, dvc_snapshot_digest
from signlab.reproducibility.stages import (
    FIXTURE_IMPLEMENTATION,
    FIXTURE_PROFILE,
    STAGE_NAMES,
    STAGE_REGISTRY,
)

type JsonObject = dict[str, object]
type DigestMap = dict[str, str]
type Phase = Literal[
    "preflight",
    "clone-producer",
    "reproduce",
    "push",
    "clone-consumer",
    "pull",
    "compare",
    "report",
]

REPORT_SCHEMA: Final = "dvc-clean-room-proof/1"
CONSENT_STATUS: Final = "not_checked"
_REMOTE_NAME: Final = "clean-room"
_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_ENVIRONMENT_PREFIXES: Final = (
    "AWS_",
    "AZURE_",
    "DVC_",
    "GOOGLE_",
    "MLFLOW_",
    "SIGNLAB_DVC_",
)
_PHASES: Final[tuple[Phase, ...]] = (
    "preflight",
    "clone-producer",
    "reproduce",
    "push",
    "clone-consumer",
    "pull",
    "compare",
    "report",
)


class CleanRoomVerificationError(RuntimeError):
    """A sanitized failure containing only an allowlisted workflow phase."""

    def __init__(self, phase: Phase, completed: Sequence[Phase] = ()) -> None:
        self.phase = phase
        self.completed = tuple(completed)
        super().__init__(f"clean-room phase failed: {phase}")


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError("command failed")
    return result.stdout


def _git(repository: Path, environment: Mapping[str, str], *arguments: str) -> str:
    return _run(("git", *arguments), cwd=repository, environment=environment).strip()


def _dvc(repository: Path, environment: Mapping[str, str], *arguments: str) -> str:
    return _run(
        (sys.executable, "-I", "-m", "dvc", *arguments),
        cwd=repository,
        environment=environment,
    ).strip()


def _environment(repository: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(_PRIVATE_ENVIRONMENT_PREFIXES)
    }
    interpreter_directory = str(Path(sys.executable).parent)
    inherited_path = environment.get("PATH", "")
    environment.update(
        {
            "DVC_EXP_AUTO_PUSH": "false",
            "DVC_NO_ANALYTICS": "true",
            "DVC_STUDIO_OFFLINE": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(repository / "src"),
            "PYTHONUTF8": "1",
            "PATH": os.pathsep.join(
                part for part in (interpreter_directory, inherited_path) if part
            ),
            "TZ": "UTC",
        }
    )
    return environment


def _phase[T](name: Phase, completed: list[Phase], operation: Callable[[], T]) -> T:
    sys.stdout.write(f"Clean-room phase started: {name}.\n")
    try:
        result = operation()
    except CleanRoomVerificationError as error:
        raise CleanRoomVerificationError(error.phase, completed) from error
    except Exception as error:
        raise CleanRoomVerificationError(name, completed) from error
    completed.append(name)
    sys.stdout.write(f"Clean-room phase passed: {name}.\n")
    return result


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _output_hashes(repository: Path) -> DigestMap:
    hashes: DigestMap = {}
    for spec in STAGE_REGISTRY:
        payload = (repository / spec.output_path).read_bytes()
        document = parse_json_object(payload)
        if (
            document.get("fixture_only") is not True
            or document.get("implementation") != FIXTURE_IMPLEMENTATION
            or document.get("profile") != FIXTURE_PROFILE
            or document.get("stage") != spec.name
        ):
            raise ValueError("invalid fixture receipt")
        hashes[spec.name] = _sha256(payload)
    return hashes


def _normalize_dvc_lock_newlines(repository: Path) -> None:
    """Restore the LF form required by ``.gitattributes`` after Windows DVC writes."""

    path = repository / "dvc.lock"
    payload = path.read_bytes()
    normalized = payload.replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise ValueError("dvc.lock contains invalid newlines")
    if normalized != payload:
        path.write_bytes(normalized)


def _dvc_is_clean(repository: Path, environment: Mapping[str, str]) -> bool:
    return parse_json_object(_dvc(repository, environment, "status", "--json")) == {}


def _git_is_clean(repository: Path, environment: Mapping[str, str]) -> bool:
    return _git(repository, environment, "status", "--porcelain=v1", "--untracked-files=all") == ""


def _clone(source: Path, destination: Path, environment: Mapping[str, str], commit: str) -> None:
    _run(
        (
            "git",
            "clone",
            "--quiet",
            "--no-local",
            "--no-hardlinks",
            "--",
            str(source),
            str(destination),
        ),
        cwd=destination.parent,
        environment=environment,
    )
    clone_environment = _environment(destination)
    if _git(destination, clone_environment, "rev-parse", "HEAD") != commit:
        raise ValueError("clone commit mismatch")
    if not _git_is_clean(destination, clone_environment):
        raise ValueError("clone is not clean")


def _configure_local_remote(
    repository: Path,
    environment: Mapping[str, str],
    remote: Path,
) -> None:
    _dvc(
        repository,
        environment,
        "remote",
        "add",
        "--local",
        "--force",
        "--default",
        _REMOTE_NAME,
        str(remote),
    )


def _preflight(source: Path, report_path: Path) -> tuple[Path, str, dict[str, str]]:
    root = source.resolve(strict=True)
    environment = _environment(root)
    if not _git_is_clean(root, environment):
        raise ValueError("source is not clean")
    commit = _git(root, environment, "rev-parse", "--verify", "HEAD^{commit}")
    if len(commit) != 40:
        raise ValueError("invalid commit")
    if _dvc(root, environment, "--version") != DVC_VERSION:
        raise ValueError("wrong DVC version")
    if not report_path.is_absolute() or report_path.suffix.casefold() != ".json":
        raise ValueError("invalid report path")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return root, commit, environment


def _success_report(
    repository: Path,
    commit: str,
    completed: Sequence[Phase],
    producer_hashes: DigestMap,
    pulled_hashes: DigestMap,
) -> JsonObject:
    snapshot = build_dvc_snapshot(
        repository,
        commit,
        metadata_repository_role="public-fixture",
        git_working_tree_clean=True,
        dvc_workspace_clean=True,
    )
    return {
        "consent": CONSENT_STATUS,
        "dvc_lock_sha256": snapshot.dvc_lock_sha256,
        "dvc_snapshot_sha256": dvc_snapshot_digest(snapshot),
        "dvc_version": DVC_VERSION,
        "failed_phase": None,
        "fixture_only": True,
        "git_commit": commit,
        "phases": list(completed),
        "producer_output_sha256": producer_hashes,
        "pulled_output_sha256": pulled_hashes,
        "schema_version": REPORT_SCHEMA,
        "stage_lock_sha256": {
            stage.stage_name: stage.lock_entry_sha256 for stage in snapshot.stages
        },
        "stage_names": list(STAGE_NAMES),
        "status": "passed",
    }


def _failure_report(completed: Sequence[Phase], failed_phase: Phase) -> JsonObject:
    return {
        "consent": CONSENT_STATUS,
        "failed_phase": failed_phase,
        "fixture_only": True,
        "phases": list(completed),
        "schema_version": REPORT_SCHEMA,
        "status": "failed",
    }


def _write_report(report_path: Path, report: Mapping[str, object]) -> None:
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")


def validate_report(document: Mapping[str, object]) -> JsonObject:
    """Validate the stable public fields of a successful clean-room report."""

    required = {
        "consent",
        "dvc_lock_sha256",
        "dvc_snapshot_sha256",
        "dvc_version",
        "failed_phase",
        "fixture_only",
        "git_commit",
        "phases",
        "producer_output_sha256",
        "pulled_output_sha256",
        "schema_version",
        "stage_lock_sha256",
        "stage_names",
        "status",
    }
    phases = document.get("phases")
    stage_names = document.get("stage_names")
    producer_hashes = document.get("producer_output_sha256")
    pulled_hashes = document.get("pulled_output_sha256")
    stage_hashes = document.get("stage_lock_sha256")
    digest_maps = (producer_hashes, pulled_hashes, stage_hashes)
    if (
        set(document) != required
        or document.get("schema_version") != REPORT_SCHEMA
        or document.get("status") != "passed"
        or document.get("failed_phase") is not None
        or document.get("fixture_only") is not True
        or document.get("consent") != CONSENT_STATUS
        or document.get("dvc_version") != DVC_VERSION
        or type(document.get("git_commit")) is not str
        or _COMMIT_PATTERN.fullmatch(cast(str, document.get("git_commit"))) is None
        or any(
            type(document.get(field)) is not str
            or _DIGEST_PATTERN.fullmatch(cast(str, document.get(field))) is None
            for field in ("dvc_lock_sha256", "dvc_snapshot_sha256")
        )
        or type(phases) is not list
        or tuple(cast(list[object], phases)) != _PHASES
        or type(stage_names) is not list
        or tuple(cast(list[object], stage_names)) != STAGE_NAMES
        or any(
            type(values) is not dict
            or tuple(cast(dict[object, object], values)) != STAGE_NAMES
            or any(
                type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None
                for value in cast(dict[object, object], values).values()
            )
            for values in digest_maps
        )
        or producer_hashes != pulled_hashes
    ):
        raise CleanRoomVerificationError("report")
    return dict(document)


def canonical_report_bytes(document: Mapping[str, object]) -> bytes:
    """Canonicalize a validated successful report."""

    return canonical_json_bytes(validate_report(document)) + b"\n"


def run_clean_room(source_repository: Path, report_path: Path) -> JsonObject:
    """Run the producer/push/empty-consumer/pull proof."""

    completed: list[Phase] = []
    source, commit, source_environment = _phase(
        "preflight",
        completed,
        lambda: _preflight(source_repository, report_path),
    )
    with tempfile.TemporaryDirectory(prefix="signlab-dvc-clean-room-") as temporary:
        workspace = Path(temporary)
        producer = workspace / "producer"
        consumer = workspace / "consumer"
        remote = workspace / "remote"
        remote.mkdir()

        _phase(
            "clone-producer",
            completed,
            lambda: _clone(source, producer, source_environment, commit),
        )
        producer_environment = _environment(producer)

        def reproduce() -> DigestMap:
            _dvc(producer, producer_environment, "repro", "--force", "--no-run-cache")
            _normalize_dvc_lock_newlines(producer)
            hashes = _output_hashes(producer)
            if not _dvc_is_clean(producer, producer_environment):
                raise ValueError("producer DVC state is not clean")
            return hashes

        producer_hashes = _phase("reproduce", completed, reproduce)

        def push() -> None:
            _configure_local_remote(producer, producer_environment, remote)
            _dvc(producer, producer_environment, "push")
            if not any(path.is_file() for path in remote.rglob("*")):
                raise ValueError("remote is empty")

        _phase("push", completed, push)
        _phase(
            "clone-consumer",
            completed,
            lambda: _clone(source, consumer, source_environment, commit),
        )
        consumer_environment = _environment(consumer)

        def pull() -> DigestMap:
            if any((consumer / spec.output_path).exists() for spec in STAGE_REGISTRY):
                raise ValueError("consumer outputs are not empty")
            _configure_local_remote(consumer, consumer_environment, remote)
            _dvc(consumer, consumer_environment, "pull")
            return _output_hashes(consumer)

        pulled_hashes = _phase("pull", completed, pull)

        def compare() -> None:
            if producer_hashes != pulled_hashes:
                raise ValueError("output hashes differ")
            for repository, environment in (
                (producer, producer_environment),
                (consumer, consumer_environment),
            ):
                if not _dvc_is_clean(repository, environment) or not _git_is_clean(
                    repository, environment
                ):
                    raise ValueError("clone state is not clean")

        _phase("compare", completed, compare)

        def report() -> JsonObject:
            value = _success_report(
                producer,
                commit,
                (*completed, "report"),
                producer_hashes,
                pulled_hashes,
            )
            _write_report(report_path, value)
            return validate_report(value)

        return _phase("report", completed, report)


def _usage() -> str:
    return "Usage: verify_dvc_clean_room.py --report ABSOLUTE_JSON_PATH\n"


def main(arguments: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if values in (["-h"], ["--help"]):
        sys.stdout.write(_usage())
        return 0
    if len(values) != 2 or values[0] != "--report":
        sys.stderr.write(_usage())
        return 2
    report_path = Path(values[1])
    try:
        run_clean_room(Path.cwd(), report_path)
    except CleanRoomVerificationError as error:
        try:
            if report_path.is_absolute():
                report_path.parent.mkdir(parents=True, exist_ok=True)
                _write_report(report_path, _failure_report(error.completed, error.phase))
        except OSError:
            pass
        sys.stderr.write(f"Clean-room verification failed during phase: {error.phase}.\n")
        return 1
    except Exception:
        sys.stderr.write("Clean-room verification failed during phase: preflight.\n")
        return 1
    sys.stdout.write("Clean-room verification passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
