"""Prove the public synthetic DVC graph from two isolated local clones.

The verifier is intentionally a standalone release check. It never reads a configured
private remote, never publishes a machine path, and never treats a cache hit as proof
that the six stage commands can execute from committed source.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Final, Never, cast

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.reproducibility.provenance import (
    DVC_VERSION,
    EXPECTED_DVC_STAGES,
    PUBLIC_REPOSITORY_ORIGINS,
    DvcSnapshotV1,
    build_dvc_snapshot,
    dvc_snapshot_digest,
)
from signlab.reproducibility.stages import STAGE_REGISTRY

type JsonObject = dict[str, object]
type DigestMap = dict[str, str]

REPORT_SCHEMA: Final = "dvc-clean-room-proof/1"
CONSENT_STATUS: Final = "not_checked"
MAX_JSON_BYTES: Final = 1_048_576
COMMAND_TIMEOUT_SECONDS: Final = 300
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_STAGE_NAMES: Final = tuple(EXPECTED_DVC_STAGES)
_CONTROL_FILES: Final = ("dvc.yaml", "dvc.lock", "uv.lock")
_DVC_REMOTE_NAME: Final = "clean-room"
_REPARSE_POINT: Final = 0x400
_REPORT_BOOLEAN_FIELDS: Final = (
    "producer_all_stages_executed",
    "producer_metadata_unchanged",
    "producer_dvc_clean",
    "producer_git_clean",
    "remote_push_verified",
    "consumer_cache_initially_empty",
    "consumer_outputs_initially_absent",
    "remote_pull_verified",
    "consumer_dvc_clean_after_pull",
    "offline_remote_unavailable",
    "offline_all_stages_executed",
    "offline_reproduction_verified",
    "consumer_metadata_unchanged",
    "consumer_dvc_clean",
    "consumer_git_clean",
)
_REPORT_DIGEST_FIELDS: Final = (
    "stage_lock_sha256",
    "producer_output_sha256",
    "pulled_output_sha256",
    "offline_output_sha256",
)
_REPORT_KEYS: Final = frozenset(
    {
        "schema_version",
        "fixture_only",
        "git_commit",
        "dvc_version",
        "uv_lock_sha256",
        "dvc_yaml_sha256",
        "dvc_lock_sha256",
        "dvc_snapshot_sha256",
        "stage_names",
        "consent",
        *_REPORT_BOOLEAN_FIELDS,
        *_REPORT_DIGEST_FIELDS,
    }
)


class CleanRoomVerificationError(RuntimeError):
    """Sanitized failure at the clean-room proof boundary."""


def _fail(message: str = "clean-room verification failed") -> Never:
    raise CleanRoomVerificationError(message) from None


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        _fail("required path is unavailable")
    attributes = getattr(details, "st_file_attributes", 0)
    return stat.S_ISLNK(details.st_mode) or bool(attributes & _REPARSE_POINT)


def _require_directory(path: Path) -> None:
    try:
        details = path.lstat()
    except OSError:
        _fail("required directory is unavailable")
    if _is_link_or_reparse(path) or not stat.S_ISDIR(details.st_mode):
        _fail("required directory is unsafe")


def _require_regular_file(path: Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> None:
    try:
        details = path.lstat()
    except OSError:
        _fail("required file is unavailable")
    if (
        _is_link_or_reparse(path)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or not 0 < details.st_size <= maximum_bytes
    ):
        _fail("required file is unsafe")


def _resolved_directory(path: Path) -> Path:
    if not path.is_absolute():
        _fail("directory must be absolute")
    _require_directory(path)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("directory could not be resolved")
    if resolved != path.absolute():
        _fail("directory must not traverse links")
    return resolved


def _safe_repository_path(root: Path, relative_path: str, *, must_exist: bool) -> Path:
    if not relative_path or "\\" in relative_path:
        _fail("repository path is invalid")
    parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail("repository path is invalid")
    candidate = root.joinpath(*parts)
    current = root
    missing_ancestor = False
    for part in parts:
        current /= part
        if missing_ancestor:
            continue
        if current.exists() or current.is_symlink():
            if _is_link_or_reparse(current):
                _fail("repository path uses a link")
        elif must_exist:
            _fail("repository path is unavailable")
        else:
            missing_ancestor = True
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError):
        _fail("repository path could not be resolved")
    if not resolved.is_relative_to(root):
        _fail("repository path escapes the clone")
    return candidate


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sanitized_environment(clone: Path, environment_home: Path) -> dict[str, str]:
    """Build a credential-free environment whose Python imports come from one clone."""

    try:
        environment_home.mkdir(parents=True, exist_ok=True)
    except OSError:
        _fail("isolated environment could not be created")
    isolated_home = _resolved_directory(environment_home)
    directories = {
        name: isolated_home / name
        for name in (
            "appdata",
            "cache",
            "dvc-global",
            "dvc-site-cache",
            "dvc-system",
            "localappdata",
            "temp",
        )
    }
    for directory in directories.values():
        try:
            directory.mkdir(exist_ok=True)
        except OSError:
            _fail("isolated environment could not be created")
        _resolved_directory(directory)
    executable_directory = str(Path(sys.executable).resolve().parent)
    inherited_path = os.environ.get("PATH", "")
    path_value = os.pathsep.join(part for part in (executable_directory, inherited_path) if part)
    environment: dict[str, str] = {
        "APPDATA": str(directories["appdata"]),
        "DVC_EXP_AUTO_PUSH": "false",
        "DVC_GLOBAL_CONFIG_DIR": str(directories["dvc-global"]),
        "DVC_NO_ANALYTICS": "true",
        "DVC_SITE_CACHE_DIR": str(directories["dvc-site-cache"]),
        "DVC_STUDIO_OFFLINE": "true",
        "DVC_SYSTEM_CONFIG_DIR": str(directories["dvc-system"]),
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(isolated_home),
        "LANG": "C",
        "LC_ALL": "C",
        "LOCALAPPDATA": str(directories["localappdata"]),
        "LOGNAME": "signlab-clean-room",
        "PATH": path_value,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(clone / "src"),
        "PYTHONUTF8": "1",
        "TEMP": str(directories["temp"]),
        "TMP": str(directories["temp"]),
        "TZ": "UTC",
        "USER": "signlab-clean-room",
        "USERNAME": "signlab-clean-room",
        "USERPROFILE": str(isolated_home),
        "XDG_CACHE_HOME": str(directories["cache"]),
        "XDG_CONFIG_HOME": str(directories["appdata"]),
    }
    for name in ("COMSPEC", "PATHEXT", "SYSTEMDRIVE", "SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _offline_environment(environment: Mapping[str, str]) -> dict[str, str]:
    offline = dict(environment)
    unavailable_proxy = "http://127.0.0.1:9"
    offline.update(
        {
            "ALL_PROXY": unavailable_proxy,
            "HTTPS_PROXY": unavailable_proxy,
            "HTTP_PROXY": unavailable_proxy,
            "NO_PROXY": "",
            "all_proxy": unavailable_proxy,
            "https_proxy": unavailable_proxy,
            "http_proxy": unavailable_proxy,
            "no_proxy": "",
        }
    )
    return offline


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS,
) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        _fail("required command could not run")
    if result.returncode != 0:
        _fail("required command failed")
    return result.stdout


def _run_git(
    repository: Path,
    environment: Mapping[str, str],
    *arguments: str,
) -> str:
    return _run(("git", *arguments), cwd=repository, environment=environment)


def _run_dvc(
    repository: Path,
    environment: Mapping[str, str],
    *arguments: str,
) -> str:
    return _run(
        (sys.executable, "-I", "-m", "dvc", *arguments),
        cwd=repository,
        environment=environment,
    )


def _git_is_clean(repository: Path, environment: Mapping[str, str]) -> bool:
    status = _run_git(
        repository,
        environment,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    return status == ""


def _git_commit(repository: Path, environment: Mapping[str, str]) -> str:
    commit = _run_git(repository, environment, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        _fail("Git commit identity is invalid")
    return commit


def _dvc_status_document(
    repository: Path,
    environment: Mapping[str, str],
    *,
    cloud: bool = False,
) -> JsonObject:
    arguments = ("status", "--cloud", "--json") if cloud else ("status", "--json")
    output = _run_dvc(repository, environment, *arguments)
    try:
        document = parse_json_object(output)
    except (TypeError, ValueError):
        _fail("DVC status is invalid")
    return cast(JsonObject, document)


def _dvc_is_clean(repository: Path, environment: Mapping[str, str], *, cloud: bool = False) -> bool:
    return _dvc_status_document(repository, environment, cloud=cloud) == {}


def _dvc_version(repository: Path, environment: Mapping[str, str]) -> str:
    version = _run_dvc(repository, environment, "--version").strip()
    if version != DVC_VERSION:
        _fail("DVC version is not the locked release")
    return version


def _control_file_bytes(repository: Path) -> dict[str, bytes]:
    captured: dict[str, bytes] = {}
    for name in _CONTROL_FILES:
        path = _safe_repository_path(repository, name, must_exist=True)
        _require_regular_file(path)
        try:
            captured[name] = _normalized_control_bytes(path.read_bytes())
        except OSError:
            _fail("control file could not be read")
    return captured


def _normalized_control_bytes(payload: bytes) -> bytes:
    if payload.startswith(b"\xef\xbb\xbf") or b"\0" in payload:
        _fail("control file encoding is invalid")
    if b"\r\n" in payload:
        remainder = payload.replace(b"\r\n", b"")
        if b"\r" in remainder or b"\n" in remainder:
            _fail("control file line endings are mixed")
        payload = payload.replace(b"\r\n", b"\n")
    elif b"\r" in payload:
        _fail("control file line endings are invalid")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("control file encoding is invalid")
    return payload


def _control_files_unchanged(repository: Path, baseline: Mapping[str, bytes]) -> bool:
    return _control_file_bytes(repository) == dict(baseline)


def _refresh_validated_lock_index(
    repository: Path,
    environment: Mapping[str, str],
    baseline: Mapping[str, bytes],
) -> None:
    """Refresh only DVC's native Windows line-ending rewrite after proving equivalence."""

    if not _control_files_unchanged(repository, baseline):
        _fail("committed DVC metadata changed")
    _run_git(repository, environment, "diff", "--quiet", "--", *_CONTROL_FILES)
    # DVC writes its YAML lock with native newlines on Windows. Git's eol=lf
    # clean filter proves the blob is unchanged; adding it only refreshes index
    # stat data. The cached-diff check below fails closed if content changed.
    _run_git(repository, environment, "add", "--", "dvc.lock")
    _run_git(repository, environment, "diff", "--cached", "--quiet", "--", *_CONTROL_FILES)
    if not _control_files_unchanged(repository, baseline):
        _fail("committed DVC metadata changed")


def _require_all_stages_executed(output: str) -> None:
    markers = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("Synthetic reproduction stage completed:")
    ]
    expected = [f"Synthetic reproduction stage completed: {stage}." for stage in _STAGE_NAMES]
    if markers != expected:
        _fail("the exact synthetic stage inventory did not execute")


def _parse_fixture_output(payload: bytes, stage: str) -> None:
    try:
        document = parse_json_object(payload)
        canonical = canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError):
        _fail("synthetic fixture output is invalid")
    expected_keys = {
        "fixture_only",
        "implementation",
        "payload",
        "profile",
        "schema_version",
        "stage",
        "upstream_sha256",
    }
    if (
        canonical != payload
        or frozenset(document) != frozenset(expected_keys)
        or document["fixture_only"] is not True
        or document["implementation"] != "fixture-only/1"
        or document["profile"] != "public-synthetic-reproducibility"
        or document["schema_version"] != "synthetic-dvc-stage/1"
        or document["stage"] != stage
        or type(document["payload"]) is not dict
        or not isinstance(document["upstream_sha256"], str)
        or _SHA256_PATTERN.fullmatch(document["upstream_sha256"]) is None
    ):
        _fail("synthetic fixture output is invalid")


def _fixture_output_hashes(repository: Path) -> DigestMap:
    hashes: DigestMap = {}
    for spec in STAGE_REGISTRY:
        path = _safe_repository_path(repository, spec.output_path, must_exist=True)
        _require_regular_file(path)
        try:
            payload = path.read_bytes()
        except OSError:
            _fail("synthetic fixture output could not be read")
        _parse_fixture_output(payload, spec.name)
        hashes[spec.name] = _sha256(payload)
    if tuple(hashes) != _STAGE_NAMES:
        _fail("synthetic fixture inventory is invalid")
    return hashes


def _tree_is_empty_or_absent(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return True
    if _is_link_or_reparse(path):
        _fail("cache path is unsafe")
    _require_directory(path)
    try:
        with os.scandir(path) as entries:
            return next(entries, None) is None
    except OSError:
        _fail("cache path could not be inspected")


def _consumer_state_is_empty(repository: Path) -> tuple[bool, bool]:
    cache = _safe_repository_path(repository, ".dvc/cache", must_exist=False)
    cache_empty = _tree_is_empty_or_absent(cache)
    outputs_absent = all(
        not _safe_repository_path(repository, spec.output_path, must_exist=False).exists()
        for spec in STAGE_REGISTRY
    )
    return cache_empty, outputs_absent


def _assert_under_workspace(path: Path, workspace: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved_workspace = workspace.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("temporary workspace target is unavailable")
    if not resolved.is_relative_to(resolved_workspace) or resolved == resolved_workspace:
        _fail("temporary workspace target is outside the proof sandbox")


def _delete_validated_consumer_state(repository: Path, workspace: Path) -> None:
    _assert_under_workspace(repository, workspace)
    cache = _safe_repository_path(repository, ".dvc/cache", must_exist=False)
    if cache.exists() or cache.is_symlink():
        if _is_link_or_reparse(cache):
            _fail("cache path is unsafe")
        _assert_under_workspace(cache, workspace)
        _require_directory(cache)
        if not _tree_is_empty_or_absent(cache) and not _remote_contains_only_regular_files(cache):
            _fail("cache tree is unsafe")
        try:
            shutil.rmtree(cache, onexc=_clear_readonly_and_retry)
        except OSError:
            _fail("temporary cache could not be removed")
    for spec in STAGE_REGISTRY:
        output = _safe_repository_path(repository, spec.output_path, must_exist=True)
        _assert_under_workspace(output, workspace)
        _require_regular_file(output)
        try:
            output.unlink()
        except PermissionError:
            try:
                output.chmod(stat.S_IRUSR | stat.S_IWUSR)
                output.unlink()
            except OSError:
                _fail("temporary output could not be removed")
        except OSError:
            _fail("temporary output could not be removed")


def _clear_readonly_and_retry(
    function: Callable[[str], object],
    path: str,
    error: BaseException,
) -> None:
    if not isinstance(error, PermissionError):
        _fail("temporary cache could not be removed")
    candidate = Path(path)
    if _is_link_or_reparse(candidate):
        _fail("cache tree is unsafe")
    try:
        candidate.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        function(path)
    except OSError:
        _fail("temporary cache could not be removed")


def _remote_contains_only_regular_files(remote: Path) -> bool:
    _require_directory(remote)
    found_file = False
    try:
        for directory_name, directory_names, file_names in os.walk(remote, followlinks=False):
            directory = Path(directory_name)
            for name in directory_names:
                child = directory / name
                if _is_link_or_reparse(child):
                    return False
                _require_directory(child)
            for name in file_names:
                child = directory / name
                if _is_link_or_reparse(child):
                    return False
                _require_regular_file(child, maximum_bytes=2**31 - 1)
                found_file = True
    except OSError:
        _fail("temporary remote could not be inspected")
    return found_file


def _configure_local_remote(
    repository: Path,
    environment: Mapping[str, str],
    remote: Path,
) -> None:
    _run_dvc(
        repository,
        environment,
        "remote",
        "add",
        "--local",
        "--default",
        _DVC_REMOTE_NAME,
        str(remote),
    )
    local_config = _safe_repository_path(repository, ".dvc/config.local", must_exist=True)
    _require_regular_file(local_config)
    ignored = _run_git(
        repository,
        environment,
        "check-ignore",
        "--quiet",
        "--",
        ".dvc/config.local",
    )
    if ignored != "":
        _fail("local DVC configuration is not isolated")


def _clone_repository(
    source: Path,
    destination: Path,
    environment: Mapping[str, str],
) -> None:
    if destination.exists() or destination.is_symlink():
        _fail("clone destination already exists")
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
    _resolved_directory(destination)


def _verify_clone_identity(
    repository: Path,
    environment: Mapping[str, str],
    expected_commit: str,
) -> None:
    top_level = _run_git(repository, environment, "rev-parse", "--show-toplevel").strip()
    try:
        top_level_path = Path(top_level).resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("clone Git root is invalid")
    if top_level_path != repository or _git_commit(repository, environment) != expected_commit:
        _fail("clone identity does not match source")
    if not _git_is_clean(repository, environment):
        _fail("clone is not Git-clean")


def _source_preflight(source: Path, sandbox: Path) -> tuple[Path, str, str]:
    root = _resolved_directory(source)
    if sandbox.is_relative_to(root) or root.is_relative_to(sandbox):
        _fail("proof workspace must be outside the source checkout")
    home = sandbox / "source-home"
    home.mkdir()
    environment = _sanitized_environment(root, home)
    top_level = _run_git(root, environment, "rev-parse", "--show-toplevel").strip()
    try:
        top_level_path = Path(top_level).resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("source Git root is invalid")
    if top_level_path != root or not _git_is_clean(root, environment):
        _fail("source must be a clean committed checkout")
    origin = _run_git(root, environment, "remote", "get-url", "origin").strip()
    if origin not in PUBLIC_REPOSITORY_ORIGINS:
        _fail("source repository identity is invalid")
    for name in _CONTROL_FILES:
        _run_git(root, environment, "ls-files", "--error-unmatch", "--", name)
    commit = _git_commit(root, environment)
    version = _dvc_version(root, environment)
    return root, commit, version


def _snapshot(repository: Path, commit: str) -> DvcSnapshotV1:
    try:
        return build_dvc_snapshot(
            repository,
            commit,
            metadata_repository_role="public-fixture",
            git_working_tree_clean=True,
            dvc_workspace_clean=True,
        )
    except (TypeError, ValueError):
        _fail("DVC provenance snapshot is invalid")


def _digest_map(value: object) -> DigestMap:
    if type(value) is not dict:
        _fail("report digest inventory is invalid")
    raw = cast(dict[object, object], value)
    if frozenset(raw) != frozenset(_STAGE_NAMES):
        _fail("report digest inventory is invalid")
    normalized: DigestMap = {}
    for stage in _STAGE_NAMES:
        digest = raw[stage]
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            _fail("report digest inventory is invalid")
        normalized[stage] = digest
    return normalized


def validate_report(document: Mapping[str, object]) -> JsonObject:
    """Strictly validate and normalize the path-free public proof report."""

    if frozenset(document) != _REPORT_KEYS:
        _fail("report shape is invalid")
    if (
        document["schema_version"] != REPORT_SCHEMA
        or document["fixture_only"] is not True
        or document["dvc_version"] != DVC_VERSION
        or document["consent"] != CONSENT_STATUS
    ):
        _fail("report identity is invalid")
    commit = document["git_commit"]
    if type(commit) is not str or _COMMIT_PATTERN.fullmatch(commit) is None:
        _fail("report commit is invalid")
    for field in ("uv_lock_sha256", "dvc_yaml_sha256", "dvc_lock_sha256", "dvc_snapshot_sha256"):
        digest = document[field]
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            _fail("report digest is invalid")
    stage_names = document["stage_names"]
    if type(stage_names) is not list or tuple(stage_names) != _STAGE_NAMES:
        _fail("report stage inventory is invalid")
    if any(document[field] is not True for field in _REPORT_BOOLEAN_FIELDS):
        _fail("report proof is incomplete")
    normalized: JsonObject = {
        "schema_version": REPORT_SCHEMA,
        "fixture_only": True,
        "git_commit": commit,
        "dvc_version": DVC_VERSION,
        "uv_lock_sha256": document["uv_lock_sha256"],
        "dvc_yaml_sha256": document["dvc_yaml_sha256"],
        "dvc_lock_sha256": document["dvc_lock_sha256"],
        "dvc_snapshot_sha256": document["dvc_snapshot_sha256"],
        "stage_names": list(_STAGE_NAMES),
        **{field: _digest_map(document[field]) for field in _REPORT_DIGEST_FIELDS},
        **{field: True for field in _REPORT_BOOLEAN_FIELDS},
        "consent": CONSENT_STATUS,
    }
    return normalized


def canonical_report_bytes(document: Mapping[str, object]) -> bytes:
    """Return canonical JSON only after the public report passes its allowlist."""

    try:
        return canonical_json_bytes(validate_report(document))
    except (TypeError, ValueError):
        _fail("report is not canonicalizable")


def _prepare_report_path(report_path: Path, source: Path, sandbox: Path) -> Path:
    if not report_path.is_absolute() or report_path.suffix.lower() != ".json":
        _fail("report target must be a new absolute JSON file")
    if report_path.exists() or report_path.is_symlink():
        _fail("report target already exists")
    parent = report_path.parent
    _require_directory(parent)
    try:
        target = report_path.resolve(strict=False)
        resolved_parent = parent.resolve(strict=True)
        resolved_source = source.resolve(strict=True)
        resolved_sandbox = sandbox.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("report target could not be resolved")
    if (
        target.parent != resolved_parent
        or target.is_relative_to(resolved_source)
        or target.is_relative_to(resolved_sandbox)
    ):
        _fail("report target must be outside repositories and proof workspace")
    return target


def _write_report(report_path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(report_path, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        with suppress(OSError):
            report_path.unlink(missing_ok=True)
        _fail("report could not be published")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _build_report(
    snapshot: DvcSnapshotV1,
    snapshot_sha256: str,
    producer_hashes: DigestMap,
    pulled_hashes: DigestMap,
    offline_hashes: DigestMap,
) -> JsonObject:
    stage_digests = {stage.stage_name: stage.lock_entry_sha256 for stage in snapshot.stages}
    return validate_report(
        {
            "schema_version": REPORT_SCHEMA,
            "fixture_only": True,
            "git_commit": snapshot.metadata_git_commit,
            "dvc_version": snapshot.dvc_version,
            "uv_lock_sha256": snapshot.uv_lock_sha256,
            "dvc_yaml_sha256": snapshot.dvc_yaml_sha256,
            "dvc_lock_sha256": snapshot.dvc_lock_sha256,
            "dvc_snapshot_sha256": snapshot_sha256,
            "stage_names": list(_STAGE_NAMES),
            "stage_lock_sha256": stage_digests,
            "producer_output_sha256": producer_hashes,
            "pulled_output_sha256": pulled_hashes,
            "offline_output_sha256": offline_hashes,
            **{field: True for field in _REPORT_BOOLEAN_FIELDS},
            "consent": CONSENT_STATUS,
        }
    )


def run_clean_room(source_repository: Path, report_path: Path) -> JsonObject:
    """Run the complete synthetic producer/pull/offline reproduction proof."""

    with tempfile.TemporaryDirectory(prefix="signlab-dvc-clean-room-") as temporary:
        sandbox = _resolved_directory(Path(temporary))
        source, commit, _ = _source_preflight(source_repository, sandbox)
        target = _prepare_report_path(report_path, source, sandbox)

        producer = sandbox / "producer"
        consumer = sandbox / "consumer"
        remote = sandbox / "remote"
        producer_home = sandbox / "producer-home"
        consumer_home = sandbox / "consumer-home"
        producer_home.mkdir()
        consumer_home.mkdir()
        remote.mkdir()

        producer_environment = _sanitized_environment(producer, producer_home)
        consumer_environment = _sanitized_environment(consumer, consumer_home)
        _clone_repository(source, producer, producer_environment)
        _clone_repository(source, consumer, consumer_environment)
        _verify_clone_identity(producer, producer_environment, commit)
        _verify_clone_identity(consumer, consumer_environment, commit)
        _dvc_version(producer, producer_environment)
        _dvc_version(consumer, consumer_environment)

        producer_baseline = _control_file_bytes(producer)
        producer_output = _run_dvc(
            producer,
            producer_environment,
            "repro",
            "--force",
            "--no-run-cache",
        )
        _require_all_stages_executed(producer_output)
        producer_hashes = _fixture_output_hashes(producer)
        _refresh_validated_lock_index(producer, producer_environment, producer_baseline)
        if not _dvc_is_clean(producer, producer_environment):
            _fail("producer is not DVC-clean")
        if not _git_is_clean(producer, producer_environment):
            _fail("producer is not Git-clean")
        producer_snapshot = _snapshot(producer, commit)
        snapshot_sha256 = dvc_snapshot_digest(producer_snapshot)

        _configure_local_remote(producer, producer_environment, remote)
        if not _control_files_unchanged(producer, producer_baseline):
            _fail("local remote configuration changed committed metadata")
        _run_dvc(producer, producer_environment, "push")
        if not _remote_contains_only_regular_files(remote):
            _fail("temporary remote did not receive the fixture")
        if not _dvc_is_clean(producer, producer_environment, cloud=True):
            _fail("temporary remote does not match producer")
        if not _git_is_clean(producer, producer_environment):
            _fail("producer is not Git-clean after push")

        cache_empty, outputs_absent = _consumer_state_is_empty(consumer)
        if not cache_empty or not outputs_absent:
            _fail("consumer clone did not start empty")
        consumer_baseline = _control_file_bytes(consumer)
        _configure_local_remote(consumer, consumer_environment, remote)
        if not _control_files_unchanged(consumer, consumer_baseline):
            _fail("consumer local remote changed committed metadata")
        _run_dvc(consumer, consumer_environment, "pull")
        pulled_hashes = _fixture_output_hashes(consumer)
        if pulled_hashes != producer_hashes:
            _fail("pulled outputs do not match producer")
        if not _dvc_is_clean(consumer, consumer_environment):
            _fail("consumer is not DVC-clean after pull")
        if not _dvc_is_clean(consumer, consumer_environment, cloud=True):
            _fail("consumer and temporary remote differ")

        unavailable_remote = sandbox / "remote-unavailable"
        try:
            remote.rename(unavailable_remote)
        except OSError:
            _fail("temporary remote could not be made unavailable")
        if remote.exists() or remote.is_symlink():
            _fail("temporary remote is still available")
        _delete_validated_consumer_state(consumer, sandbox)
        empty_after_delete, absent_after_delete = _consumer_state_is_empty(consumer)
        if not empty_after_delete or not absent_after_delete:
            _fail("consumer state could not be cleared")

        offline_environment = _offline_environment(consumer_environment)
        offline_output = _run_dvc(
            consumer,
            offline_environment,
            "repro",
            "--force",
            "--no-run-cache",
        )
        _require_all_stages_executed(offline_output)
        offline_hashes = _fixture_output_hashes(consumer)
        if offline_hashes != producer_hashes:
            _fail("offline outputs do not match producer")
        _refresh_validated_lock_index(consumer, offline_environment, consumer_baseline)
        if not _dvc_is_clean(consumer, offline_environment):
            _fail("consumer is not DVC-clean after offline reproduction")
        if not _git_is_clean(consumer, offline_environment):
            _fail("consumer is not Git-clean after offline reproduction")
        consumer_snapshot = _snapshot(consumer, commit)
        if consumer_snapshot != producer_snapshot:
            _fail("producer and consumer provenance identities differ")

        report = _build_report(
            producer_snapshot,
            snapshot_sha256,
            producer_hashes,
            pulled_hashes,
            offline_hashes,
        )
        _write_report(target, canonical_report_bytes(report))
        return report


def _usage() -> str:
    return "Usage: verify_dvc_clean_room.py --report ABSOLUTE_JSON_PATH\n"


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI boundary with generic output that cannot disclose a failing path or command."""

    values = list(sys.argv[1:] if arguments is None else arguments)
    if values in (["-h"], ["--help"]):
        sys.stdout.write(_usage())
        return 0
    if len(values) != 2 or values[0] != "--report":
        sys.stderr.write(_usage())
        return 2
    try:
        run_clean_room(Path.cwd(), Path(values[1]))
    except Exception:
        sys.stderr.write("Clean-room verification failed.\n")
        return 1
    sys.stdout.write("Clean-room verification passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
