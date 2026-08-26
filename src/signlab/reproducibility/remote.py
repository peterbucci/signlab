"""Configure SignLab's private DVC remote without persisting credentials."""

from __future__ import annotations

import configparser
import ipaddress
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import TracebackType
from typing import Final
from urllib.parse import SplitResult, urlsplit

from signlab.reproducibility.provenance import DVC_VERSION

DVC_REMOTE_URL_ENV: Final = "SIGNLAB_DVC_REMOTE_URL"
DVC_ENDPOINT_URL_ENV: Final = "SIGNLAB_DVC_ENDPOINT_URL"
DVC_REGION_ENV: Final = "SIGNLAB_DVC_REGION"
DVC_REMOTE_NAME: Final = "private"

_MAX_CONFIGURATION_BYTES: Final = 1024 * 1024
_DVC_COMMAND_TIMEOUT_SECONDS: Final = 30.0
_LOCK_RELATIVE_PATH: Final = ".dvc/tmp/signlab-private-remote.lock"
_LOCK_TOKEN_BYTES: Final = 32
_INHERITED_PROCESS_ENVIRONMENT: Final = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
)
_BUCKET_PATTERN: Final = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,253}[a-z0-9])?")
_REGION_PATTERN: Final = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_AUTHENTICATION_OPTION_NAMES: Final = frozenset(
    {
        "access_key_id",
        "accesskeyid",
        "credential_path",
        "credentialpath",
        "key_file",
        "key_id",
        "keyfile",
        "profile",
        "secret_access_key",
        "secretaccesskey",
        "session_token",
    }
)
_AUTHENTICATION_OPTION_FRAGMENTS: Final = ("credential", "password", "secret", "token")

_INVALID_REPOSITORY = "repository root must be a real initialized Git and DVC repository"
_UNSAFE_LOCAL_CONFIGURATION = ".dvc/config.local must be an ignored credential-free regular file"
_CONFIGURATION_FAILED = "private DVC remote configuration failed; local configuration was restored"
_ROLLBACK_FAILED = (
    "private DVC remote configuration failed and local configuration could not be restored"
)
_LOCK_UNAVAILABLE = "private DVC remote configuration lock is unavailable"


class DvcRemoteConfigurationError(RuntimeError):
    """Raised when a private DVC remote cannot be configured safely."""


@dataclass(frozen=True, slots=True)
class DvcRemoteConfigurationResult:
    """A credential-free summary of a successful local remote configuration."""

    remote_name: str
    endpoint_configured: bool
    region_configured: bool


@dataclass(frozen=True, slots=True)
class _LocalConfigurationSnapshot:
    data: bytes | None
    mode: int | None
    access_time_ns: int | None
    modification_time_ns: int | None


@dataclass(frozen=True, slots=True)
class _ConfigurationLock:
    path: Path
    token: bytes
    file_status: os.stat_result


def _is_link_or_reparse(file_status: os.stat_result) -> bool:
    if stat.S_ISLNK(file_status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(file_status, "st_file_attributes", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _safe_lstat(path: Path, *, error_message: str) -> os.stat_result:
    try:
        file_status = path.lstat()
    except OSError:
        raise DvcRemoteConfigurationError(error_message) from None
    if _is_link_or_reparse(file_status):
        raise DvcRemoteConfigurationError(error_message)
    return file_status


def _read_regular_file(path: Path, *, error_message: str) -> tuple[bytes, os.stat_result]:
    before = _safe_lstat(path, error_message=error_message)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > _MAX_CONFIGURATION_BYTES
    ):
        raise DvcRemoteConfigurationError(error_message)
    try:
        data = path.read_bytes()
        after = path.lstat()
    except OSError:
        raise DvcRemoteConfigurationError(error_message) from None
    if (
        len(data) > _MAX_CONFIGURATION_BYTES
        or _is_link_or_reparse(after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or not _same_file(before, after)
    ):
        raise DvcRemoteConfigurationError(error_message)
    return data, after


def _gitignore_protects_local_configuration(ignore_bytes: bytes) -> bool:
    try:
        lines = ignore_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return False

    ignored = False
    canonical_rule_seen = False
    for raw_line in lines:
        line = raw_line.strip()
        if line in {"/config.local", "config.local"}:
            ignored = True
            canonical_rule_seen = True
        elif line in {"!/config.local", "!config.local"}:
            ignored = False
            canonical_rule_seen = True
        elif ignored and line.startswith("!"):
            # Fail closed instead of trying to duplicate Git's full pattern engine.
            return False
    return canonical_rule_seen and ignored


def _configuration_contains_authentication(data: bytes) -> bool:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return True

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "[")):
            continue
        equals_index = line.find("=")
        colon_index = line.find(":")
        separator_indexes = tuple(index for index in (equals_index, colon_index) if index >= 0)
        if not separator_indexes:
            continue
        key = line[: min(separator_indexes)].strip().lower().replace("-", "_")
        if key in _AUTHENTICATION_OPTION_NAMES or any(
            fragment in key for fragment in _AUTHENTICATION_OPTION_FRAGMENTS
        ):
            return True
    return False


def _validated_repository(repo_root: Path) -> tuple[Path, Path, os.stat_result]:
    supplied_status = _safe_lstat(repo_root, error_message=_INVALID_REPOSITORY)
    if not stat.S_ISDIR(supplied_status.st_mode):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)
    try:
        root = repo_root.resolve(strict=True)
    except OSError:
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY) from None

    git_status = _safe_lstat(root / ".git", error_message=_INVALID_REPOSITORY)
    if not (stat.S_ISDIR(git_status.st_mode) or stat.S_ISREG(git_status.st_mode)):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)

    dvc_directory = root / ".dvc"
    dvc_status = _safe_lstat(dvc_directory, error_message=_INVALID_REPOSITORY)
    if not stat.S_ISDIR(dvc_status.st_mode):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)

    ignore_bytes, _ = _read_regular_file(
        dvc_directory / ".gitignore",
        error_message=_UNSAFE_LOCAL_CONFIGURATION,
    )
    if not _gitignore_protects_local_configuration(ignore_bytes):
        raise DvcRemoteConfigurationError(_UNSAFE_LOCAL_CONFIGURATION)
    return root, dvc_directory, dvc_status


def _build_isolated_subprocess_environment(root: Path) -> dict[str, str]:
    """Build a private environment for local-only Git and DVC commands."""

    home = root / "home"
    app_data = root / "appdata"
    local_app_data = root / "localappdata"
    xdg_config = root / "xdg-config"
    xdg_cache = root / "xdg-cache"
    xdg_data = root / "xdg-data"
    xdg_state = root / "xdg-state"
    temporary = root / "tmp"
    dvc_global_config = root / "dvc-global-config"
    dvc_system_config = root / "dvc-system-config"
    dvc_site_cache = root / "dvc-site-cache"
    for directory in (
        home,
        app_data,
        local_app_data,
        xdg_config,
        xdg_cache,
        xdg_data,
        xdg_state,
        temporary,
        dvc_global_config,
        dvc_system_config,
        dvc_site_cache,
    ):
        directory.mkdir(mode=0o700)

    environment = {
        name: value
        for name in _INHERITED_PROCESS_ENVIRONMENT
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "APPDATA": str(app_data),
            "CI": "1",
            "DVC_EXP_AUTO_PUSH": "false",
            "DVC_GLOBAL_CONFIG_DIR": str(dvc_global_config),
            "DVC_NO_ANALYTICS": "true",
            "DVC_SITE_CACHE_DIR": str(dvc_site_cache),
            "DVC_STUDIO_OFFLINE": "true",
            "DVC_SYSTEM_CONFIG_DIR": str(dvc_system_config),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(home),
            "LOCALAPPDATA": str(local_app_data),
            "PYTHONUTF8": "1",
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "TMPDIR": str(temporary),
            "USERPROFILE": str(home),
            "XDG_CACHE_HOME": str(xdg_cache),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_DATA_HOME": str(xdg_data),
            "XDG_STATE_HOME": str(xdg_state),
        }
    )
    return environment


@contextmanager
def _isolated_subprocess_environment() -> Iterator[dict[str, str]]:
    try:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="signlab-dvc-environment-",
            ignore_cleanup_errors=True,
        )
    except OSError:
        raise DvcRemoteConfigurationError(_CONFIGURATION_FAILED) from None
    try:
        try:
            environment = _build_isolated_subprocess_environment(Path(temporary_directory.name))
        except OSError:
            raise DvcRemoteConfigurationError(_CONFIGURATION_FAILED) from None
        yield environment
    finally:
        temporary_directory.cleanup()


def _run_identity_command(
    arguments: list[str],
    *,
    cwd: Path,
    check: bool,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True,
            shell=False,
            timeout=_DVC_COMMAND_TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY) from None


def _resolved_reported_root(value: str, *, cwd: Path) -> Path:
    lines = value.strip().splitlines()
    if len(lines) != 1 or not lines[0]:
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)
    try:
        reported = Path(lines[0])
        if not reported.is_absolute():
            reported = cwd / reported
        return reported.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY) from None


def _assert_untracked_ignored_path(
    root: Path,
    relative_path: str,
    *,
    environment: Mapping[str, str],
) -> None:
    ignored = _run_identity_command(
        ["git", "check-ignore", "--quiet", "--no-index", "--", relative_path],
        cwd=root,
        check=False,
        environment=environment,
    )
    tracked = _run_identity_command(
        ["git", "ls-files", "--error-unmatch", "--", relative_path],
        cwd=root,
        check=False,
        environment=environment,
    )
    if ignored.returncode != 0 or tracked.returncode != 1:
        raise DvcRemoteConfigurationError(_UNSAFE_LOCAL_CONFIGURATION)


def _assert_repository_identity(
    root: Path,
    *,
    environment: Mapping[str, str],
) -> None:
    git_root = _run_identity_command(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=True,
        environment=environment,
    )
    dvc_root = _run_identity_command(
        [sys.executable, "-I", "-m", "dvc", "root"],
        cwd=root,
        check=True,
        environment=environment,
    )
    if (
        _resolved_reported_root(git_root.stdout, cwd=root) != root
        or _resolved_reported_root(dvc_root.stdout, cwd=root) != root
    ):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)
    _assert_untracked_ignored_path(
        root,
        ".dvc/config.local",
        environment=environment,
    )
    _assert_untracked_ignored_path(
        root,
        _LOCK_RELATIVE_PATH,
        environment=environment,
    )


def _safe_lock_directory(dvc_directory: Path, dvc_status: os.stat_result) -> Path:
    current_dvc_status = _safe_lstat(dvc_directory, error_message=_INVALID_REPOSITORY)
    if not stat.S_ISDIR(current_dvc_status.st_mode) or not _same_file(
        dvc_status, current_dvc_status
    ):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)
    lock_directory = dvc_directory / "tmp"
    try:
        lock_directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError:
        raise DvcRemoteConfigurationError(_LOCK_UNAVAILABLE) from None
    lock_directory_status = _safe_lstat(lock_directory, error_message=_LOCK_UNAVAILABLE)
    if not stat.S_ISDIR(lock_directory_status.st_mode):
        raise DvcRemoteConfigurationError(_LOCK_UNAVAILABLE)
    current_dvc_status = _safe_lstat(dvc_directory, error_message=_INVALID_REPOSITORY)
    if not _same_file(dvc_status, current_dvc_status):
        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)
    return lock_directory


def _unlink_owned_lock(lock: _ConfigurationLock) -> None:
    data, file_status = _read_regular_file(lock.path, error_message=_LOCK_UNAVAILABLE)
    if not _same_file(lock.file_status, file_status) or data != lock.token:
        raise DvcRemoteConfigurationError(_LOCK_UNAVAILABLE)
    try:
        lock.path.unlink()
    except OSError:
        raise DvcRemoteConfigurationError(_LOCK_UNAVAILABLE) from None


def _acquire_configuration_lock(
    dvc_directory: Path,
    dvc_status: os.stat_result,
) -> _ConfigurationLock:
    lock_directory = _safe_lock_directory(dvc_directory, dvc_status)
    lock_path = lock_directory / Path(_LOCK_RELATIVE_PATH).name
    token = secrets.token_hex(_LOCK_TOKEN_BYTES).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor: int | None = None
    created_status: os.stat_result | None = None
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        created_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(created_status.st_mode)
            or created_status.st_nlink != 1
            or created_status.st_size != 0
        ):
            raise OSError
        view = memoryview(token)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError:
        raise DvcRemoteConfigurationError(_LOCK_UNAVAILABLE) from None
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created_status is not None:
            with suppress(DvcRemoteConfigurationError):
                _unlink_owned_lock(
                    _ConfigurationLock(
                        path=lock_path,
                        token=token,
                        file_status=created_status,
                    )
                )
        raise DvcRemoteConfigurationError(_LOCK_UNAVAILABLE) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    file_status = _safe_lstat(lock_path, error_message=_LOCK_UNAVAILABLE)
    if (
        created_status is None
        or not stat.S_ISREG(file_status.st_mode)
        or file_status.st_nlink != 1
        or not _same_file(created_status, file_status)
    ):
        raise DvcRemoteConfigurationError(_LOCK_UNAVAILABLE)
    return _ConfigurationLock(path=lock_path, token=token, file_status=file_status)


@contextmanager
def _configuration_lock(
    dvc_directory: Path,
    dvc_status: os.stat_result,
) -> Iterator[None]:
    lock = _acquire_configuration_lock(dvc_directory, dvc_status)
    try:
        yield
    except BaseException:
        # Preserve interrupts and SystemExit. A replaced lock is left in place so a
        # later process fails closed instead of assuming mutual exclusion.
        with suppress(DvcRemoteConfigurationError):
            _unlink_owned_lock(lock)
        raise
    else:
        _unlink_owned_lock(lock)


def _has_forbidden_url_character(value: str) -> bool:
    return (
        not value
        or value != value.strip()
        or len(value) > 2048
        or any(character.isspace() or ord(character) < 32 for character in value)
        or "\\" in value
    )


def _split_url(value: str, *, error_message: str) -> SplitResult:
    if _has_forbidden_url_character(value):
        raise DvcRemoteConfigurationError(error_message)
    try:
        return urlsplit(value)
    except ValueError:
        raise DvcRemoteConfigurationError(error_message) from None


def _validated_remote_url(value: str | None) -> str:
    missing_message = f"{DVC_REMOTE_URL_ENV} is required"
    invalid_message = f"{DVC_REMOTE_URL_ENV} must be a credential-free s3:// URL"
    if value is None or not value.strip():
        raise DvcRemoteConfigurationError(missing_message)
    parts = _split_url(value, error_message=invalid_message)
    try:
        port = parts.port
    except ValueError:
        raise DvcRemoteConfigurationError(invalid_message) from None
    if (
        not value.startswith("s3://")
        or parts.scheme != "s3"
        or parts.username is not None
        or parts.password is not None
        or port is not None
        or parts.query
        or parts.fragment
        or not _BUCKET_PATTERN.fullmatch(parts.netloc)
        or "%" in value
    ):
        raise DvcRemoteConfigurationError(invalid_message)

    path_segments = parts.path.split("/")[1:]
    if any(segment in {"", ".", ".."} for segment in path_segments) and parts.path != "/":
        raise DvcRemoteConfigurationError(invalid_message)
    return value


def _is_loopback_hostname(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validated_endpoint_url(value: str | None) -> str | None:
    if value is None:
        return None
    invalid_message = (
        f"{DVC_ENDPOINT_URL_ENV} must be a credential-free HTTPS origin "
        "(HTTP is allowed only for localhost)"
    )
    parts = _split_url(value, error_message=invalid_message)
    try:
        port = parts.port
    except ValueError:
        raise DvcRemoteConfigurationError(invalid_message) from None
    hostname = parts.hostname
    if (
        parts.scheme not in {"http", "https"}
        or not value.startswith(f"{parts.scheme}://")
        or hostname is None
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path not in {"", "/"}
        or "%" in value
        or port == 0
    ):
        raise DvcRemoteConfigurationError(invalid_message)
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError:
        raise DvcRemoteConfigurationError(invalid_message) from None
    if parts.scheme == "http" and not _is_loopback_hostname(hostname):
        raise DvcRemoteConfigurationError(invalid_message)
    return value


def _validated_region(value: str | None) -> str | None:
    if value is None:
        return None
    invalid_message = f"{DVC_REGION_ENV} must be a canonical region name"
    if not _REGION_PATTERN.fullmatch(value):
        raise DvcRemoteConfigurationError(invalid_message)
    return value


def _snapshot_local_configuration(path: Path) -> _LocalConfigurationSnapshot:
    try:
        path.lstat()
    except FileNotFoundError:
        return _LocalConfigurationSnapshot(None, None, None, None)
    except OSError:
        raise DvcRemoteConfigurationError(_UNSAFE_LOCAL_CONFIGURATION) from None

    data, file_status = _read_regular_file(path, error_message=_UNSAFE_LOCAL_CONFIGURATION)
    if _configuration_contains_authentication(data):
        raise DvcRemoteConfigurationError(_UNSAFE_LOCAL_CONFIGURATION)
    return _LocalConfigurationSnapshot(
        data=data,
        mode=stat.S_IMODE(file_status.st_mode),
        access_time_ns=file_status.st_atime_ns,
        modification_time_ns=file_status.st_mtime_ns,
    )


def _restore_local_configuration(path: Path, snapshot: _LocalConfigurationSnapshot) -> None:
    if snapshot.data is None:
        try:
            current_status = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISDIR(current_status.st_mode) and not stat.S_ISLNK(current_status.st_mode):
            raise OSError("unsafe replacement")
        path.unlink()
        return

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".config.local.restore-",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(snapshot.data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if snapshot.mode is not None:
            temporary_path.chmod(snapshot.mode)
        os.replace(temporary_path, path)
        if snapshot.access_time_ns is not None and snapshot.modification_time_ns is not None:
            # Some Windows filesystems cannot set link-safe timestamps. The bytes and
            # access mode are already restored atomically, so timestamps are best effort.
            with suppress(OSError, NotImplementedError):
                os.utime(
                    path,
                    ns=(snapshot.access_time_ns, snapshot.modification_time_ns),
                    follow_symlinks=False,
                )
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)


def _validate_generated_configuration(
    path: Path,
    *,
    remote_url: str,
    endpoint_url: str | None,
    region: str | None,
) -> None:
    data, _ = _read_regular_file(path, error_message=_UNSAFE_LOCAL_CONFIGURATION)
    if _configuration_contains_authentication(data):
        raise DvcRemoteConfigurationError(_UNSAFE_LOCAL_CONFIGURATION)
    try:
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(data.decode("utf-8"))
    except (configparser.Error, UnicodeError):
        raise DvcRemoteConfigurationError(_UNSAFE_LOCAL_CONFIGURATION) from None
    expected_remote = {"url": remote_url}
    if endpoint_url is not None:
        expected_remote["endpointurl"] = endpoint_url
    if region is not None:
        expected_remote["region"] = region
    actual = {section: dict(parser[section]) for section in parser.sections()}
    if actual != {
        "core": {"remote": DVC_REMOTE_NAME},
        f"'remote \"{DVC_REMOTE_NAME}\"'": expected_remote,
    }:
        raise DvcRemoteConfigurationError(_UNSAFE_LOCAL_CONFIGURATION)


def _run_dvc(
    arguments: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        timeout=_DVC_COMMAND_TIMEOUT_SECONDS,
        env=environment,
    )
    return result.stdout


def configure_private_dvc_remote(repo_root: Path) -> DvcRemoteConfigurationResult:
    """Configure the ``private`` DVC remote in ignored local configuration.

    Provider credentials remain entirely in the parent process environment and are
    intentionally not forwarded to these local-only configuration commands. Later
    ``dvc pull`` and ``dvc push`` invocations use the provider credential chain.
    """

    remote_url = _validated_remote_url(os.environ.get(DVC_REMOTE_URL_ENV))
    endpoint_url = _validated_endpoint_url(os.environ.get(DVC_ENDPOINT_URL_ENV))
    region = _validated_region(os.environ.get(DVC_REGION_ENV))
    try:
        installed_dvc_version = distribution_version("dvc")
    except PackageNotFoundError:
        installed_dvc_version = None
    if installed_dvc_version != DVC_VERSION:
        raise DvcRemoteConfigurationError(_CONFIGURATION_FAILED)

    root, dvc_directory, dvc_status = _validated_repository(Path(repo_root))
    with _isolated_subprocess_environment() as environment:
        _assert_repository_identity(root, environment=environment)
        with _configuration_lock(dvc_directory, dvc_status):
            # Re-prove the checkout after winning the process lock so validation from a
            # competing invocation cannot be reused against changed repository state.
            _assert_repository_identity(root, environment=environment)
            local_configuration = dvc_directory / "config.local"
            snapshot = _snapshot_local_configuration(local_configuration)

            commands = [
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "dvc",
                    "remote",
                    "add",
                    "--local",
                    "--default",
                    "--force",
                    DVC_REMOTE_NAME,
                    remote_url,
                ]
            ]
            if endpoint_url is not None:
                commands.append(
                    [
                        sys.executable,
                        "-I",
                        "-m",
                        "dvc",
                        "remote",
                        "modify",
                        "--local",
                        DVC_REMOTE_NAME,
                        "endpointurl",
                        endpoint_url,
                    ]
                )
            if region is not None:
                commands.append(
                    [
                        sys.executable,
                        "-I",
                        "-m",
                        "dvc",
                        "remote",
                        "modify",
                        "--local",
                        DVC_REMOTE_NAME,
                        "region",
                        region,
                    ]
                )

            failure: BaseException | None = None
            failure_traceback: TracebackType | None = None
            rollback_failed = False
            try:
                for command in commands:
                    _run_dvc(command, cwd=root, environment=environment)
                current_dvc_status = _safe_lstat(
                    dvc_directory,
                    error_message=_INVALID_REPOSITORY,
                )
                if not stat.S_ISDIR(current_dvc_status.st_mode) or not _same_file(
                    dvc_status, current_dvc_status
                ):
                    raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)
                _assert_repository_identity(root, environment=environment)
                _validate_generated_configuration(
                    local_configuration,
                    remote_url=remote_url,
                    endpoint_url=endpoint_url,
                    region=region,
                )
            except BaseException as error:
                failure = error
                failure_traceback = error.__traceback__
                try:
                    current_dvc_status = _safe_lstat(
                        dvc_directory,
                        error_message=_INVALID_REPOSITORY,
                    )
                    if not _same_file(dvc_status, current_dvc_status):
                        raise DvcRemoteConfigurationError(_INVALID_REPOSITORY)
                    _restore_local_configuration(local_configuration, snapshot)
                except BaseException:
                    rollback_failed = True

            if failure is not None:
                if not isinstance(failure, Exception):
                    if rollback_failed:
                        failure.add_note(_ROLLBACK_FAILED)
                    raise failure.with_traceback(failure_traceback)
                if rollback_failed:
                    raise DvcRemoteConfigurationError(_ROLLBACK_FAILED)
                raise DvcRemoteConfigurationError(_CONFIGURATION_FAILED)

    return DvcRemoteConfigurationResult(
        remote_name=DVC_REMOTE_NAME,
        endpoint_configured=endpoint_url is not None,
        region_configured=region is not None,
    )
