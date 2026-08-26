"""Configure a credential-free private DVC remote in ignored local state."""

from __future__ import annotations

import configparser
import ipaddress
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final
from urllib.parse import SplitResult, urlsplit

from signlab.reproducibility import DVC_VERSION

DVC_REMOTE_URL_ENV: Final = "SIGNLAB_DVC_REMOTE_URL"
DVC_ENDPOINT_URL_ENV: Final = "SIGNLAB_DVC_ENDPOINT_URL"
DVC_REGION_ENV: Final = "SIGNLAB_DVC_REGION"
DVC_REMOTE_NAME: Final = "private"

_REGION_PATTERN: Final = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_BUCKET_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_SECRET_KEY_PARTS: Final = ("credential", "password", "secret", "token", "access_key")


class DvcRemoteConfigurationError(RuntimeError):
    """Raised without echoing a remote location or credential value."""


@dataclass(frozen=True, slots=True)
class DvcRemoteConfigurationResult:
    """Non-sensitive summary of the local configuration."""

    remote_name: str
    endpoint_configured: bool
    region_configured: bool


def _split_url(value: str) -> SplitResult:
    if not value or value != value.strip() or any(character.isspace() for character in value):
        raise DvcRemoteConfigurationError("private DVC remote configuration is invalid")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise DvcRemoteConfigurationError("private DVC remote configuration is invalid") from error
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DvcRemoteConfigurationError("private DVC remote configuration is invalid")
    return parsed


def _validate_remote_url(value: str) -> str:
    parsed = _split_url(value)
    bucket = parsed.hostname or ""
    segments = tuple(segment for segment in parsed.path.split("/") if segment)
    if (
        parsed.scheme != "s3"
        or parsed.port is not None
        or _BUCKET_PATTERN.fullmatch(bucket) is None
        or any(segment in {".", ".."} for segment in segments)
    ):
        raise DvcRemoteConfigurationError("private DVC remote configuration is invalid")
    return value


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_endpoint_url(value: str) -> str:
    parsed = _split_url(value)
    host = parsed.hostname or ""
    if not host or (
        parsed.scheme != "https" and not (parsed.scheme == "http" and _is_loopback(host))
    ):
        raise DvcRemoteConfigurationError("private DVC remote configuration is invalid")
    return value.rstrip("/")


def _validate_region(value: str) -> str:
    if _REGION_PATTERN.fullmatch(value) is None:
        raise DvcRemoteConfigurationError("private DVC remote configuration is invalid")
    return value


def _require_repository(repository_root: Path) -> Path:
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise DvcRemoteConfigurationError("Git/DVC repository is unavailable") from error
    if not (root / ".git").exists() or not (root / ".dvc").is_dir():
        raise DvcRemoteConfigurationError("Git/DVC repository is unavailable")
    ignored = subprocess.run(
        ("git", "check-ignore", "--quiet", "--", ".dvc/config.local"),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise DvcRemoteConfigurationError("local DVC configuration is not ignored")
    return root


def _run_dvc(repository: Path, arguments: Sequence[str]) -> None:
    environment = dict(os.environ)
    environment["DVC_NO_ANALYTICS"] = "true"
    try:
        result = subprocess.run(
            (sys.executable, "-I", "-m", "dvc", *arguments),
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DvcRemoteConfigurationError("DVC configuration command failed") from error
    if result.returncode != 0:
        raise DvcRemoteConfigurationError("DVC configuration command failed")


def _validate_local_config(
    path: Path,
    remote_url: str,
    endpoint: str | None,
    region: str | None,
) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(path, encoding="utf-8")
    except (OSError, UnicodeError, configparser.Error) as error:
        raise DvcRemoteConfigurationError("local DVC configuration is invalid") from error
    for section in parser.sections():
        for option in parser.options(section):
            normalized = option.casefold().replace("-", "_")
            if any(part in normalized for part in _SECRET_KEY_PARTS):
                raise DvcRemoteConfigurationError("credentials must not be stored in DVC config")
    remote_section = f'remote "{DVC_REMOTE_NAME}"'
    if (
        not parser.has_section(remote_section)
        or parser.get(remote_section, "url", fallback=None) != remote_url
    ):
        raise DvcRemoteConfigurationError("local DVC configuration is invalid")
    expected = {"url"}
    if endpoint is not None:
        expected.add("endpointurl")
        if parser.get(remote_section, "endpointurl", fallback=None) != endpoint:
            raise DvcRemoteConfigurationError("local DVC configuration is invalid")
    if region is not None:
        expected.add("region")
        if parser.get(remote_section, "region", fallback=None) != region:
            raise DvcRemoteConfigurationError("local DVC configuration is invalid")
    if set(parser.options(remote_section)) != expected:
        raise DvcRemoteConfigurationError("local DVC configuration is invalid")


def _restore_local_config(path: Path, previous: bytes | None) -> None:
    try:
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(previous)
    except OSError as error:
        raise DvcRemoteConfigurationError(
            "local DVC configuration could not be restored"
        ) from error


def configure_private_dvc_remote(
    repository_root: Path,
    environment: Mapping[str, str] | None = None,
) -> DvcRemoteConfigurationResult:
    """Configure remote metadata locally; provider credentials remain in the environment."""

    try:
        if version("dvc") != DVC_VERSION:
            raise DvcRemoteConfigurationError("locked DVC version is unavailable")
    except PackageNotFoundError as error:
        raise DvcRemoteConfigurationError("locked DVC version is unavailable") from error
    values = os.environ if environment is None else environment
    try:
        remote_url = _validate_remote_url(values[DVC_REMOTE_URL_ENV])
    except KeyError as error:
        raise DvcRemoteConfigurationError("private DVC remote URL is required") from error
    endpoint_value = values.get(DVC_ENDPOINT_URL_ENV)
    region_value = values.get(DVC_REGION_ENV)
    endpoint = _validate_endpoint_url(endpoint_value) if endpoint_value else None
    region = _validate_region(region_value) if region_value else None

    root = _require_repository(repository_root)
    local_config = root / ".dvc" / "config.local"
    if local_config.is_symlink() or (local_config.exists() and not local_config.is_file()):
        raise DvcRemoteConfigurationError("local DVC configuration is invalid")
    try:
        previous = local_config.read_bytes() if local_config.exists() else None
    except OSError as error:
        raise DvcRemoteConfigurationError("local DVC configuration is unavailable") from error

    try:
        _run_dvc(
            root,
            ("remote", "add", "--local", "--force", "--default", DVC_REMOTE_NAME, remote_url),
        )
        if endpoint is not None:
            _run_dvc(
                root,
                (
                    "remote",
                    "modify",
                    "--local",
                    DVC_REMOTE_NAME,
                    "endpointurl",
                    endpoint,
                ),
            )
        if region is not None:
            _run_dvc(root, ("remote", "modify", "--local", DVC_REMOTE_NAME, "region", region))
        _validate_local_config(local_config, remote_url, endpoint, region)
    except (DvcRemoteConfigurationError, OSError):
        _restore_local_config(local_config, previous)
        raise DvcRemoteConfigurationError("private DVC remote configuration failed") from None

    return DvcRemoteConfigurationResult(
        remote_name=DVC_REMOTE_NAME,
        endpoint_configured=endpoint is not None,
        region_configured=region is not None,
    )


__all__ = [
    "DVC_ENDPOINT_URL_ENV",
    "DVC_REGION_ENV",
    "DVC_REMOTE_NAME",
    "DVC_REMOTE_URL_ENV",
    "DvcRemoteConfigurationError",
    "DvcRemoteConfigurationResult",
    "configure_private_dvc_remote",
]
