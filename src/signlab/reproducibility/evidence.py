"""Capture and publish sanitized DVC experiment metadata from verified repository state."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.reproducibility.provenance import (
    DVC_VERSION,
    MAX_CONTROL_FILE_BYTES,
    PUBLIC_REPOSITORY_ORIGINS,
    DvcMetadataRepositoryRole,
    DvcSnapshotV1,
    build_dvc_snapshot,
)

_COMMAND_TIMEOUT_SECONDS: Final = 30.0
_REPARSE_POINT: Final = 0x400
_REPORT_PREFIX: Final = "reports/reproduction/"


class DvcEvidenceError(ValueError):
    """Stable failure that never reveals a command, path, or external value."""

    def __init__(self) -> None:
        self.code = "dvc.evidence.capture.invalid"
        super().__init__("DVC reproduction evidence could not be captured safely")


def _safe_environment(environment_home: Path) -> dict[str, str]:
    try:
        environment_home.mkdir(parents=True)
        directories = {
            name: environment_home / name
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
            directory.mkdir()
    except OSError:
        raise DvcEvidenceError from None
    environment: dict[str, str] = {}
    path_value = os.environ.get("PATH")
    if path_value:
        environment["PATH"] = path_value
    for name in ("COMSPEC", "PATHEXT", "SYSTEMDRIVE", "SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "APPDATA": str(directories["appdata"]),
            "DVC_EXP_AUTO_PUSH": "false",
            "DVC_GLOBAL_CONFIG_DIR": str(directories["dvc-global"]),
            "DVC_NO_ANALYTICS": "true",
            "DVC_SITE_CACHE_DIR": str(directories["dvc-site-cache"]),
            "DVC_STUDIO_OFFLINE": "true",
            "DVC_SYSTEM_CONFIG_DIR": str(directories["dvc-system"]),
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(environment_home),
            "LANG": "C",
            "LC_ALL": "C",
            "LOCALAPPDATA": str(directories["localappdata"]),
            "LOGNAME": "signlab-evidence",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "TEMP": str(directories["temp"]),
            "TMP": str(directories["temp"]),
            "TZ": "UTC",
            "USER": "signlab-evidence",
            "USERNAME": "signlab-evidence",
            "USERPROFILE": str(environment_home),
            "XDG_CACHE_HOME": str(directories["cache"]),
            "XDG_CONFIG_HOME": str(directories["appdata"]),
        }
    )
    return environment


def _run(arguments: list[str], root: Path, environment: Mapping[str, str]) -> str:
    try:
        result = subprocess.run(
            arguments,
            cwd=root,
            check=True,
            capture_output=True,
            encoding="utf-8",
            env=dict(environment),
            shell=False,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise DvcEvidenceError from None
    return result.stdout


def _git_state(root: Path, environment: Mapping[str, str]) -> tuple[str, bool]:
    top_level = _run(["git", "rev-parse", "--show-toplevel"], root, environment).strip()
    try:
        if Path(top_level).resolve(strict=True) != root:
            raise DvcEvidenceError
    except (OSError, RuntimeError):
        raise DvcEvidenceError from None
    commit = _run(["git", "rev-parse", "--verify", "HEAD"], root, environment).strip()
    clean = not _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        root,
        environment,
    )
    return commit, clean


def _git_origin(
    root: Path,
    environment: Mapping[str, str],
    metadata_repository_role: DvcMetadataRepositoryRole,
) -> str:
    origin = _run(["git", "remote", "get-url", "origin"], root, environment).strip()
    if (
        not origin
        or len(origin) > 2048
        or len(origin.splitlines()) != 1
        or (origin in PUBLIC_REPOSITORY_ORIGINS) != (metadata_repository_role == "public-fixture")
    ):
        raise DvcEvidenceError
    return origin


def _dvc_state(root: Path, environment: Mapping[str, str]) -> bool:
    command = [sys.executable, "-I", "-m", "dvc"]
    version = _run([*command, "--version"], root, environment).strip()
    cache_type = _run(
        [*command, "config", "cache.type"],
        root,
        environment,
    ).strip()
    status = _run(
        [*command, "status", "--json"],
        root,
        environment,
    ).strip()
    try:
        parsed_status = json.loads(status)
    except (json.JSONDecodeError, RecursionError):
        raise DvcEvidenceError from None
    return version == DVC_VERSION and cache_type == "reflink,copy" and parsed_status == {}


def _snapshot_from_commit(
    root: Path,
    commit: str,
    environment: Mapping[str, str],
    temporary_root: Path,
    metadata_repository_role: DvcMetadataRepositoryRole,
) -> DvcSnapshotV1:
    controls = temporary_root / "committed-controls"
    try:
        controls.mkdir()
        for name in ("uv.lock", "dvc.yaml", "dvc.lock"):
            content = _run(["git", "show", f"{commit}:{name}"], root, environment)
            encoded = content.encode("utf-8", errors="strict")
            if not 0 < len(encoded) <= MAX_CONTROL_FILE_BYTES:
                raise DvcEvidenceError
            (controls / name).write_bytes(encoded)
        return build_dvc_snapshot(
            controls,
            commit,
            metadata_repository_role=metadata_repository_role,
            git_working_tree_clean=True,
            dvc_workspace_clean=True,
        )
    except DvcEvidenceError:
        raise
    except (OSError, TypeError, UnicodeError, ValueError):
        raise DvcEvidenceError from None


def capture_dvc_snapshot(
    repository_root: str | Path,
    *,
    metadata_repository_role: DvcMetadataRepositoryRole,
) -> DvcSnapshotV1:
    """Capture one race-resistant snapshot from a clean Git and DVC workspace."""

    root_input = Path(repository_root)
    try:
        details = root_input.lstat()
        root = root_input.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DvcEvidenceError from None
    if (
        stat.S_ISLNK(details.st_mode)
        or getattr(details, "st_file_attributes", 0) & _REPARSE_POINT
        or not stat.S_ISDIR(details.st_mode)
    ):
        raise DvcEvidenceError

    try:
        with tempfile.TemporaryDirectory(prefix="signlab-dvc-evidence-") as temporary:
            temporary_root = Path(temporary).resolve(strict=True)
            environment = _safe_environment(temporary_root / "environment")
            before_origin = _git_origin(root, environment, metadata_repository_role)
            before_commit, before_git_clean = _git_state(root, environment)
            before_dvc_clean = _dvc_state(root, environment)
            if not before_git_clean or not before_dvc_clean:
                raise DvcEvidenceError
            before = build_dvc_snapshot(
                root,
                before_commit,
                metadata_repository_role=metadata_repository_role,
                git_working_tree_clean=True,
                dvc_workspace_clean=True,
            )
            committed = _snapshot_from_commit(
                root,
                before_commit,
                environment,
                temporary_root,
                metadata_repository_role,
            )
            if before != committed:
                raise DvcEvidenceError
            after_origin = _git_origin(root, environment, metadata_repository_role)
            after_commit, after_git_clean = _git_state(root, environment)
            after_dvc_clean = _dvc_state(root, environment)
            after = build_dvc_snapshot(
                root,
                after_commit,
                metadata_repository_role=metadata_repository_role,
                git_working_tree_clean=after_git_clean,
                dvc_workspace_clean=after_dvc_clean,
            )
    except DvcEvidenceError:
        raise
    except (OSError, TypeError, UnicodeError, ValueError):
        raise DvcEvidenceError from None
    if before_origin != after_origin or before != after:
        raise DvcEvidenceError
    return before


def _is_link_or_reparse(path: Path) -> bool:
    details = path.lstat()
    return stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _safe_report_destination(root: Path, relative_path: str) -> Path:
    try:
        locator = WorkspaceRelativeLocatorV1.model_validate(
            {"kind": "workspace_relative", "path": relative_path},
            strict=True,
        )
    except ValidationError:
        raise DvcEvidenceError from None
    if not locator.path.startswith(_REPORT_PREFIX) or not locator.path.endswith(".json"):
        raise DvcEvidenceError
    destination = root.joinpath(*locator.path.split("/"))
    current = root
    for segment in destination.parent.relative_to(root).parts:
        current /= segment
        if current.exists() or current.is_symlink():
            try:
                if _is_link_or_reparse(current) or not current.is_dir():
                    raise DvcEvidenceError
            except OSError:
                raise DvcEvidenceError from None
        else:
            try:
                current.mkdir()
            except OSError:
                raise DvcEvidenceError from None
    if destination.exists() or destination.is_symlink():
        raise DvcEvidenceError
    return destination


def write_dvc_snapshot(
    snapshot: DvcSnapshotV1,
    repository_root: str | Path,
    relative_path: str,
) -> Path:
    """Atomically write one new ignored evidence file beneath ``reports/reproduction``."""

    root = Path(repository_root).resolve(strict=True)
    destination = _safe_report_destination(root, relative_path)
    temporary_path: Path | None = None
    try:
        payload = canonical_json_bytes(snapshot) + b"\n"
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        # Linking a same-directory temporary file publishes atomically without
        # the overwrite behavior of os.replace if another writer wins the race.
        os.link(temporary_path, destination, follow_symlinks=False)
        temporary_path.unlink()
        temporary_path = None
    except (OSError, TypeError, ValueError):
        raise DvcEvidenceError from None
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
    return destination


__all__ = [
    "DvcEvidenceError",
    "capture_dvc_snapshot",
    "write_dvc_snapshot",
]
