"""Strict, portable DVC provenance snapshots for experiment metadata."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Never, Self, cast

import yaml
from pydantic import Field, StringConstraints, ValidationError, model_validator
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from signlab.contracts.canonical import (
    CanonicalizationError,
    JsonValue,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    MAX_SAFE_INTEGER,
    GitCommit,
    NonNegativeSafeInteger,
    StrictContractModel,
    WorkspacePath,
)
from signlab.contracts.taxonomy import Sha256Digest

DVC_VERSION: Final = "3.67.1"
PUBLIC_REPOSITORY: Final = "https://github.com/peterbucci/signlab"
PUBLIC_REPOSITORY_ORIGINS: Final = frozenset(
    {
        PUBLIC_REPOSITORY,
        f"{PUBLIC_REPOSITORY}.git",
        "git@github.com:peterbucci/signlab.git",
        "ssh://git@github.com/peterbucci/signlab.git",
    }
)
MAX_CONTROL_FILE_BYTES: Final = 1_048_576
_REPARSE_POINT: Final = 0x400

type DvcStageName = Literal["ingest", "validate", "extract", "quality", "split", "feature"]
type DvcMetadataRepositoryRole = Literal["public-fixture", "protected-metadata"]
type DvcDependencyPath = WorkspacePath | Literal[".python-version"]
EXPECTED_DVC_STAGES: Final[tuple[DvcStageName, ...]] = (
    "ingest",
    "validate",
    "extract",
    "quality",
    "split",
    "feature",
)

DvcMd5 = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{32}(?:\.dir)?$"),
]
ParameterName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$",
    ),
]

type DvcProvenanceErrorCategory = Literal[
    "control_file.invalid",
    "lock.invalid",
    "snapshot.invalid",
]

_ERROR_MESSAGES: Final[dict[DvcProvenanceErrorCategory, str]] = {
    "control_file.invalid": "DVC provenance control files are unavailable or unsafe",
    "lock.invalid": "DVC lock metadata is invalid or unsupported",
    "snapshot.invalid": "DVC provenance snapshot is invalid",
}
_STAGE_KEYS: Final = frozenset({"cmd", "deps", "outs", "params"})
_REQUIRED_STAGE_KEYS: Final = frozenset({"cmd", "deps", "outs"})
_ARTIFACT_KEYS: Final = frozenset({"path", "hash", "md5", "size", "nfiles"})
_REQUIRED_ARTIFACT_KEYS: Final = frozenset({"path", "hash", "md5", "size"})
_CONTROL_FILE_NAMES: Final = ("uv.lock", "dvc.yaml", "dvc.lock")
_COMMAND_FORBIDDEN_PATHS: Final = (
    re.compile(r"(?i)(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    re.compile(r"(?i)(?:^|\s)/(?:home|users)/"),
    re.compile(r"(?i)file:///"),
)


class DvcProvenanceError(ValueError):
    """A stable failure that never echoes a path, command, or lock value."""

    def __init__(self, category: DvcProvenanceErrorCategory) -> None:
        self.category = category
        self.code = f"dvc.provenance.{category}"
        super().__init__(_ERROR_MESSAGES[category])


class _UnsafeDvcYaml(ValueError):
    """Internal marker for YAML syntax outside the accepted JSON-compatible subset."""


class _StrictDvcLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects aliases, anchors, merges, and duplicate keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            raise _UnsafeDvcYaml("aliases are forbidden")
        event = self.peek_event()  # type: ignore[no-untyped-call]
        if getattr(event, "anchor", None) is not None:
            raise _UnsafeDvcYaml("anchors are forbidden")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Hashable, Any]:
        if not isinstance(node, MappingNode):
            raise _UnsafeDvcYaml("mapping shape is invalid")
        result: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge" or key_node.value == "<<":
                raise _UnsafeDvcYaml("merge keys are forbidden")
            key = self.construct_object(key_node, deep=deep)
            if type(key) is not str:
                raise _UnsafeDvcYaml("mapping keys must be strings")
            if key in result:
                raise _UnsafeDvcYaml("duplicate mapping key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class DvcPathIdentityV1(StrictContractModel):
    """One portable DVC output with its native content identity."""

    path: WorkspacePath
    hash_type: Literal["md5"]
    hash_value: DvcMd5
    size_bytes: NonNegativeSafeInteger
    file_count: NonNegativeSafeInteger | None = None

    @model_validator(mode="after")
    def _require_consistent_directory_identity(self) -> Self:
        is_directory_hash = self.hash_value.endswith(".dir")
        if is_directory_hash != (self.file_count is not None):
            raise ValueError("DVC directory hashes and file counts must be recorded together")
        return self


class DvcDependencyIdentityV1(StrictContractModel):
    """One portable dependency, including the exact pinned-interpreter control."""

    path: DvcDependencyPath
    hash_type: Literal["md5"]
    hash_value: DvcMd5
    size_bytes: NonNegativeSafeInteger
    file_count: NonNegativeSafeInteger | None = None

    @model_validator(mode="after")
    def _require_consistent_directory_identity(self) -> Self:
        is_directory_hash = self.hash_value.endswith(".dir")
        if is_directory_hash != (self.file_count is not None):
            raise ValueError("DVC directory hashes and file counts must be recorded together")
        return self


class DvcStageIdentityV1(StrictContractModel):
    """Canonical identity of one exact DVC lock entry and its data edges."""

    stage_name: DvcStageName
    lock_entry_sha256: Sha256Digest
    dependencies: tuple[DvcDependencyIdentityV1, ...] = Field(min_length=1, max_length=256)
    outputs: tuple[DvcPathIdentityV1, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _require_canonical_unique_edges(self) -> Self:
        for field_name in ("dependencies", "outputs"):
            identities = getattr(self, field_name)
            paths = tuple(identity.path for identity in identities)
            canonical = tuple(sorted(paths, key=lambda value: (value.casefold(), value)))
            if paths != canonical or len({path.casefold() for path in paths}) != len(paths):
                raise ValueError("DVC stage paths must be cross-platform unique and sorted")
        dependency_paths = {identity.path.casefold() for identity in self.dependencies}
        output_paths = {identity.path.casefold() for identity in self.outputs}
        if dependency_paths & output_paths:
            raise ValueError("a DVC stage cannot use the same path as input and output")
        return self


class DvcSnapshotV1(StrictContractModel):
    """Immutable DVC repository state attached to an experiment or evidence report."""

    schema_version: Literal["dvc-snapshot/1"]
    metadata_repository_role: DvcMetadataRepositoryRole
    metadata_repository: Literal["https://github.com/peterbucci/signlab"] | None
    metadata_git_commit: GitCommit
    git_working_tree_clean: Literal[True]
    dvc_workspace_clean: Literal[True]
    uv_lock_sha256: Sha256Digest
    dvc_yaml_sha256: Sha256Digest
    dvc_lock_sha256: Sha256Digest
    dvc_version: Literal["3.67.1"]
    stages: tuple[DvcStageIdentityV1, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def _require_exact_stage_inventory(self) -> Self:
        if self.metadata_repository_role == "public-fixture":
            if self.metadata_repository != PUBLIC_REPOSITORY:
                raise ValueError("public fixture metadata must name the public repository")
        elif self.metadata_repository is not None:
            raise ValueError("protected metadata repositories must remain opaque")
        names = tuple(stage.stage_name for stage in self.stages)
        if names != EXPECTED_DVC_STAGES:
            raise ValueError("DVC snapshot must contain the exact canonical stage inventory")
        output_paths = [output.path.casefold() for stage in self.stages for output in stage.outputs]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("DVC stage outputs must be globally unique")
        return self


@dataclass(frozen=True, slots=True)
class DvcMlflowProjection:
    """Sanitized immutable MLflow params plus low-cardinality search tags."""

    params: Mapping[str, str]
    tags: Mapping[str, str]


type DvcSnapshotInput = DvcSnapshotV1 | str | bytes | bytearray | Mapping[str, object]
type DvcLockInput = str | bytes | bytearray


def _fail(category: DvcProvenanceErrorCategory) -> Never:
    raise DvcProvenanceError(category) from None


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _normalized_utf8_text(encoded: bytes, *, category: DvcProvenanceErrorCategory) -> str:
    if encoded.startswith(b"\xef\xbb\xbf") or b"\0" in encoded:
        _fail(category)
    if b"\r\n" in encoded:
        remainder = encoded.replace(b"\r\n", b"")
        if b"\r" in remainder or b"\n" in remainder:
            _fail(category)
        encoded = encoded.replace(b"\r\n", b"\n")
    elif b"\r" in encoded:
        _fail(category)
    try:
        return encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail(category)


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_POINT)


def _directory_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        stat.S_IFMT(details.st_mode),
        details.st_ctime_ns,
        getattr(details, "st_file_attributes", 0),
    )


def _file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        stat.S_IFMT(details.st_mode),
        details.st_nlink,
        details.st_size,
        details.st_mtime_ns,
        getattr(details, "st_file_attributes", 0),
    )


def _path_file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (*_file_identity(details), details.st_ctime_ns)


def _require_directory(details: os.stat_result) -> None:
    if stat.S_ISLNK(details.st_mode) or _is_reparse(details) or not stat.S_ISDIR(details.st_mode):
        _fail("control_file.invalid")


def _require_control_file(details: os.stat_result) -> None:
    if (
        stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or not 0 < details.st_size <= MAX_CONTROL_FILE_BYTES
    ):
        _fail("control_file.invalid")


def _validated_text_bytes(value: DvcLockInput) -> tuple[bytes, str]:
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            _fail("lock.invalid")
    elif isinstance(value, (bytes, bytearray)):
        encoded = bytes(value)
    else:  # pragma: no cover - excluded by the public type, retained for runtime callers.
        _fail("lock.invalid")
    if not encoded or len(encoded) > MAX_CONTROL_FILE_BYTES:
        _fail("lock.invalid")
    decoded = _normalized_utf8_text(encoded, category="lock.invalid")
    return encoded, decoded


def _load_yaml_object(document: DvcLockInput) -> dict[str, object]:
    _, decoded = _validated_text_bytes(document)
    try:
        loaded = yaml.load(decoded, Loader=_StrictDvcLoader)
    except (
        _UnsafeDvcYaml,
        UnicodeError,
        yaml.YAMLError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        _fail("lock.invalid")
    if type(loaded) is not dict:
        _fail("lock.invalid")
    return cast(dict[str, object], loaded)


def _mapping(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _fail("lock.invalid")
    return cast(dict[str, object], value)


def _sequence(value: object) -> list[object]:
    if type(value) is not list:
        _fail("lock.invalid")
    return cast(list[object], value)


def _safe_integer(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        _fail("lock.invalid")
    return value


type _DvcArtifactIdentity = DvcDependencyIdentityV1 | DvcPathIdentityV1


def _parse_artifact(
    value: object,
    *,
    dependency: bool,
) -> tuple[dict[str, JsonValue], _DvcArtifactIdentity]:
    raw = _mapping(value)
    keys = frozenset(raw)
    if not keys >= _REQUIRED_ARTIFACT_KEYS or not keys <= _ARTIFACT_KEYS:
        _fail("lock.invalid")
    path = raw["path"]
    hash_type = raw["hash"]
    hash_value = raw["md5"]
    if type(path) is not str or hash_type != "md5" or type(hash_value) is not str:
        _fail("lock.invalid")
    try:
        identity: _DvcArtifactIdentity
        values = {
            "path": path,
            "hash_type": hash_type,
            "hash_value": hash_value,
            "size_bytes": _safe_integer(raw["size"]),
            "file_count": _safe_integer(raw["nfiles"]) if "nfiles" in raw else None,
        }
        if dependency:
            identity = DvcDependencyIdentityV1.model_validate(values, strict=True)
        else:
            identity = DvcPathIdentityV1.model_validate(values, strict=True)
    except (KeyError, TypeError, ValueError, ValidationError):
        _fail("lock.invalid")
    normalized: dict[str, JsonValue] = {
        "path": identity.path,
        "hash": identity.hash_type,
        "md5": identity.hash_value,
        "size": identity.size_bytes,
    }
    if identity.file_count is not None:
        normalized["nfiles"] = identity.file_count
    return normalized, identity


def _parse_artifacts(
    value: object,
    *,
    dependency: bool,
) -> tuple[list[JsonValue], tuple[_DvcArtifactIdentity, ...]]:
    raw_items = _sequence(value)
    if not 1 <= len(raw_items) <= 256:
        _fail("lock.invalid")
    parsed = [_parse_artifact(item, dependency=dependency) for item in raw_items]
    parsed.sort(key=lambda item: (item[1].path.casefold(), item[1].path))
    identities = tuple(item[1] for item in parsed)
    if len({identity.path.casefold() for identity in identities}) != len(identities):
        _fail("lock.invalid")
    normalized: list[JsonValue] = [item[0] for item in parsed]
    return normalized, identities


def _parse_params(value: object) -> dict[str, JsonValue]:
    sources = _mapping(value)
    normalized: dict[str, JsonValue] = {}
    for source_path, raw_parameters in sources.items():
        try:
            # Reuse the public path contract without allowing Pydantic coercion.
            path_holder = DvcPathIdentityV1(
                path=source_path,
                hash_type="md5",
                hash_value="0" * 32,
                size_bytes=0,
                file_count=None,
            )
        except (TypeError, ValueError, ValidationError):
            _fail("lock.invalid")
        parameters = _mapping(raw_parameters)
        normalized_parameters: dict[str, JsonValue] = {}
        for parameter_name, parameter_value in parameters.items():
            try:
                name_model = _ParameterNameModel(name=parameter_name)
                checked_value = parse_json_object({"value": parameter_value})["value"]
            except (CanonicalizationError, TypeError, ValueError, ValidationError):
                _fail("lock.invalid")
            normalized_parameters[name_model.name] = checked_value
        normalized[path_holder.path] = normalized_parameters
    return normalized


class _ParameterNameModel(StrictContractModel):
    name: ParameterName


def _parse_command(value: object) -> str:
    if type(value) is not str:
        _fail("lock.invalid")
    command = value
    if (
        not command
        or len(command) > 4096
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in command)
        or any(pattern.search(command) for pattern in _COMMAND_FORBIDDEN_PATHS)
        or "\\" in command
    ):
        _fail("lock.invalid")
    return command


def _parse_stage(stage_name: DvcStageName, value: object) -> DvcStageIdentityV1:
    raw = _mapping(value)
    keys = frozenset(raw)
    if not keys >= _REQUIRED_STAGE_KEYS or not keys <= _STAGE_KEYS:
        _fail("lock.invalid")
    command = _parse_command(raw["cmd"])
    normalized_deps, parsed_dependencies = _parse_artifacts(raw["deps"], dependency=True)
    normalized_outs, parsed_outputs = _parse_artifacts(raw["outs"], dependency=False)
    dependencies = cast(tuple[DvcDependencyIdentityV1, ...], parsed_dependencies)
    outputs = cast(tuple[DvcPathIdentityV1, ...], parsed_outputs)
    dependency_paths = {identity.path.casefold() for identity in dependencies}
    output_paths = {identity.path.casefold() for identity in outputs}
    if dependency_paths & output_paths:
        _fail("lock.invalid")
    normalized_entry: dict[str, JsonValue] = {
        "cmd": command,
        "deps": normalized_deps,
        "outs": normalized_outs,
    }
    if "params" in raw:
        normalized_entry["params"] = _parse_params(raw["params"])
    try:
        entry_sha256 = canonical_sha256(
            normalized_entry,
            domain=f"dvc-stage-lock/1/{stage_name}",
        )
        return DvcStageIdentityV1(
            stage_name=stage_name,
            lock_entry_sha256=entry_sha256,
            dependencies=dependencies,
            outputs=outputs,
        )
    except (CanonicalizationError, TypeError, ValueError, ValidationError):
        _fail("lock.invalid")


def parse_dvc_lock(document: DvcLockInput) -> tuple[DvcStageIdentityV1, ...]:
    """Parse one exact DVC 3.67.1 lockfile without accepting YAML conveniences."""

    try:
        payload = _load_yaml_object(document)
        if frozenset(payload) != frozenset({"schema", "stages"}):
            _fail("lock.invalid")
        if payload["schema"] != "2.0":
            _fail("lock.invalid")
        raw_stages = _mapping(payload["stages"])
        if frozenset(raw_stages) != frozenset(EXPECTED_DVC_STAGES):
            _fail("lock.invalid")
        identities = tuple(
            _parse_stage(stage_name, raw_stages[stage_name]) for stage_name in EXPECTED_DVC_STAGES
        )
        output_paths = [output.path.casefold() for stage in identities for output in stage.outputs]
        if len(output_paths) != len(set(output_paths)):
            _fail("lock.invalid")
        return identities
    except DvcProvenanceError:
        raise
    except (CanonicalizationError, KeyError, TypeError, ValueError, ValidationError):
        _fail("lock.invalid")


def _read_control_file(repository_root: Path, name: str) -> bytes:
    try:
        root_status = os.lstat(repository_root)
        _require_directory(root_status)
        resolved_root = repository_root.resolve(strict=True)
        resolved_root_status = os.lstat(resolved_root)
        _require_directory(resolved_root_status)
        if _directory_identity(root_status) != _directory_identity(resolved_root_status):
            _fail("control_file.invalid")
        path = repository_root / name
        path_status = os.lstat(path)
        _require_control_file(path_status)
        resolved_path = path.resolve(strict=True)
        if resolved_path.parent != resolved_root:
            _fail("control_file.invalid")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        try:
            opened_status = os.fstat(descriptor)
            _require_control_file(opened_status)
            chunks: list[bytes] = []
            remaining = MAX_CONTROL_FILE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65_536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            captured = b"".join(chunks)
            final_status = os.fstat(descriptor)
            _require_control_file(final_status)
        finally:
            os.close(descriptor)
        path_after = os.lstat(path)
        _require_control_file(path_after)
        root_after = os.lstat(repository_root)
        _require_directory(root_after)
        if (
            _file_identity(path_status) != _file_identity(opened_status)
            or _file_identity(opened_status) != _file_identity(final_status)
            or _path_file_identity(path_status) != _path_file_identity(path_after)
            or _directory_identity(root_status) != _directory_identity(root_after)
            or repository_root.resolve(strict=True) != resolved_root
            or path.resolve(strict=True) != resolved_path
            or len(captured) > MAX_CONTROL_FILE_BYTES
            or len(captured) != path_status.st_size
        ):
            _fail("control_file.invalid")
        normalized = _normalized_utf8_text(captured, category="control_file.invalid")
        # Git stores these reviewed text controls with LF. Normalize DVC's native
        # Windows CRLF writer so one commit has one portable provenance identity.
        return normalized.encode("utf-8")
    except DvcProvenanceError:
        raise
    except (OSError, RuntimeError, UnicodeError, ValueError):
        _fail("control_file.invalid")


def build_dvc_snapshot(
    repository_root: str | Path,
    git_commit: str,
    *,
    metadata_repository_role: DvcMetadataRepositoryRole,
    git_working_tree_clean: bool,
    dvc_workspace_clean: bool,
) -> DvcSnapshotV1:
    """Build a snapshot from explicit verified state without consulting the active branch."""

    try:
        root = Path(repository_root)
        if git_working_tree_clean is not True or dvc_workspace_clean is not True:
            _fail("snapshot.invalid")
        control_files = {name: _read_control_file(root, name) for name in _CONTROL_FILE_NAMES}
        stages = parse_dvc_lock(control_files["dvc.lock"])
        return DvcSnapshotV1(
            schema_version="dvc-snapshot/1",
            metadata_repository_role=metadata_repository_role,
            metadata_repository=(
                PUBLIC_REPOSITORY if metadata_repository_role == "public-fixture" else None
            ),
            metadata_git_commit=git_commit,
            git_working_tree_clean=git_working_tree_clean,
            dvc_workspace_clean=dvc_workspace_clean,
            uv_lock_sha256=_sha256(control_files["uv.lock"]),
            dvc_yaml_sha256=_sha256(control_files["dvc.yaml"]),
            dvc_lock_sha256=_sha256(control_files["dvc.lock"]),
            dvc_version=DVC_VERSION,
            stages=stages,
        )
    except DvcProvenanceError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, ValidationError):
        _fail("snapshot.invalid")


def validate_dvc_snapshot(document: DvcSnapshotInput) -> DvcSnapshotV1:
    """Strictly validate one JSON DVC snapshot with a sanitized failure surface."""

    try:
        if isinstance(document, DvcSnapshotV1):
            payload = document.model_dump(mode="json", round_trip=True)
        else:
            payload = parse_json_object(document)
        if payload.get("schema_version") != "dvc-snapshot/1":
            _fail("snapshot.invalid")
        return DvcSnapshotV1.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except DvcProvenanceError:
        raise
    except (CanonicalizationError, TypeError, ValueError, ValidationError):
        _fail("snapshot.invalid")


def dvc_snapshot_digest(document: DvcSnapshotInput) -> str:
    """Return the portable domain-separated identity of one validated snapshot."""

    snapshot = validate_dvc_snapshot(document)
    try:
        return canonical_sha256(snapshot, domain="dvc-snapshot/1")
    except CanonicalizationError:
        _fail("snapshot.invalid")


def dvc_snapshot_mlflow_projection(document: DvcSnapshotInput) -> DvcMlflowProjection:
    """Project safe immutable identities without exporting lock paths or commands."""

    snapshot = validate_dvc_snapshot(document)
    params = {
        "signlab.dvc.schema_version": snapshot.schema_version,
        "signlab.dvc.metadata_repository_role": snapshot.metadata_repository_role,
        "signlab.dvc.metadata_git_commit": snapshot.metadata_git_commit,
        "signlab.dvc.git_working_tree_clean": "true",
        "signlab.dvc.workspace_clean": "true",
        "signlab.dvc.version": snapshot.dvc_version,
        "signlab.dvc.uv_lock_sha256": snapshot.uv_lock_sha256,
        "signlab.dvc.yaml_sha256": snapshot.dvc_yaml_sha256,
        "signlab.dvc.lock_sha256": snapshot.dvc_lock_sha256,
        **{
            f"signlab.dvc.stage.{stage.stage_name}.sha256": stage.lock_entry_sha256
            for stage in snapshot.stages
        },
    }
    tags = {
        "signlab.provenance.schema": snapshot.schema_version,
        "signlab.dvc.metadata_repository_role": snapshot.metadata_repository_role,
        "signlab.dvc.metadata_git_commit": snapshot.metadata_git_commit,
        "signlab.dvc.version": snapshot.dvc_version,
        "signlab.dvc.reproducible": "true",
    }
    if snapshot.metadata_repository is not None:
        tags["signlab.dvc.metadata_repository"] = "peterbucci/signlab"
    return DvcMlflowProjection(
        params=MappingProxyType(params),
        tags=MappingProxyType(tags),
    )


__all__ = [
    "DVC_VERSION",
    "EXPECTED_DVC_STAGES",
    "MAX_CONTROL_FILE_BYTES",
    "PUBLIC_REPOSITORY_ORIGINS",
    "DvcDependencyIdentityV1",
    "DvcDependencyPath",
    "DvcMetadataRepositoryRole",
    "DvcMlflowProjection",
    "DvcPathIdentityV1",
    "DvcProvenanceError",
    "DvcSnapshotV1",
    "DvcStageIdentityV1",
    "build_dvc_snapshot",
    "dvc_snapshot_digest",
    "dvc_snapshot_mlflow_projection",
    "parse_dvc_lock",
    "validate_dvc_snapshot",
]
