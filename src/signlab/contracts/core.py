"""Portable primitives shared by SignLab's versioned pipeline contracts."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Annotated, Final, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from signlab.contracts.taxonomy import Sha256Digest, TaxonomyRef

SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
JCS_ALGORITHM: Final = "rfc8785/1"
MIN_SAFE_INTEGER: Final = -(2**53) + 1
MAX_SAFE_INTEGER: Final = 2**53 - 1

ContractKind = Literal[
    "dataset",
    "split",
    "preprocessing",
    "resolved_configuration",
    "run",
    "model",
]
CURRENT_CONTRACT_SCHEMAS: Final[dict[ContractKind, str]] = {
    "dataset": "dataset-manifest/1",
    "split": "split-manifest/1",
    "preprocessing": "preprocessing-plan/1",
    "resolved_configuration": "resolved-configuration/1",
    "run": "run-record/1",
    "model": "model-manifest/1",
}
SUPPORTED_CONTRACT_REFERENCE_SCHEMAS: Final[dict[ContractKind, frozenset[str]]] = {
    "dataset": frozenset({"dataset-manifest/1"}),
    "split": frozenset({"split-manifest/1"}),
    "preprocessing": frozenset({"preprocessing-plan/1"}),
    "resolved_configuration": frozenset({"resolved-configuration/1"}),
    "run": frozenset({"run-record/1"}),
    "model": frozenset({"model-manifest/1"}),
}
if any(
    schema_version not in SUPPORTED_CONTRACT_REFERENCE_SCHEMAS[kind]
    for kind, schema_version in CURRENT_CONTRACT_SCHEMAS.items()
):
    raise RuntimeError("every current contract writer must have a retained reference reader")

StableId = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$",
    ),
]
SemanticVersion = Annotated[
    str,
    StringConstraints(
        max_length=64,
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$",
    ),
]
DottedId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$",
    ),
]
MediaType = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=127,
        pattern=r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$",
    ),
]
SchemaName = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=128,
        pattern=r"^[a-z][a-z0-9-]*/[1-9][0-9]*$",
    ),
]
GitCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


def _normalize_safe_integer(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("boolean values are not integers")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return value
        value = int(value)
    if isinstance(value, int) and not MIN_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
        raise ValueError("integer is outside the exact interoperable JSON range")
    return value


SafeInteger = Annotated[
    int,
    Field(ge=MIN_SAFE_INTEGER, le=MAX_SAFE_INTEGER),
    BeforeValidator(_normalize_safe_integer),
]
NonNegativeSafeInteger = Annotated[
    int,
    Field(ge=0, le=MAX_SAFE_INTEGER),
    BeforeValidator(_normalize_safe_integer),
]
PositiveSafeInteger = Annotated[
    int,
    Field(gt=0, le=MAX_SAFE_INTEGER),
    BeforeValidator(_normalize_safe_integer),
]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def _validate_utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(
            "timestamp must be a real UTC second in YYYY-MM-DDTHH:MM:SSZ form"
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("timestamp must use canonical UTC form")
    return value


UtcTimestamp = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"),
    AfterValidator(_validate_utc_timestamp),
]

_PORTABLE_PATH_PATTERN: Final = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
)
_WINDOWS_RESERVED_NAMES: Final = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _validate_workspace_path(value: str) -> str:
    if len(value) > 1024 or not _PORTABLE_PATH_PATTERN.fullmatch(value):
        raise ValueError("path must be a normalized workspace-relative POSIX path")
    for segment in value.split("/"):
        if segment in {".", ".."} or segment.endswith((".", " ")):
            raise ValueError("path contains a non-portable segment")
        stem = segment.split(".", 1)[0].casefold()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise ValueError("path contains a reserved cross-platform segment")
    return value


WorkspacePath = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=1024,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$",
    ),
    AfterValidator(_validate_workspace_path),
]

_ARTIFACT_URI_PATTERN: Final = re.compile(
    r"^signlab://[a-z][a-z0-9-]{0,63}(?:/[a-z][a-z0-9._-]{0,127})+$"
)


def _validate_artifact_uri(value: str) -> str:
    if len(value) > 1024 or not _ARTIFACT_URI_PATTERN.fullmatch(value):
        raise ValueError("URI must use canonical signlab:// artifact syntax")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("artifact URI contains an invalid authority") from error
    if (
        parsed.scheme != "signlab"
        or not parsed.netloc
        or parsed.netloc != parsed.netloc.casefold()
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or "%" in value
        or "\\" in value
    ):
        raise ValueError("artifact URI must be logical, canonical, and credential-free")
    segments = parsed.path.removeprefix("/").split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("artifact URI contains a non-canonical path segment")
    return value


ArtifactUri = Annotated[
    str,
    StringConstraints(
        min_length=12,
        max_length=1024,
        pattern=r"^signlab://[a-z][a-z0-9-]{0,63}(/[a-z][a-z0-9._-]{0,127})+$",
    ),
    AfterValidator(_validate_artifact_uri),
]


def contract_config(schema_filename: str) -> ConfigDict:
    """Return the shared fail-closed Pydantic configuration for a public contract."""

    return ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        json_schema_extra={"$id": f"{SCHEMA_BASE}{schema_filename}"},
    )


class StrictContractModel(BaseModel):
    """Closed, immutable-by-construction base for nested contract records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )


class WorkspaceRelativeLocatorV1(StrictContractModel):
    """A normalized path interpreted only relative to an explicit workspace root."""

    kind: Literal["workspace_relative"]
    path: WorkspacePath


class ArtifactUriLocatorV1(StrictContractModel):
    """A storage-independent logical URI resolved by a future artifact adapter."""

    kind: Literal["artifact_uri"]
    uri: ArtifactUri


PortableLocatorV1 = Annotated[
    WorkspaceRelativeLocatorV1 | ArtifactUriLocatorV1,
    Field(discriminator="kind"),
]


class ArtifactRefV1(StrictContractModel):
    """Exact bytes plus a portable location; the location is never the identity."""

    schema_version: Literal["artifact-reference/1"]
    artifact_id: StableId
    role: StableId
    media_type: MediaType
    sha256: Sha256Digest
    size_bytes: NonNegativeSafeInteger
    locator: PortableLocatorV1


class ContractRefV1(StrictContractModel):
    """Digest-bound identity of another versioned SignLab contract."""

    schema_version: Literal["contract-reference/1"]
    kind: ContractKind
    contract_schema_version: SchemaName
    contract_id: StableId
    contract_version: SemanticVersion
    canonicalization: Literal["rfc8785/1"]
    sha256: Sha256Digest
    locator: PortableLocatorV1

    @model_validator(mode="after")
    def _require_supported_schema_for_kind(self) -> Self:
        supported = SUPPORTED_CONTRACT_REFERENCE_SCHEMAS[self.kind]
        if self.contract_schema_version not in supported:
            supported_text = ", ".join(sorted(supported))
            raise ValueError(
                f"{self.kind} references must use a retained contract schema: {supported_text}"
            )
        return self


ParameterScalar = bool | SafeInteger | FiniteFloat | StableId
ParameterList = Annotated[tuple[ParameterScalar, ...], Field(min_length=1, max_length=64)]


class ParameterV1(StrictContractModel):
    """One immutable, JSON-native component setting without code or path injection."""

    name: DottedId
    value: ParameterScalar | ParameterList


class ComponentSpecV1(StrictContractModel):
    """A registered implementation and its fully resolved scalar settings."""

    schema_version: Literal["component-spec/1"]
    role: Literal["model", "optimizer", "trainer", "evaluator"]
    implementation_id: StableId
    implementation_version: SemanticVersion
    parameters: tuple[ParameterV1, ...]

    @model_validator(mode="after")
    def _require_canonical_parameters(self) -> Self:
        names = tuple(parameter.name for parameter in self.parameters)
        if names != tuple(sorted(set(names))):
            raise ValueError("component parameters must have unique names in sorted order")
        return self


def same_contract_reference(left: ContractRefV1, right: ContractRefV1) -> bool:
    """Compare contract identity while deliberately ignoring storage location."""

    return (
        left.kind,
        left.contract_schema_version,
        left.contract_id,
        left.contract_version,
        left.canonicalization,
        left.sha256,
    ) == (
        right.kind,
        right.contract_schema_version,
        right.contract_id,
        right.contract_version,
        right.canonicalization,
        right.sha256,
    )


def same_artifact_reference(left: ArtifactRefV1, right: ArtifactRefV1) -> bool:
    """Compare exact artifact bytes while deliberately ignoring storage location."""

    return (
        left.artifact_id,
        left.role,
        left.media_type,
        left.sha256,
        left.size_bytes,
    ) == (
        right.artifact_id,
        right.role,
        right.media_type,
        right.sha256,
        right.size_bytes,
    )


__all__ = [
    "CURRENT_CONTRACT_SCHEMAS",
    "JCS_ALGORITHM",
    "MAX_SAFE_INTEGER",
    "MIN_SAFE_INTEGER",
    "SUPPORTED_CONTRACT_REFERENCE_SCHEMAS",
    "ArtifactRefV1",
    "ArtifactUri",
    "ArtifactUriLocatorV1",
    "ComponentSpecV1",
    "ContractKind",
    "ContractRefV1",
    "DottedId",
    "FiniteFloat",
    "GitCommit",
    "MediaType",
    "NonNegativeSafeInteger",
    "ParameterV1",
    "PortableLocatorV1",
    "PositiveSafeInteger",
    "SafeInteger",
    "SchemaName",
    "SemanticVersion",
    "StableId",
    "StrictContractModel",
    "TaxonomyRef",
    "UtcTimestamp",
    "WorkspacePath",
    "WorkspaceRelativeLocatorV1",
    "contract_config",
    "same_artifact_reference",
    "same_contract_reference",
]
