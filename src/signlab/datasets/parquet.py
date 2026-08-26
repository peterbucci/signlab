"""Deterministic, fail-closed Parquet storage for versioned dataset tables.

The JSON/Pydantic contracts remain the semantic authority.  This module is a
storage adapter: it validates those contracts before Arrow conversion, writes
one deliberately pinned Parquet profile, and verifies captured bytes before it
reconstructs a strict contract on read.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
)
from signlab.contracts.core import (
    ArtifactUriLocatorV1,
    WorkspaceRelativeLocatorV1,
)
from signlab.contracts.dataset import (
    DATASET_TABLE_PRIMARY_KEYS,
    DATASET_TABLE_ROW_MODELS,
    DATASET_TABLE_SCHEMA_VERSIONS,
    DATASET_TABLE_WRAPPER_MODELS,
    AnnotationRowV1,
    ClipRowV1,
    DatasetContractError,
    DatasetTable,
    DatasetTableRefV1,
    DerivedArtifactRowV1,
    ParticipantRowV1,
    RecordingRowV1,
    SessionRowV1,
    TableName,
    TableSchemaVersion,
    dataset_table_digest,
    validate_dataset_table,
)

type DatasetRow = (
    ParticipantRowV1
    | SessionRowV1
    | RecordingRowV1
    | ClipRowV1
    | AnnotationRowV1
    | DerivedArtifactRowV1
)
type DatasetRowInput = DatasetRow | Mapping[str, object]


class DatasetParquetError(ValueError):
    """Raised when dataset Parquet bytes or their storage boundary are invalid."""


@dataclass(frozen=True, slots=True)
class ParquetTableResult:
    """The independently computed semantic and byte evidence for one table file."""

    table_name: TableName
    table_schema_version: TableSchemaVersion
    row_count: int
    content_sha256: str
    sha256: str
    size_bytes: int
    path: Path


_FIELD_ID_KEY: Final = b"PARQUET:field_id"
_TABLE_KIND_KEY: Final = b"signlab:table_kind"
_SCHEMA_VERSION_KEY: Final = b"signlab:schema_version"
_SCHEMA_SHA256_KEY: Final = b"signlab:schema_sha256"
_CONTENT_SHA256_KEY: Final = b"signlab:content_sha256"
_ARROW_SCHEMA_KEY: Final = b"ARROW:schema"
_STATIC_SCHEMA_METADATA_KEYS: Final = frozenset(
    {_TABLE_KIND_KEY, _SCHEMA_VERSION_KEY, _SCHEMA_SHA256_KEY}
)
_FILE_SCHEMA_METADATA_KEYS: Final = _STATIC_SCHEMA_METADATA_KEYS | {_CONTENT_SHA256_KEY}
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_ROW_GROUP_SIZE: Final = 65_536
_THRIFT_STRING_SIZE_LIMIT: Final = 16 * 1024 * 1024
_THRIFT_CONTAINER_SIZE_LIMIT: Final = 1_000_000


class _FieldIds:
    """Allocate stable, positive, schema-local Parquet field IDs depth first."""

    def __init__(self) -> None:
        self._next = 1

    def take(self) -> int:
        value = self._next
        self._next += 1
        return value


def _field(
    ids: _FieldIds,
    name: str,
    data_type: pa.DataType,
    *,
    nullable: bool = False,
) -> pa.Field:
    return pa.field(
        name,
        data_type,
        nullable=nullable,
        metadata={_FIELD_ID_KEY: str(ids.take()).encode("ascii")},
    )


def _struct_field(
    ids: _FieldIds,
    name: str,
    children_builder: Callable[[_FieldIds], tuple[pa.Field, ...]],
    *,
    nullable: bool = False,
) -> pa.Field:
    field_id = ids.take()
    return pa.field(
        name,
        pa.struct(children_builder(ids)),
        nullable=nullable,
        metadata={_FIELD_ID_KEY: str(field_id).encode("ascii")},
    )


def _list_field(
    ids: _FieldIds,
    name: str,
    element_type: pa.DataType,
    *,
    nullable: bool = False,
) -> pa.Field:
    field_id = ids.take()
    element = _field(ids, "element", element_type)
    return pa.field(
        name,
        pa.list_(element),
        nullable=nullable,
        metadata={_FIELD_ID_KEY: str(field_id).encode("ascii")},
    )


def _locator_children(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "kind", pa.string()),
        _field(ids, "path", pa.string(), nullable=True),
        _field(ids, "uri", pa.string(), nullable=True),
    )


def _artifact_children(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "schema_version", pa.string()),
        _field(ids, "artifact_id", pa.string()),
        _field(ids, "role", pa.string()),
        _field(ids, "media_type", pa.string()),
        _field(ids, "sha256", pa.string()),
        _field(ids, "size_bytes", pa.int64()),
        _struct_field(ids, "locator", _locator_children),
    )


def _interval_children(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "schema_version", pa.string()),
        _field(ids, "start_us", pa.int64()),
        _field(ids, "end_us", pa.int64()),
    )


def _taxonomy_children(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "schema_version", pa.string()),
        _field(ids, "id", pa.string()),
        _field(ids, "version", pa.string()),
        _field(ids, "sha256", pa.string()),
    )


def _consent_scope_children(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "schema_version", pa.string()),
        _field(ids, "research_use", pa.bool_()),
        _field(ids, "raw_media_capture", pa.bool_()),
        _field(ids, "model_training", pa.bool_()),
        _field(ids, "model_evaluation", pa.bool_()),
        _field(ids, "public_demonstration", pa.bool_()),
        _field(ids, "model_weights_redistribution", pa.bool_()),
        _field(ids, "raw_media_retention", pa.bool_()),
        _field(ids, "raw_media_redistribution", pa.bool_()),
        _field(ids, "derived_features", pa.bool_()),
        _field(ids, "derived_features_redistribution", pa.bool_()),
        _field(ids, "evaluation_results_redistribution", pa.bool_()),
        _field(ids, "same_purpose_future_research", pa.bool_()),
        _field(ids, "withdrawal_supported", pa.bool_()),
        _field(ids, "audio_collection", pa.bool_()),
        _field(ids, "minor_participation", pa.bool_()),
        _field(ids, "identity_inference", pa.bool_()),
        _field(ids, "commercial_sale", pa.bool_()),
    )


def _consent_grant_children(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "schema_version", pa.string()),
        _field(ids, "grant_id", pa.string()),
        _field(ids, "recording_id", pa.string()),
        _field(ids, "participant_id", pa.string()),
        _field(ids, "receipt_id", pa.string()),
        _field(ids, "purpose_id", pa.string()),
        _field(ids, "study_id", pa.string()),
        _struct_field(ids, "taxonomy", _taxonomy_children),
        _struct_field(ids, "scope", _consent_scope_children),
        _field(ids, "scope_sha256", pa.string()),
        _field(ids, "receipt_scope_sha256", pa.string()),
        _field(ids, "issued_at", pa.timestamp("us", tz="UTC")),
        _field(ids, "captured_at", pa.timestamp("us", tz="UTC")),
    )


def _participants_fields(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "participant_id", pa.string()),
        _field(ids, "handedness", pa.string()),
    )


def _sessions_fields(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "session_id", pa.string()),
        _field(ids, "participant_id", pa.string()),
        _field(ids, "device_id", pa.string()),
        _field(ids, "started_at", pa.timestamp("us", tz="UTC")),
        _field(ids, "finished_at", pa.timestamp("us", tz="UTC")),
        _field(ids, "capture_mode", pa.string()),
        _field(ids, "capture_software_version", pa.string()),
        _field(ids, "camera_facing", pa.string()),
        _field(ids, "frame_width_px", pa.int64()),
        _field(ids, "frame_height_px", pa.int64()),
        _field(ids, "frame_rate_numerator", pa.int64()),
        _field(ids, "frame_rate_denominator", pa.int64()),
        _field(ids, "rotation_degrees", pa.int16()),
        _field(ids, "mirror_state", pa.string()),
    )


def _recordings_fields(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "recording_id", pa.string()),
        _field(ids, "participant_id", pa.string()),
        _field(ids, "session_id", pa.string()),
        _field(ids, "device_id", pa.string()),
        _field(ids, "captured_at", pa.timestamp("us", tz="UTC")),
        _field(ids, "duration_us", pa.int64()),
        _field(ids, "handedness", pa.string()),
        _field(ids, "mirror_state", pa.string()),
        _field(ids, "rotation_degrees", pa.int16()),
        _field(ids, "audio_present", pa.bool_()),
        _struct_field(ids, "media", _artifact_children),
        _struct_field(ids, "consent_grant", _consent_grant_children),
    )


def _clips_fields(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "clip_id", pa.string()),
        _field(ids, "participant_id", pa.string()),
        _field(ids, "session_id", pa.string()),
        _field(ids, "source_recording_id", pa.string()),
        _struct_field(ids, "interval", _interval_children),
        _field(ids, "handedness", pa.string()),
        _field(ids, "mirror_state", pa.string()),
        _struct_field(ids, "artifact", _artifact_children, nullable=True),
    )


def _annotations_fields(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "annotation_id", pa.string()),
        _field(ids, "participant_id", pa.string()),
        _field(ids, "session_id", pa.string()),
        _field(ids, "source_recording_id", pa.string()),
        _field(ids, "clip_id", pa.string(), nullable=True),
        _struct_field(ids, "interval", _interval_children),
        _field(ids, "disposition", pa.string()),
        _field(ids, "label_id", pa.string(), nullable=True),
        _field(ids, "other_kind", pa.string(), nullable=True),
        _field(ids, "reason_code", pa.string(), nullable=True),
        _field(ids, "review_status", pa.string()),
        _field(ids, "eligible_for_training", pa.bool_()),
    )


def _derived_artifacts_fields(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "derived_artifact_id", pa.string()),
        _field(ids, "derivation_kind", pa.string()),
        _list_field(ids, "parent_artifact_ids", pa.string()),
        _field(ids, "participant_id", pa.string()),
        _field(ids, "session_id", pa.string()),
        _field(ids, "source_recording_id", pa.string()),
        _field(ids, "clip_id", pa.string(), nullable=True),
        _field(ids, "annotation_id", pa.string(), nullable=True),
        _field(ids, "sample_id", pa.string(), nullable=True),
        _field(ids, "label_id", pa.string(), nullable=True),
        _field(ids, "split_id", pa.string(), nullable=True),
        _field(ids, "partition", pa.string(), nullable=True),
        _field(ids, "handedness", pa.string()),
        _field(ids, "mirror_state", pa.string()),
        _field(ids, "operation_id", pa.string()),
        _field(ids, "operation_version", pa.string()),
        _struct_field(ids, "artifact", _artifact_children),
    )


_FIELD_BUILDERS: Final = {
    "participants": _participants_fields,
    "sessions": _sessions_fields,
    "recordings": _recordings_fields,
    "clips": _clips_fields,
    "annotations": _annotations_fields,
    "derived_artifacts": _derived_artifacts_fields,
}


def _type_snapshot(data_type: pa.DataType) -> str | dict[str, object]:
    if pa.types.is_string(data_type):
        return "utf8"
    if pa.types.is_boolean(data_type):
        return "bool"
    if pa.types.is_int16(data_type):
        return "int16"
    if pa.types.is_int64(data_type):
        return "int64"
    if pa.types.is_timestamp(data_type):
        timestamp = cast(pa.TimestampType, data_type)
        return f"timestamp[{timestamp.unit}, tz={timestamp.tz}]"
    if pa.types.is_struct(data_type):
        struct_type = cast(pa.StructType, data_type)
        return {
            "kind": "struct",
            "fields": [_field_snapshot(field) for field in struct_type],
        }
    if pa.types.is_list(data_type):
        list_type = cast(pa.ListType, data_type)
        return {
            "kind": "list",
            "element": _field_snapshot(list_type.value_field),
        }
    raise RuntimeError("the Parquet schema contains an unsupported Arrow type")


def _field_id(field: pa.Field) -> int:
    metadata = field.metadata or {}
    raw = metadata.get(_FIELD_ID_KEY)
    if raw is None:
        raise RuntimeError("every Arrow field must have a Parquet field ID")
    try:
        field_id = int(raw.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("Parquet field IDs must be positive ASCII integers") from error
    if field_id <= 0 or metadata != {_FIELD_ID_KEY: raw}:
        raise RuntimeError("Arrow field metadata must contain only one positive field ID")
    return field_id


def _field_snapshot(field: pa.Field) -> dict[str, object]:
    return {
        "name": field.name,
        "type": _type_snapshot(field.type),
        "nullable": field.nullable,
        "field_id": _field_id(field),
    }


def _schema_identity_payload(
    table_name: TableName,
    schema_version: TableSchemaVersion,
    fields: tuple[pa.Field, ...],
) -> dict[str, object]:
    return {
        "format": "arrow-schema-snapshot/1",
        "table_name": table_name,
        "schema_version": schema_version,
        "fields": [_field_snapshot(field) for field in fields],
    }


def _build_schema(table_name: TableName) -> pa.Schema:
    schema_version = DATASET_TABLE_SCHEMA_VERSIONS[table_name]
    ids = _FieldIds()
    fields = _FIELD_BUILDERS[table_name](ids)
    schema_sha256 = canonical_sha256(
        _schema_identity_payload(table_name, schema_version, fields),
        domain="parquet-arrow-schema/1",
    )
    return pa.schema(
        fields,
        metadata={
            _TABLE_KIND_KEY: table_name.encode("ascii"),
            _SCHEMA_VERSION_KEY: schema_version.encode("ascii"),
            _SCHEMA_SHA256_KEY: schema_sha256.encode("ascii"),
        },
    )


_SCHEMAS_BY_TABLE: Final = {
    table_name: _build_schema(table_name) for table_name in DATASET_TABLE_SCHEMA_VERSIONS
}
DATASET_PARQUET_SCHEMAS: Final[Mapping[TableSchemaVersion, pa.Schema]] = MappingProxyType(
    {
        DATASET_TABLE_SCHEMA_VERSIONS[table_name]: schema
        for table_name, schema in _SCHEMAS_BY_TABLE.items()
    }
)
DATASET_PARQUET_SCHEMAS_BY_TABLE: Final[Mapping[TableName, pa.Schema]] = MappingProxyType(
    _SCHEMAS_BY_TABLE
)

_ROW_MODELS: Final[Mapping[TableName, type[BaseModel]]] = MappingProxyType(DATASET_TABLE_ROW_MODELS)
_PRIMARY_KEYS: Final[Mapping[TableName, str]] = MappingProxyType(DATASET_TABLE_PRIMARY_KEYS)
_TABLE_NAMES_BY_SCHEMA: Final[Mapping[TableSchemaVersion, TableName]] = MappingProxyType(
    {version: name for name, version in DATASET_TABLE_SCHEMA_VERSIONS.items()}
)


def _all_fields(fields: Iterable[pa.Field]) -> Iterable[pa.Field]:
    for field in fields:
        yield field
        if pa.types.is_struct(field.type):
            yield from _all_fields(cast(pa.StructType, field.type))
        elif pa.types.is_list(field.type):
            yield from _all_fields((cast(pa.ListType, field.type).value_field,))


def _verify_authoritative_schemas() -> None:
    for table_name, schema in DATASET_PARQUET_SCHEMAS_BY_TABLE.items():
        field_ids = tuple(_field_id(field) for field in _all_fields(schema))
        if len(field_ids) != len(set(field_ids)):
            raise RuntimeError(f"{table_name} Parquet field IDs must be globally unique")
        metadata = schema.metadata or {}
        if set(metadata) != _STATIC_SCHEMA_METADATA_KEYS:
            raise RuntimeError(f"{table_name} Parquet schema metadata is not allow-listed")
        if metadata[_TABLE_KIND_KEY] != table_name.encode("ascii"):
            raise RuntimeError(f"{table_name} Parquet schema kind is inconsistent")
        schema_version = DATASET_TABLE_SCHEMA_VERSIONS[table_name]
        if metadata[_SCHEMA_VERSION_KEY] != schema_version.encode("ascii"):
            raise RuntimeError(f"{table_name} Parquet schema version is inconsistent")
        expected_schema_sha256 = canonical_sha256(
            _schema_identity_payload(table_name, schema_version, tuple(schema)),
            domain="parquet-arrow-schema/1",
        ).encode("ascii")
        if metadata[_SCHEMA_SHA256_KEY] != expected_schema_sha256:
            raise RuntimeError(f"{table_name} Parquet schema digest is invalid")


_verify_authoritative_schemas()


def parquet_schema_snapshot(schema_version: TableSchemaVersion | str) -> dict[str, object]:
    """Return a JSON-native, human-readable snapshot of one authoritative schema."""

    schema = DATASET_PARQUET_SCHEMAS.get(cast(TableSchemaVersion, schema_version))
    table_name = _TABLE_NAMES_BY_SCHEMA.get(cast(TableSchemaVersion, schema_version))
    if schema is None or table_name is None:
        raise DatasetParquetError("unsupported dataset Parquet schema version")
    metadata = schema.metadata or {}
    return {
        **_schema_identity_payload(
            table_name,
            cast(TableSchemaVersion, schema_version),
            tuple(schema),
        ),
        "allowed_schema_metadata": {
            _TABLE_KIND_KEY.decode("ascii"): metadata[_TABLE_KIND_KEY].decode("ascii"),
            _SCHEMA_VERSION_KEY.decode("ascii"): metadata[_SCHEMA_VERSION_KEY].decode("ascii"),
            _SCHEMA_SHA256_KEY.decode("ascii"): metadata[_SCHEMA_SHA256_KEY].decode("ascii"),
            _CONTENT_SHA256_KEY.decode("ascii"): "sha256:<64 lowercase hex characters>",
        },
        "writer_generated_file_metadata": {
            _ARROW_SCHEMA_KEY.decode("ascii"): "authoritative Arrow schema encoding",
        },
    }


def _table_name(document: DatasetTable) -> TableName:
    try:
        return _TABLE_NAMES_BY_SCHEMA[document.schema_version]
    except KeyError as error:  # pragma: no cover - validate_dataset_table guards this.
        raise DatasetParquetError("unsupported dataset table schema version") from error


def build_dataset_table(
    table_name: TableName,
    rows: Iterable[DatasetRowInput],
) -> DatasetTable:
    """Strictly validate rows, reject duplicate keys, and put them in canonical order."""

    if table_name not in DATASET_TABLE_SCHEMA_VERSIONS:
        raise DatasetParquetError("unsupported dataset table name")
    row_model = _ROW_MODELS[table_name]
    primary_key = _PRIMARY_KEYS[table_name]
    checked_rows: list[BaseModel] = []
    try:
        for row in rows:
            payload: Mapping[str, object]
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="json", round_trip=True)
            elif isinstance(row, Mapping):
                payload = _json_compatible_row_mapping(row)
            else:
                raise DatasetParquetError("dataset rows must be models or mappings")
            checked_rows.append(
                row_model.model_validate_json(canonical_json_bytes(payload), strict=True)
            )
    except (CanonicalizationError, TypeError, ValidationError) as error:
        raise DatasetParquetError(f"invalid {table_name} row") from error

    identities = [cast(str, getattr(row, primary_key)) for row in checked_rows]
    if len(identities) != len(set(identities)):
        raise DatasetParquetError(f"duplicate {primary_key} values are not allowed")
    checked_rows.sort(key=lambda row: cast(str, getattr(row, primary_key)))

    schema_version = DATASET_TABLE_SCHEMA_VERSIONS[table_name]
    wrapper_model = DATASET_TABLE_WRAPPER_MODELS[table_name]
    try:
        wrapper = wrapper_model.model_validate(
            {"schema_version": schema_version, "rows": tuple(checked_rows)},
            strict=True,
        )
    except ValidationError as error:  # pragma: no cover - rows were validated above.
        raise DatasetParquetError(f"invalid {table_name} table") from error
    return cast(DatasetTable, wrapper)


def _json_compatible_row_value(value: object) -> object:
    """Normalize nested contract models/tuples without accepting non-JSON values."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", round_trip=True)
    if isinstance(value, Mapping):
        return _json_compatible_row_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_compatible_row_value(nested) for nested in value]
    return value


def _json_compatible_row_mapping(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise CanonicalizationError("dataset row object member names must be strings")
        result[key] = _json_compatible_row_value(nested)
    return result


def semantic_table_sha256(table_name: TableName, rows: Iterable[DatasetRowInput]) -> str:
    """Return the RFC 8785 domain-separated identity of canonical logical rows."""

    return dataset_table_digest(build_dataset_table(table_name, rows))


def _timestamp_to_arrow(value: object) -> datetime:
    if not isinstance(value, str):
        raise DatasetParquetError("validated timestamp was not canonical text")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:  # pragma: no cover - strict row validation happens first.
        raise DatasetParquetError("validated timestamp was not canonical UTC") from error


def _to_arrow_value(value: object, data_type: pa.DataType) -> object:
    if value is None:
        return None
    if pa.types.is_timestamp(data_type):
        return _timestamp_to_arrow(value)
    if pa.types.is_struct(data_type):
        if not isinstance(value, Mapping):
            raise DatasetParquetError("validated struct value was not an object")
        struct_type = cast(pa.StructType, data_type)
        return {
            child.name: _to_arrow_value(value.get(child.name), child.type) for child in struct_type
        }
    if pa.types.is_list(data_type):
        if not isinstance(value, (list, tuple)):
            raise DatasetParquetError("validated list value was not an array")
        element_type = cast(pa.ListType, data_type).value_type
        return [_to_arrow_value(item, element_type) for item in value]
    return value


def _arrow_rows(document: DatasetTable, schema: pa.Schema) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in document.rows:
        payload = row.model_dump(mode="json", round_trip=True)
        result.append(
            {field.name: _to_arrow_value(payload[field.name], field.type) for field in schema}
        )
    return result


def _schema_with_content(schema: pa.Schema, content_sha256: str) -> pa.Schema:
    if _SHA256_PATTERN.fullmatch(content_sha256) is None:
        raise DatasetParquetError("expected content digest is not canonical SHA-256")
    metadata = dict(schema.metadata or {})
    metadata[_CONTENT_SHA256_KEY] = content_sha256.encode("ascii")
    return schema.with_metadata(metadata)


def _parquet_bytes(document: DatasetTable) -> tuple[bytes, str]:
    table_name = _table_name(document)
    schema_version = document.schema_version
    content_sha256 = dataset_table_digest(document)
    schema = _schema_with_content(DATASET_PARQUET_SCHEMAS[schema_version], content_sha256)
    try:
        table = pa.Table.from_pylist(_arrow_rows(document, schema), schema=schema)
        table.validate(full=True)
        sink = pa.BufferOutputStream()
        sorting_columns = pq.SortingColumn.from_ordering(
            schema,
            [(_PRIMARY_KEYS[table_name], "ascending")],
        )
        pq.write_table(
            table,
            sink,
            row_group_size=_ROW_GROUP_SIZE,
            version="2.6",
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=True,
            use_deprecated_int96_timestamps=False,
            coerce_timestamps="us",
            allow_truncated_timestamps=False,
            data_page_version="1.0",
            use_compliant_nested_type=True,
            store_schema=True,
            write_page_checksum=True,
            sorting_columns=sorting_columns,
        )
        captured = sink.getvalue().to_pybytes()
    except (ArrowException, OSError, ValueError) as error:
        raise DatasetParquetError(f"could not encode {table_name} Parquet bytes") from error
    return captured, content_sha256


# PyArrow exposes a hierarchy whose base name differs across stub/runtime releases.
ArrowException = getattr(pa, "ArrowException", Exception)


def _persist_parquet_bytes(path: Path, captured: bytes) -> None:
    """Durably replace one regular destination without opening special files."""

    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise OSError("destination is not a regular file")
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(captured)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        if os.name != "nt":
            with suppress(OSError):
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    except OSError as error:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()
        raise DatasetParquetError("could not persist dataset Parquet bytes") from error


def write_dataset_table(document: DatasetTable, destination: str | Path) -> ParquetTableResult:
    """Write one already-versioned table and return hashes over logical and exact content."""

    try:
        checked = validate_dataset_table(document)
    except DatasetContractError as error:
        raise DatasetParquetError("invalid dataset table contract") from error
    captured, content_sha256 = _parquet_bytes(checked)
    path = Path(destination)
    _persist_parquet_bytes(path, captured)
    table_name = _table_name(checked)
    return ParquetTableResult(
        table_name=table_name,
        table_schema_version=checked.schema_version,
        row_count=len(checked.rows),
        content_sha256=content_sha256,
        sha256=f"sha256:{hashlib.sha256(captured).hexdigest()}",
        size_bytes=len(captured),
        path=path,
    )


def write_parquet_table(
    table_name: TableName,
    rows: Iterable[DatasetRowInput],
    destination: str | Path,
) -> ParquetTableResult:
    """Validate, canonically sort, and write logical rows as deterministic Parquet."""

    return write_dataset_table(build_dataset_table(table_name, rows), destination)


def _canonical_sha256(value: str, *, what: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DatasetParquetError(f"expected {what} is not canonical SHA-256")
    return value


def _expected_size(value: int) -> int:
    if type(value) is not int or value < 0:
        raise DatasetParquetError("expected Parquet size must be a non-negative integer")
    return value


def _read_captured_bytes(path: Path, *, expected_size_bytes: int, expected_sha256: str) -> bytes:
    expected_size = _expected_size(expected_size_bytes)
    expected_digest = _canonical_sha256(expected_sha256, what="Parquet byte digest")
    try:
        path_status = path.stat()
        if not stat.S_ISREG(path_status.st_mode):
            raise DatasetParquetError("dataset Parquet artifact must be a regular file")
        if path_status.st_size != expected_size:
            raise DatasetParquetError("dataset Parquet byte size does not match its reference")
        captured = path.read_bytes()
    except DatasetParquetError:
        raise
    except OSError as error:
        raise DatasetParquetError("could not read dataset Parquet bytes") from error
    if len(captured) != expected_size:
        raise DatasetParquetError("dataset Parquet bytes changed while being read")
    actual_digest = f"sha256:{hashlib.sha256(captured).hexdigest()}"
    if actual_digest != expected_digest:
        raise DatasetParquetError("dataset Parquet byte digest does not match its reference")
    return captured


def _timestamp_from_arrow(value: object) -> str:
    if not isinstance(value, datetime):
        raise DatasetParquetError("Parquet timestamp is not a UTC instant")
    # Arrow's verified source type is timestamp[us, tz=UTC].  The decoding table
    # deliberately drops the timezone label so Windows does not need an external
    # IANA tzdata package merely to materialize UTC values.
    utc_value = value.replace(tzinfo=UTC)
    if utc_value.microsecond != 0:
        raise DatasetParquetError("Parquet timestamp exceeds the contract's second precision")
    return (
        f"{utc_value.year:04d}-{utc_value.month:02d}-{utc_value.day:02d}T"
        f"{utc_value.hour:02d}:{utc_value.minute:02d}:{utc_value.second:02d}Z"
    )


def _from_arrow_value(value: object, data_type: pa.DataType, *, field_name: str) -> object:
    if value is None:
        return None
    if pa.types.is_timestamp(data_type):
        return _timestamp_from_arrow(value)
    if pa.types.is_struct(data_type):
        if not isinstance(value, Mapping):  # pragma: no cover - Arrow owns this conversion.
            raise DatasetParquetError("Parquet struct did not decode to an object")
        struct_type = cast(pa.StructType, data_type)
        result = {
            child.name: _from_arrow_value(
                value.get(child.name),
                child.type,
                field_name=child.name,
            )
            for child in struct_type
        }
        if field_name == "locator":
            kind = result.get("kind")
            if kind == "workspace_relative":
                result.pop("uri", None)
            elif kind == "artifact_uri":
                result.pop("path", None)
        return result
    if pa.types.is_list(data_type):
        if not isinstance(value, list):  # pragma: no cover - Arrow owns this conversion.
            raise DatasetParquetError("Parquet list did not decode to an array")
        element_type = cast(pa.ListType, data_type).value_type
        return [_from_arrow_value(item, element_type, field_name="element") for item in value]
    return value


def _contract_from_arrow(
    table_name: TableName,
    table: pa.Table,
) -> DatasetTable:
    schema_version = DATASET_TABLE_SCHEMA_VERSIONS[table_name]
    rows: list[dict[str, object]] = []
    decoding_schema = pa.schema(
        [
            pa.field(
                field.name,
                _timezone_free_type(field.type),
                nullable=field.nullable,
            )
            for field in table.schema
        ]
    )
    try:
        decoded_rows = table.cast(decoding_schema).to_pylist()
    except (ArrowException, ValueError) as error:
        raise DatasetParquetError("Parquet rows could not be decoded safely") from error
    for raw in decoded_rows:
        rows.append(
            {
                field.name: _from_arrow_value(
                    raw[field.name],
                    field.type,
                    field_name=field.name,
                )
                for field in table.schema
            }
        )
    try:
        return validate_dataset_table({"schema_version": schema_version, "rows": rows})
    except DatasetContractError as error:
        raise DatasetParquetError("Parquet rows violate the strict dataset contract") from error


def _timezone_free_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_timestamp(data_type):
        return pa.timestamp(cast(pa.TimestampType, data_type).unit)
    if pa.types.is_struct(data_type):
        return pa.struct(
            [
                pa.field(
                    field.name,
                    _timezone_free_type(field.type),
                    nullable=field.nullable,
                )
                for field in cast(pa.StructType, data_type)
            ]
        )
    if pa.types.is_list(data_type):
        list_type = cast(pa.ListType, data_type)
        element = list_type.value_field
        return pa.list_(
            pa.field(
                element.name,
                _timezone_free_type(element.type),
                nullable=element.nullable,
            )
        )
    return data_type


def read_parquet_table(
    table_name: TableName,
    source: str | Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_content_sha256: str,
    expected_row_count: int | None = None,
) -> DatasetTable:
    """Verify exact bytes, Arrow schema, and logical content before returning rows."""

    if table_name not in DATASET_TABLE_SCHEMA_VERSIONS:
        raise DatasetParquetError("unsupported dataset table name")
    expected_content = _canonical_sha256(
        expected_content_sha256,
        what="semantic content digest",
    )
    expected_rows = _expected_size(expected_row_count) if expected_row_count is not None else None
    captured = _read_captured_bytes(
        Path(source),
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
    )
    schema_version = DATASET_TABLE_SCHEMA_VERSIONS[table_name]
    expected_schema = _schema_with_content(
        DATASET_PARQUET_SCHEMAS[schema_version],
        expected_content,
    )
    try:
        # Arrow verifies every page CRC that is present. The interchange reader
        # deliberately permits equivalent encodings without CRCs because exact
        # outer SHA-256 evidence remains mandatory for every captured file.
        parquet_file = pq.ParquetFile(
            pa.BufferReader(captured),
            page_checksum_verification=True,
            thrift_string_size_limit=_THRIFT_STRING_SIZE_LIMIT,
            thrift_container_size_limit=_THRIFT_CONTAINER_SIZE_LIMIT,
            arrow_extensions_enabled=False,
        )
        if expected_rows is not None and parquet_file.metadata.num_rows != expected_rows:
            raise DatasetParquetError("dataset Parquet row count does not match its reference")
        file_metadata = parquet_file.metadata.metadata or {}
        if set(file_metadata) != _FILE_SCHEMA_METADATA_KEYS | {_ARROW_SCHEMA_KEY}:
            raise DatasetParquetError("dataset Parquet file metadata is not allow-listed")
        if not parquet_file.schema_arrow.equals(expected_schema, check_metadata=True):
            raise DatasetParquetError("dataset Parquet Arrow schema or metadata is incompatible")
        table = parquet_file.read()
        if not table.schema.equals(expected_schema, check_metadata=True):
            raise DatasetParquetError("decoded dataset Parquet schema or metadata changed")
        table.validate(full=True)
    except DatasetParquetError:
        raise
    except (ArrowException, OSError, ValueError) as error:
        raise DatasetParquetError("dataset Parquet bytes are invalid") from error

    checked = _contract_from_arrow(table_name, table)
    if len(checked.rows) != table.num_rows:
        raise DatasetParquetError("decoded dataset Parquet row count is inconsistent")
    if dataset_table_digest(checked) != expected_content:
        raise DatasetParquetError("dataset Parquet semantic content digest does not match")
    return checked


def resolve_workspace_locator(
    workspace_root: str | Path,
    locator: WorkspaceRelativeLocatorV1 | ArtifactUriLocatorV1 | str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a normalized locator under a root and reject every escape route."""

    if isinstance(locator, ArtifactUriLocatorV1):
        raise DatasetParquetError("artifact URIs require an explicit storage adapter")
    if isinstance(locator, str):
        try:
            checked_locator = WorkspaceRelativeLocatorV1.model_validate(
                {"kind": "workspace_relative", "path": locator},
                strict=True,
            )
        except ValidationError as error:
            raise DatasetParquetError("invalid workspace-relative locator") from error
    elif isinstance(locator, WorkspaceRelativeLocatorV1):
        try:
            checked_locator = WorkspaceRelativeLocatorV1.model_validate(
                locator.model_dump(mode="json", round_trip=True),
                strict=True,
            )
        except ValidationError as error:  # pragma: no cover - immutable model is already valid.
            raise DatasetParquetError("invalid workspace-relative locator") from error
    else:
        raise DatasetParquetError("unsupported artifact locator")

    root_input = Path(workspace_root)
    try:
        root = root_input.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DatasetParquetError("workspace root does not exist") from error
    if not root.is_dir():
        raise DatasetParquetError("workspace root must be a directory")
    candidate = root.joinpath(*checked_locator.path.split("/"))
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        raise DatasetParquetError("workspace-relative artifact does not exist") from error
    if not resolved.is_relative_to(root):
        raise DatasetParquetError("workspace-relative locator escapes its explicit root")
    if (must_exist or resolved.exists()) and not resolved.is_file():
        raise DatasetParquetError("workspace-relative artifact must be a regular file")
    return resolved


def read_dataset_table(
    reference: DatasetTableRefV1,
    workspace_root: str | Path,
) -> DatasetTable:
    """Resolve and verify a manifest-bound table reference from an explicit workspace."""

    try:
        checked_reference = DatasetTableRefV1.model_validate(
            reference.model_dump(mode="json", round_trip=True),
            strict=True,
        )
    except ValidationError as error:
        raise DatasetParquetError("invalid dataset table reference") from error
    path = resolve_workspace_locator(workspace_root, checked_reference.artifact.locator)
    return read_parquet_table(
        checked_reference.table_name,
        path,
        expected_size_bytes=checked_reference.artifact.size_bytes,
        expected_sha256=checked_reference.artifact.sha256,
        expected_content_sha256=checked_reference.content_sha256,
        expected_row_count=checked_reference.row_count,
    )


__all__ = [
    "DATASET_PARQUET_SCHEMAS",
    "DATASET_PARQUET_SCHEMAS_BY_TABLE",
    "DatasetParquetError",
    "ParquetTableResult",
    "build_dataset_table",
    "parquet_schema_snapshot",
    "read_dataset_table",
    "read_parquet_table",
    "resolve_workspace_locator",
    "semantic_table_sha256",
    "write_dataset_table",
    "write_parquet_table",
]
