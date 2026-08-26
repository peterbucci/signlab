"""Deterministic Parquet storage for raw landmark-frame observations."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from signlab.contracts.canonical import canonical_sha256
from signlab.contracts.extraction import (
    LandmarkFramesTableV1,
    landmark_frames_table_digest,
    validate_landmark_frames_table,
)


class LandmarkParquetError(ValueError):
    """Raised when landmark Parquet bytes or schema evidence are invalid."""


@dataclass(frozen=True, slots=True)
class LandmarkParquetResult:
    """Semantic and exact-byte evidence for one landmark sequence."""

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
_STATIC_METADATA_KEYS: Final = frozenset({_TABLE_KIND_KEY, _SCHEMA_VERSION_KEY, _SCHEMA_SHA256_KEY})
_FILE_METADATA_KEYS: Final = _STATIC_METADATA_KEYS | {_CONTENT_SHA256_KEY, _ARROW_SCHEMA_KEY}
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_ROW_GROUP_SIZE: Final = 65_536
_THRIFT_STRING_SIZE_LIMIT: Final = 16 * 1024 * 1024
_THRIFT_CONTAINER_SIZE_LIMIT: Final = 4_000_000
ArrowException = getattr(pa, "ArrowException", Exception)


class _FieldIds:
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


def _struct(
    ids: _FieldIds,
    name: str,
    children: tuple[pa.Field, ...],
    *,
    nullable: bool = False,
) -> pa.Field:
    field_id = ids.take()
    return pa.field(
        name,
        pa.struct(children),
        nullable=nullable,
        metadata={_FIELD_ID_KEY: str(field_id).encode("ascii")},
    )


def _fixed_list(
    ids: _FieldIds,
    name: str,
    element_type: pa.DataType,
    size: int,
    *,
    nullable: bool = False,
    element_nullable: bool = False,
) -> pa.Field:
    field_id = ids.take()
    element = _field(ids, "element", element_type, nullable=element_nullable)
    return pa.field(
        name,
        pa.list_(element, list_size=size),
        nullable=nullable,
        metadata={_FIELD_ID_KEY: str(field_id).encode("ascii")},
    )


def _point_type(ids: _FieldIds) -> pa.StructType:
    return pa.struct(
        (
            _field(ids, "x", pa.float64()),
            _field(ids, "y", pa.float64()),
            _field(ids, "z", pa.float64()),
            _field(ids, "visibility", pa.float64(), nullable=True),
            _field(ids, "presence", pa.float64(), nullable=True),
        )
    )


def _hand_type(ids: _FieldIds) -> pa.StructType:
    return pa.struct(
        (
            _field(ids, "slot_id", pa.string()),
            _field(ids, "present", pa.bool_()),
            _field(ids, "tracking_id", pa.string(), nullable=True),
            _field(ids, "detector_index", pa.int16(), nullable=True),
            _field(ids, "handedness", pa.string(), nullable=True),
            _field(ids, "handedness_confidence", pa.float64(), nullable=True),
            _fixed_list(
                ids,
                "image_landmarks",
                _point_type(ids),
                21,
                element_nullable=True,
            ),
            _fixed_list(
                ids,
                "world_landmarks",
                _point_type(ids),
                21,
                element_nullable=True,
            ),
        )
    )


def _anchor_type(ids: _FieldIds) -> pa.StructType:
    return pa.struct(
        (
            _field(ids, "name", pa.string()),
            _field(ids, "present", pa.bool_()),
            _struct(ids, "image_point", tuple(_point_type(ids)), nullable=True),
            _struct(ids, "world_point", tuple(_point_type(ids)), nullable=True),
        )
    )


def _frame_fields(ids: _FieldIds) -> tuple[pa.Field, ...]:
    return (
        _field(ids, "schema_version", pa.string()),
        _field(ids, "source_recording_id", pa.string()),
        _field(ids, "frame_index", pa.int64()),
        _field(ids, "source_pts", pa.int64()),
        _field(ids, "source_time_base_numerator", pa.int64()),
        _field(ids, "source_time_base_denominator", pa.int64()),
        _field(ids, "relative_timestamp_us", pa.int64()),
        _field(ids, "task_timestamp_ms", pa.int64()),
        _field(ids, "invalid", pa.bool_()),
        _field(ids, "invalid_reason", pa.string(), nullable=True),
        _fixed_list(ids, "hands", _hand_type(ids), 2),
        _fixed_list(ids, "body_anchors", _anchor_type(ids), 6),
        _field(ids, "observed_hand_count", pa.int8()),
        _field(ids, "observed_body_anchor_count", pa.int8()),
    )


def _field_id(field: pa.Field) -> int:
    metadata = field.metadata or {}
    raw = metadata.get(_FIELD_ID_KEY)
    if raw is None or metadata != {_FIELD_ID_KEY: raw}:
        raise RuntimeError("landmark Arrow fields require one stable field ID")
    try:
        value = int(raw.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("landmark Arrow field IDs must be positive ASCII integers") from error
    if value <= 0:
        raise RuntimeError("landmark Arrow field IDs must be positive")
    return value


def _type_snapshot(data_type: pa.DataType) -> str | dict[str, object]:
    if pa.types.is_string(data_type):
        return "utf8"
    if pa.types.is_boolean(data_type):
        return "bool"
    if pa.types.is_int8(data_type):
        return "int8"
    if pa.types.is_int16(data_type):
        return "int16"
    if pa.types.is_int64(data_type):
        return "int64"
    if pa.types.is_float64(data_type):
        return "float64"
    if pa.types.is_struct(data_type):
        return {
            "kind": "struct",
            "fields": [_field_snapshot(field) for field in cast(pa.StructType, data_type)],
        }
    if pa.types.is_fixed_size_list(data_type):
        fixed = cast(pa.FixedSizeListType, data_type)
        return {
            "kind": "fixed_size_list",
            "size": fixed.list_size,
            "element": _field_snapshot(fixed.value_field),
        }
    raise RuntimeError("landmark Arrow schema contains an unsupported type")


def _field_snapshot(field: pa.Field) -> dict[str, object]:
    return {
        "name": field.name,
        "type": _type_snapshot(field.type),
        "nullable": field.nullable,
        "field_id": _field_id(field),
    }


def _all_fields(fields: tuple[pa.Field, ...]) -> tuple[pa.Field, ...]:
    collected: list[pa.Field] = []
    for field in fields:
        collected.append(field)
        if pa.types.is_struct(field.type):
            collected.extend(_all_fields(tuple(cast(pa.StructType, field.type))))
        elif pa.types.is_fixed_size_list(field.type):
            element = cast(pa.FixedSizeListType, field.type).value_field
            collected.extend(_all_fields((element,)))
    return tuple(collected)


def _schema_identity(fields: tuple[pa.Field, ...]) -> dict[str, object]:
    return {
        "format": "arrow-schema-snapshot/1",
        "table_name": "landmark_frames",
        "schema_version": "landmark-frames-table/1",
        "fields": [_field_snapshot(field) for field in fields],
    }


def _build_schema() -> pa.Schema:
    fields = _frame_fields(_FieldIds())
    schema_sha256 = canonical_sha256(
        _schema_identity(fields),
        domain="parquet-arrow-schema/1",
    )
    return pa.schema(
        fields,
        metadata={
            _TABLE_KIND_KEY: b"landmark_frames",
            _SCHEMA_VERSION_KEY: b"landmark-frames-table/1",
            _SCHEMA_SHA256_KEY: schema_sha256.encode("ascii"),
        },
    )


LANDMARK_PARQUET_SCHEMA: Final = _build_schema()
LANDMARK_PARQUET_SCHEMAS: Final = MappingProxyType(
    {"landmark-frames-table/1": LANDMARK_PARQUET_SCHEMA}
)


def _verify_schema() -> None:
    fields = tuple(LANDMARK_PARQUET_SCHEMA)
    ids = tuple(_field_id(field) for field in _all_fields(fields))
    if len(ids) != len(set(ids)):
        raise RuntimeError("landmark Arrow field IDs must be globally unique")
    metadata = LANDMARK_PARQUET_SCHEMA.metadata or {}
    if set(metadata) != _STATIC_METADATA_KEYS:
        raise RuntimeError("landmark Arrow metadata is not allow-listed")
    expected = canonical_sha256(
        _schema_identity(fields),
        domain="parquet-arrow-schema/1",
    ).encode("ascii")
    if metadata[_SCHEMA_SHA256_KEY] != expected:
        raise RuntimeError("landmark Arrow schema digest is invalid")


_verify_schema()


def landmark_parquet_schema_snapshot() -> dict[str, object]:
    """Return the reviewable, JSON-native authoritative Arrow schema."""

    metadata = LANDMARK_PARQUET_SCHEMA.metadata or {}
    return {
        **_schema_identity(tuple(LANDMARK_PARQUET_SCHEMA)),
        "allowed_schema_metadata": {
            _TABLE_KIND_KEY.decode(): metadata[_TABLE_KIND_KEY].decode(),
            _SCHEMA_VERSION_KEY.decode(): metadata[_SCHEMA_VERSION_KEY].decode(),
            _SCHEMA_SHA256_KEY.decode(): metadata[_SCHEMA_SHA256_KEY].decode(),
            _CONTENT_SHA256_KEY.decode(): "sha256:<64 lowercase hex characters>",
        },
        "writer_generated_file_metadata": {
            _ARROW_SCHEMA_KEY.decode(): "authoritative Arrow schema encoding"
        },
    }


def _schema_with_content(content_sha256: str) -> pa.Schema:
    if _SHA256_PATTERN.fullmatch(content_sha256) is None:
        raise LandmarkParquetError("landmark semantic digest is invalid")
    metadata = dict(LANDMARK_PARQUET_SCHEMA.metadata or {})
    metadata[_CONTENT_SHA256_KEY] = content_sha256.encode("ascii")
    return LANDMARK_PARQUET_SCHEMA.with_metadata(metadata)


def _parquet_bytes(document: LandmarkFramesTableV1) -> tuple[bytes, str]:
    content_sha256 = landmark_frames_table_digest(document)
    schema = _schema_with_content(content_sha256)
    rows = [_physical_row(row.model_dump(mode="json", round_trip=True)) for row in document.rows]
    try:
        table = pa.Table.from_pylist(rows, schema=schema)
        table.validate(full=True)
        sink = pa.BufferOutputStream()
        pq.write_table(
            table,
            sink,
            compression="zstd",
            compression_level=9,
            data_page_version="2.0",
            version="2.6",
            use_dictionary=True,
            write_batch_size=1_024,
            row_group_size=_ROW_GROUP_SIZE,
            write_page_checksum=True,
            store_schema=True,
        )
        captured = sink.getvalue().to_pybytes()
    except (ArrowException, OSError, ValueError) as error:
        raise LandmarkParquetError("landmark rows could not be encoded as Parquet") from error
    return captured, content_sha256


def _physical_row(row: dict[str, object]) -> dict[str, object]:
    """Encode absent landmark lists without nullable fixed-size-list ambiguity.

    Parquet cannot portably round-trip a null fixed-size list through every
    supported PyArrow build. The explicit ``present`` mask remains authoritative,
    while an absent hand is represented physically by exactly 21 null elements.
    """

    hands = cast(list[dict[str, object]], row["hands"])
    for hand in hands:
        if hand["present"] is False:
            hand["image_landmarks"] = [None] * 21
            hand["world_landmarks"] = [None] * 21
    return row


def _semantic_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Restore contract-level nulls and reject data hidden behind absence masks."""

    for row in rows:
        hands = cast(list[dict[str, object]], row["hands"])
        for hand in hands:
            if hand["present"] is False:
                for field_name in ("image_landmarks", "world_landmarks"):
                    points = hand[field_name]
                    if (
                        not isinstance(points, list)
                        or len(points) != 21
                        or any(point is not None for point in points)
                    ):
                        raise LandmarkParquetError(
                            "absent landmark storage does not match its explicit mask"
                        )
                    hand[field_name] = None
    return rows


def _persist(path: Path, captured: bytes) -> None:
    temporary: Path | None = None
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
            temporary = Path(stream.name)
            stream.write(captured)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            with suppress(OSError):
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    except OSError as error:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()
        raise LandmarkParquetError("landmark Parquet bytes could not be persisted") from error


def write_landmark_frames(
    document: LandmarkFramesTableV1,
    destination: str | Path,
) -> LandmarkParquetResult:
    """Validate and atomically write one deterministic landmark frame table."""

    try:
        checked = validate_landmark_frames_table(document)
    except (TypeError, ValueError) as error:
        raise LandmarkParquetError("landmark frame table is invalid") from error
    captured, content_sha256 = _parquet_bytes(checked)
    path = Path(destination)
    _persist(path, captured)
    return LandmarkParquetResult(
        row_count=len(checked.rows),
        content_sha256=content_sha256,
        sha256=f"sha256:{hashlib.sha256(captured).hexdigest()}",
        size_bytes=len(captured),
        path=path,
    )


def _captured_bytes(path: Path, expected_size_bytes: int, expected_sha256: str) -> bytes:
    if type(expected_size_bytes) is not int or expected_size_bytes < 0:
        raise LandmarkParquetError("landmark Parquet size is invalid")
    if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise LandmarkParquetError("landmark Parquet digest is invalid")
    try:
        status = path.stat()
        if not stat.S_ISREG(status.st_mode) or status.st_size != expected_size_bytes:
            raise LandmarkParquetError("landmark Parquet bytes do not match their reference")
        captured = path.read_bytes()
    except LandmarkParquetError:
        raise
    except OSError as error:
        raise LandmarkParquetError("landmark Parquet bytes are unavailable") from error
    if len(captured) != expected_size_bytes:
        raise LandmarkParquetError("landmark Parquet bytes changed while being read")
    if f"sha256:{hashlib.sha256(captured).hexdigest()}" != expected_sha256:
        raise LandmarkParquetError("landmark Parquet bytes do not match their reference")
    return captured


def read_landmark_frames(
    source: str | Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_content_sha256: str,
    expected_row_count: int,
) -> LandmarkFramesTableV1:
    """Verify exact bytes, Arrow schema, semantic rows, and row count."""

    if _SHA256_PATTERN.fullmatch(expected_content_sha256) is None:
        raise LandmarkParquetError("landmark semantic digest is invalid")
    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise LandmarkParquetError("landmark row count is invalid")
    captured = _captured_bytes(Path(source), expected_size_bytes, expected_sha256)
    expected_schema = _schema_with_content(expected_content_sha256)
    try:
        parquet_file = pq.ParquetFile(
            pa.BufferReader(captured),
            page_checksum_verification=True,
            thrift_string_size_limit=_THRIFT_STRING_SIZE_LIMIT,
            thrift_container_size_limit=_THRIFT_CONTAINER_SIZE_LIMIT,
            arrow_extensions_enabled=False,
        )
        if parquet_file.metadata.num_rows != expected_row_count:
            raise LandmarkParquetError("landmark Parquet row count does not match")
        if set(parquet_file.metadata.metadata or {}) != _FILE_METADATA_KEYS:
            raise LandmarkParquetError("landmark Parquet metadata is not allow-listed")
        if not parquet_file.schema_arrow.equals(expected_schema, check_metadata=True):
            raise LandmarkParquetError("landmark Parquet Arrow schema is incompatible")
        table = parquet_file.read()
        table.validate(full=True)
        if not table.schema.equals(expected_schema, check_metadata=True):
            raise LandmarkParquetError("decoded landmark Arrow schema changed")
        rows = cast(list[dict[str, object]], table.to_pylist())
        checked = validate_landmark_frames_table(
            {"schema_version": "landmark-frames-table/1", "rows": _semantic_rows(rows)}
        )
    except LandmarkParquetError:
        raise
    except (ArrowException, OSError, TypeError, ValueError) as error:
        raise LandmarkParquetError("landmark Parquet bytes are invalid") from error
    if len(checked.rows) != expected_row_count:
        raise LandmarkParquetError("decoded landmark row count is inconsistent")
    if landmark_frames_table_digest(checked) != expected_content_sha256:
        raise LandmarkParquetError("landmark Parquet semantic digest does not match")
    return checked


__all__ = [
    "LANDMARK_PARQUET_SCHEMA",
    "LANDMARK_PARQUET_SCHEMAS",
    "LandmarkParquetError",
    "LandmarkParquetResult",
    "landmark_parquet_schema_snapshot",
    "read_landmark_frames",
    "write_landmark_frames",
]
