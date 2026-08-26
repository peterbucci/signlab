"""Determinism and fail-closed tests for the dataset Parquet boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from signlab.contracts.core import (
    ArtifactRefV1,
    ArtifactUriLocatorV1,
    WorkspaceRelativeLocatorV1,
)
from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    DatasetTable,
    DatasetTableRefV1,
    DerivedArtifactRowV1,
    SessionsTableV1,
    TableName,
    dataset_table_digest,
)
from signlab.datasets.parquet import (
    DATASET_PARQUET_SCHEMAS,
    DATASET_PARQUET_SCHEMAS_BY_TABLE,
    DatasetParquetError,
    build_dataset_table,
    parquet_schema_snapshot,
    read_dataset_table,
    read_parquet_table,
    resolve_workspace_locator,
    semantic_table_sha256,
    write_dataset_table,
    write_parquet_table,
)
from signlab.governance.resources import build_example_recording_grant

_PARTICIPANT_ID = "participant_00000000000000000000000000000001"
_SESSION_ID = "session_00000000000000000000000000000001"
_DEVICE_ID = "device_00000000000000000000000000000001"
_RECORDING_ID = "recording_00000000000000000000000000000031"
_CLIP_ID = "clip_00000000000000000000000000000001"
_ANNOTATION_ID = "annotation_00000000000000000000000000000001"
_DERIVED_ID = "derived_artifact_00000000000000000000000000000001"
_SAMPLE_ID = "sample_00000000000000000000000000000001"


def _sha(number: int) -> str:
    return f"sha256:{number:064x}"


def _artifact(
    artifact_id: str,
    role: str,
    path: str,
    *,
    media_type: str = "application/octet-stream",
    number: int = 1,
) -> ArtifactRefV1:
    sha256 = _sha(number)
    if role == "dataset_table":
        table_name = path.rsplit("/", maxsplit=1)[-1].removesuffix(".parquet")
        canonical_path = f"tables/{table_name}.parquet"
    else:
        digest = sha256.removeprefix("sha256:")
        canonical_path = f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{artifact_id}"
    return ArtifactRefV1.model_validate(
        {
            "schema_version": "artifact-reference/1",
            "artifact_id": artifact_id,
            "role": role,
            "media_type": media_type,
            "sha256": sha256,
            "size_bytes": 100 + number,
            "locator": {"kind": "workspace_relative", "path": canonical_path},
        },
        strict=True,
    )


def _rows() -> dict[TableName, list[dict[str, object]]]:
    grant = build_example_recording_grant()
    return {
        "participants": [
            {"participant_id": _PARTICIPANT_ID, "handedness": "right"},
        ],
        "sessions": [
            {
                "session_id": _SESSION_ID,
                "participant_id": _PARTICIPANT_ID,
                "device_id": _DEVICE_ID,
                "started_at": "2026-08-26T12:00:00Z",
                "finished_at": "2026-08-26T12:30:00Z",
                "capture_mode": "continuous",
                "capture_software_version": "1.0.0",
                "camera_facing": "front",
                "frame_width_px": 1920,
                "frame_height_px": 1080,
                "frame_rate_numerator": 30_000,
                "frame_rate_denominator": 1001,
                "rotation_degrees": 0,
                "mirror_state": "mirrored",
            }
        ],
        "recordings": [
            {
                "recording_id": _RECORDING_ID,
                "participant_id": _PARTICIPANT_ID,
                "session_id": _SESSION_ID,
                "device_id": _DEVICE_ID,
                "captured_at": "2026-08-26T12:10:00Z",
                "duration_us": 5_000_000,
                "handedness": "right",
                "mirror_state": "mirrored",
                "rotation_degrees": 0,
                "audio_present": False,
                "media": _artifact(
                    _RECORDING_ID,
                    "raw_recording",
                    f"data/raw/{_RECORDING_ID}.mp4",
                    media_type="video/mp4",
                    number=2,
                ),
                "consent_grant": grant,
            }
        ],
        "clips": [
            {
                "clip_id": _CLIP_ID,
                "participant_id": _PARTICIPANT_ID,
                "session_id": _SESSION_ID,
                "source_recording_id": _RECORDING_ID,
                "interval": {
                    "schema_version": "media-interval/1",
                    "start_us": 100_000,
                    "end_us": 2_000_000,
                },
                "handedness": "right",
                "mirror_state": "mirrored",
                "artifact": None,
            }
        ],
        "annotations": [
            {
                "annotation_id": _ANNOTATION_ID,
                "participant_id": _PARTICIPANT_ID,
                "session_id": _SESSION_ID,
                "source_recording_id": _RECORDING_ID,
                "clip_id": _CLIP_ID,
                "interval": {
                    "schema_version": "media-interval/1",
                    "start_us": 200_000,
                    "end_us": 1_500_000,
                },
                "disposition": "class_label",
                "label_id": "other",
                "other_kind": "oov_gesture",
                "reason_code": None,
                "review_status": "reviewed",
                "eligible_for_training": True,
            }
        ],
        "derived_artifacts": [
            {
                "derived_artifact_id": _DERIVED_ID,
                "derivation_kind": "window",
                "parent_artifact_ids": [_CLIP_ID],
                "participant_id": _PARTICIPANT_ID,
                "session_id": _SESSION_ID,
                "source_recording_id": _RECORDING_ID,
                "clip_id": _CLIP_ID,
                "annotation_id": _ANNOTATION_ID,
                "sample_id": _SAMPLE_ID,
                "label_id": "other",
                "split_id": "split_primary",
                "partition": "train",
                "handedness": "right",
                "mirror_state": "mirrored",
                "operation_id": "window_landmarks",
                "operation_version": "1.0.0",
                "artifact": _artifact(
                    _SAMPLE_ID,
                    "sample_data",
                    "data/samples/sample.parquet",
                    media_type="application/vnd.apache.parquet",
                    number=3,
                ),
            }
        ],
    }


def _field_ids(fields: object) -> list[int]:
    result: list[int] = []
    for field in fields:  # type: ignore[attr-defined]
        typed_field = cast(pa.Field, field)
        result.append(int((typed_field.metadata or {})[b"PARQUET:field_id"]))
        if pa.types.is_struct(typed_field.type):
            result.extend(_field_ids(typed_field.type))
        elif pa.types.is_list(typed_field.type):
            result.extend(_field_ids((typed_field.type.value_field,)))
    return result


def _assert_same_contract(left: DatasetTable, right: DatasetTable) -> None:
    assert left.model_dump(mode="json", round_trip=True) == right.model_dump(
        mode="json", round_trip=True
    )


def _write_untrusted_parquet(
    table_name: TableName,
    rows: list[dict[str, object]],
    path: Path,
    *,
    claimed_content_sha256: str,
    write_page_checksum: bool = True,
) -> tuple[int, str]:
    """Build adversarial bytes without going through the validating writer."""

    schema = DATASET_PARQUET_SCHEMAS_BY_TABLE[table_name]
    schema = schema.with_metadata(
        {
            **(schema.metadata or {}),
            b"signlab:content_sha256": claimed_content_sha256.encode(),
        }
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        version="2.6",
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_page_checksum=write_page_checksum,
        store_schema=True,
    )
    captured = path.read_bytes()
    return len(captured), f"sha256:{hashlib.sha256(captured).hexdigest()}"


def test_authoritative_registry_has_exact_six_versioned_schemas() -> None:
    assert set(DATASET_PARQUET_SCHEMAS) == set(DATASET_TABLE_SCHEMA_VERSIONS.values())
    assert set(DATASET_PARQUET_SCHEMAS_BY_TABLE) == set(DATASET_TABLE_SCHEMA_VERSIONS)

    for table_name, schema_version in DATASET_TABLE_SCHEMA_VERSIONS.items():
        schema = DATASET_PARQUET_SCHEMAS[schema_version]
        assert schema is DATASET_PARQUET_SCHEMAS_BY_TABLE[table_name]
        assert set(schema.metadata or {}) == {
            b"signlab:table_kind",
            b"signlab:schema_version",
            b"signlab:schema_sha256",
        }
        assert schema.metadata[b"signlab:table_kind"] == table_name.encode()
        assert schema.metadata[b"signlab:schema_version"] == schema_version.encode()
        field_ids = _field_ids(schema)
        assert all(field_id > 0 for field_id in field_ids)
        assert len(field_ids) == len(set(field_ids))
        assert all(set(field.metadata or {}) == {b"PARQUET:field_id"} for field in schema)


def test_arrow_top_level_field_order_types_and_nullability_are_explicit() -> None:
    session = DATASET_PARQUET_SCHEMAS_BY_TABLE["sessions"]
    assert session.names == [
        "session_id",
        "participant_id",
        "device_id",
        "started_at",
        "finished_at",
        "capture_mode",
        "capture_software_version",
        "camera_facing",
        "frame_width_px",
        "frame_height_px",
        "frame_rate_numerator",
        "frame_rate_denominator",
        "rotation_degrees",
        "mirror_state",
    ]
    assert session.field("started_at").type == pa.timestamp("us", tz="UTC")
    assert session.field("rotation_degrees").type == pa.int16()
    assert all(not field.nullable for field in session)

    annotation = DATASET_PARQUET_SCHEMAS_BY_TABLE["annotations"]
    assert annotation.names == list(_rows()["annotations"][0])
    assert {field.name for field in annotation if field.nullable} == {
        "clip_id",
        "label_id",
        "other_kind",
        "reason_code",
    }
    assert annotation.field("interval").type.field("start_us").type == pa.int64()


def test_schema_snapshot_is_json_native_and_rejects_unknown_versions() -> None:
    snapshot = parquet_schema_snapshot("derived-artifacts-table/1")
    assert snapshot["table_name"] == "derived_artifacts"
    assert snapshot["schema_version"] == "derived-artifacts-table/1"
    assert cast(list[dict[str, object]], snapshot["fields"])[0] == {
        "name": "derived_artifact_id",
        "type": "utf8",
        "nullable": False,
        "field_id": 1,
    }
    metadata = cast(dict[str, str], snapshot["allowed_schema_metadata"])
    assert metadata["signlab:content_sha256"] == "sha256:<64 lowercase hex characters>"

    with pytest.raises(DatasetParquetError, match="unsupported"):
        parquet_schema_snapshot("participants-table/999")


def test_build_rows_sorts_ids_and_rejects_duplicates_and_extras() -> None:
    first = {"participant_id": "participant_" + "1" * 32, "handedness": "left"}
    second = {"participant_id": "participant_" + "2" * 32, "handedness": "right"}
    checked = build_dataset_table("participants", [second, first])
    assert [row.participant_id for row in checked.rows] == [
        first["participant_id"],
        second["participant_id"],
    ]

    with pytest.raises(DatasetParquetError, match="duplicate participant_id"):
        build_dataset_table("participants", [first, first])
    with pytest.raises(DatasetParquetError, match="invalid participants row"):
        build_dataset_table("participants", [{**first, "private_name": "must fail"}])
    with pytest.raises(DatasetParquetError, match="models or mappings"):
        build_dataset_table("participants", cast(list[Mapping[str, object]], ["bad"]))
    with pytest.raises(DatasetParquetError, match="invalid participants row"):
        build_dataset_table(
            "participants",
            cast(
                list[Mapping[str, object]],
                [{1: _PARTICIPANT_ID, "handedness": "right"}],
            ),
        )
    with pytest.raises(DatasetParquetError, match="unsupported dataset table name"):
        build_dataset_table(cast(TableName, "unknown"), [])


def test_base_model_and_json_mapping_rows_share_the_strict_json_boundary() -> None:
    mapping = _rows()["derived_artifacts"][0]
    from_mapping = build_dataset_table("derived_artifacts", [mapping])
    model = cast(DerivedArtifactRowV1, from_mapping.rows[0])
    from_model = build_dataset_table("derived_artifacts", [model])
    _assert_same_contract(from_mapping, from_model)
    assert model.parent_artifact_ids == (_CLIP_ID,)


def test_artifact_uri_locator_round_trips_without_union_placeholder_fields(
    tmp_path: Path,
) -> None:
    mapping = _rows()["derived_artifacts"][0]
    artifact = cast(ArtifactRefV1, mapping["artifact"])
    mapping["artifact"] = artifact.model_copy(
        update={
            "locator": ArtifactUriLocatorV1(
                kind="artifact_uri",
                uri=(
                    "signlab://objects/sha256/"
                    f"p-{artifact.sha256[7:9]}/{artifact.sha256.replace(':', '-')}/"
                    f"{artifact.artifact_id}"
                ),
            )
        }
    )
    expected = build_dataset_table("derived_artifacts", [mapping])
    result = write_dataset_table(expected, tmp_path / "derived.parquet")

    actual = read_parquet_table(
        "derived_artifacts",
        result.path,
        expected_size_bytes=result.size_bytes,
        expected_sha256=result.sha256,
        expected_content_sha256=result.content_sha256,
    )
    _assert_same_contract(actual, expected)


@pytest.mark.parametrize("table_name", tuple(DATASET_TABLE_SCHEMA_VERSIONS))
def test_every_table_round_trips_through_strict_contracts(
    table_name: TableName,
    tmp_path: Path,
) -> None:
    expected = build_dataset_table(table_name, _rows()[table_name])
    result = write_dataset_table(expected, tmp_path / f"{table_name}.parquet")
    actual = read_parquet_table(
        table_name,
        result.path,
        expected_size_bytes=result.size_bytes,
        expected_sha256=result.sha256,
        expected_content_sha256=result.content_sha256,
        expected_row_count=result.row_count,
    )

    _assert_same_contract(actual, expected)
    assert semantic_table_sha256(table_name, _rows()[table_name]) == result.content_sha256


def test_early_utc_year_round_trips_with_portable_zero_padding(tmp_path: Path) -> None:
    rows = _rows()["sessions"]
    rows[0] = {
        **rows[0],
        "started_at": "0001-01-01T00:00:00Z",
        "finished_at": "0001-01-01T00:00:01Z",
    }
    expected = build_dataset_table("sessions", rows)
    result = write_dataset_table(expected, tmp_path / "early-year.parquet")

    actual = read_parquet_table(
        "sessions",
        result.path,
        expected_size_bytes=result.size_bytes,
        expected_sha256=result.sha256,
        expected_content_sha256=result.content_sha256,
        expected_row_count=1,
    )

    _assert_same_contract(actual, expected)
    assert cast(SessionsTableV1, actual).rows[0].started_at == "0001-01-01T00:00:00Z"


def test_writer_is_byte_deterministic_and_records_pinned_profile(tmp_path: Path) -> None:
    rows = _rows()["participants"]
    first = write_parquet_table("participants", rows, tmp_path / "first.parquet")
    second = write_parquet_table("participants", rows, tmp_path / "second.parquet")

    assert first.sha256 == second.sha256
    assert first.content_sha256 == second.content_sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    parquet_file = pq.ParquetFile(first.path, page_checksum_verification=True)
    assert parquet_file.metadata.format_version == "2.6"
    assert parquet_file.metadata.row_group(0).column(0).compression == "ZSTD"
    assert parquet_file.metadata.row_group(0).sorting_columns[0].column_index == 0
    assert parquet_file.schema_arrow.metadata[b"signlab:content_sha256"] == (
        first.content_sha256.encode()
    )


def test_empty_optional_table_round_trips(tmp_path: Path) -> None:
    result = write_parquet_table("clips", [], tmp_path / "empty-clips.parquet")
    assert pq.read_metadata(result.path).num_rows == 0
    checked = read_parquet_table(
        "clips",
        result.path,
        expected_size_bytes=result.size_bytes,
        expected_sha256=result.sha256,
        expected_content_sha256=result.content_sha256,
        expected_row_count=0,
    )
    assert checked.rows == ()


def test_writer_rejects_an_invalid_wrapper_and_unwritable_destination(tmp_path: Path) -> None:
    with pytest.raises(DatasetParquetError, match="invalid dataset table contract"):
        write_dataset_table(cast(DatasetTable, object()), tmp_path / "invalid.parquet")

    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_bytes(b"occupied")
    checked = build_dataset_table("participants", _rows()["participants"])
    with pytest.raises(DatasetParquetError, match="could not persist"):
        write_dataset_table(checked, blocking_file / "table.parquet")

    special_destination = tmp_path / "special-destination.parquet"
    special_destination.mkdir()
    with pytest.raises(DatasetParquetError, match="could not persist"):
        write_dataset_table(checked, special_destination)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_size_bytes": 1}, "size"),
        ({"expected_sha256": _sha(99)}, "digest"),
        ({"expected_content_sha256": _sha(99)}, "schema or metadata"),
        ({"expected_row_count": 2}, "row count"),
    ],
)
def test_reader_rejects_mismatched_evidence(
    overrides: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    result = write_parquet_table(
        "participants", _rows()["participants"], tmp_path / "participants.parquet"
    )
    arguments: dict[str, object] = {
        "expected_size_bytes": result.size_bytes,
        "expected_sha256": result.sha256,
        "expected_content_sha256": result.content_sha256,
        "expected_row_count": result.row_count,
        **overrides,
    }
    with pytest.raises(DatasetParquetError, match=message):
        read_parquet_table("participants", result.path, **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        {"expected_sha256": "not-a-digest"},
        {"expected_content_sha256": "SHA256:" + "0" * 64},
        {"expected_size_bytes": -1},
        {"expected_size_bytes": True},
        {"expected_row_count": -1},
    ],
)
def test_reader_rejects_malformed_evidence(
    arguments: dict[str, object],
    tmp_path: Path,
) -> None:
    result = write_parquet_table(
        "participants", _rows()["participants"], tmp_path / "participants.parquet"
    )
    expected: dict[str, object] = {
        "expected_size_bytes": result.size_bytes,
        "expected_sha256": result.sha256,
        "expected_content_sha256": result.content_sha256,
        "expected_row_count": result.row_count,
        **arguments,
    }
    with pytest.raises(DatasetParquetError, match="expected"):
        read_parquet_table("participants", result.path, **expected)  # type: ignore[arg-type]

    with pytest.raises(DatasetParquetError, match="unsupported dataset table name"):
        read_parquet_table(
            cast(TableName, "unknown"),
            result.path,
            expected_size_bytes=result.size_bytes,
            expected_sha256=result.sha256,
            expected_content_sha256=result.content_sha256,
        )


def test_reader_sanitizes_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(DatasetParquetError, match="could not read"):
        read_parquet_table(
            "participants",
            tmp_path / "missing.parquet",
            expected_size_bytes=1,
            expected_sha256=_sha(1),
            expected_content_sha256=_sha(2),
        )


def test_reader_rejects_corrupt_captured_bytes_before_arrow(tmp_path: Path) -> None:
    result = write_parquet_table(
        "participants", _rows()["participants"], tmp_path / "participants.parquet"
    )
    corrupt = bytearray(result.path.read_bytes())
    corrupt[-1] ^= 1
    result.path.write_bytes(corrupt)

    with pytest.raises(DatasetParquetError, match="digest"):
        read_parquet_table(
            "participants",
            result.path,
            expected_size_bytes=len(corrupt),
            expected_sha256=result.sha256,
            expected_content_sha256=result.content_sha256,
        )


def test_reader_verifies_page_crc_even_when_outer_byte_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "participant_id": f"participant_{number:032x}",
            "handedness": "right",
        }
        for number in range(1, 100)
    ]
    result = write_parquet_table("participants", rows, tmp_path / "participants.parquet")
    metadata = pq.read_metadata(result.path)
    data_page_offset = metadata.row_group(0).column(0).data_page_offset
    corrupt = bytearray(result.path.read_bytes())
    corrupt[data_page_offset + 10] ^= 1
    result.path.write_bytes(corrupt)
    recomputed_outer_hash = f"sha256:{hashlib.sha256(corrupt).hexdigest()}"

    with pytest.raises(DatasetParquetError, match="bytes are invalid") as caught:
        read_parquet_table(
            "participants",
            result.path,
            expected_size_bytes=len(corrupt),
            expected_sha256=recomputed_outer_hash,
            expected_content_sha256=result.content_sha256,
        )

    assert caught.value.__cause__ is not None
    assert "CRC checksum verification failed" in str(caught.value.__cause__)


def test_reader_accepts_checksum_free_reencoding_with_exact_outer_evidence(
    tmp_path: Path,
) -> None:
    expected = build_dataset_table("participants", _rows()["participants"])
    content_sha256 = dataset_table_digest(expected)
    path = tmp_path / "checksum-free.parquet"
    size_bytes, byte_sha256 = _write_untrusted_parquet(
        "participants",
        _rows()["participants"],
        path,
        claimed_content_sha256=content_sha256,
        write_page_checksum=False,
    )

    actual = read_parquet_table(
        "participants",
        path,
        expected_size_bytes=size_bytes,
        expected_sha256=byte_sha256,
        expected_content_sha256=content_sha256,
        expected_row_count=1,
    )

    _assert_same_contract(actual, expected)


def test_reader_rejects_valid_parquet_with_schema_metadata_drift(tmp_path: Path) -> None:
    result = write_parquet_table(
        "participants", _rows()["participants"], tmp_path / "participants.parquet"
    )
    table = pq.read_table(result.path)
    drifted = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"unsafe:location": b"private-machine-value"}
    )
    pq.write_table(drifted, result.path, version="2.6", compression="zstd")
    captured = result.path.read_bytes()

    with pytest.raises(DatasetParquetError, match="metadata"):
        read_parquet_table(
            "participants",
            result.path,
            expected_size_bytes=len(captured),
            expected_sha256=f"sha256:{hashlib.sha256(captured).hexdigest()}",
            expected_content_sha256=result.content_sha256,
        )


def test_reader_reconstructs_rows_strictly_after_arrow_validation(tmp_path: Path) -> None:
    claimed_content = _sha(80)
    path = tmp_path / "invalid-row.parquet"
    size_bytes, byte_sha256 = _write_untrusted_parquet(
        "participants",
        [
            {
                "participant_id": _PARTICIPANT_ID,
                "handedness": "not-a-registered-value",
            }
        ],
        path,
        claimed_content_sha256=claimed_content,
    )

    with pytest.raises(DatasetParquetError, match="strict dataset contract"):
        read_parquet_table(
            "participants",
            path,
            expected_size_bytes=size_bytes,
            expected_sha256=byte_sha256,
            expected_content_sha256=claimed_content,
        )


def test_reader_rejects_valid_rows_with_a_false_semantic_digest(tmp_path: Path) -> None:
    claimed_content = _sha(81)
    path = tmp_path / "false-content.parquet"
    size_bytes, byte_sha256 = _write_untrusted_parquet(
        "participants",
        _rows()["participants"],
        path,
        claimed_content_sha256=claimed_content,
    )

    with pytest.raises(DatasetParquetError, match="semantic content digest"):
        read_parquet_table(
            "participants",
            path,
            expected_size_bytes=size_bytes,
            expected_sha256=byte_sha256,
            expected_content_sha256=claimed_content,
        )


def test_reader_rejects_timestamp_precision_outside_contract(tmp_path: Path) -> None:
    claimed_content = _sha(82)
    session = _rows()["sessions"][0]
    arrow_row = {
        **session,
        "started_at": datetime(2026, 8, 26, 12, 0, 0, 1, tzinfo=UTC),
        "finished_at": datetime(2026, 8, 26, 12, 30, tzinfo=UTC),
    }
    path = tmp_path / "subsecond.parquet"
    size_bytes, byte_sha256 = _write_untrusted_parquet(
        "sessions",
        [arrow_row],
        path,
        claimed_content_sha256=claimed_content,
    )

    with pytest.raises(DatasetParquetError, match="second precision"):
        read_parquet_table(
            "sessions",
            path,
            expected_size_bytes=size_bytes,
            expected_sha256=byte_sha256,
            expected_content_sha256=claimed_content,
        )


def test_workspace_locator_rejects_absolute_traversal_and_missing_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "data" / "table.parquet"
    artifact.parent.mkdir()
    artifact.write_bytes(b"bytes")

    assert resolve_workspace_locator(workspace, "data/table.parquet") == artifact.resolve()
    with pytest.raises(DatasetParquetError, match="invalid workspace-relative"):
        resolve_workspace_locator(workspace, "../outside.parquet")
    with pytest.raises(DatasetParquetError, match="invalid workspace-relative"):
        resolve_workspace_locator(workspace, str(artifact.resolve()))
    with pytest.raises(DatasetParquetError, match="does not exist"):
        resolve_workspace_locator(workspace, "data/missing.parquet")
    with pytest.raises(DatasetParquetError, match="unsupported artifact locator"):
        resolve_workspace_locator(workspace, cast(str, 42))


def test_workspace_locator_requires_an_existing_directory_root(tmp_path: Path) -> None:
    with pytest.raises(DatasetParquetError, match="root does not exist"):
        resolve_workspace_locator(tmp_path / "missing", "table.parquet")

    root_file = tmp_path / "root-file"
    root_file.write_bytes(b"not a directory")
    with pytest.raises(DatasetParquetError, match="root must be a directory"):
        resolve_workspace_locator(root_file, "table.parquet", must_exist=False)

    artifact_directory = tmp_path / "artifact-directory"
    artifact_directory.mkdir()
    with pytest.raises(DatasetParquetError, match="regular file"):
        resolve_workspace_locator(tmp_path, artifact_directory.name)
    with pytest.raises(DatasetParquetError, match="regular file"):
        resolve_workspace_locator(tmp_path, artifact_directory.name, must_exist=False)


def test_reader_rejects_a_nonregular_artifact_before_reading(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "participants.parquet"
    artifact_directory.mkdir()

    with pytest.raises(DatasetParquetError, match="regular file"):
        read_parquet_table(
            "participants",
            artifact_directory,
            expected_size_bytes=artifact_directory.stat().st_size,
            expected_sha256=_sha(1),
            expected_content_sha256=_sha(2),
            expected_row_count=1,
        )


@pytest.mark.parametrize(
    ("failing_resolution", "message"),
    [(1, "root does not exist"), (2, "artifact does not exist")],
    ids=["root-loop", "artifact-loop"],
)
def test_workspace_locator_sanitizes_symlink_resolution_loops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_resolution: int,
    message: str,
) -> None:
    workspace = tmp_path / "workspace"
    artifact = workspace / "data" / "table.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"bytes")
    real_resolve = Path.resolve
    resolution_count = 0

    def seeded_loop(path: Path, strict: bool = False) -> Path:
        nonlocal resolution_count
        resolution_count += 1
        if resolution_count == failing_resolution:
            raise RuntimeError("private symlink loop detail")
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", seeded_loop)

    with pytest.raises(DatasetParquetError, match=message) as caught:
        resolve_workspace_locator(workspace, "data/table.parquet")

    assert "private symlink loop detail" not in str(caught.value)


def test_workspace_locator_rejects_symlink_escape_when_supported(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "private.parquet").write_bytes(b"private")
    link = workspace / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not available for this account")

    with pytest.raises(DatasetParquetError, match="escapes"):
        resolve_workspace_locator(workspace, "linked/private.parquet")


def test_manifest_reference_resolves_and_verifies_exact_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    result = write_parquet_table(
        "participants",
        _rows()["participants"],
        workspace / "tables" / "participants.parquet",
    )
    artifact = _artifact(
        "participants_table",
        "dataset_table",
        "data/participants.parquet",
        media_type="application/vnd.apache.parquet",
    ).model_copy(
        update={"sha256": result.sha256, "size_bytes": result.size_bytes},
    )
    reference = DatasetTableRefV1(
        schema_version="dataset-table-reference/1",
        table_name="participants",
        table_schema_version="participants-table/1",
        row_count=result.row_count,
        content_sha256=result.content_sha256,
        artifact=artifact,
    )

    actual = read_dataset_table(reference, workspace)
    _assert_same_contract(actual, build_dataset_table("participants", _rows()["participants"]))


def test_manifest_reader_rejects_logical_uri_without_an_adapter(tmp_path: Path) -> None:
    result = write_parquet_table(
        "participants", _rows()["participants"], tmp_path / "participants.parquet"
    )
    artifact = _artifact(
        "participants_table",
        "dataset_table",
        "data/participants.parquet",
        media_type="application/vnd.apache.parquet",
    ).model_copy(
        update={
            "sha256": result.sha256,
            "size_bytes": result.size_bytes,
            "locator": {
                "kind": "artifact_uri",
                "uri": "signlab://tables/participants",
            },
        }
    )
    reference = DatasetTableRefV1(
        schema_version="dataset-table-reference/1",
        table_name="participants",
        table_schema_version="participants-table/1",
        row_count=result.row_count,
        content_sha256=result.content_sha256,
        artifact=artifact,
    )

    with pytest.raises(DatasetParquetError, match="storage adapter"):
        read_dataset_table(reference, tmp_path)


def test_manifest_reader_revalidates_a_copied_reference(tmp_path: Path) -> None:
    result = write_parquet_table(
        "participants", _rows()["participants"], tmp_path / "participants.parquet"
    )
    artifact = _artifact(
        "participants_table",
        "dataset_table",
        "participants.parquet",
        media_type="application/vnd.apache.parquet",
    ).model_copy(update={"sha256": result.sha256, "size_bytes": result.size_bytes})
    reference = DatasetTableRefV1(
        schema_version="dataset-table-reference/1",
        table_name="participants",
        table_schema_version="participants-table/1",
        row_count=result.row_count,
        content_sha256=result.content_sha256,
        artifact=artifact,
    ).model_copy(update={"row_count": -1})

    with pytest.raises(DatasetParquetError, match="invalid dataset table reference"):
        read_dataset_table(reference, tmp_path)


def test_workspace_locator_can_safely_resolve_a_future_write_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    destination = resolve_workspace_locator(
        workspace,
        WorkspaceRelativeLocatorV1(
            kind="workspace_relative",
            path="data/new-table.parquet",
        ),
        must_exist=False,
    )
    assert destination == (workspace / "data" / "new-table.parquet").resolve()
