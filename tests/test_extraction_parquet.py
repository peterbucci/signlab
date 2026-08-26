from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    LandmarkFramesTableV1,
    landmark_frames_table_digest,
)
from signlab.extraction.parquet import (
    LANDMARK_PARQUET_SCHEMA,
    LandmarkParquetError,
    landmark_parquet_schema_snapshot,
    read_landmark_frames,
    write_landmark_frames,
)

RECORDING_ID = "recording_00000000000000000000000000000001"


def _point(*, pose: bool = False) -> dict[str, object]:
    return {
        "x": 0.25,
        "y": 0.5,
        "z": -0.125,
        "visibility": 0.9 if pose else None,
        "presence": 0.8 if pose else None,
    }


def _hand(slot_id: str, *, present: bool) -> dict[str, object]:
    points = [_point() for _ in range(21)] if present else None
    return {
        "slot_id": slot_id,
        "present": present,
        "detector_index": 0 if present else None,
        "tracking_id": slot_id if present else None,
        "handedness": "right" if present else None,
        "handedness_confidence": 0.95 if present else None,
        "image_landmarks": points,
        "world_landmarks": points,
    }


def _anchors(*, present: bool) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "present": present,
            "image_point": _point(pose=True) if present else None,
            "world_point": _point(pose=True) if present else None,
        }
        for name in BODY_ANCHOR_NAMES
    ]


def _frame(index: int, *, invalid: bool = False) -> dict[str, object]:
    relative_us = index * 33_333
    return {
        "schema_version": "landmark-frame/1",
        "source_recording_id": RECORDING_ID,
        "frame_index": index,
        "source_pts": 100 + index,
        "source_time_base_numerator": 1,
        "source_time_base_denominator": 30,
        "relative_timestamp_us": relative_us,
        "task_timestamp_ms": relative_us // 1_000,
        "invalid": invalid,
        "invalid_reason": "task_inference_failed" if invalid else None,
        "hands": [
            _hand("hand_0", present=not invalid),
            _hand("hand_1", present=False),
        ],
        "body_anchors": _anchors(present=not invalid),
        "observed_hand_count": 0 if invalid else 1,
        "observed_body_anchor_count": 0 if invalid else 6,
    }


def _table() -> LandmarkFramesTableV1:
    return LandmarkFramesTableV1.model_validate_json(
        json.dumps(
            {
                "schema_version": "landmark-frames-table/1",
                "rows": [_frame(0), _frame(1, invalid=True)],
            }
        ),
        strict=True,
    )


def _read(path: Path, result: Any) -> LandmarkFramesTableV1:
    return read_landmark_frames(
        path,
        expected_size_bytes=result.size_bytes,
        expected_sha256=result.sha256,
        expected_content_sha256=result.content_sha256,
        expected_row_count=result.row_count,
    )


def _read_rewritten(
    path: Path,
    *,
    expected_content_sha256: str,
    expected_row_count: int,
) -> LandmarkFramesTableV1:
    captured = path.read_bytes()
    return read_landmark_frames(
        path,
        expected_size_bytes=len(captured),
        expected_sha256=f"sha256:{hashlib.sha256(captured).hexdigest()}",
        expected_content_sha256=expected_content_sha256,
        expected_row_count=expected_row_count,
    )


def test_landmark_parquet_is_deterministic_and_round_trips_explicit_masks(
    tmp_path: Path,
) -> None:
    table = _table()
    first = write_landmark_frames(table, tmp_path / "first.parquet")
    second = write_landmark_frames(table, tmp_path / "second.parquet")

    assert first.sha256 == second.sha256
    assert first.content_sha256 == landmark_frames_table_digest(table)
    assert first.path.read_bytes() == second.path.read_bytes()
    assert _read(first.path, first) == table
    assert _read(first.path, first).rows[1].hands[0].image_landmarks is None


@pytest.mark.golden
def test_landmark_parquet_canonical_fixture_has_cross_platform_identity(tmp_path: Path) -> None:
    result = write_landmark_frames(_table(), tmp_path / "canonical.parquet")

    assert result.content_sha256 == (
        "sha256:9d46f221370585d0722d5f7de9536f3d1a90fb5b837f37bb94c01ab8f4574672"
    )
    assert result.sha256 == (
        "sha256:d1d896bc62793eeadc67b1e7a0fb110a8cc12ab2cdaeffcd452ab00334947f07"
    )
    assert result.size_bytes == 19_134


def test_landmark_parquet_schema_is_stable_and_reviewable(tmp_path: Path) -> None:
    result = write_landmark_frames(_table(), tmp_path / "frames.parquet")
    parquet = pq.ParquetFile(result.path)
    snapshot = landmark_parquet_schema_snapshot()

    assert parquet.schema_arrow.equals(
        LANDMARK_PARQUET_SCHEMA.with_metadata(parquet.schema_arrow.metadata),
        check_metadata=True,
    )
    assert snapshot["format"] == "arrow-schema-snapshot/1"
    assert snapshot["schema_version"] == "landmark-frames-table/1"
    assert len(snapshot["fields"]) == 14  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "content_sha256", "row_count"])
def test_landmark_parquet_rejects_reference_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    result = write_landmark_frames(_table(), tmp_path / "frames.parquet")
    arguments: dict[str, object] = {
        "expected_size_bytes": result.size_bytes,
        "expected_sha256": result.sha256,
        "expected_content_sha256": result.content_sha256,
        "expected_row_count": result.row_count,
    }
    arguments[f"expected_{field}"] = (
        "sha256:" + "0" * 64 if "sha256" in field else result.row_count + 1
    )

    with pytest.raises(LandmarkParquetError):
        read_landmark_frames(result.path, **arguments)  # type: ignore[arg-type]


def test_landmark_parquet_rejects_tampered_bytes(tmp_path: Path) -> None:
    result = write_landmark_frames(_table(), tmp_path / "frames.parquet")
    captured = bytearray(result.path.read_bytes())
    captured[len(captured) // 2] ^= 1
    result.path.write_bytes(captured)

    with pytest.raises(LandmarkParquetError, match="bytes"):
        _read(result.path, result)


def test_landmark_parquet_rejects_data_hidden_behind_absence_mask() -> None:
    table = _table()
    original = copy.deepcopy(table.model_dump(mode="json", round_trip=True))
    original["rows"][1]["hands"][0]["image_landmarks"] = [_point() for _ in range(21)]
    original["rows"][1]["hands"][0]["world_landmarks"] = [_point() for _ in range(21)]

    # The semantic contract blocks this before storage; this assertion protects
    # the same fail-closed boundary independently of Arrow's nullable behavior.
    with pytest.raises(ValueError, match="absent hand"):
        LandmarkFramesTableV1.model_validate_json(json.dumps(original), strict=True)


def test_landmark_parquet_write_rejects_invalid_document_and_destination(
    tmp_path: Path,
) -> None:
    with pytest.raises(LandmarkParquetError, match="table is invalid"):
        write_landmark_frames({"schema_version": "wrong"}, tmp_path / "invalid.parquet")  # type: ignore[arg-type]

    destination = tmp_path / "directory.parquet"
    destination.mkdir()
    with pytest.raises(LandmarkParquetError, match="could not be persisted"):
        write_landmark_frames(_table(), destination)


def test_landmark_parquet_write_wraps_encoder_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("private encoder detail")

    monkeypatch.setattr("signlab.extraction.parquet.pq.write_table", fail_write)

    with pytest.raises(LandmarkParquetError, match="could not be encoded") as captured:
        write_landmark_frames(_table(), tmp_path / "frames.parquet")
    assert "private encoder detail" not in str(captured.value)


def test_landmark_parquet_atomic_replace_failure_preserves_destination_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "frames.parquet"
    destination.write_bytes(b"previous bytes")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("private replace detail")

    monkeypatch.setattr("signlab.extraction.parquet.os.replace", fail_replace)

    with pytest.raises(LandmarkParquetError, match="could not be persisted"):
        write_landmark_frames(_table(), destination)

    assert destination.read_bytes() == b"previous bytes"
    assert list(tmp_path.glob(".frames.parquet.*.tmp")) == []


@pytest.mark.parametrize(
    ("expected_size", "expected_sha256", "message"),
    [
        (True, "sha256:" + "0" * 64, "size is invalid"),
        (0, "not-a-digest", "digest is invalid"),
    ],
)
def test_landmark_parquet_rejects_malformed_byte_reference_types(
    tmp_path: Path,
    expected_size: object,
    expected_sha256: object,
    message: str,
) -> None:
    source = tmp_path / "frames.parquet"
    source.write_bytes(b"")

    with pytest.raises(LandmarkParquetError, match=message):
        read_landmark_frames(
            source,
            expected_size_bytes=expected_size,  # type: ignore[arg-type]
            expected_sha256=expected_sha256,  # type: ignore[arg-type]
            expected_content_sha256="sha256:" + "0" * 64,
            expected_row_count=1,
        )


@pytest.mark.parametrize(
    ("content_sha256", "row_count", "message"),
    [
        ("not-a-digest", 1, "semantic digest is invalid"),
        ("sha256:" + "0" * 64, True, "row count is invalid"),
        ("sha256:" + "0" * 64, 0, "row count is invalid"),
    ],
)
def test_landmark_parquet_rejects_malformed_semantic_reference_types(
    tmp_path: Path,
    content_sha256: str,
    row_count: object,
    message: str,
) -> None:
    with pytest.raises(LandmarkParquetError, match=message):
        read_landmark_frames(
            tmp_path / "missing.parquet",
            expected_size_bytes=0,
            expected_sha256="sha256:" + "0" * 64,
            expected_content_sha256=content_sha256,
            expected_row_count=row_count,  # type: ignore[arg-type]
        )


def test_landmark_parquet_rejects_missing_source_and_invalid_parquet_with_exact_hash(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(LandmarkParquetError, match="bytes are unavailable"):
        read_landmark_frames(
            missing,
            expected_size_bytes=1,
            expected_sha256="sha256:" + "0" * 64,
            expected_content_sha256="sha256:" + "0" * 64,
            expected_row_count=1,
        )

    invalid = tmp_path / "invalid.parquet"
    invalid.write_bytes(b"not parquet")
    with pytest.raises(LandmarkParquetError, match="bytes are invalid"):
        _read_rewritten(
            invalid,
            expected_content_sha256="sha256:" + "0" * 64,
            expected_row_count=1,
        )


def test_landmark_parquet_rejects_file_changed_between_stat_and_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = write_landmark_frames(_table(), tmp_path / "frames.parquet")
    captured = result.path.read_bytes()
    original_read_bytes = Path.read_bytes

    def shortened_read(path: Path) -> bytes:
        return captured[:-1] if path == result.path else original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", shortened_read)
    with pytest.raises(LandmarkParquetError, match="changed while being read"):
        _read(result.path, result)


def test_landmark_parquet_rejects_physical_row_count_metadata_and_schema_changes(
    tmp_path: Path,
) -> None:
    result = write_landmark_frames(_table(), tmp_path / "source.parquet")
    table = pq.read_table(result.path)

    short_path = tmp_path / "short.parquet"
    pq.write_table(table.slice(0, 1), short_path)
    with pytest.raises(LandmarkParquetError, match="row count does not match"):
        _read_rewritten(
            short_path,
            expected_content_sha256=result.content_sha256,
            expected_row_count=2,
        )

    metadata_path = tmp_path / "metadata.parquet"
    metadata = dict(table.schema.metadata or {})
    metadata[b"unexpected"] = b"not-allowed"
    pq.write_table(table.replace_schema_metadata(metadata), metadata_path)
    with pytest.raises(LandmarkParquetError, match="metadata is not allow-listed"):
        _read_rewritten(
            metadata_path,
            expected_content_sha256=result.content_sha256,
            expected_row_count=2,
        )

    schema_path = tmp_path / "schema.parquet"
    fields = list(table.schema)
    field_index = table.schema.get_field_index("observed_hand_count")
    original_field = fields[field_index]
    fields[field_index] = pa.field(
        original_field.name,
        pa.int16(),
        nullable=original_field.nullable,
        metadata=original_field.metadata,
    )
    changed_schema = pa.schema(fields, metadata=table.schema.metadata)
    pq.write_table(table.cast(changed_schema), schema_path)
    with pytest.raises(LandmarkParquetError, match="schema is incompatible"):
        _read_rewritten(
            schema_path,
            expected_content_sha256=result.content_sha256,
            expected_row_count=2,
        )


def test_landmark_parquet_rejects_invalid_semantic_rows_and_digest_changes(
    tmp_path: Path,
) -> None:
    result = write_landmark_frames(_table(), tmp_path / "source.parquet")
    table = pq.read_table(result.path)

    invalid_rows = table.to_pylist()
    invalid_rows[1]["frame_index"] = 9
    invalid_table = pa.Table.from_pylist(invalid_rows, schema=table.schema)
    invalid_path = tmp_path / "invalid-row.parquet"
    pq.write_table(invalid_table, invalid_path)
    with pytest.raises(LandmarkParquetError, match="bytes are invalid"):
        _read_rewritten(
            invalid_path,
            expected_content_sha256=result.content_sha256,
            expected_row_count=2,
        )

    changed_rows = table.to_pylist()
    changed_rows[0]["hands"][0]["image_landmarks"][0]["x"] = 0.75
    changed_table = pa.Table.from_pylist(changed_rows, schema=table.schema)
    changed_path = tmp_path / "changed-semantic.parquet"
    pq.write_table(changed_table, changed_path)
    with pytest.raises(LandmarkParquetError, match="semantic digest does not match"):
        _read_rewritten(
            changed_path,
            expected_content_sha256=result.content_sha256,
            expected_row_count=2,
        )


def test_landmark_parquet_rejects_observations_hidden_behind_physical_absence_mask(
    tmp_path: Path,
) -> None:
    result = write_landmark_frames(_table(), tmp_path / "source.parquet")
    table = pq.read_table(result.path)
    rows = table.to_pylist()
    rows[1]["hands"][0]["image_landmarks"][0] = _point()
    hidden = pa.Table.from_pylist(rows, schema=table.schema)
    hidden_path = tmp_path / "hidden.parquet"
    pq.write_table(hidden, hidden_path)

    with pytest.raises(LandmarkParquetError, match="explicit mask"):
        _read_rewritten(
            hidden_path,
            expected_content_sha256=result.content_sha256,
            expected_row_count=2,
        )
