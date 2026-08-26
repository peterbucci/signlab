from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from signlab.contracts.ingest import (
    capture_identifier_set_digest,
    validate_capture_identifier_set,
)
from signlab.datasets.capture import CaptureAllocationError, allocate_capture_identifiers


def _tokens(start: int = 1) -> Iterator[str]:
    index = start
    while True:
        yield f"{index:032x}"
        index += 1


def _factory(tokens: Iterator[str]) -> Callable[[int], str]:
    def token_hex(size: int) -> str:
        assert size == 16
        return next(tokens)

    return token_hex


def test_allocate_capture_identifiers_persists_and_reuses_exact_ids(tmp_path: Path) -> None:
    destination = tmp_path / "private" / "capture-ids.json"
    created = allocate_capture_identifiers(
        destination,
        token_hex=_factory(_tokens()),
    )

    assert created.status == "created"
    assert created.identifiers.collection_id == "collection_00000000000000000000000000000001"
    assert created.identifiers.visit_id == "visit_00000000000000000000000000000002"
    assert created.identifiers.store_id.startswith("store-")
    assert destination.read_bytes().endswith(b"\n")
    assert validate_capture_identifier_set(destination.read_bytes()) == created.identifiers
    assert created.identifiers_sha256 == capture_identifier_set_digest(created.identifiers)

    def must_not_allocate(_size: int) -> str:
        raise AssertionError("an idempotent read must not allocate new identifiers")

    repeated = allocate_capture_identifiers(destination, token_hex=must_not_allocate)
    assert repeated.status == "unchanged"
    assert repeated.identifiers == created.identifiers
    assert repeated.identifiers_sha256 == created.identifiers_sha256


def test_retry_reuses_workflow_ids_but_allocates_new_byte_identity(tmp_path: Path) -> None:
    original_path = tmp_path / "original.json"
    retry_path = tmp_path / "retry.json"
    original = allocate_capture_identifiers(
        original_path,
        token_hex=_factory(_tokens()),
    ).identifiers
    retry = allocate_capture_identifiers(
        retry_path,
        retry_of=original_path,
        token_hex=_factory(_tokens(100)),
    ).identifiers

    original_payload = original.model_dump(mode="json", round_trip=True)
    retry_payload = retry.model_dump(mode="json", round_trip=True)
    for changed in ("recording_id", "attempt_id", "source_key"):
        assert retry_payload.pop(changed) != original_payload.pop(changed)
    assert retry_payload == original_payload


def test_existing_retry_is_unchanged_only_for_the_requested_parent(tmp_path: Path) -> None:
    first_parent_path = tmp_path / "first-parent.json"
    second_parent_path = tmp_path / "second-parent.json"
    retry_path = tmp_path / "retry.json"
    first_parent = allocate_capture_identifiers(
        first_parent_path,
        token_hex=_factory(_tokens()),
    ).identifiers
    allocate_capture_identifiers(
        second_parent_path,
        token_hex=_factory(_tokens(200)),
    )
    created_retry = allocate_capture_identifiers(
        retry_path,
        retry_of=first_parent_path,
        token_hex=_factory(_tokens(100)),
    )

    def must_not_allocate(_size: int) -> str:
        raise AssertionError("a same-parent retry no-op must not allocate identifiers")

    repeated = allocate_capture_identifiers(
        retry_path,
        retry_of=first_parent_path,
        token_hex=must_not_allocate,
    )
    assert repeated.status == "unchanged"
    assert repeated.identifiers == created_retry.identifiers

    captured = retry_path.read_bytes()
    with pytest.raises(CaptureAllocationError):
        allocate_capture_identifiers(
            retry_path,
            retry_of=second_parent_path,
            token_hex=must_not_allocate,
        )
    assert retry_path.read_bytes() == captured

    same_identity_path = tmp_path / "not-a-retry.json"
    same_identity_path.write_bytes(first_parent_path.read_bytes())
    with pytest.raises(CaptureAllocationError):
        allocate_capture_identifiers(
            same_identity_path,
            retry_of=first_parent_path,
            token_hex=must_not_allocate,
        )
    assert validate_capture_identifier_set(same_identity_path.read_bytes()) == first_parent


@pytest.mark.parametrize(
    "invalid_content",
    [b"not-json", b"{}", b'[{"participant_name":"private"}]'],
)
def test_existing_invalid_destination_fails_without_replacement(
    tmp_path: Path,
    invalid_content: bytes,
) -> None:
    destination = tmp_path / "private-identifiers.json"
    destination.write_bytes(invalid_content)

    with pytest.raises(CaptureAllocationError) as captured:
        allocate_capture_identifiers(destination)

    assert captured.value.code == "dataset.capture.identifiers.invalid"
    assert str(captured.value) == "capture identifiers could not be allocated or validated"
    assert destination.read_bytes() == invalid_content


def test_identifier_file_is_path_free_and_contains_only_opaque_values(tmp_path: Path) -> None:
    destination = tmp_path / "person-name-private.json"
    result = allocate_capture_identifiers(destination, token_hex=_factory(_tokens()))
    payload = json.loads(destination.read_text(encoding="utf-8"))
    rendered = json.dumps(payload, sort_keys=True)

    assert set(payload) == set(type(result.identifiers).model_fields)
    assert "person-name-private" not in rendered
    assert str(tmp_path) not in rendered
    assert "@" not in rendered
