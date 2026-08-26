"""Durable allocation of opaque identifiers used by capture workflows."""

from __future__ import annotations

import os
import secrets
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from signlab.contracts.canonical import CanonicalizationError, canonical_json_bytes
from signlab.contracts.ingest import (
    CaptureIdentifierSetV1,
    IngestContractError,
    capture_identifier_set_digest,
    validate_capture_identifier_set,
)

type TokenHexFactory = Callable[[int], str]
type CaptureAllocationStatus = Literal["created", "unchanged"]


class CaptureAllocationError(ValueError):
    """A privacy-safe failure while allocating or loading capture identifiers."""

    def __init__(self) -> None:
        self.code = "dataset.capture.identifiers.invalid"
        super().__init__("capture identifiers could not be allocated or validated")


@dataclass(frozen=True, slots=True)
class CaptureIdentifierAllocation:
    """A newly persisted or previously persisted identifier set."""

    status: CaptureAllocationStatus
    identifiers: CaptureIdentifierSetV1
    identifiers_sha256: str


def _opaque_id(prefix: str, token_hex: TokenHexFactory, *, separator: str = "_") -> str:
    return f"{prefix}{separator}{token_hex(16)}"


def _new_identifier_set(token_hex: TokenHexFactory) -> CaptureIdentifierSetV1:
    return CaptureIdentifierSetV1(
        schema_version="capture-identifier-set/1",
        collection_id=_opaque_id("collection", token_hex),
        visit_id=_opaque_id("visit", token_hex),
        participant_id=_opaque_id("participant", token_hex),
        session_id=_opaque_id("session", token_hex),
        device_id=_opaque_id("device", token_hex),
        recording_id=_opaque_id("recording", token_hex),
        attempt_id=_opaque_id("attempt", token_hex),
        source_key=_opaque_id("source", token_hex),
        prompt_occurrence_id=_opaque_id("occurrence", token_hex),
        annotation_id=_opaque_id("annotation", token_hex),
        annotator_actor_id=_opaque_id("actor", token_hex),
        reviewer_actor_id=_opaque_id("actor", token_hex),
        adjudicator_actor_id=_opaque_id("actor", token_hex),
        annotator_decision_id=_opaque_id("decision", token_hex),
        reviewer_decision_id=_opaque_id("decision", token_hex),
        adjudicator_decision_id=_opaque_id("decision", token_hex),
        dataset_id=_opaque_id("dataset", token_hex),
        store_id=_opaque_id("store", token_hex, separator="-"),
        inventory_id=_opaque_id("inventory", token_hex),
    )


def _retry_identifier_set(
    previous: CaptureIdentifierSetV1,
    token_hex: TokenHexFactory,
) -> CaptureIdentifierSetV1:
    """Keep collection identity while allocating a new physical capture attempt."""

    payload = previous.model_dump(mode="json", round_trip=True)
    payload.update(
        {
            "recording_id": _opaque_id("recording", token_hex),
            "attempt_id": _opaque_id("attempt", token_hex),
            "source_key": _opaque_id("source", token_hex),
        }
    )
    return CaptureIdentifierSetV1.model_validate(payload, strict=True)


def _require_retry_matches_parent(
    retry: CaptureIdentifierSetV1,
    parent: CaptureIdentifierSetV1,
) -> None:
    """Reject an existing retry that does not derive from the requested parent."""

    retry_payload = retry.model_dump(mode="json", round_trip=True)
    parent_payload = parent.model_dump(mode="json", round_trip=True)
    for changed_field in ("recording_id", "attempt_id", "source_key"):
        if retry_payload.pop(changed_field) == parent_payload.pop(changed_field):
            raise CaptureAllocationError
    if retry_payload != parent_payload:
        raise CaptureAllocationError


def _load_identifier_set(path: Path) -> CaptureIdentifierSetV1:
    try:
        if path.is_symlink() or not path.is_file():
            raise CaptureAllocationError
        return validate_capture_identifier_set(path.read_bytes())
    except CaptureAllocationError:
        raise
    except (IngestContractError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise CaptureAllocationError from error


def _publish_new(path: Path, content: bytes) -> bool:
    """Publish without replacement; return false when another writer won the race."""

    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    except OSError as error:
        raise CaptureAllocationError from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def allocate_capture_identifiers(
    destination: str | Path,
    *,
    retry_of: str | Path | None = None,
    token_hex: TokenHexFactory = secrets.token_hex,
) -> CaptureIdentifierAllocation:
    """Persist stable workflow IDs once, or derive a retry's new byte identity.

    Repeating the call for an existing valid destination is an idempotent read. A
    retry reuses the collection, prompt, annotation, and reviewer identities while
    assigning a distinct attempt, recording, and private source key.
    """

    path = Path(destination)
    try:
        if path.is_symlink():
            raise CaptureAllocationError
        retry_parent = None if retry_of is None else _load_identifier_set(Path(retry_of))
        if path.exists():
            checked = _load_identifier_set(path)
            if retry_parent is not None:
                _require_retry_matches_parent(checked, retry_parent)
            return CaptureIdentifierAllocation(
                status="unchanged",
                identifiers=checked,
                identifiers_sha256=capture_identifier_set_digest(checked),
            )
        identifiers = (
            _new_identifier_set(token_hex)
            if retry_parent is None
            else _retry_identifier_set(retry_parent, token_hex)
        )
        content = canonical_json_bytes(identifiers) + b"\n"
        created = _publish_new(path, content)
        checked = identifiers if created else _load_identifier_set(path)
        if retry_parent is not None:
            _require_retry_matches_parent(checked, retry_parent)
        return CaptureIdentifierAllocation(
            status="created" if created else "unchanged",
            identifiers=checked,
            identifiers_sha256=capture_identifier_set_digest(checked),
        )
    except CaptureAllocationError:
        raise
    except (CanonicalizationError, IngestContractError, TypeError, ValueError) as error:
        raise CaptureAllocationError from error


__all__ = [
    "CaptureAllocationError",
    "CaptureAllocationStatus",
    "CaptureIdentifierAllocation",
    "TokenHexFactory",
    "allocate_capture_identifiers",
]
