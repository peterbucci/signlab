"""Atomic, UI-independent capture-attempt updates for collection sidecars."""

from __future__ import annotations

import errno
import hashlib
import importlib
import os
import stat
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from signlab.contracts.canonical import CanonicalizationError, canonical_json_bytes
from signlab.contracts.governance import RecordingConsentGrantV1, validate_recording_consent_grant
from signlab.contracts.ingest import (
    CaptureAttemptV1,
    CaptureIdentifierSetV1,
    CollectionSidecarV1,
    IngestContractError,
    collection_sidecar_digest,
    validate_capture_identifier_set,
    validate_collection_sidecar,
)

type CaptureLedgerStatus = Literal["appended", "unchanged"]
type CaptureLedgerErrorCategory = Literal[
    "sidecar.invalid",
    "identifiers.invalid",
    "media.invalid",
    "attempt.invalid",
    "replay.conflict",
    "publication.failed",
]
type ContractInput = str | bytes | bytearray | Mapping[str, object]


class CaptureLedgerError(ValueError):
    """A privacy-safe failure at the capture-ledger persistence boundary."""

    def __init__(self, category: CaptureLedgerErrorCategory) -> None:
        self.category = category
        super().__init__(f"capture ledger update failed: {category}")


@dataclass(frozen=True, slots=True)
class CaptureLedgerUpdate:
    """The validated result of one idempotent attempt append."""

    status: CaptureLedgerStatus
    sidecar: CollectionSidecarV1
    attempt: CaptureAttemptV1
    collection_sidecar_sha256: str


@dataclass(frozen=True, slots=True)
class _MediaFingerprint:
    sha256: str
    size_bytes: int


def _is_linklike(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_point)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _fingerprint_media(path: Path) -> _MediaFingerprint:
    """Hash one stable regular file without exposing its operational path."""

    try:
        if _is_linklike(path):
            raise CaptureLedgerError("media.invalid")
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise CaptureLedgerError("media.invalid")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if _file_identity(opened) != _file_identity(before):
                raise CaptureLedgerError("media.invalid")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat(follow_symlinks=False)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(current) != _file_identity(opened)
            or _is_linklike(path)
        ):
            raise CaptureLedgerError("media.invalid")
        return _MediaFingerprint(
            sha256=f"sha256:{digest.hexdigest()}",
            size_bytes=opened.st_size,
        )
    except CaptureLedgerError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CaptureLedgerError("media.invalid") from error


def _load_sidecar(path: Path) -> tuple[bytes, CollectionSidecarV1]:
    try:
        if _is_linklike(path) or not path.is_file():
            raise CaptureLedgerError("sidecar.invalid")
        document = path.read_bytes()
        return document, validate_collection_sidecar(document)
    except CaptureLedgerError:
        raise
    except (IngestContractError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise CaptureLedgerError("sidecar.invalid") from error


def _load_identifiers(document: ContractInput | CaptureIdentifierSetV1) -> CaptureIdentifierSetV1:
    try:
        return validate_capture_identifier_set(document)
    except (IngestContractError, TypeError, ValueError) as error:
        raise CaptureLedgerError("identifiers.invalid") from error


def _load_consent_grant(
    document: ContractInput | RecordingConsentGrantV1 | None,
) -> RecordingConsentGrantV1 | None:
    if document is None:
        return None
    try:
        return validate_recording_consent_grant(document)
    except (TypeError, ValueError) as error:
        raise CaptureLedgerError("attempt.invalid") from error


def _require_identifier_context(
    sidecar: CollectionSidecarV1,
    identifiers: CaptureIdentifierSetV1,
) -> int:
    if (
        identifiers.collection_id != sidecar.collection_id
        or identifiers.dataset_id != sidecar.dataset_id
        or identifiers.store_id != sidecar.store_id
        or identifiers.inventory_id != sidecar.inventory_id
    ):
        raise CaptureLedgerError("identifiers.invalid")

    occurrence_indexes = tuple(
        index
        for index, occurrence in enumerate(sidecar.occurrences)
        if occurrence.prompt_occurrence_id == identifiers.prompt_occurrence_id
    )
    sessions = tuple(
        session for session in sidecar.sessions if session.session_id == identifiers.session_id
    )
    plans = tuple(
        plan for plan in sidecar.session_plans if plan.session_id == identifiers.session_id
    )
    participants = tuple(
        participant
        for participant in sidecar.participants
        if participant.participant_id == identifiers.participant_id
    )
    if not (len(occurrence_indexes) == len(sessions) == len(plans) == len(participants) == 1):
        raise CaptureLedgerError("identifiers.invalid")

    occurrence = sidecar.occurrences[occurrence_indexes[0]]
    session = sessions[0]
    plan = plans[0]
    if (
        occurrence.participant_id != identifiers.participant_id
        or occurrence.session_id != identifiers.session_id
        or session.participant_id != identifiers.participant_id
        or session.device_id != identifiers.device_id
        or plan.visit_id != identifiers.visit_id
    ):
        raise CaptureLedgerError("identifiers.invalid")
    return occurrence_indexes[0]


def _desired_attempt(
    sidecar: CollectionSidecarV1,
    identifiers: CaptureIdentifierSetV1,
    occurrence_index: int,
    fingerprint: _MediaFingerprint,
    *,
    outcome: str,
    reason_code: str | None,
    recorded_at: str,
    media_type: str,
    duration_us: int,
    handedness: str,
    mirror_state: str,
    rotation_degrees: int,
    consent_grant: RecordingConsentGrantV1 | None,
) -> CaptureAttemptV1:
    occurrence = sidecar.occurrences[occurrence_index]
    existing = next(
        (
            attempt
            for attempt in occurrence.attempts
            if attempt.attempt_id == identifiers.attempt_id
        ),
        None,
    )
    previous = occurrence.attempts[-1] if occurrence.attempts else None
    if existing is None and (
        occurrence.state != "pending" or (previous is not None and previous.outcome != "retry")
    ):
        raise CaptureLedgerError("replay.conflict")
    retry_of_attempt_id = (
        existing.retry_of_attempt_id
        if existing is not None
        else previous.attempt_id
        if previous is not None
        else None
    )
    try:
        return CaptureAttemptV1.model_validate(
            {
                "schema_version": "capture-attempt/1",
                "attempt_id": identifiers.attempt_id,
                "recording_id": identifiers.recording_id,
                "source_key": identifiers.source_key,
                "outcome": outcome,
                "reason_code": reason_code,
                "retry_of_attempt_id": retry_of_attempt_id,
                "recorded_at": recorded_at,
                "media_type": media_type,
                "expected_sha256": fingerprint.sha256,
                "expected_size_bytes": fingerprint.size_bytes,
                "duration_us": duration_us,
                "handedness": handedness,
                "mirror_state": mirror_state,
                "rotation_degrees": rotation_degrees,
                "audio_present": False,
                "consent_grant": consent_grant,
            },
            strict=True,
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise CaptureLedgerError("attempt.invalid") from error


def _append_attempt(
    sidecar: CollectionSidecarV1,
    attempt: CaptureAttemptV1,
    occurrence_index: int,
) -> CollectionSidecarV1:
    payload = sidecar.model_dump(mode="json", round_trip=True)
    occurrences = cast(list[dict[str, object]], payload["occurrences"])
    occurrence = occurrences[occurrence_index]
    attempts = cast(list[dict[str, object]], occurrence["attempts"])
    attempts.append(attempt.model_dump(mode="json", round_trip=True))
    occurrence["state"] = {
        "accepted": "accepted",
        "retry": "pending",
        "quarantined": "quarantined",
    }[attempt.outcome]
    payload["updated_at"] = max(cast(str, payload["updated_at"]), attempt.recorded_at)
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)
    try:
        return validate_collection_sidecar(payload)
    except (IngestContractError, TypeError, ValueError) as error:
        raise CaptureLedgerError("attempt.invalid") from error


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_advisory_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | os.O_EXCL, 0o600)
        os.write(descriptor, b"\0")
        os.fsync(descriptor)
        return descriptor
    except FileExistsError:
        descriptor = os.open(path, flags, 0o600)
        if os.fstat(descriptor).st_size == 0:
            os.ftruncate(descriptor, 1)
            os.fsync(descriptor)
        return descriptor


def _try_lock(descriptor: int) -> None:
    if os.name == "nt":
        windows_locking = importlib.import_module("msvcrt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        windows_locking.locking(descriptor, windows_locking.LK_NBLCK, 1)
        return
    file_control = importlib.import_module("fcntl")
    file_control.flock(descriptor, file_control.LOCK_EX | file_control.LOCK_NB)


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        windows_locking = importlib.import_module("msvcrt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        windows_locking.locking(descriptor, windows_locking.LK_UNLCK, 1)
        return
    file_control = importlib.import_module("fcntl")
    file_control.flock(descriptor, file_control.LOCK_UN)


@contextmanager
def _sidecar_write_lock(sidecar_path: Path) -> Iterator[None]:
    """Serialize writers using a crash-released OS lock on a stable sibling inode."""

    lock_path = sidecar_path.with_name(f".{sidecar_path.name}.lock")
    descriptor: int | None = None
    locked = False
    try:
        if _is_linklike(lock_path):
            raise CaptureLedgerError("publication.failed")
        descriptor = _open_advisory_lock(lock_path)
        while True:
            try:
                _try_lock(descriptor)
                locked = True
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                time.sleep(0.01)
        yield
    except CaptureLedgerError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CaptureLedgerError("publication.failed") from error
    finally:
        if descriptor is not None:
            if locked:
                with suppress(OSError):
                    _unlock(descriptor)
            os.close(descriptor)


def _publish_sidecar(path: Path, expected: bytes, sidecar: CollectionSidecarV1) -> None:
    temporary: Path | None = None
    try:
        content = canonical_json_bytes(sidecar) + b"\n"
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
        if _is_linklike(path) or path.read_bytes() != expected:
            raise CaptureLedgerError("replay.conflict")
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    except CaptureLedgerError:
        raise
    except (CanonicalizationError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise CaptureLedgerError("publication.failed") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def append_capture_attempt(
    sidecar_path: str | Path,
    identifiers: ContractInput | CaptureIdentifierSetV1,
    *,
    media_path: str | Path,
    outcome: str,
    reason_code: str | None,
    recorded_at: str,
    media_type: str,
    duration_us: int,
    handedness: str,
    mirror_state: str,
    rotation_degrees: int,
    consent_grant: ContractInput | RecordingConsentGrantV1 | None = None,
) -> CaptureLedgerUpdate:
    """Append one preallocated attempt, or validate an exact idempotent replay.

    The mutable sidecar must remain active or paused. Media paths are used only to
    compute byte identity and are never placed in the sidecar or returned errors.
    """

    path = Path(sidecar_path)
    with _sidecar_write_lock(path):
        original, sidecar = _load_sidecar(path)
        if sidecar.state == "complete":
            raise CaptureLedgerError("sidecar.invalid")
        checked_identifiers = _load_identifiers(identifiers)
        occurrence_index = _require_identifier_context(sidecar, checked_identifiers)
        fingerprint = _fingerprint_media(Path(media_path))
        checked_grant = _load_consent_grant(consent_grant)
        desired = _desired_attempt(
            sidecar,
            checked_identifiers,
            occurrence_index,
            fingerprint,
            outcome=outcome,
            reason_code=reason_code,
            recorded_at=recorded_at,
            media_type=media_type,
            duration_us=duration_us,
            handedness=handedness,
            mirror_state=mirror_state,
            rotation_degrees=rotation_degrees,
            consent_grant=checked_grant,
        )

        attempts = tuple(attempt for item in sidecar.occurrences for attempt in item.attempts)
        existing = next(
            (
                attempt
                for attempt in attempts
                if attempt.attempt_id == checked_identifiers.attempt_id
            ),
            None,
        )
        if existing is not None:
            if existing != desired:
                raise CaptureLedgerError("replay.conflict")
            return CaptureLedgerUpdate(
                status="unchanged",
                sidecar=sidecar,
                attempt=existing,
                collection_sidecar_sha256=sidecar.collection_sidecar_sha256,
            )
        if any(
            attempt.recording_id == checked_identifiers.recording_id
            or attempt.source_key == checked_identifiers.source_key
            for attempt in attempts
        ):
            raise CaptureLedgerError("replay.conflict")

        updated = _append_attempt(sidecar, desired, occurrence_index)
        _publish_sidecar(path, original, updated)
        return CaptureLedgerUpdate(
            status="appended",
            sidecar=updated,
            attempt=desired,
            collection_sidecar_sha256=updated.collection_sidecar_sha256,
        )


__all__ = [
    "CaptureLedgerError",
    "CaptureLedgerErrorCategory",
    "CaptureLedgerStatus",
    "CaptureLedgerUpdate",
    "append_capture_attempt",
]
