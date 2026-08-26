from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.governance import RecordingConsentGrantV1
from signlab.contracts.ingest import (
    CAPTURE_CHECKLIST_IDS,
    CONSENT_CHECKLIST_IDS,
    CaptureIdentifierSetV1,
    CollectionSidecarV1,
    collection_sidecar_digest,
    validate_collection_sidecar,
)
from signlab.datasets import ledger
from signlab.datasets.ledger import (
    CaptureLedgerError,
    CaptureLedgerUpdate,
    append_capture_attempt,
)
from signlab.governance.resources import build_example_recording_grant, build_governance_policy

_PARTICIPANT_ID = "participant_00000000000000000000000000000001"
_SESSION_ID = "session_00000000000000000000000000000001"
_DEVICE_ID = "device_00000000000000000000000000000001"
_RECORDED_AT = "2026-08-26T12:10:00Z"


def _identifiers(
    *,
    attempt: int = 31,
    recording: int = 31,
    source: int = 1,
) -> CaptureIdentifierSetV1:
    return CaptureIdentifierSetV1(
        schema_version="capture-identifier-set/1",
        collection_id="collection_00000000000000000000000000000001",
        visit_id="visit_00000000000000000000000000000001",
        participant_id=_PARTICIPANT_ID,
        session_id=_SESSION_ID,
        device_id=_DEVICE_ID,
        recording_id=f"recording_{recording:032d}",
        attempt_id=f"attempt_{attempt:032d}",
        source_key=f"source_{source:032d}",
        prompt_occurrence_id="occurrence_00000000000000000000000000000001",
        annotation_id="annotation_00000000000000000000000000000001",
        annotator_actor_id="actor_00000000000000000000000000000001",
        reviewer_actor_id="actor_00000000000000000000000000000002",
        adjudicator_actor_id="actor_00000000000000000000000000000003",
        annotator_decision_id="decision_00000000000000000000000000000001",
        reviewer_decision_id="decision_00000000000000000000000000000002",
        adjudicator_decision_id="decision_00000000000000000000000000000003",
        dataset_id="dataset_00000000000000000000000000000001",
        store_id="store-00000000000000000000000000000001",
        inventory_id="inventory_00000000000000000000000000000001",
    )


def _checklist(check_ids: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "checklist-result/1",
            "check_id": check_id,
            "status": "not_applicable",
            "reason_code": "synthetic_no_person_no_camera",
        }
        for check_id in check_ids
    ]


def _write_sidecar(
    path: Path,
    identifiers: CaptureIdentifierSetV1,
    *,
    state: str = "active",
) -> CollectionSidecarV1:
    policy = build_governance_policy()
    payload: dict[str, object] = {
        "schema_version": "collection-sidecar/1",
        "collection_id": identifiers.collection_id,
        "dataset_id": identifiers.dataset_id,
        "dataset_version": "1.0.0",
        "store_id": identifiers.store_id,
        "inventory_id": identifiers.inventory_id,
        "generated_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:00:00Z",
        "finalized_at": None,
        "state": state,
        "fixture_only": True,
        "taxonomy": policy.taxonomy.model_dump(mode="json", round_trip=True),
        "protocol": {
            "schema_version": "collection-protocol-reference/1",
            "protocol_id": "signlab-collection-protocol",
            "version": "0.1.0",
            "sha256": "sha256:" + "1" * 64,
        },
        "governance_policy": policy.model_dump(mode="json", round_trip=True),
        "participants": [
            {
                "participant_id": identifiers.participant_id,
                "handedness": "right",
            }
        ],
        "sessions": [
            {
                "session_id": identifiers.session_id,
                "participant_id": identifiers.participant_id,
                "device_id": identifiers.device_id,
                "started_at": "2026-08-26T12:00:00Z",
                "finished_at": "2026-08-26T12:30:00Z",
                "capture_mode": "continuous",
                "capture_software_version": "1.0.0",
                "camera_facing": "front",
                "frame_width_px": 1280,
                "frame_height_px": 720,
                "frame_rate_numerator": 30,
                "frame_rate_denominator": 1,
                "rotation_degrees": 0,
                "mirror_state": "mirrored",
            }
        ],
        "session_plans": [
            {
                "schema_version": "collection-session-plan/1",
                "visit_id": identifiers.visit_id,
                "session_id": identifiers.session_id,
                "condition_profile_id": "fixture_condition",
                "prompt_randomization": {
                    "schema_version": "prompt-randomization/1",
                    "algorithm_id": "fixture_order",
                    "algorithm_version": "1.0.0",
                    "seed_sha256": "sha256:" + "2" * 64,
                    "realized_order_authoritative": True,
                    "rerolled_for_performance": False,
                },
                "consent_checklist": _checklist(CONSENT_CHECKLIST_IDS),
                "capture_checklist": _checklist(CAPTURE_CHECKLIST_IDS),
            }
        ],
        "occurrences": [
            {
                "schema_version": "prompt-occurrence/1",
                "prompt_occurrence_id": identifiers.prompt_occurrence_id,
                "ordinal": 1,
                "repetition": 1,
                "prompt_label_id": "hello",
                "participant_id": identifiers.participant_id,
                "session_id": identifiers.session_id,
                "state": "pending",
                "skip_reason_code": None,
                "attempts": [],
            }
        ],
        "annotations": [],
        "collection_sidecar_sha256": "sha256:" + "0" * 64,
    }
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)
    checked = validate_collection_sidecar(payload)
    path.write_bytes(canonical_json_bytes(checked) + b"\n")
    return checked


def _write_identifiers(path: Path, identifiers: CaptureIdentifierSetV1) -> None:
    path.write_bytes(canonical_json_bytes(identifiers) + b"\n")


def _replace_identifiers(
    identifiers: CaptureIdentifierSetV1,
    **updates: object,
) -> CaptureIdentifierSetV1:
    payload = identifiers.model_dump(mode="json", round_trip=True)
    payload.update(updates)
    return CaptureIdentifierSetV1.model_validate(payload, strict=True)


def _append(
    sidecar: Path,
    identifiers: CaptureIdentifierSetV1 | bytes,
    media: Path,
    *,
    outcome: str,
    reason_code: str | None,
    recorded_at: str = _RECORDED_AT,
    consent_grant: RecordingConsentGrantV1 | None = None,
) -> CaptureLedgerUpdate:
    return append_capture_attempt(
        sidecar,
        identifiers,
        media_path=media,
        outcome=outcome,
        reason_code=reason_code,
        recorded_at=recorded_at,
        media_type="video/webm",
        duration_us=5_000_000,
        handedness="right",
        mirror_state="mirrored",
        rotation_degrees=0,
        consent_grant=consent_grant,
    )


def test_accepted_append_computes_checksum_and_persists_a_valid_sidecar(tmp_path: Path) -> None:
    identifiers = _identifiers()
    sidecar = tmp_path / "collection-sidecar.json"
    media = tmp_path / "capture.webm"
    payload = b"explicitly synthetic captured bytes"
    _write_sidecar(sidecar, identifiers)
    media.write_bytes(payload)

    result = _append(
        sidecar,
        identifiers,
        media,
        outcome="accepted",
        reason_code=None,
        consent_grant=build_example_recording_grant(),
    )

    checked = validate_collection_sidecar(sidecar.read_bytes())
    attempt = checked.occurrences[0].attempts[0]
    assert result.status == "appended"
    assert attempt.expected_sha256 == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert attempt.expected_size_bytes == len(payload)
    assert attempt == result.attempt
    assert checked.collection_sidecar_sha256 == result.collection_sidecar_sha256
    assert str(media) not in sidecar.read_text(encoding="utf-8")


def test_exact_resume_on_paused_sidecar_is_idempotent(tmp_path: Path) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    sidecar = tmp_path / "paused-sidecar.json"
    media = tmp_path / "retry.webm"
    _write_sidecar(sidecar, identifiers, state="paused")
    media.write_bytes(b"synthetic retry bytes")

    first = _append(
        sidecar,
        identifiers,
        media,
        outcome="retry",
        reason_code="camera_interruption",
    )
    captured = sidecar.read_bytes()
    second = _append(
        sidecar,
        identifiers,
        media,
        outcome="retry",
        reason_code="camera_interruption",
    )

    assert first.status == "appended"
    assert second.status == "unchanged"
    assert second.sidecar.state == "paused"
    assert sidecar.read_bytes() == captured


def test_retry_can_be_followed_by_a_direct_quarantine(tmp_path: Path) -> None:
    first_ids = _identifiers(attempt=30, recording=30, source=1)
    second_ids = _identifiers(attempt=32, recording=32, source=2)
    sidecar = tmp_path / "collection-sidecar.json"
    first_media = tmp_path / "first.webm"
    second_media = tmp_path / "second.webm"
    _write_sidecar(sidecar, first_ids)
    first_media.write_bytes(b"synthetic technical retry")
    second_media.write_bytes(b"synthetic quarantined bytes")

    _append(
        sidecar,
        first_ids,
        first_media,
        outcome="retry",
        reason_code="camera_interruption",
    )
    result = _append(
        sidecar,
        second_ids,
        second_media,
        outcome="quarantined",
        reason_code="third_party_presence",
        recorded_at="2026-08-26T12:11:00Z",
    )

    occurrence = result.sidecar.occurrences[0]
    assert occurrence.state == "quarantined"
    assert tuple(attempt.outcome for attempt in occurrence.attempts) == (
        "retry",
        "quarantined",
    )
    assert occurrence.attempts[1].retry_of_attempt_id == occurrence.attempts[0].attempt_id


def test_conflicting_replay_fails_without_replacing_sidecar(tmp_path: Path) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    sidecar = tmp_path / "collection-sidecar.json"
    media = tmp_path / "capture.webm"
    _write_sidecar(sidecar, identifiers)
    media.write_bytes(b"first synthetic bytes")
    _append(
        sidecar,
        identifiers,
        media,
        outcome="retry",
        reason_code="camera_interruption",
    )
    persisted = sidecar.read_bytes()
    media.write_bytes(b"different synthetic bytes")

    with pytest.raises(CaptureLedgerError) as raised:
        _append(
            sidecar,
            identifiers,
            media,
            outcome="retry",
            reason_code="camera_interruption",
        )

    assert raised.value.category == "replay.conflict"
    assert sidecar.read_bytes() == persisted


def test_concurrent_distinct_attempts_serialize_and_retain_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_ids = _identifiers(attempt=30, recording=30, source=1)
    second_ids = _replace_identifiers(
        first_ids,
        attempt_id="attempt_00000000000000000000000000000032",
        recording_id="recording_00000000000000000000000000000032",
        source_key="source_00000000000000000000000000000002",
        prompt_occurrence_id="occurrence_00000000000000000000000000000002",
    )
    sidecar = tmp_path / "collection-sidecar.json"
    first_media = tmp_path / "first.webm"
    second_media = tmp_path / "second.webm"
    checked = _write_sidecar(sidecar, first_ids)
    payload = checked.model_dump(mode="json", round_trip=True)
    occurrences = payload["occurrences"]
    assert isinstance(occurrences, list)
    occurrences.append(
        {
            "schema_version": "prompt-occurrence/1",
            "prompt_occurrence_id": second_ids.prompt_occurrence_id,
            "ordinal": 2,
            "repetition": 1,
            "prompt_label_id": "no",
            "participant_id": second_ids.participant_id,
            "session_id": second_ids.session_id,
            "state": "pending",
            "skip_reason_code": None,
            "attempts": [],
        }
    )
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)
    sidecar.write_bytes(canonical_json_bytes(validate_collection_sidecar(payload)) + b"\n")
    first_media.write_bytes(b"first concurrent synthetic media")
    second_media.write_bytes(b"second concurrent synthetic media")

    first_loaded = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_loaded = threading.Event()
    observed_attempt_counts: list[int] = []
    observations_lock = threading.Lock()
    original_load = ledger._load_sidecar

    def observed_load(path: Path) -> tuple[bytes, CollectionSidecarV1]:
        document, loaded = original_load(path)
        attempt_count = sum(len(item.attempts) for item in loaded.occurrences)
        with observations_lock:
            call_index = len(observed_attempt_counts)
            observed_attempt_counts.append(attempt_count)
        if call_index == 0:
            first_loaded.set()
            if not release_first.wait(timeout=5):
                raise AssertionError("first writer was not released")
        else:
            second_loaded.set()
        return document, loaded

    monkeypatch.setattr(ledger, "_load_sidecar", observed_load)

    def append_first() -> CaptureLedgerUpdate:
        return _append(
            sidecar,
            first_ids,
            first_media,
            outcome="retry",
            reason_code="camera_interruption",
        )

    def append_second() -> CaptureLedgerUpdate:
        second_started.set()
        return _append(
            sidecar,
            second_ids,
            second_media,
            outcome="retry",
            reason_code="prompt_display_failure",
            recorded_at="2026-08-26T12:11:00Z",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(append_first)
        assert first_loaded.wait(timeout=5)
        second = executor.submit(append_second)
        assert second_started.wait(timeout=5)
        try:
            assert not second_loaded.wait(timeout=0.2)
        finally:
            release_first.set()
        assert first.result(timeout=5).status == "appended"
        assert second.result(timeout=5).status == "appended"

    final = validate_collection_sidecar(sidecar.read_bytes())
    assert observed_attempt_counts == [0, 1]
    assert tuple(item.attempts[0].attempt_id for item in final.occurrences) == (
        first_ids.attempt_id,
        second_ids.attempt_id,
    )


def test_empty_advisory_lock_file_is_repaired(tmp_path: Path) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    sidecar = tmp_path / "collection-sidecar.json"
    media = tmp_path / "capture.webm"
    _write_sidecar(sidecar, identifiers)
    media.write_bytes(b"synthetic media")
    lock_path = sidecar.with_name(f".{sidecar.name}.lock")
    lock_path.touch()

    result = _append(
        sidecar,
        identifiers,
        media,
        outcome="retry",
        reason_code="camera_interruption",
    )

    assert result.status == "appended"
    assert lock_path.read_bytes() == b"\0"


def test_media_failure_does_not_leak_path_or_modify_sidecar(tmp_path: Path) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    sidecar = tmp_path / "collection-sidecar.json"
    identifying_media = tmp_path / "private-person-name.webm"
    _write_sidecar(sidecar, identifiers)
    persisted = sidecar.read_bytes()

    with pytest.raises(CaptureLedgerError) as raised:
        _append(
            sidecar,
            identifiers,
            identifying_media,
            outcome="retry",
            reason_code="missing_source",
        )

    assert raised.value.category == "media.invalid"
    assert "private-person-name" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)
    assert sidecar.read_bytes() == persisted


def test_cli_appends_without_printing_paths_or_opaque_ids(tmp_path: Path) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    sidecar = tmp_path / "private-sidecar.json"
    identifier_path = tmp_path / "private-identifiers.json"
    media = tmp_path / "private-media.webm"
    _write_sidecar(sidecar, identifiers)
    _write_identifiers(identifier_path, identifiers)
    media.write_bytes(b"synthetic retry bytes")

    result = CliRunner(env={"NO_COLOR": "1"}).invoke(
        cli.app,
        [
            "data",
            "append-capture-attempt",
            str(sidecar),
            "--identifiers",
            str(identifier_path),
            "--media",
            str(media),
            "--outcome",
            "retry",
            "--reason-code",
            "camera_interruption",
            "--recorded-at",
            _RECORDED_AT,
            "--media-type",
            "video/webm",
            "--duration-us",
            "5000000",
            "--handedness",
            "right",
            "--mirror-state",
            "mirrored",
            "--rotation-degrees",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert result.output.splitlines()[:2] == [
        "Capture attempt: appended.",
        "Capture outcome: retry.",
    ]
    assert "Collection sidecar SHA-256: sha256:" in result.output
    assert identifiers.attempt_id not in result.output
    assert sidecar.name not in result.output
    assert media.name not in result.output
    assert str(tmp_path) not in result.output
    assert validate_collection_sidecar(sidecar.read_bytes()).occurrences[0].attempts


@pytest.mark.parametrize("media_kind", ["empty", "directory"])
def test_non_regular_or_empty_media_is_rejected_without_a_path_leak(
    tmp_path: Path,
    media_kind: str,
) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    sidecar = tmp_path / "private-sidecar.json"
    media = tmp_path / "private-person-media"
    _write_sidecar(sidecar, identifiers)
    persisted = sidecar.read_bytes()
    if media_kind == "empty":
        media.write_bytes(b"")
    else:
        media.mkdir()

    with pytest.raises(CaptureLedgerError) as raised:
        _append(
            sidecar,
            identifiers,
            media,
            outcome="retry",
            reason_code="corrupt_source",
        )

    assert raised.value.category == "media.invalid"
    assert "private-person-media" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)
    assert sidecar.read_bytes() == persisted


def test_media_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    sidecar = tmp_path / "private-sidecar.json"
    target = tmp_path / "target.webm"
    link = tmp_path / "private-person-link.webm"
    _write_sidecar(sidecar, identifiers)
    target.write_bytes(b"synthetic target bytes")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("file links are unavailable for this account")

    with pytest.raises(CaptureLedgerError) as raised:
        _append(
            sidecar,
            identifiers,
            link,
            outcome="retry",
            reason_code="corrupt_source",
        )

    assert raised.value.category == "media.invalid"
    assert "private-person-link" not in str(raised.value)
    assert validate_collection_sidecar(sidecar.read_bytes()).occurrences[0].attempts == ()


def test_invalid_sidecar_and_identifier_documents_are_sanitized(tmp_path: Path) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    media = tmp_path / "private-person-media.webm"
    media.write_bytes(b"synthetic media")
    invalid_sidecar = tmp_path / "private-person-sidecar.json"
    invalid_sidecar.write_text('{"participant_name":"private-person"}', encoding="utf-8")

    with pytest.raises(CaptureLedgerError) as sidecar_error:
        _append(
            invalid_sidecar,
            identifiers,
            media,
            outcome="retry",
            reason_code="camera_interruption",
        )

    assert sidecar_error.value.category == "sidecar.invalid"
    assert "private-person" not in str(sidecar_error.value)
    valid_sidecar = tmp_path / "valid-sidecar.json"
    _write_sidecar(valid_sidecar, identifiers)
    persisted = valid_sidecar.read_bytes()

    with pytest.raises(CaptureLedgerError) as identifier_error:
        _append(
            valid_sidecar,
            b'{"participant_name":"private-person"}',
            media,
            outcome="retry",
            reason_code="camera_interruption",
        )

    assert identifier_error.value.category == "identifiers.invalid"
    assert "private-person" not in str(identifier_error.value)
    assert valid_sidecar.read_bytes() == persisted


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collection_id", "collection_ffffffffffffffffffffffffffffffff"),
        ("prompt_occurrence_id", "occurrence_ffffffffffffffffffffffffffffffff"),
        ("device_id", "device_ffffffffffffffffffffffffffffffff"),
    ],
)
def test_identifier_context_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    identifiers = _identifiers(attempt=30, recording=30)
    mismatched = _replace_identifiers(identifiers, **{field: value})
    sidecar = tmp_path / "collection-sidecar.json"
    media = tmp_path / "capture.webm"
    _write_sidecar(sidecar, identifiers)
    media.write_bytes(b"synthetic media")
    persisted = sidecar.read_bytes()

    with pytest.raises(CaptureLedgerError) as raised:
        _append(
            sidecar,
            mismatched,
            media,
            outcome="retry",
            reason_code="camera_interruption",
        )

    assert raised.value.category == "identifiers.invalid"
    assert sidecar.read_bytes() == persisted


def test_invalid_consent_and_accepted_without_consent_are_rejected(tmp_path: Path) -> None:
    identifiers = _identifiers()
    sidecar = tmp_path / "collection-sidecar.json"
    media = tmp_path / "private-person-media.webm"
    _write_sidecar(sidecar, identifiers)
    media.write_bytes(b"synthetic accepted media")
    persisted = sidecar.read_bytes()

    with pytest.raises(CaptureLedgerError) as invalid_grant:
        append_capture_attempt(
            sidecar,
            identifiers,
            media_path=media,
            outcome="accepted",
            reason_code=None,
            recorded_at=_RECORDED_AT,
            media_type="video/webm",
            duration_us=5_000_000,
            handedness="right",
            mirror_state="mirrored",
            rotation_degrees=0,
            consent_grant=b'{"participant_name":"private-person"}',
        )
    assert invalid_grant.value.category == "attempt.invalid"
    assert "private-person" not in str(invalid_grant.value)

    with pytest.raises(CaptureLedgerError) as missing_grant:
        _append(
            sidecar,
            identifiers,
            media,
            outcome="accepted",
            reason_code=None,
        )
    assert missing_grant.value.category == "attempt.invalid"
    assert sidecar.read_bytes() == persisted


def test_completed_sidecar_refuses_further_attempts(tmp_path: Path) -> None:
    identifiers = _identifiers()
    sidecar = tmp_path / "collection-sidecar.json"
    media = tmp_path / "capture.webm"
    _write_sidecar(sidecar, identifiers)
    media.write_bytes(b"synthetic accepted media")
    appended = _append(
        sidecar,
        identifiers,
        media,
        outcome="accepted",
        reason_code=None,
        consent_grant=build_example_recording_grant(),
    )
    payload = appended.sidecar.model_dump(mode="json", round_trip=True)
    payload["state"] = "complete"
    payload["finalized_at"] = "2026-08-26T12:15:00Z"
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)
    completed = validate_collection_sidecar(payload)
    sidecar.write_bytes(canonical_json_bytes(completed) + b"\n")
    persisted = sidecar.read_bytes()

    with pytest.raises(CaptureLedgerError) as raised:
        _append(
            sidecar,
            identifiers,
            media,
            outcome="accepted",
            reason_code=None,
            consent_grant=build_example_recording_grant(),
        )

    assert raised.value.category == "sidecar.invalid"
    assert sidecar.read_bytes() == persisted


@pytest.mark.parametrize("duplicate_field", ["recording", "source"])
def test_new_attempt_cannot_reuse_recording_or_source_identity(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    first_ids = _identifiers(attempt=30, recording=30, source=1)
    second_ids = _identifiers(
        attempt=32,
        recording=30 if duplicate_field == "recording" else 32,
        source=1 if duplicate_field == "source" else 2,
    )
    sidecar = tmp_path / "collection-sidecar.json"
    first_media = tmp_path / "first.webm"
    second_media = tmp_path / "second.webm"
    _write_sidecar(sidecar, first_ids)
    first_media.write_bytes(b"first synthetic media")
    second_media.write_bytes(b"second synthetic media")
    _append(
        sidecar,
        first_ids,
        first_media,
        outcome="retry",
        reason_code="camera_interruption",
    )
    persisted = sidecar.read_bytes()

    with pytest.raises(CaptureLedgerError) as raised:
        _append(
            sidecar,
            second_ids,
            second_media,
            outcome="quarantined",
            reason_code="third_party_presence",
            recorded_at="2026-08-26T12:11:00Z",
        )

    assert raised.value.category == "replay.conflict"
    assert sidecar.read_bytes() == persisted
