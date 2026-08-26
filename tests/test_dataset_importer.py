from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from signlab.contracts.dataset import AnnotationsTableV1, RecordingsTableV1
from signlab.contracts.governance import RecordingConsentGrantV1
from signlab.contracts.ingest import (
    CollectionSidecarV1,
    collection_sidecar_digest,
    validate_collection_sidecar,
)
from signlab.contracts.quarantine import discover_quarantined_recording_ids
from signlab.datasets import importer
from signlab.datasets.importer import DatasetImportError, import_collection_sidecar
from signlab.datasets.raw_bundle import validate_raw_dataset_bundle
from signlab.governance.resources import (
    build_example_recording_grant,
    build_governance_policy,
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "public" / "ingest"
_PROTOCOL_PATH = Path(__file__).parents[1] / "docs" / "collection-protocol.md"
_PROTOCOL_SHA256 = "sha256:530f7cc60544324ff8fe419a3a35c7349fb47f3531f7ce4603a4ed6badcf6bf6"
_PARTICIPANT_ID = "participant_00000000000000000000000000000001"
_SESSION_ID = "session_00000000000000000000000000000001"
_DEVICE_ID = "device_00000000000000000000000000000001"
_ACCEPTED_RECORDING_ID = "recording_00000000000000000000000000000031"
_ACCEPTED_SOURCE_KEY = "source_00000000000000000000000000000001"


def _sha(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _grant() -> RecordingConsentGrantV1:
    original = build_example_recording_grant()
    return RecordingConsentGrantV1.model_validate(
        original.model_dump(mode="json", round_trip=True),
        strict=True,
    )


def _attempt(
    number: int,
    payload: bytes,
    outcome: str,
    recorded_at: str,
    *,
    retry_of: str | None = None,
    source_number: int | None = None,
) -> dict[str, object]:
    accepted = outcome == "accepted"
    suffix = f"{number:032d}"
    source_suffix = f"{source_number if source_number is not None else number:032d}"
    return {
        "schema_version": "capture-attempt/1",
        "attempt_id": f"attempt_{suffix}",
        "recording_id": f"recording_{suffix}",
        "source_key": f"source_{source_suffix}",
        "outcome": outcome,
        "reason_code": None if accepted else "camera_interruption",
        "retry_of_attempt_id": retry_of,
        "recorded_at": recorded_at,
        "media_type": "video/webm",
        "expected_sha256": _sha(payload),
        "expected_size_bytes": len(payload),
        "duration_us": 5_000_000,
        "handedness": "right",
        "mirror_state": "mirrored",
        "rotation_degrees": 0,
        "audio_present": False,
        "consent_grant": _grant().model_dump(mode="json", round_trip=True) if accepted else None,
    }


def _sidecar(
    accepted_payload: bytes,
    *,
    retry_payload: bytes | None = None,
    quarantined_payload: bytes | None = None,
    include_skipped: bool = False,
    state: str = "complete",
    fixture_only: bool = True,
) -> CollectionSidecarV1:
    assert b"\r\n" not in _PROTOCOL_PATH.read_bytes()
    assert _sha(_PROTOCOL_PATH.read_bytes()) == _PROTOCOL_SHA256
    policy = build_governance_policy()
    accepted = _attempt(
        31,
        accepted_payload,
        "accepted",
        "2026-08-26T12:10:00Z",
        retry_of="attempt_00000000000000000000000000000030" if retry_payload is not None else None,
        source_number=1,
    )
    accepted_attempts: list[dict[str, object]] = []
    if retry_payload is not None:
        accepted_attempts.append(
            _attempt(
                30,
                retry_payload,
                "retry",
                "2026-08-26T12:09:00Z",
            )
        )
    accepted_attempts.append(accepted)
    occurrences: list[dict[str, object]] = [
        {
            "schema_version": "prompt-occurrence/1",
            "prompt_occurrence_id": "occurrence_00000000000000000000000000000001",
            "ordinal": 1,
            "repetition": 1,
            "prompt_label_id": "hello",
            "participant_id": _PARTICIPANT_ID,
            "session_id": _SESSION_ID,
            "state": "accepted",
            "skip_reason_code": None,
            "attempts": accepted_attempts,
        }
    ]
    if quarantined_payload is not None:
        occurrences.append(
            {
                "schema_version": "prompt-occurrence/1",
                "prompt_occurrence_id": "occurrence_00000000000000000000000000000002",
                "ordinal": 2,
                "repetition": 1,
                "prompt_label_id": "no",
                "participant_id": _PARTICIPANT_ID,
                "session_id": _SESSION_ID,
                "state": "quarantined",
                "skip_reason_code": None,
                "attempts": [
                    _attempt(
                        32,
                        quarantined_payload,
                        "quarantined",
                        "2026-08-26T12:11:00Z",
                    )
                ],
            }
        )
    if include_skipped:
        occurrences.append(
            {
                "schema_version": "prompt-occurrence/1",
                "prompt_occurrence_id": "occurrence_00000000000000000000000000000003",
                "ordinal": len(occurrences) + 1,
                "repetition": 1,
                "prompt_label_id": "please",
                "participant_id": _PARTICIPANT_ID,
                "session_id": _SESSION_ID,
                "state": "skipped",
                "skip_reason_code": "session_stopped",
                "attempts": [],
            }
        )
    payload: dict[str, object] = {
        "schema_version": "collection-sidecar/1",
        "collection_id": "collection_00000000000000000000000000000001",
        "dataset_id": "dataset_00000000000000000000000000000001",
        "dataset_version": "1.0.0",
        "store_id": "store-00000000000000000000000000000001",
        "inventory_id": "inventory_00000000000000000000000000000001",
        "generated_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:15:00Z",
        "finalized_at": "2026-08-26T12:15:00Z" if state == "complete" else None,
        "state": state,
        "fixture_only": fixture_only,
        "taxonomy": policy.taxonomy.model_dump(mode="json", round_trip=True),
        "protocol": {
            "schema_version": "collection-protocol-reference/1",
            "protocol_id": "signlab-collection-protocol",
            "version": "0.1.0",
            "sha256": _PROTOCOL_SHA256,
        },
        "governance_policy": policy.model_dump(mode="json", round_trip=True),
        "participants": [{"participant_id": _PARTICIPANT_ID, "handedness": "right"}],
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
                "visit_id": "visit_00000000000000000000000000000001",
                "session_id": _SESSION_ID,
                "condition_profile_id": "fixture_condition",
                "prompt_randomization": {
                    "schema_version": "prompt-randomization/1",
                    "algorithm_id": "fixture_order",
                    "algorithm_version": "1.0.0",
                    "seed_sha256": "sha256:" + "2" * 64,
                    "realized_order_authoritative": True,
                    "rerolled_for_performance": False,
                },
                "consent_checklist": [
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": check_id,
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    }
                    for check_id in (
                        "authenticated_receipt_is_current",
                        "collection_readiness_is_ready",
                        "purpose_is_authorized_before_capture",
                    )
                ],
                "capture_checklist": [
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": check_id,
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    }
                    for check_id in (
                        "camera_and_lens_ready",
                        "framing_and_lighting_usable",
                        "no_third_party_present",
                        "orientation_and_mirror_recorded",
                        "timing_and_playback_checked",
                    )
                ],
            }
        ],
        "occurrences": occurrences,
        "annotations": [
            {
                "schema_version": "capture-annotation/1",
                "annotation_id": "annotation_00000000000000000000000000000001",
                "source_recording_id": _ACCEPTED_RECORDING_ID,
                "decisions": [
                    {
                        "schema_version": "annotation-decision/1",
                        "decision_id": "decision_00000000000000000000000000000001",
                        "actor_id": "actor_00000000000000000000000000000001",
                        "role": "annotator",
                        "decided_at": "2026-08-26T12:12:00Z",
                        "proposal": {
                            "schema_version": "annotation-proposal/1",
                            "interval": {
                                "schema_version": "media-interval/1",
                                "start_us": 100_000,
                                "end_us": 1_000_000,
                            },
                            "disposition": "class_label",
                            "label_id": "hello",
                            "other_kind": None,
                            "reason_code": None,
                        },
                    },
                    {
                        "schema_version": "annotation-decision/1",
                        "decision_id": "decision_00000000000000000000000000000002",
                        "actor_id": "actor_00000000000000000000000000000002",
                        "role": "reviewer",
                        "decided_at": "2026-08-26T12:13:00Z",
                        "proposal": {
                            "schema_version": "annotation-proposal/1",
                            "interval": {
                                "schema_version": "media-interval/1",
                                "start_us": 100_000,
                                "end_us": 1_000_000,
                            },
                            "disposition": "class_label",
                            "label_id": "hello",
                            "other_kind": None,
                            "reason_code": None,
                        },
                    },
                ],
            }
        ],
        "collection_sidecar_sha256": "sha256:" + "0" * 64,
    }
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)
    return validate_collection_sidecar(payload)


def _public_fixture() -> tuple[CollectionSidecarV1, dict[str, str]]:
    payload = (_FIXTURE_ROOT / f"{_ACCEPTED_SOURCE_KEY}.webm").read_bytes()
    source_map = json.loads((_FIXTURE_ROOT / "source-map.json").read_text(encoding="utf-8"))
    assert isinstance(source_map, dict)
    return _sidecar(payload), source_map


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def test_fixture_protocol_reference_matches_the_published_document() -> None:
    sidecar, _source_map = _public_fixture()
    document_bytes = _PROTOCOL_PATH.read_bytes()
    document = document_bytes.decode("utf-8")

    assert sidecar.protocol.protocol_id == "signlab-collection-protocol"
    assert sidecar.protocol.version == "0.1.0"
    assert sidecar.protocol.sha256 == _PROTOCOL_SHA256 == _sha(document_bytes)
    assert f"- Protocol ID: `{sidecar.protocol.protocol_id}`" in document
    assert f"- Draft version: `{sidecar.protocol.version}`" in document


@pytest.mark.integration
def test_fixture_import_builds_a_fully_verified_raw_bundle(tmp_path: Path) -> None:
    sidecar, source_map = _public_fixture()
    destination = tmp_path / "raw"

    result = import_collection_sidecar(
        sidecar,
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=destination,
    )
    loaded = validate_raw_dataset_bundle(result.manifest, destination)

    assert result.status == "published"
    assert result.accepted_recordings == 1
    assert result.retry_attempts == 0
    assert result.quarantined_attempts == 0
    assert result.skipped_occurrences == 0
    assert result.validation.artifact_byte_integrity == "verified"
    assert result.validation.quarantine_inventory_integrity == "verified"
    assert result.validation.consent_authorization == "not_checked"
    assert loaded.manifest == result.manifest
    recordings = loaded.tables["recordings"]
    assert isinstance(recordings, RecordingsTableV1)
    assert len(recordings.rows) == 1
    assert recordings.rows[0].recording_id == _ACCEPTED_RECORDING_ID
    assert recordings.rows[0].media.sha256 == _sha(
        (_FIXTURE_ROOT / f"{_ACCEPTED_SOURCE_KEY}.webm").read_bytes()
    )
    assert loaded.tables["clips"].rows == ()
    assert loaded.tables["derived_artifacts"].rows == ()
    annotations = loaded.tables["annotations"]
    assert isinstance(annotations, AnnotationsTableV1)
    assert annotations.rows[0].review_status == "reviewed"
    assert loaded.quarantine_inventory.assets == ()


@pytest.mark.golden
def test_identical_replay_is_a_verified_noop_with_stable_tree(tmp_path: Path) -> None:
    sidecar, source_map = _public_fixture()
    destination = tmp_path / "raw"
    first = import_collection_sidecar(
        sidecar,
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=destination,
    )
    before = _tree_hashes(destination)

    reordered = dict(reversed(list(sidecar.model_dump(mode="json", round_trip=True).items())))
    second = import_collection_sidecar(
        json.dumps(reordered, indent=7).encode("utf-8"),
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=destination,
    )

    assert second.status == "unchanged"
    assert second.manifest == first.manifest
    assert second.manifest.raw_data_sha256 == first.manifest.raw_data_sha256
    assert _tree_hashes(destination) == before
    assert not tuple(tmp_path.glob(".raw.staging-*"))


def test_source_filename_location_and_mtime_do_not_affect_identity(tmp_path: Path) -> None:
    sidecar, _source_map = _public_fixture()
    payload = (_FIXTURE_ROOT / f"{_ACCEPTED_SOURCE_KEY}.webm").read_bytes()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    (second_root / "nested").mkdir(parents=True)
    (first_root / "opaque-a.webm").write_bytes(payload)
    second_path = second_root / "nested" / "opaque-b.webm"
    second_path.write_bytes(payload)
    second_path.touch()

    first = import_collection_sidecar(
        sidecar,
        source_root=first_root,
        source_map={_ACCEPTED_SOURCE_KEY: "opaque-a.webm"},
        destination=tmp_path / "out-a",
    )
    second = import_collection_sidecar(
        sidecar,
        source_root=second_root,
        source_map={_ACCEPTED_SOURCE_KEY: Path("nested") / "opaque-b.webm"},
        destination=tmp_path / "out-b",
    )

    assert first.manifest == second.manifest
    assert _tree_hashes(tmp_path / "out-a") == _tree_hashes(tmp_path / "out-b")
    published_names = "\n".join(_tree_hashes(tmp_path / "out-b"))
    published_metadata = b"".join(path.read_bytes() for path in (tmp_path / "out-b").glob("*.json"))
    assert "opaque-b" not in published_names
    assert b"opaque-b" not in published_metadata


def test_failure_before_publication_leaves_nothing_and_retry_is_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar, source_map = _public_fixture()
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()

    with monkeypatch.context() as context:
        context.setattr(
            importer,
            "_checkpoint",
            lambda phase: (_ for _ in ()).throw(OSError("seeded")) if phase == "metadata" else None,
        )
        with pytest.raises(DatasetImportError) as raised:
            import_collection_sidecar(
                sidecar,
                source_root=_FIXTURE_ROOT,
                source_map=source_map,
                destination=interrupted,
            )

    assert raised.value.category == "publication.failed"
    assert interrupted.is_dir()
    assert not tuple(interrupted.iterdir())
    assert not tuple(tmp_path.glob(".interrupted.staging-*"))
    resumed = import_collection_sidecar(
        sidecar,
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=interrupted,
    )
    uninterrupted = import_collection_sidecar(
        sidecar,
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=tmp_path / "uninterrupted",
    )
    assert resumed.manifest == uninterrupted.manifest
    assert _tree_hashes(interrupted) == _tree_hashes(tmp_path / "uninterrupted")


@pytest.mark.parametrize(
    "unsafe_location",
    [
        "../outside.webm",
        "C" + ":/private.webm",
        "CON/file.webm",
        "missing.webm",
    ],
)
def test_source_paths_fail_closed_without_echoing_locations(
    tmp_path: Path,
    unsafe_location: str,
) -> None:
    sidecar, _source_map = _public_fixture()

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=_FIXTURE_ROOT,
            source_map={_ACCEPTED_SOURCE_KEY: unsafe_location},
            destination=tmp_path / "raw",
        )

    assert raised.value.category == "source.invalid"
    assert unsafe_location not in str(raised.value)
    assert str(_FIXTURE_ROOT) not in str(raised.value)
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_source_root_must_be_an_existing_directory(
    tmp_path: Path,
    root_kind: str,
) -> None:
    sidecar, source_map = _public_fixture()
    source_root = tmp_path / "source-root"
    if root_kind == "file":
        source_root.write_bytes(b"not a directory")

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=source_root,
            source_map=source_map,
            destination=tmp_path / "raw",
        )

    assert raised.value.category == "source.invalid"
    assert str(source_root) not in str(raised.value)


@pytest.mark.parametrize("map_kind", ["missing", "not_mapping", "empty_path", "directory"])
def test_source_map_must_be_exact_and_resolve_to_regular_files(
    tmp_path: Path,
    map_kind: str,
) -> None:
    sidecar, _source_map = _public_fixture()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    directory = source_root / "opaque-directory"
    directory.mkdir()
    source_map: Mapping[str, importer.SourceLocation]
    if map_kind == "missing":
        source_map = {}
    elif map_kind == "not_mapping":
        source_map = cast(Mapping[str, importer.SourceLocation], object())
    elif map_kind == "empty_path":
        source_map = {_ACCEPTED_SOURCE_KEY: ""}
    else:
        source_map = {_ACCEPTED_SOURCE_KEY: "opaque-directory"}

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=source_root,
            source_map=source_map,
            destination=tmp_path / "raw",
        )

    assert raised.value.category == "source.invalid"
    assert str(source_root) not in str(raised.value)


def test_source_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    sidecar, _source_map = _public_fixture()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    link = source_root / "opaque.webm"
    try:
        link.symlink_to(_FIXTURE_ROOT / f"{_ACCEPTED_SOURCE_KEY}.webm")
    except OSError:
        pytest.skip("file links are unavailable for this account")

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=source_root,
            source_map={_ACCEPTED_SOURCE_KEY: "opaque.webm"},
            destination=tmp_path / "raw",
        )

    assert raised.value.category == "source.invalid"


def test_source_hash_mismatch_and_midcopy_mutation_never_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"expected synthetic media"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "opaque.webm"
    source.write_bytes(b"wrong synthetic media")
    sidecar = _sidecar(expected)

    with pytest.raises(DatasetImportError) as mismatch:
        import_collection_sidecar(
            sidecar,
            source_root=source_root,
            source_map={_ACCEPTED_SOURCE_KEY: "opaque.webm"},
            destination=tmp_path / "mismatch",
        )
    assert mismatch.value.category == "source.invalid"

    source.write_bytes(expected)
    real_copy = importer._copy_verified

    def mutate_before_copy(
        fingerprint: importer.SourceFingerprint,
        destination: Path,
    ) -> None:
        fingerprint.path.write_bytes(expected + b"changed")
        real_copy(fingerprint, destination)

    monkeypatch.setattr(importer, "_copy_verified", mutate_before_copy)
    with pytest.raises(DatasetImportError) as mutation:
        import_collection_sidecar(
            sidecar,
            source_root=source_root,
            source_map={_ACCEPTED_SOURCE_KEY: "opaque.webm"},
            destination=tmp_path / "mutation",
        )
    assert mutation.value.category == "source.invalid"
    assert not (tmp_path / "mismatch").exists()
    assert not (tmp_path / "mutation").exists()


@pytest.mark.parametrize("mode", ["paused", "real"])
def test_import_refuses_incomplete_or_nonfixture_collection(
    tmp_path: Path,
    mode: str,
) -> None:
    payload = (_FIXTURE_ROOT / f"{_ACCEPTED_SOURCE_KEY}.webm").read_bytes()
    sidecar = _sidecar(
        payload,
        state="paused" if mode == "paused" else "complete",
        fixture_only=mode != "real",
    )

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=_FIXTURE_ROOT,
            source_map={_ACCEPTED_SOURCE_KEY: f"{_ACCEPTED_SOURCE_KEY}.webm"},
            destination=tmp_path / "raw",
        )

    assert raised.value.category == "sidecar.invalid"
    assert not (tmp_path / "raw").exists()


def test_import_requires_the_exact_builtin_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar, source_map = _public_fixture()
    monkeypatch.setattr(importer, "taxonomy_reference", lambda _taxonomy: object())

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=_FIXTURE_ROOT,
            source_map=source_map,
            destination=tmp_path / "raw",
        )

    assert raised.value.category == "sidecar.invalid"
    assert not (tmp_path / "raw").exists()


def test_retry_and_quarantine_bytes_are_auditable_but_not_normalized(
    tmp_path: Path,
) -> None:
    accepted = b"synthetic accepted media"
    retry = b"synthetic technical retry"
    quarantined = b"synthetic quarantined media"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    mapping: dict[str, str] = {}
    for number, source_number, payload in (
        (30, 30, retry),
        (31, 1, accepted),
        (32, 32, quarantined),
    ):
        filename = f"opaque-{number}.webm"
        (source_root / filename).write_bytes(payload)
        mapping[f"source_{source_number:032d}"] = filename
    sidecar = _sidecar(
        accepted,
        retry_payload=retry,
        quarantined_payload=quarantined,
        include_skipped=True,
    )

    result = import_collection_sidecar(
        sidecar,
        source_root=source_root,
        source_map=mapping,
        destination=tmp_path / "raw",
    )
    loaded = validate_raw_dataset_bundle(result.manifest, tmp_path / "raw")

    recordings = loaded.tables["recordings"]
    assert isinstance(recordings, RecordingsTableV1)
    assert tuple(row.recording_id for row in recordings.rows) == (_ACCEPTED_RECORDING_ID,)
    assert result.accepted_recordings == 1
    assert result.retry_attempts == 1
    assert result.quarantined_attempts == 1
    assert result.skipped_occurrences == 1
    assert len(tuple((tmp_path / "raw" / "quarantine").rglob("attempt_*"))) == 2
    inventory = loaded.quarantine_inventory
    assert tuple(asset.recording_id for asset in inventory.assets) == (
        "recording_00000000000000000000000000000030",
        "recording_00000000000000000000000000000032",
    )
    assert tuple(asset.outcome for asset in inventory.assets) == ("retry", "quarantined")
    assert all(asset.lifecycle_state == "quarantined" for asset in inventory.assets)
    assert all(asset.consent_evidence_status == "absent" for asset in inventory.assets)
    assert discover_quarantined_recording_ids(inventory, _PARTICIPANT_ID) == tuple(
        asset.recording_id for asset in inventory.assets
    )
    assert (
        discover_quarantined_recording_ids(
            inventory,
            "participant_00000000000000000000000000000002",
        )
        == ()
    )
    serialized = json.dumps(inventory.model_dump(mode="json", round_trip=True))
    assert "receipt_id" not in serialized
    assert "grant_id" not in serialized


def test_changed_input_conflicts_without_mutating_existing_bundle(tmp_path: Path) -> None:
    sidecar, source_map = _public_fixture()
    destination = tmp_path / "raw"
    import_collection_sidecar(
        sidecar,
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=destination,
    )
    before = _tree_hashes(destination)
    changed = b"different synthetic bytes"
    source_root = tmp_path / "changed"
    source_root.mkdir()
    (source_root / "opaque.webm").write_bytes(changed)
    changed_sidecar = _sidecar(changed)

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            changed_sidecar,
            source_root=source_root,
            source_map={_ACCEPTED_SOURCE_KEY: "opaque.webm"},
            destination=destination,
        )

    assert raised.value.category == "destination.conflict"
    assert _tree_hashes(destination) == before


def test_unrecognized_nonempty_destination_is_never_overwritten(tmp_path: Path) -> None:
    sidecar, source_map = _public_fixture()
    destination = tmp_path / "raw"
    destination.mkdir()
    sentinel = destination / "opaque-existing-artifact"
    sentinel.write_bytes(b"existing bytes")

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=_FIXTURE_ROOT,
            source_map=source_map,
            destination=destination,
        )

    assert raised.value.category == "destination.invalid"
    assert sentinel.read_bytes() == b"existing bytes"
    assert not tuple(tmp_path.glob(".raw.staging-*"))


def test_file_destination_is_rejected_before_publication(tmp_path: Path) -> None:
    sidecar, source_map = _public_fixture()
    destination = tmp_path / "raw"
    destination.write_bytes(b"existing file")

    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            sidecar,
            source_root=_FIXTURE_ROOT,
            source_map=source_map,
            destination=destination,
        )

    assert raised.value.category == "destination.invalid"
    assert destination.read_bytes() == b"existing file"


def test_malformed_sidecar_has_one_sanitized_boundary(tmp_path: Path) -> None:
    with pytest.raises(DatasetImportError) as raised:
        import_collection_sidecar(
            b"{}",
            source_root=tmp_path,
            source_map={},
            destination=tmp_path / "raw",
        )

    assert raised.value.category == "sidecar.invalid"
    assert str(tmp_path) not in str(raised.value)
