from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import RecordingsTableV1
from signlab.contracts.ingest import (
    IngestContractError,
    RawDatasetManifestV1,
    collection_sidecar_digest,
    validate_collection_sidecar,
)
from signlab.contracts.quarantine import quarantine_inventory_digest
from signlab.datasets import raw_bundle
from signlab.datasets.importer import import_collection_sidecar
from signlab.datasets.raw_bundle import (
    QUARANTINE_INVENTORY_FILENAME,
    RAW_MANIFEST_FILENAME,
    RawDatasetBundleError,
    build_raw_lineage_inventory,
    validate_raw_dataset_bundle,
)
from test_dataset_importer import (
    _ACCEPTED_SOURCE_KEY,
    _FIXTURE_ROOT,
    _public_fixture,
    _sidecar,
)


def _publish(tmp_path: Path) -> tuple[Path, RawDatasetManifestV1]:
    sidecar, source_map = _public_fixture()
    destination = tmp_path / "raw"
    result = import_collection_sidecar(
        sidecar,
        source_root=_FIXTURE_ROOT,
        source_map=source_map,
        destination=destination,
    )
    return destination, result.manifest


@pytest.mark.golden
def test_public_fixture_pins_cross_platform_raw_and_manifest_identity(tmp_path: Path) -> None:
    destination, manifest = _publish(tmp_path)

    assert (
        manifest.raw_data_sha256
        == "sha256:f1da566976da6ae0127ab95aca1dcc7a013663ceaae4ea3bd179479008d4ef7d"
    )
    assert (
        "sha256:" + hashlib.sha256((destination / RAW_MANIFEST_FILENAME).read_bytes()).hexdigest()
        == "sha256:0e3dae8abe07dcf83fc32a6a115be615af3ec67c5155fbb7c3317ee651277f4c"
    )


def test_raw_inventory_exactly_binds_each_accepted_recording(tmp_path: Path) -> None:
    destination, manifest = _publish(tmp_path)
    loaded = validate_raw_dataset_bundle(manifest, destination)
    recordings = loaded.tables["recordings"]
    assert isinstance(recordings, RecordingsTableV1)

    expected = build_raw_lineage_inventory(loaded.sidecar, recordings.rows)

    assert loaded.inventory == expected
    assert loaded.inventory.inventory_sha256 == manifest.content.lineage_inventory_sha256
    asset = loaded.inventory.assets[0]
    recording = recordings.rows[0]
    assert asset.sha256 == recording.media.sha256
    assert asset.participant_ids == (recording.participant_id,)
    assert asset.recording_ids == (recording.recording_id,)
    assert asset.receipt_ids == (recording.consent_grant.receipt_id,)
    assert asset.grant_ids == (recording.consent_grant.grant_id,)


@pytest.mark.parametrize(
    ("target", "category"),
    [
        ("raw-dataset-manifest.json", "manifest.invalid"),
        ("collection-sidecar.json", "metadata.invalid"),
        ("lineage-inventory.json", "metadata.invalid"),
        ("quarantine-inventory.json", "metadata.invalid"),
        ("tables/recordings.parquet", "table_bytes.invalid"),
        ("accepted-media", "row_artifact_bytes.invalid"),
    ],
)
def test_tampering_any_bound_bundle_layer_fails_closed(
    tmp_path: Path,
    target: str,
    category: str,
) -> None:
    destination, manifest = _publish(tmp_path)
    if target == "accepted-media":
        loaded = validate_raw_dataset_bundle(manifest, destination)
        recordings = loaded.tables["recordings"]
        assert isinstance(recordings, RecordingsTableV1)
        locator = recordings.rows[0].media.locator
        assert isinstance(locator, WorkspaceRelativeLocatorV1)
        path = destination.joinpath(*locator.path.split("/"))
    else:
        path = destination.joinpath(*target.split("/"))
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(manifest, destination)

    assert raised.value.category == category
    assert str(destination) not in str(raised.value)


@pytest.mark.parametrize("extra_kind", ["file", "directory"])
def test_unchanged_validation_rejects_every_unexpected_entry(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    destination, manifest = _publish(tmp_path)
    extra = destination / "unexpected_private_entry"
    if extra_kind == "file":
        extra.write_bytes(b"must not be ignored")
    else:
        extra.mkdir()

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(manifest, destination)

    assert raised.value.category == "metadata.invalid"
    assert "unexpected" not in str(raised.value)


@pytest.mark.parametrize("tamper", ["missing", "directory", "size", "sha256"])
def test_nonaccepted_byte_tampering_is_detected(tmp_path: Path, tamper: str) -> None:
    accepted = b"accepted synthetic bytes"
    retry = b"retry synthetic bytes"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "accepted.webm").write_bytes(accepted)
    (source_root / "retry.webm").write_bytes(retry)
    sidecar = _sidecar(accepted, retry_payload=retry)
    destination = tmp_path / "raw"
    result = import_collection_sidecar(
        sidecar,
        source_root=source_root,
        source_map={
            _ACCEPTED_SOURCE_KEY: "accepted.webm",
            "source_00000000000000000000000000000030": "retry.webm",
        },
        destination=destination,
    )
    quarantine = next((destination / "quarantine").rglob("attempt_*"))
    captured = quarantine.read_bytes()
    if tamper == "missing":
        quarantine.unlink()
    elif tamper == "directory":
        quarantine.unlink()
        quarantine.mkdir()
    elif tamper == "size":
        quarantine.write_bytes(captured + b"tampered")
    else:
        quarantine.write_bytes(bytes([captured[0] ^ 1]) + captured[1:])

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(result.manifest, destination)

    assert raised.value.category == "row_artifact_bytes.invalid"


def test_quarantine_inventory_cannot_omit_a_retained_attempt(tmp_path: Path) -> None:
    accepted = b"accepted synthetic bytes"
    retry = b"retry synthetic bytes"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "accepted.webm").write_bytes(accepted)
    (source_root / "retry.webm").write_bytes(retry)
    sidecar = _sidecar(accepted, retry_payload=retry)
    destination = tmp_path / "raw"
    result = import_collection_sidecar(
        sidecar,
        source_root=source_root,
        source_map={
            _ACCEPTED_SOURCE_KEY: "accepted.webm",
            "source_00000000000000000000000000000030": "retry.webm",
        },
        destination=destination,
    )
    inventory_path = destination / QUARANTINE_INVENTORY_FILENAME
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["assets"] = []
    payload["quarantine_inventory_sha256"] = quarantine_inventory_digest(payload)
    inventory_path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(result.manifest, destination)

    assert raised.value.category == "semantics.invalid"


def test_recording_must_fit_entirely_within_its_session() -> None:
    sidecar, _source_map = _public_fixture()
    payload = sidecar.model_dump(mode="json", round_trip=True)
    sessions = cast(list[dict[str, object]], payload["sessions"])
    sessions[0]["finished_at"] = "2026-08-26T12:10:01Z"
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)

    with pytest.raises(IngestContractError):
        validate_collection_sidecar(payload)


def test_missing_completion_manifest_is_not_a_bundle(tmp_path: Path) -> None:
    destination, manifest = _publish(tmp_path)
    manifest_path = destination / RAW_MANIFEST_FILENAME
    manifest_path.unlink()

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(manifest, destination)

    assert raised.value.category == "manifest.invalid"


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_raw_bundle_root_must_be_an_existing_directory(
    tmp_path: Path,
    root_kind: str,
) -> None:
    _destination, manifest = _publish(tmp_path)
    candidate = tmp_path / "not-a-bundle-root"
    if root_kind == "file":
        candidate.write_bytes(b"not a directory")

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(manifest, candidate)

    assert raised.value.category == "metadata.invalid"
    assert str(candidate) not in str(raised.value)


def test_on_disk_completion_marker_must_equal_supplied_manifest(tmp_path: Path) -> None:
    destination, manifest = _publish(tmp_path)
    changed = manifest.model_copy(update={"version": "1.0.1"})
    (destination / RAW_MANIFEST_FILENAME).write_text(
        json.dumps(changed.model_dump(mode="json", round_trip=True), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(manifest, destination)

    assert raised.value.category == "manifest.invalid"


def test_fixed_metadata_size_limit_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, manifest = _publish(tmp_path)
    monkeypatch.setattr(raw_bundle, "_MAX_METADATA_BYTES", 1)

    with pytest.raises(RawDatasetBundleError) as raised:
        validate_raw_dataset_bundle(manifest, destination)

    assert raised.value.category == "manifest.invalid"
