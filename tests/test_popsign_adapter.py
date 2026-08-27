from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import tarfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from signlab.contracts.external_dataset import (
    ExternalAcquisitionPlanV1,
    ExternalArchivePlanV1,
    ExternalDatasetManifestV1,
    external_acquisition_plan_digest,
    external_dataset_manifest_digest,
)
from signlab.datasets import popsign
from signlab.datasets.external_resources import render_external_dataset_json
from signlab.datasets.popsign import (
    EXTERNAL_DATASET_MANIFEST_FILENAME,
    POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    POPSIGN_PLAN_ID,
    ImportedExternalDatasetBundle,
    PopSignDatasetError,
    build_popsign_v1_plan,
    import_popsign_v1_archives,
    validate_external_dataset_bundle,
    write_external_acquisition_plan,
)


@dataclass(frozen=True, slots=True)
class _TarMember:
    name: str
    payload: bytes = b"synthetic-video"
    kind: bytes = tarfile.REGTYPE
    linkname: str = ""


def _archive_path(root: Path, archive: ExternalArchivePlanV1) -> Path:
    return root.joinpath(*archive.local_archive.path.split("/"))


def _normal_member(archive: ExternalArchivePlanV1) -> _TarMember:
    participant = f"fixture-signer-{archive.split}"
    recording = f"recording-{archive.source_label}-20260827T120000"
    return _TarMember(
        name=f"clips/{participant}--{recording}-.mp4",
        payload=(
            b"\x00\x00\x00\x18ftypmp42signlab-synthetic-video:" + archive.archive_id.encode("ascii")
        ),
    )


def _write_tar(path: Path, members: tuple[_TarMember, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as opened:
        for member in members:
            info = tarfile.TarInfo(member.name)
            info.type = member.kind
            info.mode = 0o644
            info.mtime = 0
            info.linkname = member.linkname
            if member.kind in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                info.size = len(member.payload)
                opened.addfile(info, io.BytesIO(member.payload))
            else:
                info.size = 0
                opened.addfile(info)


def _write_corpus(
    root: Path,
    plan: ExternalAcquisitionPlanV1,
    *,
    overrides: Mapping[str, tuple[_TarMember, ...]] | None = None,
) -> None:
    replacements = {} if overrides is None else overrides
    for archive in plan.archives:
        _write_tar(
            _archive_path(root, archive),
            replacements.get(archive.archive_id, (_normal_member(archive),)),
        )


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _import_fixture(
    root: Path,
) -> tuple[
    ExternalAcquisitionPlanV1,
    Path,
    Path,
    ImportedExternalDatasetBundle,
]:
    plan = build_popsign_v1_plan()
    archive_root = root / "archives"
    bundle_root = root / "bundle"
    _write_corpus(archive_root, plan)
    imported = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=bundle_root,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )
    return plan, archive_root, bundle_root, imported


def _redigest_manifest(payload: dict[str, object]) -> dict[str, object]:
    payload["content_sha256"] = external_dataset_manifest_digest(payload)
    return payload


def _assert_error(
    captured: pytest.ExceptionInfo[PopSignDatasetError],
    category: str,
    *private_values: str,
) -> None:
    error = captured.value
    assert error.category == category
    assert error.code == f"dataset.external.{category}"
    rendered = str(error)
    assert rendered
    assert category not in rendered
    assert all(value not in rendered for value in private_values)


def test_plan_is_the_exact_offline_fifteen_archive_popsign_selection() -> None:
    plan = build_popsign_v1_plan()
    expected_pairs = tuple(
        (split, label)
        for split in ("train", "val", "test")
        for label in ("hello", "no", "please", "thankyou", "yes")
    )

    assert plan.plan_id == POPSIGN_PLAN_ID
    assert plan.network_access == "forbidden"
    assert plan.preview_media == "forbidden"
    assert plan.required_license_acknowledgement == POPSIGN_LICENSE_ACKNOWLEDGEMENT
    assert tuple((archive.split, archive.source_label) for archive in plan.archives) == (
        expected_pairs
    )
    assert len(plan.archives) == 15
    for archive in plan.archives:
        assert archive.archive_id == (f"popsign_v1_game_{archive.split}_{archive.source_label}")
        assert archive.download_url == (
            "https://signdata.cc.gatech.edu/data/popsign_v1_0/"
            f"game/{archive.split}/{archive.source_label}.tar"
        )
        assert archive.local_archive.path == (
            f"archives/game/{archive.split}/{archive.source_label}.tar"
        )
        assert archive.publisher_sha256 is None
        assert archive.integrity_basis == "trust_on_first_use_then_sha256"


def test_plan_publication_is_canonical_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    destination = tmp_path / "plans" / "popsign.json"

    first = write_external_acquisition_plan(plan, destination)
    published = destination.read_bytes()
    second = write_external_acquisition_plan(published, destination)

    assert (first.status, second.status) == ("published", "unchanged")
    assert first.plan_sha256 == second.plan_sha256 == plan.plan_sha256
    assert first.archive_count == second.archive_count == 15
    assert published == render_external_dataset_json(plan).encode("utf-8")
    assert json.loads(published)["network_access"] == "forbidden"
    assert not tuple(destination.parent.glob(f".{destination.name}.staging-*"))

    destination.write_bytes(b"different-private-content")
    with pytest.raises(PopSignDatasetError) as captured:
        write_external_acquisition_plan(plan, destination)
    _assert_error(captured, "plan.conflict", str(destination), "different-private-content")
    assert destination.read_bytes() == b"different-private-content"
    assert not tuple(destination.parent.glob(f".{destination.name}.staging-*"))


def test_import_is_deterministic_idempotent_content_addressed_and_opaque(
    tmp_path: Path,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "downloaded-archives"
    first_bundle = tmp_path / "first-bundle"
    second_bundle = tmp_path / "second-bundle"
    _write_corpus(archive_root, plan)

    first = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=first_bundle,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )
    repeated = import_popsign_v1_archives(
        plan.model_dump_json(),
        archive_root=archive_root,
        destination=first_bundle,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )
    independent = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=second_bundle,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )

    assert (first.status, repeated.status, independent.status) == (
        "published",
        "unchanged",
        "published",
    )
    assert first.manifest == repeated.manifest == independent.manifest
    assert first.validation.content_sha256 == first.manifest.content_sha256
    assert first.validation.archive_count == 15
    assert first.validation.media_count == 15
    assert first.validation.semantic_integrity == "verified"
    assert first.validation.media_byte_integrity == "verified"
    assert first.validation.archive_byte_integrity == "verified"
    assert first.validation.license_authorization == "verified"
    assert (first_bundle / EXTERNAL_DATASET_MANIFEST_FILENAME).read_bytes() == (
        second_bundle / EXTERNAL_DATASET_MANIFEST_FILENAME
    ).read_bytes()

    serialized = render_external_dataset_json(first.manifest)
    assert "fixture-signer" not in serialized
    assert "20260827T120000" not in serialized
    assert "--" not in serialized
    assert first.manifest.source_metadata_retained is False
    assert first.manifest.license_acknowledgement.signlab_participant_consent == ("not_applicable")
    assert len({item.sample_id for item in first.manifest.media}) == 15
    assert len({item.recording_id for item in first.manifest.media}) == 15
    assert len({item.participant_id for item in first.manifest.media}) == 3
    assert all(item.sample_id.startswith("sample_") for item in first.manifest.media)
    assert all(item.recording_id.startswith("recording_") for item in first.manifest.media)
    assert all(item.participant_id.startswith("participant_") for item in first.manifest.media)

    expected_files = {EXTERNAL_DATASET_MANIFEST_FILENAME}
    for item in first.manifest.media:
        digest = item.sha256.removeprefix("sha256:")
        assert item.locator.path == f"media/sha256/{digest[:2]}/{digest}.mp4"
        media_path = first_bundle.joinpath(*item.locator.path.split("/"))
        assert _sha256(media_path) == item.sha256
        assert media_path.stat().st_size == item.size_bytes
        expected_files.add(item.locator.path)
    actual_files = {
        path.relative_to(first_bundle).as_posix()
        for path in first_bundle.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files
    for archive, record in zip(plan.archives, first.manifest.archives, strict=True):
        source_path = _archive_path(archive_root, archive)
        assert record.sha256 == _sha256(source_path)
        assert record.size_bytes == source_path.stat().st_size
        assert record.member_count == 1


def test_validation_survives_moving_both_relative_roots(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "original-archives"
    bundle_root = tmp_path / "original-bundle"
    _write_corpus(archive_root, plan)
    imported = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=bundle_root,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )

    moved_archives = tmp_path / "moved" / "archives"
    moved_bundle = tmp_path / "moved" / "bundle"
    moved_archives.parent.mkdir()
    archive_root.rename(moved_archives)
    bundle_root.rename(moved_bundle)

    media_only = validate_external_dataset_bundle(imported.manifest, moved_bundle)
    full = validate_external_dataset_bundle(
        (moved_bundle / EXTERNAL_DATASET_MANIFEST_FILENAME).read_bytes(),
        moved_bundle,
        archive_root=moved_archives,
    )

    assert media_only.archive_byte_integrity == "not_checked"
    assert full.archive_byte_integrity == "verified"
    assert media_only.content_sha256 == full.content_sha256 == imported.manifest.content_sha256


def test_import_rejects_a_signer_that_crosses_source_splits(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    train = plan.archives[0]
    validation = plan.archives[5]
    shared_train = _TarMember(
        name="shared-signer--train-recording-.mp4",
        payload=b"synthetic-train-video",
    )
    shared_validation = _TarMember(
        name="shared-signer--validation-recording-.mp4",
        payload=b"synthetic-validation-video",
    )
    archive_root = tmp_path / "private-signer-path" / "archives"
    _write_corpus(
        archive_root,
        plan,
        overrides={
            train.archive_id: (shared_train,),
            validation.archive_id: (shared_validation,),
        },
    )

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=tmp_path / "bundle",
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, "split.leakage", "shared-signer", str(archive_root))


@pytest.mark.parametrize(
    ("members", "category"),
    [
        pytest.param(
            (_TarMember("../signer--recording-.mp4"),),
            "archive.structure_invalid",
            id="parent-traversal",
        ),
        pytest.param(
            (_TarMember("/signer--recording-.mp4"),),
            "archive.structure_invalid",
            id="absolute-posix-path",
        ),
        pytest.param(
            (_TarMember("C:" + "/signer--recording-.mp4"),),
            "archive.structure_invalid",
            id="absolute-windows-path",
        ),
        pytest.param(
            (_TarMember(r"clips\signer--recording-.mp4"),),
            "archive.structure_invalid",
            id="backslash-path",
        ),
        pytest.param(
            (
                _TarMember(
                    "signer--recording-.mp4",
                    kind=tarfile.SYMTYPE,
                    linkname="private-target",
                ),
            ),
            "archive.structure_invalid",
            id="symbolic-link",
        ),
        pytest.param(
            (
                _TarMember(
                    "signer--recording-.mp4",
                    kind=tarfile.LNKTYPE,
                    linkname="private-target",
                ),
            ),
            "archive.structure_invalid",
            id="hard-link",
        ),
        pytest.param(
            (_TarMember("signer--recording-.mp4", kind=tarfile.FIFOTYPE),),
            "archive.structure_invalid",
            id="special-file",
        ),
        pytest.param(
            (_TarMember("signer--recording-.txt"),),
            "archive.member_invalid",
            id="non-mp4",
        ),
        pytest.param(
            (_TarMember("provider-filename.mp4"),),
            "archive.member_invalid",
            id="malformed-provider-name",
        ),
        pytest.param(
            (_TarMember("signer--recording-.mp4", payload=b""),),
            "archive.limit_exceeded",
            id="empty-media",
        ),
        pytest.param(
            (
                _TarMember("signer--recording-.mp4", payload=b"one"),
                _TarMember("signer--recording-.mp4", payload=b"two"),
            ),
            "archive.structure_invalid",
            id="duplicate-name",
        ),
        pytest.param(
            (
                _TarMember("Signer--Recording-.mp4", payload=b"one"),
                _TarMember("signer--recording-.mp4", payload=b"two"),
            ),
            "archive.structure_invalid",
            id="case-colliding-name",
        ),
    ],
)
def test_import_rejects_unsafe_or_invalid_tar_members(
    tmp_path: Path,
    members: tuple[_TarMember, ...],
    category: str,
) -> None:
    plan = build_popsign_v1_plan()
    first_archive = plan.archives[0]
    archive_root = tmp_path / "secret-archive-root"
    _write_corpus(
        archive_root,
        plan,
        overrides={first_archive.archive_id: members},
    )
    destination = tmp_path / "unpublished-bundle"

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=destination,
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, category, str(archive_root), *[item.name for item in members])
    assert not destination.exists()
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))


@pytest.mark.parametrize(
    ("limit_name", "limit", "category"),
    [
        ("_MAX_ARCHIVE_BYTES", 1, "archive.bytes_invalid"),
        ("_MAX_TAR_ENTRIES", 0, "archive.limit_exceeded"),
        ("_MAX_MEDIA_PER_ARCHIVE", 0, "archive.limit_exceeded"),
        ("_MAX_MEMBER_BYTES", 1, "archive.limit_exceeded"),
        ("_MAX_UNCOMPRESSED_BYTES", 1, "archive.limit_exceeded"),
    ],
)
def test_import_enforces_archive_and_member_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    category: str,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    _write_corpus(archive_root, plan)
    monkeypatch.setattr(popsign, limit_name, limit)

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=tmp_path / "bundle",
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, category, str(archive_root))


@pytest.mark.parametrize(
    ("tamper", "category"),
    [
        ("media", "media.bytes_invalid"),
        ("manifest", "manifest.invalid"),
        ("extra-file", "bundle.inventory_invalid"),
        ("archive-bytes", "archive.bytes_invalid"),
        ("archive-member", "archive.bytes_invalid"),
    ],
)
def test_validation_detects_bundle_and_source_tampering(
    tmp_path: Path,
    tamper: str,
    category: str,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    bundle_root = tmp_path / "bundle"
    _write_corpus(archive_root, plan)
    imported = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=bundle_root,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )

    if tamper == "media":
        media_path = bundle_root.joinpath(*imported.manifest.media[0].locator.path.split("/"))
        media_path.write_bytes(b"tampered-private-video")
    elif tamper == "manifest":
        (bundle_root / EXTERNAL_DATASET_MANIFEST_FILENAME).write_bytes(b"{}\n")
    elif tamper == "extra-file":
        (bundle_root / "unexpected-private-file.txt").write_text("private", encoding="utf-8")
    elif tamper == "archive-bytes":
        with _archive_path(archive_root, plan.archives[0]).open("ab") as stream:
            stream.write(b"tampered-trailing-archive-bytes")
    else:
        first_archive = plan.archives[0]
        replacement = _normal_member(first_archive)
        _write_tar(
            _archive_path(archive_root, first_archive),
            (
                _TarMember(
                    replacement.name,
                    payload=b"different-synthetic-video",
                ),
            ),
        )

    with pytest.raises(PopSignDatasetError) as captured:
        validate_external_dataset_bundle(
            imported.manifest,
            bundle_root,
            archive_root=archive_root,
        )

    _assert_error(captured, category, str(bundle_root), str(archive_root), "private")


def test_missing_archive_and_license_failures_are_sanitized(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    secret_root = tmp_path / "participant-alice-private-archives"
    secret_root.mkdir()
    destination = tmp_path / "bundle"

    with pytest.raises(PopSignDatasetError) as denied:
        import_popsign_v1_archives(
            plan,
            archive_root=secret_root,
            destination=destination,
            accept_license="secret-wrong-license-value",
        )
    _assert_error(
        denied,
        "license.denied",
        str(secret_root),
        "participant-alice",
        "secret-wrong-license-value",
    )

    with pytest.raises(PopSignDatasetError) as missing:
        import_popsign_v1_archives(
            plan,
            archive_root=secret_root,
            destination=destination,
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )
    _assert_error(missing, "archive.missing", str(secret_root), "participant-alice")
    assert not destination.exists()


def test_failed_import_is_atomic_and_can_be_retried_after_repair(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    first_archive = plan.archives[0]
    archive_root = tmp_path / "archives"
    destination = tmp_path / "bundle"
    _write_corpus(
        archive_root,
        plan,
        overrides={first_archive.archive_id: (_TarMember("../../private--member-.mp4"),)},
    )

    with pytest.raises(PopSignDatasetError):
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=destination,
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )
    assert not destination.exists()
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))

    _write_tar(_archive_path(archive_root, first_archive), (_normal_member(first_archive),))
    repaired = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=destination,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )

    assert repaired.status == "published"
    assert repaired.validation.media_count == 15
    assert (destination / EXTERNAL_DATASET_MANIFEST_FILENAME).is_file()
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))


def test_existing_conflicting_destination_is_never_overwritten(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    destination = tmp_path / "bundle"
    marker = destination / "private-marker.txt"
    _write_corpus(archive_root, plan)
    destination.mkdir()
    marker.write_bytes(b"must-survive")

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=destination,
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, "destination.conflict", str(destination), "private-marker")
    assert marker.read_bytes() == b"must-survive"
    assert {path.name for path in destination.iterdir()} == {marker.name}
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))


def test_empty_destination_is_atomically_replaced_by_complete_bundle(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    destination = tmp_path / "bundle"
    _write_corpus(archive_root, plan)
    destination.mkdir()

    imported = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=destination,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )

    assert imported.status == "published"
    assert (destination / EXTERNAL_DATASET_MANIFEST_FILENAME).is_file()
    assert imported.validation.media_count == 15


def test_import_and_validation_do_not_require_a_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    bundle_root = tmp_path / "bundle"
    _write_corpus(archive_root, plan)

    def forbidden_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)

    imported = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=bundle_root,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )
    validated = validate_external_dataset_bundle(
        imported.manifest,
        bundle_root,
        archive_root=archive_root,
    )

    assert validated.archive_byte_integrity == "verified"
    assert validated.media_count == 15


@pytest.mark.parametrize("document_kind", ["invalid", "oversized-bytes", "oversized-text"])
def test_plan_input_fails_closed_before_any_filesystem_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_kind: str,
) -> None:
    if document_kind == "invalid":
        document: object = b"not-json"
    elif document_kind == "oversized-bytes":
        monkeypatch.setattr(popsign, "_MAX_PLAN_BYTES", 4)
        document = b"12345"
    else:
        monkeypatch.setattr(popsign, "_MAX_PLAN_BYTES", 4)
        document = "12345"

    with pytest.raises(PopSignDatasetError) as captured:
        write_external_acquisition_plan(
            cast(str | bytes | bytearray | Mapping[str, object], document),
            tmp_path / "must-not-exist.json",
        )

    _assert_error(captured, "plan.invalid", repr(document), str(tmp_path))
    assert not (tmp_path / "must-not-exist.json").exists()


def test_a_valid_but_nonexact_mirror_plan_is_rejected(tmp_path: Path) -> None:
    plan = build_popsign_v1_plan()
    payload = plan.model_dump(mode="json", round_trip=True)
    archives = cast(list[dict[str, object]], payload["archives"])
    archives[0]["download_url"] = "https://mirror.invalid/popsign/game/train/hello.tar"
    payload["plan_sha256"] = external_acquisition_plan_digest(payload)

    with pytest.raises(PopSignDatasetError) as captured:
        write_external_acquisition_plan(payload, tmp_path / "plan.json")

    _assert_error(captured, "plan.invalid", "mirror.invalid", str(tmp_path))
    assert not (tmp_path / "plan.json").exists()


def test_plan_builder_sanitizes_packaged_resource_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_source() -> None:
        raise ValueError("private resource detail")

    monkeypatch.setattr(popsign, "load_popsign_source", unavailable_source)

    with pytest.raises(PopSignDatasetError) as captured:
        build_popsign_v1_plan()

    _assert_error(captured, "plan.invalid", "private resource detail")


@pytest.mark.parametrize("failure", ["directory", "parent-file", "link-failure"])
def test_plan_publication_rejects_invalid_destinations_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    plan = build_popsign_v1_plan()
    destination = tmp_path / "plan.json"
    if failure == "directory":
        destination.mkdir()
        expected = "plan.conflict"
    elif failure == "parent-file":
        parent = tmp_path / "not-a-directory"
        parent.write_bytes(b"private")
        destination = parent / "plan.json"
        expected = "publication.failed"
    else:
        expected = "publication.failed"

        def fail_link(
            _source: str | os.PathLike[str],
            _destination: str | os.PathLike[str],
            *,
            follow_symlinks: bool = True,
        ) -> None:
            del follow_symlinks
            raise OSError("private publication failure")

        monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(PopSignDatasetError) as captured:
        write_external_acquisition_plan(plan, destination)

    _assert_error(captured, expected, str(destination), "private publication failure")
    if destination.parent.is_dir():
        assert not tuple(destination.parent.glob(f".{destination.name}.staging-*"))


def test_plan_publication_reconciles_an_identical_create_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_popsign_v1_plan()
    destination = tmp_path / "plan.json"

    def publish_then_report_race(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        del follow_symlinks
        Path(target).write_bytes(Path(source).read_bytes())
        raise FileExistsError

    monkeypatch.setattr(os, "link", publish_then_report_race)

    result = write_external_acquisition_plan(plan, destination)

    assert result.status == "unchanged"
    assert destination.read_bytes() == render_external_dataset_json(plan).encode("utf-8")
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))


def test_invalid_archive_roots_and_destinations_fail_before_scanning(
    tmp_path: Path,
) -> None:
    plan = build_popsign_v1_plan()
    missing_root = tmp_path / "private-missing-root"
    with pytest.raises(PopSignDatasetError) as missing:
        import_popsign_v1_archives(
            plan,
            archive_root=missing_root,
            destination=tmp_path / "bundle",
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )
    _assert_error(missing, "archive.root_invalid", str(missing_root))

    root_file = tmp_path / "private-root-file"
    root_file.write_bytes(b"private")
    with pytest.raises(PopSignDatasetError) as file_root:
        import_popsign_v1_archives(
            plan,
            archive_root=root_file,
            destination=tmp_path / "bundle",
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )
    _assert_error(file_root, "archive.root_invalid", str(root_file))

    valid_root = tmp_path / "valid-root"
    valid_root.mkdir()
    invalid_destinations = [valid_root / "nested-bundle", tmp_path / "bundle-file"]
    invalid_destinations[1].write_bytes(b"private")
    parent_file = tmp_path / "parent-file"
    parent_file.write_bytes(b"private")
    invalid_destinations.append(parent_file / "bundle")
    for destination in invalid_destinations:
        with pytest.raises(PopSignDatasetError) as invalid_destination:
            import_popsign_v1_archives(
                plan,
                archive_root=valid_root,
                destination=destination,
                accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
            )
        _assert_error(
            invalid_destination,
            "destination.invalid",
            str(destination),
            "private",
        )


@pytest.mark.parametrize(
    ("defect", "category"),
    [
        ("directory", "archive.bytes_invalid"),
        ("empty", "archive.bytes_invalid"),
        ("malformed", "archive.structure_invalid"),
        ("directory-only", "archive.member_invalid"),
    ],
)
def test_import_rejects_invalid_archive_files(
    tmp_path: Path,
    defect: str,
    category: str,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    _write_corpus(archive_root, plan)
    first = _archive_path(archive_root, plan.archives[0])
    first.unlink()
    if defect == "directory":
        first.mkdir()
    elif defect == "empty":
        first.touch()
    elif defect == "malformed":
        first.write_bytes(b"this is not an uncompressed tar archive")
    else:
        _write_tar(first, (_TarMember("clips/", kind=tarfile.DIRTYPE),))

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=tmp_path / "bundle",
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, category, str(first))


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("clips//signer--recording-.mp4", "archive.structure_invalid"),
        ("clips/\x01signer--recording-.mp4", "archive.structure_invalid"),
        ("signér--recording-.mp4", "archive.member_invalid"),
        ("signer--recording--extra-.mp4", "archive.member_invalid"),
    ],
)
def test_import_rejects_ambiguous_control_unicode_and_provider_names(
    tmp_path: Path,
    name: str,
    category: str,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    _write_corpus(
        archive_root,
        plan,
        overrides={plan.archives[0].archive_id: (_TarMember(name),)},
    )

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=tmp_path / "bundle",
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, category, name)


def test_identical_media_bytes_are_stored_once_but_keep_distinct_records(
    tmp_path: Path,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    shared = b"same-synthetic-video-bytes"
    first = plan.archives[0]
    _write_corpus(
        archive_root,
        plan,
        overrides={
            first.archive_id: (
                _TarMember("signer-train--recording-one-.mp4", shared),
                _TarMember("signer-train--recording-two-.mp4", shared),
            )
        },
    )
    bundle = tmp_path / "bundle"

    imported = import_popsign_v1_archives(
        plan,
        archive_root=archive_root,
        destination=bundle,
        accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
    )
    validated = validate_external_dataset_bundle(imported.manifest, bundle)

    assert len(imported.manifest.media) == 16
    first_archive_media = [
        item for item in imported.manifest.media if item.archive_id == first.archive_id
    ]
    assert len(first_archive_media) == 2
    assert len({item.sample_id for item in first_archive_media}) == 2
    assert len({item.locator.path for item in first_archive_media}) == 1
    assert len(list((bundle / "media").rglob("*.mp4"))) == 15
    assert validated.media_count == 16


@pytest.mark.parametrize(
    "mutation",
    ["dataset-id", "archive-locator", "target-mapping"],
)
def test_validation_rejects_contract_valid_manifests_not_bound_to_this_adapter(
    tmp_path: Path,
    mutation: str,
) -> None:
    _plan, _archives, bundle, imported = _import_fixture(tmp_path)
    payload = imported.manifest.model_dump(mode="json", round_trip=True)
    if mutation == "dataset-id":
        payload["dataset_id"] = "different-dataset"
    elif mutation == "archive-locator":
        archives = cast(list[dict[str, object]], payload["archives"])
        archives[0]["local_archive"] = {
            "kind": "workspace_relative",
            "path": "archives/game/train/different.tar",
        }
    else:
        media = cast(list[dict[str, object]], payload["media"])
        hello = next(item for item in media if item["source_label"] == "hello")
        hello["target_label_id"] = "no"
    _redigest_manifest(payload)

    with pytest.raises(PopSignDatasetError) as captured:
        validate_external_dataset_bundle(payload, bundle)

    _assert_error(captured, "manifest.invalid", mutation, str(bundle))


@pytest.mark.parametrize("document_kind", ["invalid", "oversized-bytes", "oversized-text"])
def test_manifest_documents_are_bounded_and_strictly_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_kind: str,
) -> None:
    if document_kind == "invalid":
        document: object = b"not-json"
    elif document_kind == "oversized-bytes":
        monkeypatch.setattr(popsign, "_MAX_MANIFEST_BYTES", 4)
        document = b"12345"
    else:
        monkeypatch.setattr(popsign, "_MAX_MANIFEST_BYTES", 4)
        document = "12345"

    with pytest.raises(PopSignDatasetError) as captured:
        validate_external_dataset_bundle(
            cast(
                ExternalDatasetManifestV1 | str | bytes | bytearray | Mapping[str, object], document
            ),
            tmp_path / "unused-private-root",
        )

    _assert_error(captured, "manifest.invalid", repr(document), str(tmp_path))


@pytest.mark.parametrize(
    ("defect", "category"),
    [
        ("missing-root", "bundle.inventory_invalid"),
        ("root-file", "bundle.inventory_invalid"),
        ("missing-manifest", "manifest.invalid"),
        ("oversized-manifest", "manifest.invalid"),
        ("missing-media", "media.bytes_invalid"),
        ("media-directory", "media.bytes_invalid"),
        ("linked-media", "media.bytes_invalid"),
        ("linked-directory", "bundle.inventory_invalid"),
    ],
)
def test_validation_rejects_invalid_bundle_roots_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    category: str,
) -> None:
    _plan, _archives, bundle, imported = _import_fixture(tmp_path)
    validation_root = bundle
    first_media = bundle.joinpath(*imported.manifest.media[0].locator.path.split("/"))
    if defect == "missing-root":
        validation_root = tmp_path / "missing-private-bundle"
    elif defect == "root-file":
        validation_root = tmp_path / "private-bundle-file"
        validation_root.write_bytes(b"private")
    elif defect == "missing-manifest":
        (bundle / EXTERNAL_DATASET_MANIFEST_FILENAME).unlink()
    elif defect == "oversized-manifest":
        size = (bundle / EXTERNAL_DATASET_MANIFEST_FILENAME).stat().st_size
        monkeypatch.setattr(popsign, "_MAX_MANIFEST_BYTES", size - 1)
    elif defect == "missing-media":
        first_media.unlink()
    elif defect == "media-directory":
        first_media.unlink()
        first_media.mkdir()
    elif defect == "linked-media":
        original = popsign._is_linklike
        monkeypatch.setattr(
            popsign,
            "_is_linklike",
            lambda path: path == first_media or original(path),
        )
    else:
        linked = bundle / "linked-private-directory"
        linked.mkdir()
        original = popsign._is_linklike
        monkeypatch.setattr(
            popsign,
            "_is_linklike",
            lambda path: path == linked or original(path),
        )

    with pytest.raises(PopSignDatasetError) as captured:
        validate_external_dataset_bundle(imported.manifest, validation_root)

    _assert_error(captured, category, str(validation_root), "private")


@pytest.mark.parametrize("conflict", ["different-manifest", "corrupt-media", "linked-manifest"])
def test_reimport_rejects_every_nonidentical_or_invalid_existing_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conflict: str,
) -> None:
    plan, archive_root, bundle, imported = _import_fixture(tmp_path)
    if conflict == "different-manifest":
        first = plan.archives[0]
        normal = _normal_member(first)
        _write_tar(
            _archive_path(archive_root, first),
            (_TarMember(normal.name, b"changed-source-video"),),
        )
    elif conflict == "corrupt-media":
        media = bundle.joinpath(*imported.manifest.media[0].locator.path.split("/"))
        media.write_bytes(b"corrupt-existing-video")
    else:
        manifest_path = bundle / EXTERNAL_DATASET_MANIFEST_FILENAME
        original = popsign._is_linklike
        monkeypatch.setattr(
            popsign,
            "_is_linklike",
            lambda path: path == manifest_path or original(path),
        )

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=bundle,
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, "destination.conflict", conflict, str(bundle))
    assert not tuple(tmp_path.glob(f".{bundle.name}.staging-*"))


@pytest.mark.parametrize("failure", ["manifest-render", "media-publish", "bundle-rename"])
def test_import_publication_failures_leave_no_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    destination = tmp_path / "bundle"
    _write_corpus(archive_root, plan)
    if failure == "manifest-render":
        original_render = render_external_dataset_json

        def fail_manifest_render(document: BaseModel | Mapping[str, object]) -> str:
            if isinstance(document, ExternalDatasetManifestV1):
                raise ValueError("private manifest render failure")
            return original_render(document)

        monkeypatch.setattr(
            "signlab.datasets.popsign.render_external_dataset_json",
            fail_manifest_render,
        )
        expected = "publication.failed"
    elif failure == "media-publish":

        def fail_replace(
            _source: str | os.PathLike[str],
            _destination: str | os.PathLike[str],
        ) -> None:
            raise OSError("private media publish failure")

        monkeypatch.setattr(os, "replace", fail_replace)
        expected = "archive.member_invalid"
    else:
        original_rename = Path.rename

        def fail_bundle_rename(
            path: Path,
            target: str | os.PathLike[str],
        ) -> Path:
            if path.name.startswith(f".{destination.name}.staging-"):
                raise OSError("private bundle rename failure")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_bundle_rename)
        expected = "publication.failed"

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=destination,
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, expected, str(destination), "private")
    assert not destination.exists()
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))


def test_failed_publish_restores_a_preexisting_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    destination = tmp_path / "bundle"
    _write_corpus(archive_root, plan)
    destination.mkdir()
    original_rename = Path.rename

    def fail_bundle_rename(path: Path, target: str | os.PathLike[str]) -> Path:
        if path.name.startswith(f".{destination.name}.staging-"):
            raise OSError("private bundle rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_bundle_rename)

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=destination,
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, "publication.failed", str(destination), "private")
    assert destination.is_dir()
    assert not tuple(destination.iterdir())
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))


def test_archive_symlink_is_rejected_when_the_platform_allows_symlinks(
    tmp_path: Path,
) -> None:
    plan = build_popsign_v1_plan()
    archive_root = tmp_path / "archives"
    _write_corpus(archive_root, plan)
    first = _archive_path(archive_root, plan.archives[0])
    target = first.with_name("private-target.tar")
    first.rename(target)
    try:
        first.symlink_to(target.name)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(PopSignDatasetError) as captured:
        import_popsign_v1_archives(
            plan,
            archive_root=archive_root,
            destination=tmp_path / "bundle",
            accept_license=POPSIGN_LICENSE_ACKNOWLEDGEMENT,
        )

    _assert_error(captured, "archive.bytes_invalid", str(first), "private-target")
