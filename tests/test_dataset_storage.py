from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from signlab.contracts.core import (
    ArtifactRefV1,
    ArtifactUriLocatorV1,
    WorkspaceRelativeLocatorV1,
)
from signlab.datasets import storage as storage_module
from signlab.datasets.resources import build_example_dataset_bundle
from signlab.datasets.storage import (
    DatasetStorageError,
    collect_row_artifact_references,
    verify_artifact_references,
)


def _reference(
    artifact_id: str,
    payload: bytes,
    *,
    path: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> ArtifactRefV1:
    digest = hashlib.sha256(payload).hexdigest()
    locator_path = path or f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{artifact_id}"
    return ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=artifact_id,
        role="derived_data",
        media_type="application/octet-stream",
        sha256=sha256 or f"sha256:{digest}",
        size_bytes=len(payload) if size_bytes is None else size_bytes,
        locator=WorkspaceRelativeLocatorV1(kind="workspace_relative", path=locator_path),
    )


def _write(root: Path, reference: ArtifactRefV1, payload: bytes) -> Path:
    locator = reference.locator
    assert isinstance(locator, WorkspaceRelativeLocatorV1)
    path = root.joinpath(*locator.path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_stream_verifier_checks_exact_bytes_for_a_canonical_reference_set(tmp_path: Path) -> None:
    first = _reference("artifact_a", b"first-public-fixture")
    second = _reference("artifact_b", b"second-public-fixture")
    _write(tmp_path, first, b"first-public-fixture")
    _write(tmp_path, second, b"second-public-fixture")

    result = verify_artifact_references((first, second), tmp_path)

    assert result.artifact_byte_integrity == "verified"
    assert result.artifacts_verified == 2
    assert result.total_bytes_verified == len(b"first-public-fixturesecond-public-fixture")


@pytest.mark.parametrize("failure", ["missing", "size", "digest", "directory"])
def test_storage_failures_are_stable_and_do_not_echo_paths(
    tmp_path: Path,
    failure: str,
) -> None:
    payload = b"private-sentinel-bytes"
    reference = _reference("artifact_private_sentinel", payload)
    path = tmp_path.joinpath(*reference.locator.path.split("/"))  # type: ignore[union-attr]
    if failure != "missing":
        path.parent.mkdir(parents=True)
        if failure == "directory":
            path.mkdir()
        else:
            path.write_bytes(payload)
    if failure == "size":
        reference = _reference("artifact_private_sentinel", payload, size_bytes=len(payload) + 1)
    elif failure == "digest":
        reference = _reference(
            "artifact_private_sentinel",
            payload,
            sha256="sha256:" + "0" * 64,
        )

    with pytest.raises(DatasetStorageError) as raised:
        verify_artifact_references((reference,), tmp_path)

    assert raised.value.code == "dataset.storage.artifact_bytes.invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert "private" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_reference_inventory_rejects_empty_unsorted_and_colliding_inputs(tmp_path: Path) -> None:
    first = _reference("artifact_a", b"same")
    assert isinstance(first.locator, WorkspaceRelativeLocatorV1)
    second = _reference("artifact_b", b"same", path=first.locator.path)
    for references in ((), (second, first), (first, second)):
        with pytest.raises(DatasetStorageError):
            verify_artifact_references(references, tmp_path)


def test_logical_uri_requires_an_explicit_storage_adapter(tmp_path: Path) -> None:
    reference = _reference("artifact_a", b"fixture").model_copy(
        update={
            "locator": ArtifactUriLocatorV1(
                kind="artifact_uri",
                uri="signlab://objects/sha256/fixture",
            )
        }
    )

    with pytest.raises(DatasetStorageError):
        verify_artifact_references((reference,), tmp_path)


def test_linked_or_hardlinked_artifact_is_rejected_when_supported(tmp_path: Path) -> None:
    payload = b"fixture"
    reference = _reference("artifact_a", payload)
    path = tmp_path.joinpath(*reference.locator.path.split("/"))  # type: ignore[union-attr]
    path.parent.mkdir(parents=True)
    external = tmp_path / "external"
    external.write_bytes(payload)
    try:
        path.symlink_to(external)
    except OSError:
        path.hardlink_to(external)

    with pytest.raises(DatasetStorageError):
        verify_artifact_references((reference,), tmp_path)


def test_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    payload = b"fixture"
    reference = _reference("artifact_a", payload)
    path = tmp_path.joinpath(*reference.locator.path.split("/"))  # type: ignore[union-attr]
    path.parent.mkdir(parents=True)
    external = tmp_path / "external-hardlink"
    external.write_bytes(payload)
    try:
        os.link(external, path)
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")

    with pytest.raises(DatasetStorageError):
        verify_artifact_references((reference,), tmp_path)


@pytest.mark.parametrize("boundary", ["root", "parent", "artifact"])
def test_windows_reparse_points_are_rejected_at_every_storage_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    payload = b"fixture"
    reference = _reference("artifact_a", payload)
    path = _write(tmp_path, reference, payload)
    target = {"root": tmp_path, "parent": path.parent, "artifact": path}[boundary]
    target_status = os.lstat(target)

    def classify(details: os.stat_result) -> bool:
        return details.st_dev == target_status.st_dev and details.st_ino == target_status.st_ino

    monkeypatch.setattr(storage_module, "_is_reparse", classify)

    with pytest.raises(DatasetStorageError):
        verify_artifact_references((reference,), tmp_path)


def test_parent_identity_drift_after_streaming_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"fixture"
    reference = _reference("artifact_a", payload)
    path = _write(tmp_path, reference, payload)
    target_status = os.lstat(path.parent)
    original = storage_module._directory_identity
    calls = 0

    def drifting_identity(details: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        identity = original(details)
        if details.st_dev == target_status.st_dev and details.st_ino == target_status.st_ino:
            calls += 1
            if calls == 2:
                return (*identity, 1)
        return identity

    monkeypatch.setattr(storage_module, "_directory_identity", drifting_identity)

    with pytest.raises(DatasetStorageError):
        verify_artifact_references((reference,), tmp_path)


def test_final_path_identity_drift_after_streaming_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"fixture"
    reference = _reference("artifact_a", payload)
    _write(tmp_path, reference, payload)
    original = storage_module._path_file_identity
    calls = 0

    def drifting_identity(details: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        identity = original(details)
        return (*identity, 1) if calls == 1 else identity

    monkeypatch.setattr(storage_module, "_path_file_identity", drifting_identity)

    with pytest.raises(DatasetStorageError):
        verify_artifact_references((reference,), tmp_path)


def test_example_dataset_exposes_one_sorted_unique_row_artifact_inventory() -> None:
    example = build_example_dataset_bundle()

    references = collect_row_artifact_references(example.tables)

    assert references
    assert tuple(reference.artifact_id for reference in references) == tuple(
        sorted({reference.artifact_id for reference in references})
    )
    assert all(
        isinstance(reference.locator, WorkspaceRelativeLocatorV1) for reference in references
    )
