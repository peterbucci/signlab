from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from signlab.contracts.core import (
    ArtifactRefV1,
    ArtifactUriLocatorV1,
    WorkspaceRelativeLocatorV1,
)
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


def test_stream_verifier_checks_size_and_sha256_for_ordered_portable_references(
    tmp_path: Path,
) -> None:
    first_payload = b"first synthetic artifact"
    second_payload = b"second synthetic artifact"
    first = _reference("artifact_a", first_payload)
    second = _reference("artifact_b", second_payload)
    _write(tmp_path, first, first_payload)
    _write(tmp_path, second, second_payload)

    result = verify_artifact_references((first, second), tmp_path)

    assert result.artifact_byte_integrity == "verified"
    assert result.artifacts_verified == 2
    assert result.total_bytes_verified == len(first_payload) + len(second_payload)


@pytest.mark.parametrize("failure", ["missing", "size", "sha256"])
def test_missing_or_mismatched_bytes_fail_without_disclosing_the_locator(
    tmp_path: Path,
    failure: str,
) -> None:
    payload = b"private sentinel bytes"
    reference = _reference("artifact_private_sentinel", payload)
    if failure != "missing":
        _write(tmp_path, reference, payload)
    if failure == "size":
        reference = _reference(
            "artifact_private_sentinel",
            payload,
            size_bytes=len(payload) + 1,
        )
    elif failure == "sha256":
        reference = _reference(
            "artifact_private_sentinel",
            payload,
            sha256="sha256:" + "0" * 64,
        )

    with pytest.raises(DatasetStorageError) as captured:
        verify_artifact_references((reference,), tmp_path)

    assert captured.value.code == "dataset.storage.artifact_bytes.invalid"
    assert "private" not in str(captured.value)
    assert str(tmp_path) not in str(captured.value)


def test_reference_set_must_be_nonempty_sorted_unique_and_workspace_relative(
    tmp_path: Path,
) -> None:
    first = _reference("artifact_a", b"first")
    second = _reference("artifact_b", b"second")
    assert isinstance(first.locator, WorkspaceRelativeLocatorV1)
    duplicate_path = _reference("artifact_b", b"second", path=first.locator.path)
    logical = first.model_copy(
        update={
            "locator": ArtifactUriLocatorV1(
                kind="artifact_uri",
                uri="signlab://objects/sha256/fixture",
            )
        }
    )

    for references in ((), (second, first), (first, duplicate_path), (logical,)):
        with pytest.raises(DatasetStorageError):
            verify_artifact_references(references, tmp_path)


def test_workspace_escape_through_a_link_is_rejected_when_links_are_available(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    payload = b"synthetic artifact"
    reference = _reference("artifact_a", payload)
    locator = reference.locator
    assert isinstance(locator, WorkspaceRelativeLocatorV1)
    candidate = root.joinpath(*locator.path.split("/"))
    candidate.parent.mkdir(parents=True)
    outside = tmp_path / "outside-artifact"
    outside.write_bytes(payload)
    try:
        candidate.symlink_to(outside)
    except OSError:
        pytest.skip("file links are unavailable for this account")

    with pytest.raises(DatasetStorageError):
        verify_artifact_references((reference,), root)


def test_example_dataset_inventory_is_sorted_unique_and_portable() -> None:
    references = collect_row_artifact_references(build_example_dataset_bundle().tables)

    assert references
    assert tuple(reference.artifact_id for reference in references) == tuple(
        sorted({reference.artifact_id for reference in references})
    )
    assert all(
        isinstance(reference.locator, WorkspaceRelativeLocatorV1) for reference in references
    )
