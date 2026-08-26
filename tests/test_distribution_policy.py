from __future__ import annotations

import io
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest
from scripts import verify_distribution as distribution_verifier
from scripts.verify_distribution import (
    EXPECTED_GOVERNANCE_RESOURCES,
    EXPECTED_TAXONOMY_SCHEMAS,
    _inspect_sdist,
    _inspect_wheel,
    validate_member_names,
    verify_distribution,
)


def _write_sdist(path: Path, governance_resources: Iterable[str]) -> None:
    archive_root = path.name.removesuffix(".tar.gz")
    member_names = [
        f"{archive_root}/src/signlab/py.typed",
        f"{archive_root}/src/signlab/resources/governance/__init__.py",
        *(
            f"{archive_root}/src/signlab/resources/governance/{resource}"
            for resource in governance_resources
        ),
    ]
    with tarfile.open(path, mode="w:gz") as archive:
        for member_name in sorted(member_names):
            payload = b"synthetic\n"
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def _write_wheel(path: Path, governance_resources: Iterable[str]) -> None:
    member_names = [
        "signlab/py.typed",
        "signlab/commands/__init__.py",
        "signlab/resources/governance/__init__.py",
        "signlab/resources/taxonomies/signlab-five-1.0.0.json",
        *(f"signlab/resources/schemas/{name}" for name in EXPECTED_TAXONOMY_SCHEMAS),
        *(f"signlab/resources/governance/{name}" for name in governance_resources),
        "signlab-0.1.0.dist-info/METADATA",
        "signlab-0.1.0.dist-info/entry_points.txt",
    ]
    with zipfile.ZipFile(path, mode="w") as archive:
        for member_name in sorted(member_names):
            payload = (
                b"[console_scripts]\nsignlab = signlab.cli:main\n"
                if member_name.endswith("entry_points.txt")
                else b"synthetic\n"
            )
            archive.writestr(member_name, payload)


def test_distribution_policy_allows_only_portable_package_members() -> None:
    members = (
        "signlab/__init__.py",
        "signlab/commands/data.py",
        "signlab/data/__init__.py",
        "signlab/models/tcn.py",
        "signlab-0.1.0/src/signlab/artifacts/metadata.py",
        "signlab-0.1.0.dist-info/METADATA",
    )

    assert validate_member_names(members) == ()


def test_distribution_policy_rejects_traversal_and_private_artifacts() -> None:
    assert validate_member_names(("../secret.txt",)) == (
        "archive contains a non-portable member path",
    )
    assert validate_member_names(("signlab/data/champion.onnx",)) == (
        "archive contains a private or generated artifact",
    )
    assert validate_member_names(("signlab-0.1.0/data/raw/participant.json",)) == (
        "archive contains a private or generated artifact",
    )
    assert validate_member_names(("models/champion.json",)) == (
        "archive contains a private or generated artifact",
    )


def test_sdist_requires_the_exact_governance_resource_set(tmp_path: Path) -> None:
    sdist = tmp_path / "signlab-0.1.0.tar.gz"
    missing = set(EXPECTED_GOVERNANCE_RESOURCES)
    missing.remove("evidence/withdrawal-dry-run-v1.json")
    _write_sdist(sdist, missing)

    assert _inspect_sdist(sdist) == (
        "source distribution does not contain the exact participant-governance resource set",
    )


def test_sdist_accepts_the_exact_governance_resource_set(tmp_path: Path) -> None:
    sdist = tmp_path / "signlab-0.1.0.tar.gz"
    _write_sdist(sdist, set(EXPECTED_GOVERNANCE_RESOURCES))

    assert _inspect_sdist(sdist) == ()


def test_sdist_rejects_duplicate_governance_resources(tmp_path: Path) -> None:
    sdist = tmp_path / "signlab-0.1.0.tar.gz"
    resources = [
        *sorted(EXPECTED_GOVERNANCE_RESOURCES),
        "evidence/withdrawal-dry-run-v1.json",
    ]
    _write_sdist(sdist, resources)

    assert _inspect_sdist(sdist) == (
        "source distribution does not contain the exact participant-governance resource set",
    )


def test_wheel_rejects_duplicate_governance_resources(tmp_path: Path) -> None:
    wheel = tmp_path / "signlab-0.1.0-py3-none-any.whl"
    resources = [
        *sorted(EXPECTED_GOVERNANCE_RESOURCES),
        "evidence/withdrawal-dry-run-v1.json",
    ]
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_wheel(wheel, resources)

    assert _inspect_wheel(wheel) == (
        "wheel does not contain the exact participant-governance resource set",
    )


def test_distribution_verification_smoke_tests_wheel_and_sdist_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "signlab-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "signlab-0.1.0.tar.gz"
    wheel.write_bytes(b"synthetic wheel")
    sdist.write_bytes(b"synthetic sdist")
    smoke_tested: list[Path] = []

    monkeypatch.setattr(distribution_verifier, "_inspect_wheel", lambda _path: ())
    monkeypatch.setattr(distribution_verifier, "_inspect_sdist", lambda _path: ())
    monkeypatch.setattr(distribution_verifier, "_install_and_smoke_test", smoke_tested.append)

    assert verify_distribution(tmp_path) == ()
    assert smoke_tested == [wheel, sdist]
