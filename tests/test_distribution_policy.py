from __future__ import annotations

import io
import stat
import subprocess
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest
from scripts import verify_distribution as distribution_verifier
from scripts.verify_distribution import (
    EXPECTED_CONTRACT_RESOURCES,
    EXPECTED_DATASET_RESOURCES,
    EXPECTED_EXTRACTION_RESOURCES,
    EXPECTED_GOVERNANCE_RESOURCES,
    EXPECTED_TAXONOMY_SCHEMAS,
    _inspect_sdist,
    _inspect_wheel,
    validate_member_names,
    verify_distribution,
)


def _write_sdist(
    path: Path,
    governance_resources: Iterable[str],
    contract_resources: Iterable[str] = EXPECTED_CONTRACT_RESOURCES,
    dataset_resources: Iterable[str] = EXPECTED_DATASET_RESOURCES,
    extraction_resources: Iterable[str] = EXPECTED_EXTRACTION_RESOURCES,
    extra_members: Iterable[tarfile.TarInfo] = (),
) -> None:
    archive_root = path.name.removesuffix(".tar.gz")
    member_names = [
        f"{archive_root}/src/signlab/py.typed",
        f"{archive_root}/src/signlab/resources/contracts/__init__.py",
        f"{archive_root}/src/signlab/resources/datasets/__init__.py",
        f"{archive_root}/src/signlab/resources/extraction/__init__.py",
        f"{archive_root}/src/signlab/resources/governance/__init__.py",
        *(
            f"{archive_root}/src/signlab/resources/governance/{resource}"
            for resource in governance_resources
        ),
        *(
            f"{archive_root}/src/signlab/resources/contracts/{resource}"
            for resource in contract_resources
        ),
        *(
            f"{archive_root}/src/signlab/resources/datasets/{resource}"
            for resource in dataset_resources
        ),
        *(
            f"{archive_root}/src/signlab/resources/extraction/{resource}"
            for resource in extraction_resources
        ),
    ]
    with tarfile.open(path, mode="w:gz") as archive:
        for member_name in sorted(member_names):
            payload = b"synthetic\n"
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        for member in extra_members:
            archive.addfile(member)


def _write_wheel(
    path: Path,
    governance_resources: Iterable[str],
    contract_resources: Iterable[str] = EXPECTED_CONTRACT_RESOURCES,
    dataset_resources: Iterable[str] = EXPECTED_DATASET_RESOURCES,
    extraction_resources: Iterable[str] = EXPECTED_EXTRACTION_RESOURCES,
) -> None:
    member_names = [
        "signlab/py.typed",
        "signlab/commands/__init__.py",
        "signlab/resources/contracts/__init__.py",
        "signlab/resources/datasets/__init__.py",
        "signlab/resources/extraction/__init__.py",
        "signlab/resources/governance/__init__.py",
        "signlab/resources/taxonomies/signlab-five-1.0.0.json",
        *(f"signlab/resources/schemas/{name}" for name in EXPECTED_TAXONOMY_SCHEMAS),
        *(f"signlab/resources/governance/{name}" for name in governance_resources),
        *(f"signlab/resources/contracts/{name}" for name in contract_resources),
        *(f"signlab/resources/datasets/{name}" for name in dataset_resources),
        *(f"signlab/resources/extraction/{name}" for name in extraction_resources),
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
    assert validate_member_names(("signlab/models/hand_landmarker.task",)) == (
        "archive contains a private or generated artifact",
    )


@pytest.mark.parametrize(
    "member_name",
    [
        ".dvcignore",
        "dvc.lock",
        "dvc.yaml",
        "params.yaml",
        "private-data.dvc",
        "signlab-0.1.0/.dvc/config",
    ],
)
def test_distribution_policy_rejects_every_dvc_data_metadata_form(member_name: str) -> None:
    assert validate_member_names((member_name,)) == (
        "archive contains a private or generated artifact",
    )


@pytest.mark.parametrize(
    "member_name",
    [
        f"{chr(67)}:/outside.txt",
        r"..\outside.txt",
        r"\\server\share\outside.txt",
        "signlab//module.py",
        "signlab/./module.py",
        "signlab/CON.py",
        "signlab/module.py.",
        "signlab/module.py ",
        "signlab/module" + chr(10) + ".py",
        "signlab/module" + chr(127) + ".py",
    ],
)
def test_distribution_policy_rejects_nonportable_windows_and_noncanonical_paths(
    member_name: str,
) -> None:
    assert validate_member_names((member_name,)) == ("archive contains a non-portable member path",)


@pytest.mark.parametrize(
    "member_names",
    [
        ("signlab/module.py", "signlab/module.py"),
        ("signlab/Module.py", "signlab/module.py"),
        ("signlab/package", "signlab/package/"),
    ],
)
def test_distribution_policy_rejects_duplicate_and_case_colliding_paths(
    member_names: tuple[str, str],
) -> None:
    assert validate_member_names(member_names) == (
        "archive contains duplicate or case-colliding member paths",
    )


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_sdist_rejects_link_members(
    tmp_path: Path,
    member_type: bytes,
) -> None:
    sdist = tmp_path / "signlab-0.1.0.tar.gz"
    archive_root = sdist.name.removesuffix(".tar.gz")
    link = tarfile.TarInfo(f"{archive_root}/src/signlab/linked-module.py")
    link.type = member_type
    link.linkname = f"{archive_root}/src/signlab/__init__.py"
    _write_sdist(
        sdist,
        EXPECTED_GOVERNANCE_RESOURCES,
        extra_members=(link,),
    )

    assert _inspect_sdist(sdist) == ("source distribution contains a non-regular member",)


@pytest.mark.parametrize("member_name", ["signlab/linked-module.py", "signlab/linked-directory/"])
def test_wheel_rejects_symbolic_link_members(tmp_path: Path, member_name: str) -> None:
    wheel = tmp_path / "signlab-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, EXPECTED_GOVERNANCE_RESOURCES)
    link = zipfile.ZipInfo(member_name)
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(wheel, mode="a") as archive:
        archive.writestr(link, b"__init__.py")

    assert _inspect_wheel(wheel) == ("wheel contains a non-regular member",)


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
        "archive contains duplicate or case-colliding member paths",
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
        "archive contains duplicate or case-colliding member paths",
        "wheel does not contain the exact participant-governance resource set",
    )


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_archives_require_the_exact_nonduplicate_contract_resource_set(
    tmp_path: Path,
    archive_kind: str,
    mutation: str,
) -> None:
    resources = sorted(EXPECTED_CONTRACT_RESOURCES)
    if mutation == "missing":
        resources.remove("schemas/run-record-1.schema.json")
    elif mutation == "extra":
        resources.append("schemas/unexpected-1.schema.json")
    else:
        resources.append("schemas/run-record-1.schema.json")

    if archive_kind == "wheel":
        archive = tmp_path / "signlab-0.1.0-py3-none-any.whl"
        if mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                _write_wheel(archive, EXPECTED_GOVERNANCE_RESOURCES, resources)
        else:
            _write_wheel(archive, EXPECTED_GOVERNANCE_RESOURCES, resources)
        errors = _inspect_wheel(archive)
        expected_error = "wheel does not contain the exact pipeline-contract resource set"
    else:
        archive = tmp_path / "signlab-0.1.0.tar.gz"
        _write_sdist(archive, EXPECTED_GOVERNANCE_RESOURCES, resources)
        errors = _inspect_sdist(archive)
        expected_error = (
            "source distribution does not contain the exact pipeline-contract resource set"
        )

    expected_errors: tuple[str, ...] = (expected_error,)
    if mutation == "duplicate":
        expected_errors = (
            "archive contains duplicate or case-colliding member paths",
            expected_error,
        )
    assert errors == expected_errors


def test_archives_accept_the_exact_contract_resource_set(tmp_path: Path) -> None:
    wheel = tmp_path / "signlab-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "signlab-0.1.0.tar.gz"
    _write_wheel(wheel, EXPECTED_GOVERNANCE_RESOURCES)
    _write_sdist(sdist, EXPECTED_GOVERNANCE_RESOURCES)

    assert _inspect_wheel(wheel) == ()
    assert _inspect_sdist(sdist) == ()


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_archives_require_the_exact_nonduplicate_dataset_resource_set(
    tmp_path: Path,
    archive_kind: str,
    mutation: str,
) -> None:
    resources = sorted(EXPECTED_DATASET_RESOURCES)
    if mutation == "missing":
        resources.remove("schemas/participants-table-1.schema.json")
    elif mutation == "extra":
        resources.append("schemas/unexpected-table-1.schema.json")
    else:
        resources.append("schemas/participants-table-1.schema.json")

    if archive_kind == "wheel":
        archive = tmp_path / "signlab-0.1.0-py3-none-any.whl"
        if mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                _write_wheel(
                    archive,
                    EXPECTED_GOVERNANCE_RESOURCES,
                    EXPECTED_CONTRACT_RESOURCES,
                    resources,
                )
        else:
            _write_wheel(
                archive,
                EXPECTED_GOVERNANCE_RESOURCES,
                EXPECTED_CONTRACT_RESOURCES,
                resources,
            )
        errors = _inspect_wheel(archive)
        expected_error = "wheel does not contain the exact dataset resource set"
    else:
        archive = tmp_path / "signlab-0.1.0.tar.gz"
        _write_sdist(
            archive,
            EXPECTED_GOVERNANCE_RESOURCES,
            EXPECTED_CONTRACT_RESOURCES,
            resources,
        )
        errors = _inspect_sdist(archive)
        expected_error = "source distribution does not contain the exact dataset resource set"

    expected_errors: tuple[str, ...] = (expected_error,)
    if mutation == "duplicate":
        expected_errors = (
            "archive contains duplicate or case-colliding member paths",
            expected_error,
        )
    assert errors == expected_errors


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_archives_require_the_exact_nonduplicate_extraction_resource_set(
    tmp_path: Path,
    archive_kind: str,
    mutation: str,
) -> None:
    resources = sorted(EXPECTED_EXTRACTION_RESOURCES)
    if mutation == "missing":
        resources.remove("models/mediapipe-tasks-1.0.1.lock.json")
    elif mutation == "extra":
        resources.append("models/unexpected.lock.json")
    else:
        resources.append("config/mediapipe-extraction-config-1.default.json")

    if archive_kind == "wheel":
        archive = tmp_path / "signlab-0.1.0-py3-none-any.whl"
        if mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                _write_wheel(
                    archive,
                    EXPECTED_GOVERNANCE_RESOURCES,
                    extraction_resources=resources,
                )
        else:
            _write_wheel(
                archive,
                EXPECTED_GOVERNANCE_RESOURCES,
                extraction_resources=resources,
            )
        errors = _inspect_wheel(archive)
        expected_error = "wheel does not contain the exact extraction resource set"
    else:
        archive = tmp_path / "signlab-0.1.0.tar.gz"
        _write_sdist(
            archive,
            EXPECTED_GOVERNANCE_RESOURCES,
            extraction_resources=resources,
        )
        errors = _inspect_sdist(archive)
        expected_error = "source distribution does not contain the exact extraction resource set"

    expected_errors: tuple[str, ...] = (expected_error,)
    if mutation == "duplicate":
        expected_errors = (
            "archive contains duplicate or case-colliding member paths",
            expected_error,
        )
    assert errors == expected_errors


def test_isolated_smoke_tests_remove_pythonpath_and_use_temporary_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = tmp_path / "signlab-0.1.0-py3-none-any.whl"
    calls: list[tuple[tuple[str, ...], dict[str, str], Path | None]] = []

    def fake_run(
        command: Iterable[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        normalized_command = tuple(command)
        calls.append((normalized_command, dict(environment), cwd))
        stdout = "0.1.0\n" if "--version" in normalized_command else ""
        return subprocess.CompletedProcess(normalized_command, 0, stdout, "")

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "checkout"))
    monkeypatch.setattr(distribution_verifier, "_run", fake_run)

    distribution_verifier._install_and_smoke_test(distribution)

    assert calls
    assert all("PYTHONPATH" not in environment for _, environment, _ in calls)
    working_directories = {cwd for _, _, cwd in calls}
    assert len(working_directories) == 1
    assert None not in working_directories
    commands = {command[-2:] for command, _, _ in calls if len(command) >= 2}
    assert ("contracts", "--help") in commands
    assert ("contracts", "validate-resources") in commands
    assert ("data", "validate-resources") in commands


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
