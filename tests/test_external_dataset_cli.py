from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.datasets import popsign, public_corpus


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def _validation(*, archive_bytes: str = "verified") -> SimpleNamespace:
    return SimpleNamespace(
        content_sha256="sha256:" + "a" * 64,
        archive_count=15,
        media_count=30,
        semantic_integrity="verified",
        media_byte_integrity="verified",
        archive_byte_integrity=archive_bytes,
        license_authorization="verified",
    )


def test_plan_external_dataset_writes_reviewed_plan_without_network_output(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "operator-private-plan.json"
    plan = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(popsign, "build_popsign_v1_plan", lambda: plan)

    def write(document: object, destination: Path) -> object:
        calls.append((document, destination))
        return SimpleNamespace(
            status="published",
            archive_count=15,
            plan_sha256="sha256:" + "b" * 64,
        )

    monkeypatch.setattr(popsign, "write_external_acquisition_plan", write)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "plan-external-dataset",
            "popsign-asl-v1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert calls == [(plan, output)]
    assert result.output.splitlines() == [
        "External dataset plan: published.",
        "Registered source: PopSign ASL v1.0.",
        "Planned archives: 15",
        "Acquisition plan SHA-256: sha256:" + "b" * 64,
        "Network access: not used.",
    ]
    assert output.name not in result.output
    assert str(tmp_path) not in result.output


def test_import_popsign_passes_explicit_license_and_reports_aggregate_evidence(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "private-plan.json"
    plan.write_bytes(b"reviewed-plan")
    archive_root = tmp_path / "identifiable-archives"
    destination = tmp_path / "licensed-bundle"
    calls: list[tuple[bytes, Path, Path, str]] = []

    def import_archives(
        document: bytes,
        *,
        archive_root: Path,
        destination: Path,
        accept_license: str,
    ) -> object:
        calls.append((document, archive_root, destination, accept_license))
        return SimpleNamespace(status="published", validation=_validation())

    monkeypatch.setattr(popsign, "import_popsign_v1_archives", import_archives)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "import-popsign",
            str(plan),
            "--archive-root",
            str(archive_root),
            "--output",
            str(destination),
            "--accept-license",
            "CC-BY-4.0",
        ],
    )

    assert result.exit_code == 0
    assert calls == [(b"reviewed-plan", archive_root, destination, "CC-BY-4.0")]
    assert result.output.splitlines() == [
        "External dataset: published.",
        "External data SHA-256: sha256:" + "a" * 64,
        "Licensed archives: 15",
        "Imported media: 30",
        "Dataset semantics: verified",
        "Imported media bytes: verified",
        "Original archive bytes: verified",
        "License authorization: verified",
        "SignLab participant consent: not applicable to licensed public data.",
    ]
    assert str(tmp_path) not in result.output
    assert "private-plan" not in result.output
    assert "identifiable-archives" not in result.output


@pytest.mark.parametrize("with_archive_root", [False, True])
def test_validate_external_dataset_reports_whether_original_archives_were_checked(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_archive_root: bool,
) -> None:
    manifest = tmp_path / "external-dataset-manifest.json"
    manifest.write_bytes(b"external-manifest")
    workspace_root = tmp_path / "external-bundle"
    archive_root = tmp_path / "private-archives" if with_archive_root else None
    calls: list[tuple[bytes, Path, Path | None]] = []

    def validate(
        document: bytes,
        root: Path,
        *,
        archive_root: Path | None,
    ) -> object:
        calls.append((document, root, archive_root))
        return _validation(archive_bytes="verified" if archive_root is not None else "not_checked")

    monkeypatch.setattr(popsign, "validate_external_dataset_bundle", validate)
    arguments = [
        "data",
        "validate-external-dataset",
        str(manifest),
        "--workspace-root",
        str(workspace_root),
    ]
    if archive_root is not None:
        arguments.extend(["--archive-root", str(archive_root)])

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert calls == [(b"external-manifest", workspace_root, archive_root)]
    expected_archive_status = "verified" if with_archive_root else "not checked"
    assert f"Original archive bytes: {expected_archive_status}" in result.output
    assert "External dataset: verified." in result.output
    assert str(tmp_path) not in result.output


def test_build_public_corpus_delegates_and_reports_aggregate_progress(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "private-manifest.json"
    external_root = tmp_path / "private-external"
    archive_root = tmp_path / "private-archives"
    model_root = tmp_path / "private-models"
    output = tmp_path / "private-output"
    calls: list[tuple[Path, Path, Path, Path, Path | None, int, bool]] = []

    def build(
        manifest_path: Path,
        *,
        external_root: Path,
        model_root: Path,
        output_root: Path,
        archive_root: Path | None,
        max_candidates_per_group: int,
        trainable_smoke: bool,
        progress: Callable[[int, int, bool], None],
    ) -> object:
        calls.append(
            (
                manifest_path,
                external_root,
                model_root,
                output_root,
                archive_root,
                max_candidates_per_group,
                trainable_smoke,
            )
        )
        progress(1, 2, True)
        progress(2, 2, False)
        return SimpleNamespace(
            selected_count=1,
            group_count=2,
            exclusion_count=3,
            attempted_count=2,
            target_count=80,
            attempt_limit=750,
            decision="insufficient",
            corpus_sha256="sha256:" + "f" * 64,
        )

    monkeypatch.setattr(public_corpus, "build_public_corpus", build)
    result = runner.invoke(
        cli.app,
        [
            "data",
            "build-public-corpus",
            str(manifest),
            "--external-root",
            str(external_root),
            "--archive-root",
            str(archive_root),
            "--model-root",
            str(model_root),
            "--output",
            str(output),
            "--max-candidates-per-group",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert calls == [(manifest, external_root, model_root, output, archive_root, 2, False)]
    assert "Public corpus groups: 1/2 (latest selected)." in result.output
    assert "Public corpus groups: 2/2 (latest unfilled)." in result.output
    assert "Selected usable clips: 1/2 groups." in result.output
    assert "Coded unselected or unusable clips: 3." in result.output
    assert str(tmp_path) not in result.output

    trainable = runner.invoke(
        cli.app,
        [
            "data",
            "build-public-corpus",
            str(manifest),
            "--external-root",
            str(external_root),
            "--model-root",
            str(model_root),
            "--output",
            str(output),
            "--trainable-smoke",
        ],
    )
    assert trainable.exit_code == 0
    assert calls[-1] == (manifest, external_root, model_root, output, None, 5, True)
    assert "Public corpus attempts: 1/2 (latest selected)." in trainable.output
    assert "Trainable smoke decision: INSUFFICIENT." in trainable.output
    assert "Selected usable clips: 1/80." in trainable.output
    assert "Attempted videos: 2/750." in trainable.output


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            [
                "data",
                "plan-external-dataset",
                "popsign-asl-v1",
                "--output",
                "private-output.json",
            ],
            "External dataset planning failed.",
        ),
        (
            [
                "data",
                "import-popsign",
                "private-plan.json",
                "--archive-root",
                "private-archives",
                "--output",
                "private-bundle",
                "--accept-license",
                "CC-BY-4.0",
            ],
            "External dataset import failed.",
        ),
        (
            [
                "data",
                "validate-external-dataset",
                "private-manifest.json",
                "--workspace-root",
                "private-bundle",
            ],
            "External dataset validation failed.",
        ),
        (
            [
                "data",
                "build-public-corpus",
                "private-manifest.json",
                "--external-root",
                "private-bundle",
                "--model-root",
                "private-models",
                "--output",
                "private-output",
            ],
            "Public corpus build failed.",
        ),
    ],
)
def test_external_dataset_commands_redact_private_failures(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    expected: str,
) -> None:
    secret = str(tmp_path / "named-signer-2026-08-27")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise popsign.PopSignDatasetError("archive.member_invalid") from ValueError(secret)

    monkeypatch.setattr(popsign, "build_popsign_v1_plan", fail)
    monkeypatch.setattr(popsign, "import_popsign_v1_archives", fail)
    monkeypatch.setattr(popsign, "validate_external_dataset_bundle", fail)
    monkeypatch.setattr(public_corpus, "build_public_corpus", fail)

    result = runner.invoke(cli.app, command)

    assert result.exit_code == 1
    assert result.output.strip() == expected
    assert secret not in result.output
    assert "named-signer" not in result.output
    assert "Traceback" not in result.output


def test_external_dataset_usage_errors_are_privacy_safe(runner: CliRunner) -> None:
    result = runner.invoke(
        cli.app,
        [
            "data",
            "plan-external-dataset",
            "private-unknown-source",
            "--output",
            "private-plan.json",
        ],
    )

    assert result.exit_code == 2
    assert result.output.strip() == (
        "Error: invalid command usage; run --help for accepted arguments."
    )
    assert "private" not in result.output
