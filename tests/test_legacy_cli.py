"""CLI behavior for exporting and validating legacy evidence."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.commands import data
from signlab.legacy.exporter import ExportSummary, LegacyExportError
from signlab.legacy.validator import ValidationSummary


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


@pytest.fixture
def export_arguments(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    audit_snapshot = tmp_path / "legacy-state.json"
    audit_snapshot.write_text("{}\n", encoding="utf-8")
    paths = {
        "legacy_root": legacy_root,
        "audit_snapshot": audit_snapshot,
        "public_output": tmp_path / "public",
        "quarantine_output": tmp_path / "private",
    }
    arguments = [
        "data",
        "export-legacy",
        "--legacy-root",
        str(legacy_root),
        "--audit-snapshot",
        str(audit_snapshot),
        "--public-output",
        str(paths["public_output"]),
        "--quarantine-output",
        str(paths["quarantine_output"]),
    ]
    return arguments, paths


def test_export_legacy_reports_counts_and_passes_resolved_paths(
    runner: CliRunner,
    export_arguments: tuple[list[str], dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, expected_paths = export_arguments
    received: dict[str, Path] = {}

    def fake_export(
        *,
        legacy_root: Path,
        audit_snapshot: Path,
        public_output: Path,
        quarantine_output: Path,
    ) -> ExportSummary:
        received.update(
            legacy_root=legacy_root,
            audit_snapshot=audit_snapshot,
            public_output=public_output,
            quarantine_output=quarantine_output,
        )
        return ExportSummary(
            runs=464,
            attempts=532,
            annotations=408,
            detections=45,
            sessions=5,
            promoted_models=4,
            quarantined_segments=529,
            quarantine_objects=535,
        )

    monkeypatch.setattr(data, "export_legacy_evidence", fake_export)

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert received == {name: path.resolve() for name, path in expected_paths.items()}
    assert result.output.strip() == (
        "Legacy export complete: 464 runs, 532 attempts, 4 promoted models, "
        "529 quarantined segments."
    )


def test_export_legacy_returns_a_stable_failure(
    runner: CliRunner,
    export_arguments: tuple[list[str], dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _ = export_arguments

    def fail_export(**_kwargs: Path) -> ExportSummary:
        raise LegacyExportError("The audited source changed.")

    monkeypatch.setattr(data, "export_legacy_evidence", fail_export)

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 1
    assert result.output.strip() == "Legacy export failed: The audited source changed."
    assert result.exception is not None


@pytest.mark.parametrize("with_quarantine", [False, True])
def test_validate_legacy_reports_public_and_optional_private_verification(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_quarantine: bool,
) -> None:
    public_root = tmp_path / "public"
    public_root.mkdir()
    quarantine_root = tmp_path / "private"
    quarantine_root.mkdir()
    received: dict[str, Path | None] = {}

    def fake_validate(
        *,
        public_root: Path,
        quarantine_root: Path | None,
    ) -> ValidationSummary:
        received.update(public_root=public_root, quarantine_root=quarantine_root)
        return ValidationSummary(
            runs=464,
            attempts=532,
            annotations=408,
            detections=45,
            sessions=5,
            quarantine_verified=quarantine_root is not None,
        )

    monkeypatch.setattr(data, "validate_legacy_export", fake_validate)
    arguments = ["data", "validate-legacy", "--public-root", str(public_root)]
    if with_quarantine:
        arguments.extend(["--quarantine-root", str(quarantine_root)])

    result = runner.invoke(cli.app, arguments)

    expected_quarantine = quarantine_root.resolve() if with_quarantine else None
    assert result.exit_code == 0
    assert received == {
        "public_root": public_root.resolve(),
        "quarantine_root": expected_quarantine,
    }
    suffix = " and private quarantine" if with_quarantine else ""
    assert result.output.strip() == f"Validated 464 runs and 532 attempts{suffix}."


def test_validate_legacy_returns_a_stable_failure(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_root = tmp_path / "public"
    public_root.mkdir()

    def fail_validation(
        *,
        public_root: Path,
        quarantine_root: Path | None,
    ) -> ValidationSummary:
        del public_root, quarantine_root
        raise LegacyExportError("A declared component failed its integrity check.")

    monkeypatch.setattr(data, "validate_legacy_export", fail_validation)

    result = runner.invoke(
        cli.app,
        ["data", "validate-legacy", "--public-root", str(public_root)],
    )

    assert result.exit_code == 1
    assert result.output.strip() == (
        "Legacy export validation failed: A declared component failed its integrity check."
    )
    assert result.exception is not None
