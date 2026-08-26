from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from signlab import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def _manifest() -> object:
    return SimpleNamespace(
        manifest_sha256="sha256:" + "a" * 64,
        policy_sha256="sha256:" + "b" * 64,
        extraction_manifest_sha256="sha256:" + "c" * 64,
        raw_dataset_manifest_sha256="sha256:" + "d" * 64,
        dataset_report=SimpleNamespace(status="blocked"),
        participant_id="participant_ffffffffffffffffffffffffffffffff",
    )


def _validation() -> object:
    return SimpleNamespace(
        raw_bundle_integrity="verified",
        consent_boundary="synthetic_fixture_only",
        extraction_bundle_integrity="verified",
        manifest_integrity="verified",
        report_recomputation="verified",
        exact_inventory="verified",
        sequence_count=4,
        pass_count=1,
        warning_count=1,
        quarantine_count=1,
        reject_count=1,
    )


def _expected_summary(status: str) -> list[str]:
    return [
        f"Landmark quality: {status}.",
        "Quality manifest SHA-256: sha256:" + "a" * 64,
        "Quality policy SHA-256: sha256:" + "b" * 64,
        "Extraction manifest SHA-256: sha256:" + "c" * 64,
        "Raw dataset manifest SHA-256: sha256:" + "d" * 64,
        "Assessed sequences: 4",
        "Quality dispositions: 1 pass, 1 warning, 1 quarantine, 1 reject.",
        "Dataset quality status: blocked",
        "Quality report recomputation: verified",
    ]


def test_assess_quality_reads_manifests_and_delegates_stable_inputs(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from signlab.quality import batch, resources

    extraction_manifest = tmp_path / "participant-private-extraction.json"
    raw_manifest = tmp_path / "participant-private-raw.json"
    extraction_root = tmp_path / "participant-private-extraction-root"
    raw_root = tmp_path / "participant-private-raw-root"
    output = tmp_path / "participant-private-quality"
    extraction_manifest.write_bytes(b"synthetic-extraction-manifest")
    raw_manifest.write_bytes(b"synthetic-raw-manifest")
    policy = object()
    calls: list[tuple[bytes, Path, bytes, Path, object, Path]] = []
    monkeypatch.setattr(resources, "load_packaged_default_quality_policy", lambda: policy)

    def assess(
        document: bytes,
        *,
        extraction_root: Path,
        raw_manifest: bytes,
        raw_bundle_root: Path,
        policy: object,
        destination: Path,
    ) -> object:
        calls.append(
            (
                document,
                extraction_root,
                raw_manifest,
                raw_bundle_root,
                policy,
                destination,
            )
        )
        return SimpleNamespace(
            status="published",
            manifest=_manifest(),
            validation=_validation(),
        )

    monkeypatch.setattr(batch, "assess_landmark_quality", assess)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "assess-landmark-quality",
            str(extraction_manifest),
            "--extraction-root",
            str(extraction_root),
            "--raw-manifest",
            str(raw_manifest),
            "--raw-bundle-root",
            str(raw_root),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            b"synthetic-extraction-manifest",
            extraction_root,
            b"synthetic-raw-manifest",
            raw_root,
            policy,
            output,
        )
    ]
    assert result.output.splitlines() == _expected_summary("published")
    assert "participant" not in result.output.casefold()
    assert str(tmp_path) not in result.output


def test_validate_quality_reads_all_manifests_and_delegates_exact_roots(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from signlab.quality import batch

    quality_manifest = tmp_path / "participant-private-quality.json"
    extraction_manifest = tmp_path / "participant-private-extraction.json"
    raw_manifest = tmp_path / "participant-private-raw.json"
    quality_root = tmp_path / "participant-private-quality-root"
    extraction_root = tmp_path / "participant-private-extraction-root"
    raw_root = tmp_path / "participant-private-raw-root"
    quality_manifest.write_bytes(b"synthetic-quality-manifest")
    extraction_manifest.write_bytes(b"synthetic-extraction-manifest")
    raw_manifest.write_bytes(b"synthetic-raw-manifest")
    calls: list[tuple[bytes, Path, bytes, Path, bytes, Path]] = []

    def validate(
        document: bytes,
        workspace_root: Path,
        *,
        extraction_manifest: bytes,
        extraction_root: Path,
        raw_manifest: bytes,
        raw_bundle_root: Path,
    ) -> object:
        calls.append(
            (
                document,
                workspace_root,
                extraction_manifest,
                extraction_root,
                raw_manifest,
                raw_bundle_root,
            )
        )
        return SimpleNamespace(manifest=_manifest(), validation=_validation())

    monkeypatch.setattr(batch, "validate_landmark_quality_bundle", validate)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-landmark-quality",
            str(quality_manifest),
            "--workspace-root",
            str(quality_root),
            "--extraction-manifest",
            str(extraction_manifest),
            "--extraction-root",
            str(extraction_root),
            "--raw-manifest",
            str(raw_manifest),
            "--raw-bundle-root",
            str(raw_root),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            b"synthetic-quality-manifest",
            quality_root,
            b"synthetic-extraction-manifest",
            extraction_root,
            b"synthetic-raw-manifest",
            raw_root,
        )
    ]
    assert result.output.splitlines() == _expected_summary("verified")
    assert "participant" not in result.output.casefold()
    assert str(tmp_path) not in result.output


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("assess-landmark-quality", "Landmark quality assessment failed."),
        ("validate-landmark-quality", "Landmark quality validation failed."),
    ],
)
def test_quality_commands_redact_paths_and_internal_errors(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    message: str,
) -> None:
    from signlab.quality import batch, resources

    quality_manifest = tmp_path / "participant-private-quality.json"
    extraction_manifest = tmp_path / "participant-private-extraction.json"
    raw_manifest = tmp_path / "participant-private-raw.json"
    for path in (quality_manifest, extraction_manifest, raw_manifest):
        path.write_bytes(b"participant-private-content")
    monkeypatch.setattr(resources, "load_packaged_default_quality_policy", lambda: object())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise batch.QualityBatchError("bundle.invalid")

    monkeypatch.setattr(batch, "assess_landmark_quality", fail)
    monkeypatch.setattr(batch, "validate_landmark_quality_bundle", fail)
    arguments = [
        "data",
        command,
        str(extraction_manifest if command == "assess-landmark-quality" else quality_manifest),
        "--extraction-root",
        str(tmp_path / "participant-private-extraction-root"),
        "--raw-manifest",
        str(raw_manifest),
        "--raw-bundle-root",
        str(tmp_path / "participant-private-raw-root"),
    ]
    if command == "assess-landmark-quality":
        arguments.extend(["--output", str(tmp_path / "participant-private-output")])
    else:
        arguments.extend(
            [
                "--workspace-root",
                str(tmp_path / "participant-private-quality-root"),
                "--extraction-manifest",
                str(extraction_manifest),
            ]
        )

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 1
    assert result.output.strip() == message
    assert "participant-private" not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output


def test_quality_help_does_not_import_optional_or_storage_modules() -> None:
    probe = (
        "import sys; from typer.testing import CliRunner; import signlab.cli as cli; "
        "runner = CliRunner(env={'NO_COLOR': '1'}); "
        "results = [runner.invoke(cli.app, ['data', '--help']), "
        "runner.invoke(cli.app, ['data', 'assess-landmark-quality', '--help']), "
        "runner.invoke(cli.app, ['data', 'validate-landmark-quality', '--help'])]; "
        "blocked = {'pyarrow', 'signlab.quality.batch', 'signlab.quality.policy', "
        "'signlab.quality.resources'} & set(sys.modules); "
        "failed = any(result.exit_code for result in results); "
        "raise SystemExit(','.join(sorted(blocked)) if blocked else (1 if failed else 0))"
    )

    subprocess.run([sys.executable, "-c", probe], check=True)
