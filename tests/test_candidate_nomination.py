from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from signlab import candidate_nomination as nomination
from signlab import cli
from signlab.contracts.canonical import canonical_json_bytes, parse_json_object

ROOT = Path(__file__).resolve().parents[1]
DOSSIER = ROOT / "configs/evaluation/popsign-tcn-portable-export-candidate-v1.json"
CHECKPOINT = ROOT / "runs/popsign-constructed-calibration-v1/model.keras"
REPORT = ROOT / "docs/reports/popsign-tcn-portable-export-nomination-v1.json"


def _verified_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nomination,
        "_checkpoint_gate",
        lambda *_args: (
            {"gate": "checkpoint_identity", "reason": "verified", "status": "pass"},
            "local_hash_verified_not_published",
        ),
    )


def _reasons(report: dict[str, object]) -> set[str]:
    gates = cast(list[dict[str, str]], report["development_gates"])
    return {row["reason"] for row in gates}


def test_checked_report_is_export_only_and_blocks_every_release_gate() -> None:
    report = cast(dict[str, object], parse_json_object(REPORT.read_bytes()))
    assert report["candidate_status"] == "nominated_for_portable_export"
    assert report["champion_status"] == "none_blocked"
    assert report["test_status"] == "sealed_not_loaded"
    release = cast(list[dict[str, str]], report["release_gates"])
    assert len(release) == 8
    assert {row["status"] for row in release} == {"blocked"}


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="ignored research checkpoint unavailable")
def test_local_checkpoint_reproduces_checked_report() -> None:
    dossier = nomination.load_candidate_dossier(DOSSIER)
    report = nomination.build_candidate_nomination_report(
        dossier, repository_root=ROOT, checkpoint_path=CHECKPOINT
    )
    assert report == parse_json_object(REPORT.read_bytes())


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("architecture", "mamba", "unsupported_architecture"),
        ("representation", "combined", "unsupported_representation"),
        ("split_strategy", "random_rows_v1", "invalid_split"),
        ("source_run_id", "0" * 32, "provenance_identity_mismatch"),
    ],
)
def test_unsupported_candidate_fails_one_named_gate(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str, reason: str
) -> None:
    _verified_checkpoint(monkeypatch)
    dossier = nomination.load_candidate_dossier(DOSSIER).model_copy(update={field: value})
    report = nomination.build_candidate_nomination_report(
        dossier, repository_root=ROOT, checkpoint_path="unused"
    )
    assert report["candidate_status"] == "nomination_blocked"
    assert reason in _reasons(report)


def test_stale_card_and_checkpoint_bytes_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dossier = nomination.load_candidate_dossier(DOSSIER)
    stale = dossier.evidence.model_copy(
        update={
            "dataset_card": dossier.evidence.dataset_card.model_copy(
                update={"sha256": "sha256:" + "0" * 64}
            )
        }
    )
    _verified_checkpoint(monkeypatch)
    report = nomination.build_candidate_nomination_report(
        dossier.model_copy(update={"evidence": stale}),
        repository_root=ROOT,
        checkpoint_path="unused",
    )
    assert "stale_dataset_card" in _reasons(report)

    monkeypatch.undo()
    altered = tmp_path / "model.keras"
    altered.write_bytes(b"altered")
    report = nomination.build_candidate_nomination_report(
        dossier, repository_root=ROOT, checkpoint_path=altered
    )
    assert "checkpoint_identity_mismatch" in _reasons(report)
    report = nomination.build_candidate_nomination_report(
        dossier, repository_root=ROOT, checkpoint_path=tmp_path / "missing.keras"
    )
    assert "nomination_blocked_missing_artifact" in _reasons(report)


def test_invalid_dossier_has_stable_codes(tmp_path: Path) -> None:
    payload = parse_json_object(DOSSIER.read_bytes())
    del cast(dict[str, object], payload["identities"])["research_model_sha256"]
    missing = tmp_path / "missing.json"
    missing.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(nomination.CandidateNominationError) as caught:
        nomination.load_candidate_dossier(missing)
    assert caught.value.code == "missing_hash"

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(DOSSIER.read_bytes() + b"\n")
    with pytest.raises(nomination.CandidateNominationError) as caught:
        nomination.load_candidate_dossier(noncanonical)
    assert caught.value.code == "noncanonical_dossier"


def test_cli_is_path_safe_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _verified_checkpoint(monkeypatch)
    output = tmp_path / "nomination.json"
    args = [
        "evaluate",
        "candidate-nomination",
        str(DOSSIER),
        str(tmp_path / "private-model.keras"),
        str(ROOT),
        "--report",
        str(output),
    ]
    result = CliRunner(env={"NO_COLOR": "1"}).invoke(cli.app, args)
    assert result.exit_code == 0
    assert result.output.strip().endswith("champion activation remains blocked.")
    assert (
        output.read_bytes() == canonical_json_bytes(parse_json_object(output.read_bytes())) + b"\n"
    )
    assert str(tmp_path) not in output.read_text(encoding="utf-8")

    repeated = CliRunner(env={"NO_COLOR": "1"}).invoke(cli.app, args)
    assert repeated.exit_code == 1
    assert repeated.output.strip() == "Candidate nomination failed: report_exists_or_unwritable."
