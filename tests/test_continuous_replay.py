from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab import continuous_replay as replay
from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.contracts.dataset import MediaIntervalV1

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/constructed-continuous-replay-v1.json"
FIXTURE = ROOT / "tests/fixtures/public/replay/constructed-continuous-replay-v1.json"
DETECTOR = ROOT / "configs/evaluation/candidate-event-detector-v1.json"
POLICY = ROOT / "docs/reports/popsign-constructed-calibration-policy-v1.json"
REPORT = ROOT / "docs/reports/constructed-continuous-replay-v1.json"


def _inputs() -> tuple[replay.ReplayScoringConfig, replay.ContinuousReplayFixture]:
    return replay.load_replay_scoring_config(CONFIG), replay.load_continuous_replay_fixture(FIXTURE)


def _interval(start: int, end: int) -> MediaIntervalV1:
    return MediaIntervalV1(schema_version="media-interval/1", start_us=start, end_us=end)


def test_constructed_report_has_exact_scoring_and_honest_boundaries() -> None:
    config, fixture = _inputs()
    report = replay.build_continuous_replay_report(
        config,
        fixture,
        candidate_event_config_path=DETECTOR,
        decision_policy_path=POLICY,
    )

    assert report == parse_json_object(REPORT.read_bytes())
    counts = cast(dict[str, int], report["counts"])
    assert (counts["truths"], counts["temporal_matches"], counts["misses"]) == (3, 2, 1)
    assert (counts["duplicates"], counts["false_target_activations"]) == (1, 2)
    metrics = cast(dict[str, float], report["metrics"])
    assert (metrics["temporal_f1"], metrics["correct_target_rate"]) == (0.5, 0.333333)
    assert metrics["false_target_activations_per_hour"] == 2.0


def test_matching_uses_fixed_boundary_tie_break_and_reports_early_decision() -> None:
    session = replay.ReplaySession(
        session_alias="tie_session",
        group_alias="tie_group",
        duration_us=200,
        truths=(
            replay.ReplayTruth(truth_id="truth_one", interval=_interval(0, 100), label="hello"),
        ),
        decisions=(
            replay.ReplayDecision(
                decision_id="earlier",
                interval=_interval(0, 20),
                kind="target",
                label="hello",
                decision_timestamp_us=90,
            ),
            replay.ReplayDecision(
                decision_id="later",
                interval=_interval(80, 100),
                kind="target",
                label="no",
                decision_timestamp_us=110,
            ),
        ),
    )

    counts, latencies = replay._counts(session, 200_000)
    assert (counts["correct_labels"], counts["duplicates"], latencies) == (1, 1, [-10])
    assert replay._nearest_rank([], 0.5) is None


def test_group_bootstrap_and_zero_denominators_fail_honestly() -> None:
    config, fixture = _inputs()
    one_group = fixture.model_copy(
        update={
            "sessions": tuple(
                session.model_copy(update={"group_alias": "same_group"})
                for session in fixture.sessions
            )
        }
    )
    assert replay._bootstrap(config, one_group) == {
        "status": "insufficient_groups",
        "group_count": 1,
    }
    empty = {name: 0 for name in replay._counts(fixture.sessions[1], 200_000)[0]}
    assert replay._metrics(empty) == dict.fromkeys(cast(dict[str, object], replay._metrics(empty)))


def test_invalid_fixture_and_identity_mismatch_are_rejected(tmp_path: Path) -> None:
    for kind, label in (("target", None), ("other", None), ("abstain", "hello")):
        with pytest.raises(ValueError, match="label"):
            replay.ReplayDecision.model_validate(
                {
                    "decision_id": "invalid",
                    "interval": _interval(0, 1),
                    "kind": kind,
                    "label": label,
                    "decision_timestamp_us": 1,
                },
                strict=True,
            )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(FIXTURE.read_bytes() + b"\n")
    with pytest.raises(replay.ContinuousReplayError, match="fixture is invalid"):
        replay.load_continuous_replay_fixture(noncanonical)

    config, fixture = _inputs()
    changed = config.model_copy(update={"research_model_sha256": "sha256:" + "0" * 64})
    with pytest.raises(replay.ContinuousReplayError, match="do not match"):
        replay.build_continuous_replay_report(
            changed,
            fixture,
            candidate_event_config_path=DETECTOR,
            decision_policy_path=POLICY,
        )


def test_cli_writes_one_canonical_path_safe_report(tmp_path: Path) -> None:
    runner = CliRunner(env={"NO_COLOR": "1"})
    report = tmp_path / "report.json"
    arguments = [
        "evaluate",
        "continuous-replay",
        str(CONFIG),
        str(FIXTURE),
        str(DETECTOR),
        str(POLICY),
        "--report",
        str(report),
    ]
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 0
    assert result.output.strip() == (
        "Constructed replay scoring verified: 2/3 accepted target decisions matched; test sealed."
    )
    raw = report.read_bytes()
    assert raw == canonical_json_bytes(parse_json_object(raw)) + b"\n"
    assert str(tmp_path).encode() not in raw

    repeated = runner.invoke(cli.app, arguments)
    assert repeated.exit_code == 1
    assert repeated.output.strip() == (
        "Continuous replay scoring failed: inputs or output are invalid."
    )
    assert str(tmp_path) not in repeated.output
