from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from feature_fixtures import make_anchor_row, make_hand_row, make_recording, make_table
from signlab import cli
from signlab.candidate_events import (
    CandidateEvent,
    CandidateEventConfigV1,
    CandidateEventDetector,
    CandidateEventError,
    CandidateObservation,
    candidate_event_config_digest,
    load_candidate_event_config,
    project_candidate_observations,
)
from signlab.contracts.canonical import parse_json_object
from signlab.contracts.extraction import HandSlotV1, LandmarkFramesTableV1

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/evaluation/candidate-event-detector-v1.json"
FIXTURE_PATH = ROOT / "tests/fixtures/public/events/candidate-event-stream-v1.json"
CONFIG_SHA256 = "sha256:0443badf68d34347a00096682cf049b6f49b5253c12e47bf61b068a597aa162d"


@pytest.fixture
def config() -> CandidateEventConfigV1:
    return load_candidate_event_config(CONFIG_PATH)


def _observation(
    timestamp_us: int,
    motion_q: int,
    *,
    hand_present: bool = True,
    quality_ok: bool = True,
) -> CandidateObservation:
    return CandidateObservation(
        timestamp_us=timestamp_us,
        hand_present=hand_present,
        quality_ok=quality_ok,
        motion_q=motion_q,
    )


def _active_prefix(detector: CandidateEventDetector) -> None:
    for row in (
        _observation(0, 0),
        _observation(50_000, 1_000_000),
        _observation(100_000, 1_000_000),
        _observation(150_000, 1_000_000),
        _observation(200_000, 1_000_000),
    ):
        assert detector.push(row) is None
    assert detector.state == "recording"


def _config_with(config: CandidateEventConfigV1, **changes: object) -> CandidateEventConfigV1:
    payload = config.model_dump(mode="json", round_trip=True)
    payload.update(changes)
    return CandidateEventConfigV1.model_validate(payload, strict=True)


def test_config_and_observation_order_fail_closed(
    config: CandidateEventConfigV1, tmp_path: Path
) -> None:
    assert candidate_event_config_digest(config) == CONFIG_SHA256
    noncanonical = tmp_path / "config.json"
    noncanonical.write_text(CONFIG_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CandidateEventError, match="config is invalid"):
        load_candidate_event_config(noncanonical)

    detector = CandidateEventDetector(config)
    detector.push(_observation(0, 0, hand_present=False))
    with pytest.raises(CandidateEventError, match="increase strictly"):
        detector.push(_observation(0, 0, hand_present=False))
    with pytest.raises(CandidateEventError, match="config must be validated"):
        CandidateEventDetector(cast(CandidateEventConfigV1, object()))
    with pytest.raises(CandidateEventError, match="observation must be validated"):
        detector.push(cast(CandidateObservation, object()))

    invalid_cases = (
        ({"start_motion_q": config.stop_motion_q}, "motion thresholds"),
        (
            {"minimum_event_duration_us": config.maximum_event_duration_us},
            "minimum event duration",
        ),
        ({"quality_policy_sha256": "sha256:" + "0" * 64}, "published quality policy"),
    )
    for changes, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _config_with(config, **changes)


def test_constructed_fixture_has_exact_transitions_and_boundaries(
    config: CandidateEventConfigV1,
) -> None:
    payload = parse_json_object(FIXTURE_PATH.read_bytes())
    observations = cast(list[dict[str, object]], payload["observations"])
    expected_transitions = cast(list[dict[str, object]], payload["expected_transitions"])
    detector = CandidateEventDetector(config)
    events: list[CandidateEvent] = []
    transitions: list[dict[str, object]] = []
    prior = detector.state
    for frame_index, row in enumerate(observations):
        event = detector.push(CandidateObservation.model_validate(row, strict=True))
        if detector.state != prior:
            transitions.append({"frame_index": frame_index, "state": detector.state})
            prior = detector.state
        if event is not None:
            events.append(event)

    assert detector.finish() is None
    assert transitions == expected_transitions
    assert {row["state"] for row in transitions} == {
        "inactive",
        "arming",
        "recording",
        "finalizing",
        "cooldown",
    }
    assert [
        (
            event.first_timestamp_us,
            event.last_timestamp_us,
            event.termination_reason,
            event.config_sha256,
        )
        for event in events
    ] == [
        (250_000, 1_250_000, "settled", CONFIG_SHA256),
        (1_750_000, 2_000_000, "signal_gap", CONFIG_SHA256),
    ]


def test_idle_static_and_subthreshold_jitter_never_emit(
    config: CandidateEventConfigV1,
) -> None:
    detector = CandidateEventDetector(config)
    rows = (
        _observation(0, 0, hand_present=False),
        _observation(50_000, 0),
        _observation(100_000, 100_000),
        _observation(150_000, 0),
        _observation(200_000, 120_000),
    )

    assert all(detector.push(row) is None for row in rows)
    assert detector.finish() is None


def test_inactive_timestamp_gap_establishes_a_fresh_motion_baseline(
    config: CandidateEventConfigV1,
) -> None:
    detector = CandidateEventDetector(config)
    assert detector.push(_observation(0, 0)) is None
    assert detector.push(_observation(200_000, 1_000_000)) is None
    assert detector.state == "inactive"


def test_short_pause_resumes_recording_without_fragmentation(
    config: CandidateEventConfigV1,
) -> None:
    detector = CandidateEventDetector(_config_with(config, smoothing_alpha_ppm=1_000_000))
    _active_prefix(detector)
    assert detector.push(_observation(250_000, 0)) is None
    assert detector.state == "finalizing"
    assert detector.push(_observation(300_000, 1_000_000)) is None
    assert cast(str, detector.state) == "recording"

    event = detector.finish()
    assert event is not None
    assert event.termination_reason == "stream_end"
    assert (event.first_timestamp_us, event.last_timestamp_us) == (0, 300_000)


def test_arming_tolerates_bounded_gap_but_cancels_quiet_or_excess(
    config: CandidateEventConfigV1,
) -> None:
    exact = _config_with(config, smoothing_alpha_ppm=1_000_000)
    detector = CandidateEventDetector(exact)
    for row in (
        _observation(0, 0),
        _observation(50_000, 1_000_000),
        _observation(75_000, 0, quality_ok=False),
        _observation(100_000, 0, quality_ok=False),
        _observation(125_000, 1_000_000),
    ):
        assert detector.push(row) is None
    assert detector.state == "recording"

    for terminal in (
        (
            _observation(75_000, 0, quality_ok=False),
            _observation(100_000, 0, quality_ok=False),
            _observation(125_000, 0, quality_ok=False),
        ),
        (_observation(100_000, 0),),
    ):
        detector = CandidateEventDetector(exact)
        detector.push(_observation(0, 0))
        detector.push(_observation(50_000, 1_000_000))
        for row in terminal:
            detector.push(row)
        assert detector.state == "inactive"


def test_two_frame_100ms_gap_is_inclusive(config: CandidateEventConfigV1) -> None:
    detector = CandidateEventDetector(config)
    _active_prefix(detector)
    assert detector.push(_observation(225_000, 0, quality_ok=False)) is None
    assert detector.push(_observation(250_000, 0, quality_ok=False)) is None
    assert detector.push(_observation(300_000, 1_000_000)) is None
    assert detector.state == "recording"
    assert detector.finish() is not None


@pytest.mark.parametrize(
    "gaps",
    [
        (225_000, 250_000, 275_000),
        (250_000, 300_001),
    ],
)
def test_gap_frame_or_time_limit_ends_at_last_usable(
    config: CandidateEventConfigV1, gaps: tuple[int, ...]
) -> None:
    detector = CandidateEventDetector(config)
    _active_prefix(detector)
    event = None
    for timestamp_us in gaps:
        event = detector.push(_observation(timestamp_us, 0, quality_ok=False))
        if event is not None:
            break

    assert event is not None
    assert event.termination_reason == "signal_gap"
    assert (event.last_frame_index, event.last_timestamp_us) == (4, 200_000)


def test_gap_is_rechecked_when_signal_returns(config: CandidateEventConfigV1) -> None:
    direct = CandidateEventDetector(config)
    _active_prefix(direct)
    event = direct.push(_observation(300_001, 1_000_000))
    assert event is not None
    assert event.termination_reason == "signal_gap"
    assert event.last_timestamp_us == 200_000

    detector = CandidateEventDetector(config)
    _active_prefix(detector)
    detector.push(_observation(225_000, 0, quality_ok=False))
    detector.push(_observation(250_000, 0, quality_ok=False))
    event = detector.push(_observation(300_001, 1_000_000))

    assert event is not None
    assert event.termination_reason == "signal_gap"
    assert event.last_timestamp_us == 200_000


def test_max_duration_and_implausible_motion_are_bounded(
    config: CandidateEventConfigV1,
) -> None:
    bounded = _config_with(config, maximum_event_duration_us=400_000)
    detector = CandidateEventDetector(bounded)
    _active_prefix(detector)
    assert detector.push(_observation(300_000, 1_000_000)) is None
    event = detector.push(_observation(400_000, 1_000_000))
    assert event is not None
    assert event.termination_reason == "max_duration"
    assert event.last_timestamp_us == 400_000

    detector = CandidateEventDetector(bounded)
    _active_prefix(detector)
    detector.push(_observation(300_000, 1_000_000))
    detector.push(_observation(390_000, 1_000_000))
    event = detector.push(_observation(450_000, 1_000_000))
    assert event is not None
    assert event.termination_reason == "max_duration"
    assert event.last_timestamp_us == 390_000

    detector = CandidateEventDetector(bounded)
    _active_prefix(detector)
    event = detector.push(_observation(450_001, 1_000_000))
    assert event is not None
    assert event.termination_reason == "signal_gap"
    assert event.last_timestamp_us == 200_000

    detector = CandidateEventDetector(config)
    _active_prefix(detector)
    assert detector.push(_observation(225_000, 13_000_000)) is None
    assert detector.push(_observation(250_000, 13_000_000)) is None
    event = detector.push(_observation(275_000, 13_000_000))
    assert event is not None
    assert event.termination_reason == "signal_gap"


def test_finish_flushes_confirmed_event_but_drops_arming(
    config: CandidateEventConfigV1,
) -> None:
    arming = CandidateEventDetector(config)
    assert arming.push(_observation(0, 0)) is None
    assert arming.push(_observation(50_000, 1_000_000)) is None
    assert arming.state == "arming"
    assert arming.finish() is None

    recording = CandidateEventDetector(config)
    _active_prefix(recording)
    event = recording.finish()
    assert event is not None
    assert event.termination_reason == "stream_end"
    assert recording.finish() is None
    with pytest.raises(CandidateEventError, match="already finished"):
        recording.push(_observation(250_000, 1_000_000))

    exact = _config_with(config, smoothing_alpha_ppm=1_000_000)
    finalizing = CandidateEventDetector(exact)
    _active_prefix(finalizing)
    finalizing.push(_observation(250_000, 0))
    event = finalizing.finish()
    assert event is not None
    assert event.last_timestamp_us == 200_000

    minimum = _config_with(
        exact,
        minimum_event_duration_us=300_000,
        maximum_event_duration_us=400_000,
    )
    suppressed = CandidateEventDetector(minimum)
    _active_prefix(suppressed)
    suppressed.push(_observation(250_000, 0))
    assert suppressed.push(_observation(450_000, 0)) is None
    assert suppressed.state == "cooldown"


def test_landmark_projection_uses_motion_not_handedness_confidence(
    config: CandidateEventConfigV1,
) -> None:
    timestamps = (0, 100_000, 200_000)
    recording = make_recording(duration_us=300_000)
    table = make_table(
        recording,
        timestamps,
        tuple(
            make_hand_row(timestamp_us=value, image_velocity_per_second=(0.5, 0.0))
            for value in timestamps
        ),
        tuple(make_anchor_row() for _ in timestamps),
    )
    projected = project_candidate_observations(table)
    assert [(row.quality_ok, row.motion_q) for row in projected] == [
        (True, 0),
        (True, 500_000),
        (True, 500_000),
    ]

    changed_rows = []
    for frame in table.rows:
        hands = tuple(
            hand.model_copy(update={"handedness_confidence": 0.51}) if hand.present else hand
            for hand in frame.hands
        )
        changed_rows.append(
            frame.model_copy(update={"hands": cast(tuple[HandSlotV1, HandSlotV1], hands)})
        )
    changed = LandmarkFramesTableV1(
        schema_version="landmark-frames-table/1", rows=tuple(changed_rows)
    )
    assert project_candidate_observations(changed) == projected
    with pytest.raises(CandidateEventError, match="validated landmark table"):
        project_candidate_observations(cast(LandmarkFramesTableV1, object()))

    invalid_source = make_table(
        recording,
        timestamps,
        (
            make_hand_row(timestamp_us=0, first_present=False),
            *tuple(
                make_hand_row(timestamp_us=value, image_velocity_per_second=(0.5, 0.0))
                for value in timestamps[1:]
            ),
        ),
        (make_anchor_row(present=False), *tuple(make_anchor_row() for _ in timestamps[1:])),
    )
    first = invalid_source.rows[0].model_copy(
        update={"invalid": True, "invalid_reason": "source_frame_invalid"}
    )
    invalid_source = LandmarkFramesTableV1(
        schema_version="landmark-frames-table/1",
        rows=(first, *invalid_source.rows[1:]),
    )
    assert project_candidate_observations(invalid_source)[0] == CandidateObservation(
        timestamp_us=0, hand_present=False, quality_ok=False, motion_q=0
    )

    gap_timestamps = (0, 50_000, 100_000, 150_000, 175_000, 200_000, 250_000)
    gap_table = make_table(
        recording,
        gap_timestamps,
        tuple(
            make_hand_row(
                timestamp_us=value,
                first_present=index not in {4, 5},
                image_velocity_per_second=(1.0, 0.0),
            )
            for index, value in enumerate(gap_timestamps)
        ),
        tuple(make_anchor_row() for _ in gap_timestamps),
    )
    detector = CandidateEventDetector(config)
    for observation in project_candidate_observations(gap_table):
        assert detector.push(observation) is None
    assert detector.state == "recording"


def test_cli_writes_sanitized_constructed_report(tmp_path: Path) -> None:
    runner = CliRunner(env={"NO_COLOR": "1"})
    report = tmp_path / "report.md"
    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "candidate-events",
            str(CONFIG_PATH),
            str(FIXTURE_PATH),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == ("Candidate-event conformance verified: 2/2 expected events.")
    rendered = report.read_text(encoding="utf-8")
    assert "Fixture recall | 1.000" in rendered
    assert "False candidates | 0" in rendered
    assert "sealed_not_loaded" in rendered
    assert "not_applicable_constructed_fixture" in rendered
    assert str(tmp_path) not in rendered

    repeated = runner.invoke(
        cli.app,
        [
            "evaluate",
            "candidate-events",
            str(CONFIG_PATH),
            str(FIXTURE_PATH),
            "--report",
            str(report),
        ],
    )
    assert repeated.exit_code == 1
    assert repeated.output.strip() == (
        "Candidate-event evaluation failed: inputs or output are invalid."
    )
    assert str(tmp_path) not in repeated.output

    altered = FIXTURE_PATH.read_text(encoding="utf-8").replace(
        '"frame_index": 7, "state": "arming"',
        '"frame_index": 7, "state": "recording"',
    )
    bad_fixture = tmp_path / "bad-fixture.json"
    bad_fixture.write_text(altered, encoding="utf-8")
    failed = runner.invoke(
        cli.app,
        [
            "evaluate",
            "candidate-events",
            str(CONFIG_PATH),
            str(bad_fixture),
            "--report",
            str(tmp_path / "unused.md"),
        ],
    )
    assert failed.exit_code == 1
    assert "inputs or output are invalid" in failed.output
