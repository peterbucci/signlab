"""Evaluation command group."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import typer
from pydantic import Field, ValidationError, model_validator

from signlab.candidate_events import (
    CandidateEvent,
    CandidateEventDetector,
    CandidateObservation,
    CandidateState,
    candidate_event_config_digest,
    load_candidate_event_config,
)
from signlab.commands._group import create_group
from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256, parse_json_object
from signlab.contracts.core import NonNegativeSafeInteger, StableId, StrictContractModel

app = create_group(help_text="Evaluate checkpoints on locked clips and continuous replay sessions.")

_FIXTURE_DOMAIN = "signlab-candidate-event-fixture/1"


class _TruthEvent(StrictContractModel):
    first_timestamp_us: NonNegativeSafeInteger
    last_timestamp_us: NonNegativeSafeInteger

    @model_validator(mode="after")
    def _require_positive_span(self) -> Self:
        if self.last_timestamp_us <= self.first_timestamp_us:
            raise ValueError("truth event must have a positive span")
        return self


class _ExpectedTransition(StrictContractModel):
    frame_index: NonNegativeSafeInteger
    state: CandidateState


class _CandidateEventFixture(StrictContractModel):
    format: Literal["signlab-candidate-event-fixture/1"]
    fixture_id: StableId
    scenario_coverage: tuple[StableId, ...] = Field(min_length=1)
    observations: tuple[CandidateObservation, ...] = Field(min_length=2)
    truth_events: tuple[_TruthEvent, ...] = Field(min_length=1)
    expected_transitions: tuple[_ExpectedTransition, ...] = Field(min_length=5)

    @model_validator(mode="after")
    def _require_ordered_fixture(self) -> Self:
        timestamps = tuple(row.timestamp_us for row in self.observations)
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("fixture timestamps must increase strictly")
        truth = tuple((row.first_timestamp_us, row.last_timestamp_us) for row in self.truth_events)
        if truth != tuple(sorted(truth)) or any(
            left[1] >= right[0] for left, right in pairwise(truth)
        ):
            raise ValueError("truth events must be ordered and non-overlapping")
        indexes = tuple(row.frame_index for row in self.expected_transitions)
        if indexes != tuple(sorted(set(indexes))) or indexes[-1] >= len(self.observations):
            raise ValueError("expected transition indexes are invalid")
        return self


def _load_fixture(path: Path) -> _CandidateEventFixture:
    try:
        return _CandidateEventFixture.model_validate_json(
            canonical_json_bytes(parse_json_object(path.read_bytes())), strict=True
        )
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise ValueError("candidate-event fixture is invalid") from error


def _overlap_us(truth: _TruthEvent, event: CandidateEvent) -> int:
    return max(
        0,
        min(truth.last_timestamp_us, event.last_timestamp_us)
        - max(truth.first_timestamp_us, event.first_timestamp_us),
    )


def _ordered_conformance(
    truth: tuple[_TruthEvent, ...], events: tuple[CandidateEvent, ...]
) -> tuple[float, float]:
    if len(events) != len(truth) or any(
        _overlap_us(expected, emitted) <= 0 for expected, emitted in zip(truth, events, strict=True)
    ):
        raise ValueError("candidate events do not match the fixed fixture")
    onset_ms = (
        sum(
            abs(emitted.first_timestamp_us - expected.first_timestamp_us)
            for expected, emitted in zip(truth, events, strict=True)
        )
        / len(truth)
        / 1_000
    )
    offset_ms = (
        sum(
            abs(emitted.last_timestamp_us - expected.last_timestamp_us)
            for expected, emitted in zip(truth, events, strict=True)
        )
        / len(truth)
        / 1_000
    )
    return onset_ms, offset_ms


def _markdown(
    fixture: _CandidateEventFixture,
    config_sha256: str,
    fixture_sha256: str,
    event_count: int,
    onset_error_ms: float,
    offset_error_ms: float,
) -> str:
    scenarios = "\n".join(f"- `{name}`" for name in fixture.scenario_coverage)
    return f"""# Candidate-event detector constructed replay v1

> Technical conformance evidence only; no natural-use performance claim.

- Evidence kind: `constructed_replay_conformance`
- Metric claim: `none`
- Test status: `sealed_not_loaded`
- Corpus status: `not_applicable_constructed_fixture`
- Split status: `not_applicable_constructed_fixture`
- Configuration: `{config_sha256}`
- Fixture: `{fixture_sha256}`

## Result

| Measure | Result |
| --- | ---: |
| Expected events | {event_count} |
| Emitted events | {event_count} |
| Matched events | {event_count} |
| Fixture recall | 1.000 |
| Missed events | 0 |
| Fragmented truth events | 0 |
| Duplicate proposals | 0 |
| Extra proposals | 0 |
| False candidates | 0 |
| Mean absolute onset error | {onset_error_ms:.3f} ms |
| Mean absolute offset error | {offset_error_ms:.3f} ms |

## Constructed scenario coverage

{scenarios}

## Limits

The fixture contains invented numeric observations, not video, landmarks, portable
features, public-corpus samples, or participant data. It proves deterministic state
transitions, boundary handling, gap tolerance, and duplicate suppression. It does not
tune thresholds or estimate natural-session recall, false activations per hour,
generalization, classification quality, or continuous-sign recognition.
"""


@app.command("candidate-events")
def candidate_events_command(
    config_path: Annotated[Path, typer.Argument(help="Checked-in detector config JSON.")],
    fixture_path: Annotated[Path, typer.Argument(help="Constructed signal fixture JSON.")],
    report: Annotated[Path, typer.Option(help="New path for the sanitized report.")],
) -> None:
    """Verify exact candidate-event behavior on invented observations."""

    try:
        config = load_candidate_event_config(config_path)
        fixture = _load_fixture(fixture_path)
        detector = CandidateEventDetector(config)
        events: list[CandidateEvent] = []
        transitions: list[dict[str, int | str]] = []
        previous_state: CandidateState = detector.state
        for frame_index, observation in enumerate(fixture.observations):
            event = detector.push(observation)
            if detector.state != previous_state:
                transitions.append({"frame_index": frame_index, "state": detector.state})
                previous_state = detector.state
            if event is not None:
                events.append(event)
        final_event = detector.finish()
        if final_event is not None:
            events.append(final_event)
        expected = tuple(row.model_dump(mode="json") for row in fixture.expected_transitions)
        if tuple(transitions) != expected:
            raise ValueError("candidate-event transition conformance failed")
        onset_error_ms, offset_error_ms = _ordered_conformance(fixture.truth_events, tuple(events))
        fixture_sha256 = canonical_sha256(fixture, domain=_FIXTURE_DOMAIN)
        rendered = _markdown(
            fixture,
            candidate_event_config_digest(config),
            fixture_sha256,
            len(events),
            onset_error_ms,
            offset_error_ms,
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Candidate-event evaluation failed: inputs or output are invalid.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Candidate-event conformance verified: "
        f"{len(events)}/{len(fixture.truth_events)} expected events."
    )


@app.command("continuous-replay")
def continuous_replay_command(
    config_path: Annotated[Path, typer.Argument(help="Checked-in replay scoring config.")],
    fixture_path: Annotated[Path, typer.Argument(help="Constructed replay fixture JSON.")],
    candidate_event_config_path: Annotated[Path, typer.Argument(help="Detector config.")],
    decision_policy_path: Annotated[Path, typer.Argument(help="Decision policy.")],
    report: Annotated[Path, typer.Option(help="New path for the canonical JSON report.")],
) -> None:
    """Score timestamped constructed decisions without claiming live performance."""

    from signlab.continuous_replay import ContinuousReplayError, run_continuous_replay

    try:
        result = run_continuous_replay(
            config_path,
            fixture_path,
            candidate_event_config_path,
            decision_policy_path,
            report,
        )
    except ContinuousReplayError as error:
        typer.echo("Continuous replay scoring failed: inputs or output are invalid.", err=True)
        raise typer.Exit(code=1) from error
    counts = cast(dict[str, int], result["counts"])
    typer.echo(
        "Constructed replay scoring verified: "
        f"{counts['temporal_matches']}/{counts['truths']} accepted target decisions matched; "
        "test sealed."
    )
