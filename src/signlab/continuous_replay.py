"""Deterministic scoring for constructed continuous-replay traces."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Self, cast

from pydantic import Field, ValidationError, model_validator

from signlab.candidate_events import candidate_event_config_digest, load_candidate_event_config
from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    NonNegativeSafeInteger,
    PositiveSafeInteger,
    SemanticVersion,
    StableId,
    StrictContractModel,
)
from signlab.contracts.dataset import MediaIntervalV1
from signlab.contracts.taxonomy import Sha256Digest

TargetLabel = Literal["hello", "no", "please", "thank_you", "yes"]
DecisionKind = Literal["target", "other", "abstain"]

_CONFIG_DOMAIN: Final = "signlab-continuous-replay-scoring-config/1"
_FIXTURE_DOMAIN: Final = "signlab-continuous-replay-fixture/1"
_PPM: Final = 1_000_000
_US_PER_HOUR: Final = 3_600_000_000


class ContinuousReplayError(ValueError):
    pass


class ReplayScoringConfig(StrictContractModel):
    format: Literal["signlab-continuous-replay-scoring-config/1"]
    config_id: StableId
    version: SemanticVersion
    matching_rule: Literal["truth_chronological_maximum_iou_then_earliest/1"]
    minimum_iou_ppm: Annotated[PositiveSafeInteger, Field(le=_PPM)]
    bootstrap_unit: Literal["group"]
    bootstrap_replicates: Annotated[PositiveSafeInteger, Field(le=10_000)]
    bootstrap_seed: NonNegativeSafeInteger
    candidate_event_config_sha256: Sha256Digest
    decision_policy_sha256: Sha256Digest
    research_model_sha256: Sha256Digest
    test_partition_policy: Literal["sealed_not_loaded"]


class ReplayTruth(StrictContractModel):
    truth_id: StableId
    interval: MediaIntervalV1
    label: TargetLabel


class ReplayDecision(StrictContractModel):
    decision_id: StableId
    interval: MediaIntervalV1
    kind: DecisionKind
    label: TargetLabel | Literal["other"] | None
    decision_timestamp_us: NonNegativeSafeInteger

    @model_validator(mode="after")
    def _require_decision_shape(self) -> Self:
        expected = self.label in {"hello", "no", "please", "thank_you", "yes"}
        if (self.kind == "target") != expected:
            raise ValueError("only target decisions require a target label")
        if (self.kind == "other") != (self.label == "other"):
            raise ValueError("other decisions require only the other label")
        if self.kind == "abstain" and self.label is not None:
            raise ValueError("abstain decisions cannot carry a label")
        if self.decision_timestamp_us < self.interval.end_us:
            raise ValueError("decision timestamp cannot precede its candidate interval")
        return self


class ReplaySession(StrictContractModel):
    session_alias: StableId
    group_alias: StableId
    duration_us: PositiveSafeInteger
    truths: tuple[ReplayTruth, ...]
    decisions: tuple[ReplayDecision, ...]

    @model_validator(mode="after")
    def _require_bounded_ordered_rows(self) -> Self:
        for rows, identity in ((self.truths, "truth_id"), (self.decisions, "decision_id")):
            keys = tuple(
                (row.interval.start_us, row.interval.end_us, cast(str, getattr(row, identity)))
                for row in rows
            )
            if keys != tuple(sorted(keys)) or len({key[2] for key in keys}) != len(keys):
                raise ValueError("replay rows must be ordered with unique identities")
            if any(row.interval.end_us > self.duration_us for row in rows):
                raise ValueError("replay intervals must fit within the session")
        if any(
            left.interval.end_us > right.interval.start_us
            for left, right in zip(self.truths, self.truths[1:], strict=False)
        ):
            raise ValueError("truth intervals cannot overlap")
        if any(row.decision_timestamp_us > self.duration_us for row in self.decisions):
            raise ValueError("decision timestamps must fit within the session")
        return self


class ContinuousReplayFixture(StrictContractModel):
    format: Literal["signlab-continuous-replay-fixture/1"]
    fixture_id: StableId
    sessions: Annotated[tuple[ReplaySession, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def _require_fixture_coverage(self) -> Self:
        aliases = tuple(row.session_alias for row in self.sessions)
        if aliases != tuple(sorted(set(aliases))):
            raise ValueError("session aliases must be unique and sorted")
        if not any(row.truths for row in self.sessions) or not any(
            not row.truths for row in self.sessions
        ):
            raise ValueError("fixture requires positive and negative-only sessions")
        return self


@dataclass(frozen=True, slots=True)
class _Match:
    truth_index: int
    decision_index: int


def _load_canonical(
    path: Path, model: type[ReplayScoringConfig] | type[ContinuousReplayFixture]
) -> ReplayScoringConfig | ContinuousReplayFixture:
    raw = path.read_bytes()
    parsed = model.model_validate_json(canonical_json_bytes(parse_json_object(raw)), strict=True)
    if raw != canonical_json_bytes(parsed) + b"\n":
        raise ValueError("input is not canonical")
    return parsed


def load_replay_scoring_config(path: str | Path) -> ReplayScoringConfig:
    try:
        return cast(ReplayScoringConfig, _load_canonical(Path(path), ReplayScoringConfig))
    except (CanonicalizationError, OSError, ValidationError, ValueError) as error:
        raise ContinuousReplayError("replay scoring configuration is invalid") from error


def load_continuous_replay_fixture(path: str | Path) -> ContinuousReplayFixture:
    try:
        return cast(
            ContinuousReplayFixture,
            _load_canonical(Path(path), ContinuousReplayFixture),
        )
    except (CanonicalizationError, OSError, ValidationError, ValueError) as error:
        raise ContinuousReplayError("continuous replay fixture is invalid") from error


def _iou_ppm(left: MediaIntervalV1, right: MediaIntervalV1) -> int:
    overlap = max(0, min(left.end_us, right.end_us) - max(left.start_us, right.start_us))
    union = max(left.end_us, right.end_us) - min(left.start_us, right.start_us)
    return overlap * _PPM // union


def _matches(session: ReplaySession, minimum_iou_ppm: int) -> tuple[_Match, ...]:
    accepted = tuple(index for index, row in enumerate(session.decisions) if row.kind == "target")
    unused = set(accepted)
    matches: list[_Match] = []
    for truth_index, truth in enumerate(session.truths):
        eligible = [
            (
                -_iou_ppm(truth.interval, session.decisions[index].interval),
                session.decisions[index].interval.start_us,
                session.decisions[index].decision_id,
                index,
            )
            for index in unused
            if _iou_ppm(truth.interval, session.decisions[index].interval) >= minimum_iou_ppm
        ]
        if eligible:
            decision_index = min(eligible)[-1]
            unused.remove(decision_index)
            matches.append(_Match(truth_index, decision_index))
    return tuple(matches)


def _counts(session: ReplaySession, minimum_iou_ppm: int) -> tuple[dict[str, int], list[int]]:
    matches = _matches(session, minimum_iou_ppm)
    used = {row.decision_index for row in matches}
    accepted = {index for index, row in enumerate(session.decisions) if row.kind == "target"}
    unmatched = accepted - used
    duplicates = sum(
        any(
            _iou_ppm(truth.interval, session.decisions[index].interval) >= minimum_iou_ppm
            for truth in session.truths
        )
        for index in unmatched
    )
    correct = sum(
        session.truths[row.truth_index].label == session.decisions[row.decision_index].label
        for row in matches
    )
    latencies = [
        session.decisions[row.decision_index].decision_timestamp_us
        - session.truths[row.truth_index].interval.end_us
        for row in matches
    ]
    return {
        "truths": len(session.truths),
        "decisions": len(session.decisions),
        "accepted_targets": len(accepted),
        "other": sum(row.kind == "other" for row in session.decisions),
        "abstain": sum(row.kind == "abstain" for row in session.decisions),
        "temporal_matches": len(matches),
        "correct_labels": correct,
        "wrong_labels": len(matches) - correct,
        "misses": len(session.truths) - len(matches),
        "duplicates": duplicates,
        "false_target_activations": len(unmatched) - duplicates,
        "duration_us": session.duration_us,
    }, latencies


def _aggregate(
    sessions: Sequence[ReplaySession], threshold: int
) -> tuple[dict[str, int], list[int]]:
    total: defaultdict[str, int] = defaultdict(int)
    latencies: list[int] = []
    for session in sessions:
        counts, observed = _counts(session, threshold)
        for name, value in counts.items():
            total[name] += value
        latencies.extend(observed)
    return dict(total), latencies


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _metrics(counts: Mapping[str, int]) -> dict[str, float | None]:
    truths = counts["truths"]
    accepted = counts["accepted_targets"]
    matched = counts["temporal_matches"]
    correct = counts["correct_labels"]
    return {
        "temporal_precision": _ratio(matched, accepted),
        "temporal_recall": _ratio(matched, truths),
        "temporal_f1": _ratio(2 * matched, accepted + truths),
        "correct_label_precision": _ratio(correct, accepted),
        "correct_label_recall": _ratio(correct, truths),
        "correct_label_f1": _ratio(2 * correct, accepted + truths),
        "correct_target_rate": _ratio(correct, truths),
        "false_target_activations_per_hour": _ratio(
            counts["false_target_activations"] * _US_PER_HOUR,
            counts["duration_us"],
        ),
    }


def _nearest_rank(values: Sequence[int | float], fraction: float) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _bootstrap(config: ReplayScoringConfig, fixture: ContinuousReplayFixture) -> dict[str, object]:
    grouped: defaultdict[str, list[ReplaySession]] = defaultdict(list)
    for session in fixture.sessions:
        grouped[session.group_alias].append(session)
    if len(grouped) < 2:
        return {"status": "insufficient_groups", "group_count": len(grouped)}
    names = sorted(grouped)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    generator = random.Random(config.bootstrap_seed)
    for _ in range(config.bootstrap_replicates):
        selected = [session for _ in names for session in grouped[generator.choice(names)]]
        counts, _ = _aggregate(selected, config.minimum_iou_ppm)
        for name, value in _metrics(counts).items():
            if value is not None:
                samples[name].append(value)
    intervals = {
        name: {
            "valid_replicates": len(values),
            "lower": _nearest_rank(values, 0.025),
            "upper": _nearest_rank(values, 0.975),
        }
        for name, values in sorted(samples.items())
    }
    return {
        "status": "constructed_conformance_only",
        "unit": config.bootstrap_unit,
        "group_count": len(grouped),
        "replicates": config.bootstrap_replicates,
        "seed": config.bootstrap_seed,
        "intervals": intervals,
    }


def _policy_identity(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    payload = parse_json_object(raw)
    if (
        raw != canonical_json_bytes(payload) + b"\n"
        or payload.get("format") != "signlab-decision-policy/1"
    ):
        raise ValueError("decision policy is invalid")
    identities = payload.get("identities")
    if not isinstance(identities, dict) or not isinstance(identities.get("model_sha256"), str):
        raise ValueError("decision policy model identity is invalid")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}", cast(str, identities["model_sha256"])


def build_continuous_replay_report(
    config: ReplayScoringConfig,
    fixture: ContinuousReplayFixture,
    *,
    candidate_event_config_path: str | Path,
    decision_policy_path: str | Path,
) -> dict[str, object]:
    try:
        detector_sha256 = candidate_event_config_digest(
            load_candidate_event_config(candidate_event_config_path)
        )
        policy_sha256, model_sha256 = _policy_identity(Path(decision_policy_path))
    except (CanonicalizationError, OSError, TypeError, ValueError) as error:
        raise ContinuousReplayError("replay component identities are invalid") from error
    if (
        detector_sha256 != config.candidate_event_config_sha256
        or policy_sha256 != config.decision_policy_sha256
        or model_sha256 != config.research_model_sha256
    ):
        raise ContinuousReplayError("replay component identities do not match configuration")

    counts, offsets = _aggregate(fixture.sessions, config.minimum_iou_ppm)
    post_gesture_latencies = [value for value in offsets if value >= 0]
    return {
        "format": "signlab-continuous-replay-report/1",
        "evidence_kind": "constructed_continuous_replay_scoring_conformance",
        "metric_claim": "none",
        "test_status": config.test_partition_policy,
        "natural_session_status": "unavailable",
        "live_path_status": "unavailable_pending_browser_runtime",
        "model_bundle_status": "unavailable_pending_contract",
        "identities": {
            "evaluator_version": config.version,
            "config_sha256": canonical_sha256(config, domain=_CONFIG_DOMAIN),
            "fixture_sha256": canonical_sha256(fixture, domain=_FIXTURE_DOMAIN),
            "candidate_event_config_sha256": detector_sha256,
            "decision_policy_sha256": policy_sha256,
            "research_model_sha256": model_sha256,
        },
        "matching": {
            "rule": config.matching_rule,
            "decision_scope": "accepted_targets_only",
            "minimum_iou_ppm": config.minimum_iou_ppm,
            "label_independent": True,
        },
        "counts": counts,
        "metrics": _metrics(counts),
        "post_gesture_decision_latency_us": {
            "sample_count": len(post_gesture_latencies),
            "early_decisions": len(offsets) - len(post_gesture_latencies),
            "p50": _nearest_rank(post_gesture_latencies, 0.50),
            "p95": _nearest_rank(post_gesture_latencies, 0.95),
        },
        "bootstrap": _bootstrap(config, fixture),
        "unavailable_measurements": [
            "landmarker_latency",
            "preprocessing_latency",
            "inference_latency",
            "throughput",
            "frame_drops",
        ],
        "limits": [
            "Constructed intervals and decisions prove scoring mechanics only.",
            "No video, live runtime, model bundle, natural session, or locked test data was used.",
            "Reported rates, latencies, and bootstrap intervals are not natural-use estimates.",
        ],
    }


def run_continuous_replay(
    config_path: str | Path,
    fixture_path: str | Path,
    candidate_event_config_path: str | Path,
    decision_policy_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    try:
        config = load_replay_scoring_config(config_path)
        fixture = load_continuous_replay_fixture(fixture_path)
        report = build_continuous_replay_report(
            config,
            fixture,
            candidate_event_config_path=candidate_event_config_path,
            decision_policy_path=decision_policy_path,
        )
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(canonical_json_bytes(report) + b"\n")
        return report
    except (CanonicalizationError, OSError, TypeError, ValueError) as error:
        raise ContinuousReplayError("continuous replay scoring failed") from error
