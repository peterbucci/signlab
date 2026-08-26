from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from signlab.contracts import (
    BUILTIN_TAXONOMY_DIGEST,
    AnnotationsTableV1,
    SessionRowV1,
    load_builtin_taxonomy,
    taxonomy_reference,
    validate_dataset_table,
)
from signlab.contracts.taxonomy import EXPECTED_TARGET_IDS

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "public" / "collection"
PLAN_PATH = FIXTURE_ROOT / "mock-session-plan.json"
ANNOTATIONS_PATH = FIXTURE_ROOT / "annotations-table-1.mock-session.json"
SOURCE_HARD_NEGATIVE_KINDS = (
    "partial_target",
    "oov_gesture",
    "incidental_activity",
    "two_hand_non_target",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _objects(value: object) -> list[dict[str, Any]]:
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, Any]], value)


def _utc(value: object) -> datetime:
    assert isinstance(value, str)
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _checked_annotations() -> AnnotationsTableV1:
    checked = validate_dataset_table(_load_object(ANNOTATIONS_PATH))
    assert isinstance(checked, AnnotationsTableV1)
    return checked


def test_mock_annotations_pass_the_published_schema_and_authoritative_contract() -> None:
    payload = _load_object(ANNOTATIONS_PATH)
    schema = cast(
        dict[str, Any],
        json.loads(
            files("signlab.resources.datasets")
            .joinpath("schemas", "annotations-table-1.schema.json")
            .read_text(encoding="utf-8")
        ),
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    checked = _checked_annotations()

    assert checked.model_dump(mode="json", round_trip=True) == payload
    annotation_ids = tuple(row.annotation_id for row in checked.rows)
    assert annotation_ids == tuple(sorted(annotation_ids))


def test_mock_plan_binds_the_exact_immutable_taxonomy() -> None:
    plan = _load_object(PLAN_PATH)
    expected_reference = taxonomy_reference(load_builtin_taxonomy()).model_dump(
        mode="json", round_trip=True
    )

    assert plan["taxonomy"] == expected_reference
    assert expected_reference["sha256"] == BUILTIN_TAXONOMY_DIGEST
    assert tuple(expected_reference[key] for key in ("id", "version")) == (
        "signlab-five",
        "1.0.0",
    )


def test_mock_visits_are_separated_and_emit_distinct_contract_valid_sessions() -> None:
    plan = _load_object(PLAN_PATH)
    participant = cast(dict[str, Any], plan["participant"])
    visits = _objects(plan["visits"])
    validated_sessions: list[SessionRowV1] = []
    visit_starts: list[datetime] = []

    assert len(visits) == 2
    for visit in visits:
        sessions = tuple(
            SessionRowV1.model_validate(session, strict=True)
            for session in _objects(visit["sessions"])
        )
        validated_sessions.extend(sessions)
        visit_starts.append(min(_utc(session.started_at) for session in sessions))

        assert {session.capture_mode for session in sessions} == {"isolated", "continuous"}
        assert {session.participant_id for session in sessions} == {participant["participant_id"]}
        assert len({session.session_id for session in sessions}) == 2
        assert len({session.device_id for session in sessions}) == 1

    assert visit_starts[1] - visit_starts[0] >= timedelta(hours=24)
    assert len({session.session_id for session in validated_sessions}) == 4
    assert len({session.device_id for session in validated_sessions}) == 2


def test_realized_prompt_orders_are_balanced_non_adjacent_and_visit_specific() -> None:
    visits = _objects(_load_object(PLAN_PATH)["visits"])
    realized_orders: list[tuple[str, ...]] = []
    randomization_seeds: list[str] = []
    occurrence_ids: list[str] = []

    for visit in visits:
        prompts = _objects(visit["isolated_prompt_sequence"])
        labels = tuple(cast(str, prompt["label_id"]) for prompt in prompts)
        condition = cast(dict[str, Any], visit["condition_profile"])
        randomization = cast(dict[str, Any], visit["prompt_randomization"])
        visit_id = cast(str, visit["visit_id"])

        assert tuple(prompt["ordinal"] for prompt in prompts) == tuple(range(1, 26))
        assert Counter(labels) == Counter({label: 5 for label in EXPECTED_TARGET_IDS})
        assert all(left != right for left, right in pairwise(labels))
        assert condition["scope"] == "entire_visit"
        assert set(labels) == set(EXPECTED_TARGET_IDS)
        for label in EXPECTED_TARGET_IDS:
            assert sorted(
                cast(int, prompt["repetition"]) for prompt in prompts if prompt["label_id"] == label
            ) == list(range(1, 6))
        for prompt in prompts:
            occurrence_id = cast(str, prompt["prompt_occurrence_id"])
            assert occurrence_id == f"{visit_id}_prompt_{cast(int, prompt['ordinal']):02d}"
            occurrence_ids.append(occurrence_id)
        assert randomization["algorithm_id"] == "balanced_constrained_shuffle"
        assert randomization["algorithm_version"] == "mock-1"
        assert randomization["realized_order_authoritative"] is True
        assert randomization["rerolled_for_performance"] is False
        randomization_seeds.append(cast(str, randomization["mock_seed"]))
        realized_orders.append(labels)

    assert realized_orders[0] != realized_orders[1]
    assert randomization_seeds == ["signlab-mock-visit-01", "signlab-mock-visit-02"]
    assert len(occurrence_ids) == len(set(occurrence_ids)) == 50


def test_continuous_plan_covers_inactivity_targets_and_source_hard_negatives() -> None:
    visits = _objects(_load_object(PLAN_PATH)["visits"])
    hard_negative_counts: Counter[str] = Counter()

    for visit in visits:
        activities = _objects(visit["continuous_activities"])
        target_labels = {
            cast(str, activity["label_id"])
            for activity in activities
            if activity["activity"] == "target"
        }
        inactivity_seconds = sum(
            cast(int, activity["duration_seconds"])
            for activity in activities
            if activity["activity"] == "inactivity"
        )
        hard_negative_counts.update(
            cast(str, activity["other_kind"])
            for activity in activities
            if activity["activity"] == "hard_negative"
        )

        assert target_labels == set(EXPECTED_TARGET_IDS)
        assert inactivity_seconds == 60
        assert {
            cast(str, activity["position"])
            for activity in activities
            if activity["activity"] == "inactivity"
        } == {"before", "between", "after"}
        assert {activity["activity"] for activity in activities} >= {
            "direct_transition",
            "natural_mistake",
        }
        assert {
            cast(str, activity["retention"])
            for activity in activities
            if activity["activity"] in {"direct_transition", "natural_mistake"}
        } == {"context_only", "preserve"}

    assert hard_negative_counts == Counter(
        {other_kind: 2 for other_kind in SOURCE_HARD_NEGATIVE_KINDS}
    )
    assert "transition_fragment" not in hard_negative_counts


def test_mock_conditions_match_the_repeatable_pilot_profiles() -> None:
    visits = _objects(_load_object(PLAN_PATH)["visits"])
    profiles = tuple(cast(dict[str, Any], visit["condition_profile"]) for visit in visits)

    assert tuple(profile["profile_id"] for profile in profiles) == (
        "near_soft_front_landscape",
        "medium_diffuse_side_portrait",
    )
    assert tuple(profile["distance_m"] for profile in profiles) == (0.8, 1.2)
    assert tuple(profile["camera_orientation"] for profile in profiles) == (
        "landscape",
        "portrait",
    )


def test_mock_annotations_cover_reviewed_targets_negatives_and_exclusions() -> None:
    rows = _checked_annotations().rows
    target_rows = tuple(row for row in rows if row.label_id in EXPECTED_TARGET_IDS)
    other_rows = tuple(row for row in rows if row.label_id == "other")
    exclusion_rows = tuple(row for row in rows if row.disposition != "class_label")

    assert {row.label_id for row in target_rows} == set(EXPECTED_TARGET_IDS)
    assert {row.other_kind for row in other_rows} == set(SOURCE_HARD_NEGATIVE_KINDS)
    assert "transition_fragment" not in {row.other_kind for row in rows}
    assert {(row.disposition, row.reason_code) for row in exclusion_rows} == {
        ("ambiguous", "unusable_occlusion"),
        ("ignore", "third_party_presence"),
    }
    assert all(row.review_status in {"reviewed", "adjudicated"} for row in rows)
    assert all(
        row.eligible_for_training
        == (row.disposition == "class_label" and row.review_status in {"reviewed", "adjudicated"})
        for row in rows
    )


def test_mock_observations_supply_timestamps_and_evidence_without_answer_fields() -> None:
    plan = _load_object(PLAN_PATH)
    observations = _objects(plan["mock_observations"])
    annotations = _checked_annotations().rows
    answer_fields = {
        "disposition",
        "eligible_for_training",
        "label_id",
        "other_kind",
        "reason_code",
        "review_status",
    }

    assert len(observations) == len(annotations) == 11
    assert tuple(observation["interval"] for observation in observations) == tuple(
        row.interval.model_dump(mode="json", round_trip=True) for row in annotations
    )
    assert tuple(observation["annotation_id"] for observation in observations) == tuple(
        row.annotation_id for row in annotations
    )
    assert tuple(observation["session_id"] for observation in observations) == tuple(
        row.session_id for row in annotations
    )
    assert tuple(observation["source_recording_id"] for observation in observations) == tuple(
        row.source_recording_id for row in annotations
    )
    assert all(
        isinstance(observation["visible_evidence"], str) and observation["visible_evidence"]
        for observation in observations
    )
    review_status_by_path = {
        "conflict_resolved_by_adjudication": "adjudicated",
        "independent_review_agreed": "reviewed",
    }
    assert tuple(
        review_status_by_path[observation["review_path"]] for observation in observations
    ) == tuple(row.review_status for row in annotations)
    assert all(answer_fields.isdisjoint(observation) for observation in observations)


def test_evidence_is_explicitly_synthetic_and_never_claims_collection_authority() -> None:
    plan = _load_object(PLAN_PATH)
    evidence = cast(dict[str, Any], plan["evidence"])
    governance = cast(dict[str, Any], plan["governance"])
    participant = cast(dict[str, Any], plan["participant"])
    visits = _objects(plan["visits"])

    assert plan["fixture_only"] is True
    assert participant["synthetic"] is True
    assert evidence == {
        "annotation_source": "synthetic_timeline",
        "camera_used": False,
        "contains_person_data": False,
        "media_created": False,
        "purpose": "protocol_dry_run_only",
    }
    assert governance == {
        "authorization_claimed": False,
        "collection_readiness_status": "blocked",
        "real_collection_authorized": False,
    }

    continuous_session_ids = {
        session.session_id
        for visit in visits
        for session in (
            SessionRowV1.model_validate(item, strict=True) for item in _objects(visit["sessions"])
        )
        if session.capture_mode == "continuous"
    }
    annotations = _checked_annotations().rows
    assert {row.participant_id for row in annotations} == {participant["participant_id"]}
    assert {row.session_id for row in annotations} <= continuous_session_ids

    for visit in visits:
        checklists = cast(dict[str, dict[str, Any]], visit["checklists"])
        assert set(checklists) == {"capture_quality", "consent_verification"}
        assert all(item["result"] == "not_applicable" for item in checklists.values())
        assert all(
            item["reason_code"] == "synthetic_no_person_no_camera" for item in checklists.values()
        )
        assert checklists["capture_quality"]["camera_opened"] is False
        assert checklists["capture_quality"]["media_reviewed"] is False
        assert checklists["consent_verification"]["authenticated_receipt_present"] is False
        assert checklists["consent_verification"]["authorization_granted"] is False
