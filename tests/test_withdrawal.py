from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from signlab.contracts.governance import (
    GovernanceContractError,
    lineage_inventory_digest,
    validate_lineage_inventory,
    validate_withdrawal_report,
    validate_withdrawal_request,
    withdrawal_request_digest,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference
from signlab.governance.withdrawal import (
    WithdrawalPlanningError,
    plan_withdrawal_dry_run,
    render_withdrawal_report_markdown,
    withdrawal_report_json_bytes,
)

ZERO_DIGEST = "sha256:" + "0" * 64
PARTICIPANT_A = "participant_" + "1" * 32
PARTICIPANT_B = "participant_" + "2" * 32
RECEIPT_A = "receipt_" + "1" * 32
RECEIPT_B = "receipt_" + "2" * 32
RECORDING_A = "recording_" + "1" * 32
RECORDING_B = "recording_" + "2" * 32
GRANT_A = "grant_" + "1" * 32
GRANT_B = "grant_" + "2" * 32
CREATED_AT = "2026-08-26T10:00:00Z"
REQUESTED_AT = "2026-08-26T11:00:00Z"
GENERATED_AT = "2026-08-26T12:00:00Z"


def _id(prefix: str, number: int) -> str:
    return f"{prefix}_{number:032x}"


def _sha(number: int) -> str:
    return f"sha256:{number:064x}"


def _asset(
    number: int,
    kind: str,
    *,
    parents: tuple[int, ...] = (),
    participant: str = PARTICIPANT_A,
    recording: str = RECORDING_A,
    receipt: str = RECEIPT_A,
    grant: str = GRANT_A,
    shared: bool = False,
) -> dict[str, object]:
    asset_id = _id("asset", number)
    participant_ids = (PARTICIPANT_A, PARTICIPANT_B) if shared else (participant,)
    recording_ids = (RECORDING_A, RECORDING_B) if shared else (recording,)
    receipt_ids = (RECEIPT_A, RECEIPT_B) if shared else (receipt,)
    grant_ids = (GRANT_A, GRANT_B) if shared else (grant,)
    return {
        "schema_version": "governance-asset/1",
        "asset_id": asset_id,
        "asset_kind": kind,
        "logical_uri": (f"signlab://store-00000000000000000000000000000001/{kind}/{asset_id}"),
        "sha256": _sha(number),
        "taxonomy": taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json"),
        "created_at": CREATED_AT,
        "participant_ids": participant_ids,
        "recording_ids": recording_ids,
        "receipt_ids": receipt_ids,
        "grant_ids": grant_ids,
        "parent_asset_ids": tuple(_id("asset", parent) for parent in parents),
        "lifecycle_state": "active",
        "invalidated_at": None,
    }


def _inventory_document() -> dict[str, object]:
    # Two private branches meet at dataset 6. Everything from that node onward is
    # intentionally shared, proving that a withdrawal cannot stop at signer-local data.
    assets = [
        _asset(1, "raw_recording"),
        _asset(
            2,
            "raw_recording",
            participant=PARTICIPANT_B,
            recording=RECORDING_B,
            receipt=RECEIPT_B,
            grant=GRANT_B,
        ),
        _asset(3, "annotation", parents=(1,)),
        _asset(4, "derived_features", parents=(3,)),
        _asset(
            5,
            "derived_features",
            parents=(2,),
            participant=PARTICIPANT_B,
            recording=RECORDING_B,
            receipt=RECEIPT_B,
            grant=GRANT_B,
        ),
        _asset(6, "dataset_version", parents=(4, 5), shared=True),
        _asset(7, "split_version", parents=(6,), shared=True),
        _asset(8, "experiment_run", parents=(7,), shared=True),
        _asset(9, "model_artifact", parents=(8,), shared=True),
        _asset(10, "evaluation_report", parents=(9,), shared=True),
        _asset(11, "public_demo", parents=(9, 10), shared=True),
        _asset(12, "cache", parents=(4,)),
        _asset(13, "backup_copy", parents=(1,)),
        _asset(14, "withdrawal_tombstone", parents=(1,)),
    ]
    payload: dict[str, object] = {
        "schema_version": "lineage-inventory/1",
        "inventory_id": _id("inventory", 1),
        "taxonomy": taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json"),
        "generated_at": REQUESTED_AT,
        "assets": assets,
        "inventory_sha256": ZERO_DIGEST,
    }
    payload["inventory_sha256"] = lineage_inventory_digest(payload)
    return payload


def _request_document(
    *,
    participant_id: str = PARTICIPANT_A,
    receipt_ids: tuple[str, ...] = (RECEIPT_A,),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "withdrawal-request/1",
        "request_id": _id("withdrawal", 1),
        "participant_id": participant_id,
        "receipt_ids": receipt_ids,
        "requested_at": REQUESTED_AT,
        "effective_at": REQUESTED_AT,
        "target": "all_participant_data",
        "identity_verification_attestation_sha256": _sha(999),
        "request_sha256": ZERO_DIGEST,
    }
    payload["request_sha256"] = withdrawal_request_digest(payload)
    return payload


def _mutated_inventory(mutator: Callable[[dict[str, object]], None]) -> dict[str, object]:
    payload = cast(dict[str, object], json.loads(json.dumps(_inventory_document())))
    mutator(payload)
    payload["inventory_sha256"] = lineage_inventory_digest(payload)
    return payload


def test_shared_descendants_are_planned_once_and_other_private_branch_is_not() -> None:
    request = validate_withdrawal_request(_request_document())
    inventory = validate_lineage_inventory(_inventory_document())
    before = inventory.model_dump(mode="json", round_trip=True)

    report = plan_withdrawal_dry_run(request, inventory, generated_at=GENERATED_AT)

    assert validate_withdrawal_report(report) == report
    assert report.status == "complete"
    assert report.direct_asset_count == 1
    assert report.downstream_asset_count == 11
    assert report.affected_asset_count == 12
    assert report.unresolved_asset_ids == ()
    assert tuple(impact.asset_id for impact in report.impacts) == tuple(
        sorted(
            {
                _id("asset", 1),
                _id("asset", 3),
                _id("asset", 4),
                _id("asset", 6),
                _id("asset", 7),
                _id("asset", 8),
                _id("asset", 9),
                _id("asset", 10),
                _id("asset", 11),
                _id("asset", 12),
                _id("asset", 13),
                _id("asset", 14),
            }
        )
    )
    assert _id("asset", 2) not in {impact.asset_id for impact in report.impacts}
    assert _id("asset", 5) not in {impact.asset_id for impact in report.impacts}
    assert (
        next(impact for impact in report.impacts if impact.asset_id == _id("asset", 1)).relationship
        == "direct"
    )
    assert (
        next(impact for impact in report.impacts if impact.asset_id == _id("asset", 6)).relationship
        == "downstream"
    )
    assert all(
        "invalidate" in impact.planned_actions
        for impact in report.impacts
        if impact.asset_kind != "withdrawal_tombstone"
    )
    assert inventory.model_dump(mode="json", round_trip=True) == before


def test_action_policy_covers_every_governance_asset_kind() -> None:
    report = plan_withdrawal_dry_run(
        _request_document(),
        _inventory_document(),
        generated_at=GENERATED_AT,
    )
    actions = {impact.asset_kind: impact.planned_actions for impact in report.impacts}

    assert actions == {
        "raw_recording": ("delete_primary", "invalidate", "revoke_access"),
        "backup_copy": ("invalidate", "purge_backup"),
        "cache": ("delete_primary", "invalidate", "rebuild"),
        "annotation": ("delete_primary", "invalidate", "rebuild"),
        "derived_features": ("delete_primary", "invalidate", "rebuild"),
        "dataset_version": ("invalidate", "rebuild"),
        "split_version": ("invalidate", "rebuild"),
        "experiment_run": ("invalidate", "rerun"),
        "model_artifact": ("invalidate", "retire", "retrain"),
        "evaluation_report": ("invalidate", "reevaluate", "republish", "retract"),
        "public_demo": ("invalidate", "republish", "retract"),
        "withdrawal_tombstone": ("retain",),
    }


def test_report_json_markdown_and_ids_are_deterministic_and_idempotent() -> None:
    first = plan_withdrawal_dry_run(
        _request_document(),
        _inventory_document(),
        generated_at=GENERATED_AT,
    )
    second = plan_withdrawal_dry_run(
        validate_withdrawal_request(_request_document()),
        validate_lineage_inventory(_inventory_document()),
        generated_at=GENERATED_AT,
    )

    assert first == second
    assert withdrawal_report_json_bytes(first) == withdrawal_report_json_bytes(second)
    assert render_withdrawal_report_markdown(first) == render_withdrawal_report_markdown(second)
    assert json.loads(withdrawal_report_json_bytes(first))["report_sha256"] == first.report_sha256
    markdown = render_withdrawal_report_markdown(first)
    assert markdown.endswith("\n")
    assert "Storage mutations performed: `false`" in markdown
    assert "Simulated non-tombstone state: `invalidated`" in markdown
    assert "Withdrawal tombstones: `retained`" in markdown
    assert "`dataset_version`" in markdown
    assert PARTICIPANT_A not in markdown


@pytest.mark.parametrize(
    ("request_document", "message"),
    [
        (
            _request_document(participant_id="participant_" + "f" * 32),
            "participant is unknown",
        ),
        (
            _request_document(receipt_ids=("receipt_" + "f" * 32,)),
            "consent not attached",
        ),
    ],
)
def test_unknown_participant_or_consent_fails_closed(
    request_document: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(WithdrawalPlanningError, match=message):
        plan_withdrawal_dry_run(
            request_document,
            _inventory_document(),
            generated_at=GENERATED_AT,
        )


def test_partial_retry_replans_every_affected_node_with_the_full_idempotent_actions() -> None:
    def mark_partial_progress(payload: dict[str, object]) -> None:
        assets = cast_assets(payload)
        assets[0]["lifecycle_state"] = "invalidation_pending"
        assets[2].update(
            {
                "lifecycle_state": "invalidated",
                "invalidated_at": REQUESTED_AT,
            }
        )
        assets[5].update(
            {
                "lifecycle_state": "invalidated",
                "invalidated_at": REQUESTED_AT,
            }
        )

    baseline = plan_withdrawal_dry_run(
        _request_document(),
        _inventory_document(),
        generated_at=GENERATED_AT,
    )
    partial_inventory = _mutated_inventory(mark_partial_progress)
    first_retry = plan_withdrawal_dry_run(
        _request_document(),
        partial_inventory,
        generated_at=GENERATED_AT,
    )
    second_retry = plan_withdrawal_dry_run(
        validate_withdrawal_request(_request_document()),
        validate_lineage_inventory(partial_inventory),
        generated_at=GENERATED_AT,
    )

    expected_actions = {impact.asset_id: impact.planned_actions for impact in baseline.impacts}
    retried_actions = {impact.asset_id: impact.planned_actions for impact in first_retry.impacts}
    assert first_retry == second_retry
    assert first_retry.affected_asset_count == baseline.affected_asset_count == 12
    assert retried_actions == expected_actions
    assert _id("asset", 11) in retried_actions
    assert retried_actions[_id("asset", 1)] == (
        "delete_primary",
        "invalidate",
        "revoke_access",
    )
    assert retried_actions[_id("asset", 3)] == (
        "delete_primary",
        "invalidate",
        "rebuild",
    )


def test_raw_nonretention_root_still_reaches_and_replans_active_descendants() -> None:
    def mark_raw_as_already_invalidated(payload: dict[str, object]) -> None:
        # Raw media may already be gone because retention was not permitted. Its
        # lineage node still anchors every derived asset that withdrawal must reach.
        cast_assets(payload)[0].update(
            {
                "lifecycle_state": "invalidated",
                "invalidated_at": REQUESTED_AT,
            }
        )

    report = plan_withdrawal_dry_run(
        _request_document(),
        _mutated_inventory(mark_raw_as_already_invalidated),
        generated_at=GENERATED_AT,
    )
    impacts = {impact.asset_id: impact for impact in report.impacts}

    assert report.status == "complete"
    assert report.direct_asset_count == 1
    assert report.downstream_asset_count == 11
    assert report.affected_asset_count == 12
    assert impacts[_id("asset", 1)].planned_actions == (
        "delete_primary",
        "invalidate",
        "revoke_access",
    )
    assert impacts[_id("asset", 11)].planned_actions == (
        "invalidate",
        "republish",
        "retract",
    )


@pytest.mark.parametrize(
    ("mutator", "generated_at", "message"),
    [
        (
            lambda payload: payload.update({"generated_at": CREATED_AT}),
            GENERATED_AT,
            "inventory predates",
        ),
        (
            lambda payload: cast_assets(payload)[0].update({"created_at": GENERATED_AT}),
            GENERATED_AT,
            "parent asset cannot be created after",
        ),
        (
            lambda _payload: None,
            CREATED_AT,
            "report cannot predate",
        ),
    ],
)
def test_stale_or_temporally_incoherent_inventory_fails_closed(
    mutator: Callable[[dict[str, object]], None],
    generated_at: str,
    message: str,
) -> None:
    with pytest.raises(WithdrawalPlanningError, match=message):
        plan_withdrawal_dry_run(
            _request_document(),
            _mutated_inventory(mutator),
            generated_at=generated_at,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: cast_assets(payload)[3].update({"parent_asset_ids": [_id("asset", 999)]}),
        lambda payload: cast_assets(payload).append(dict(cast_assets(payload)[0])),
        lambda payload: _make_cycle(payload),
        lambda payload: cast_assets(payload)[0].update(
            {"logical_uri": "C" + ":/private/participant/video.mp4"}
        ),
    ],
)
def test_invalid_graphs_and_locators_fail_before_planning(
    mutator: Callable[[dict[str, object]], None],
) -> None:
    with pytest.raises((GovernanceContractError, WithdrawalPlanningError)):
        plan_withdrawal_dry_run(
            _request_document(),
            _mutated_inventory(mutator),
            generated_at=GENERATED_AT,
        )


def cast_assets(payload: dict[str, object]) -> list[dict[str, Any]]:
    assets = payload["assets"]
    assert isinstance(assets, list)
    return assets


def _make_cycle(payload: dict[str, object]) -> None:
    assets = cast_assets(payload)
    assets[5]["parent_asset_ids"] = [_id("asset", 8)]
    assets[7]["parent_asset_ids"] = [_id("asset", 6)]
