"""Deterministic, read-only participant-withdrawal planning."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Mapping
from typing import Final

from signlab.contracts.governance import (
    AssetKind,
    GovernanceAssetV1,
    GovernanceContractError,
    GovernanceInput,
    LineageInventoryV1,
    WithdrawalImpactV1,
    WithdrawalReportV1,
    WithdrawalRequestV1,
    validate_lineage_inventory,
    validate_withdrawal_impact,
    validate_withdrawal_report,
    validate_withdrawal_request,
    withdrawal_impact_digest,
    withdrawal_report_digest,
)
from signlab.contracts.taxonomy import canonical_json_bytes

ZERO_DIGEST: Final = "sha256:" + "0" * 64

_ACTIONS_BY_KIND: Final[dict[AssetKind, tuple[str, ...]]] = {
    "raw_recording": ("delete_primary", "invalidate", "revoke_access"),
    "backup_copy": ("invalidate", "purge_backup"),
    "cache": ("delete_primary", "invalidate", "rebuild"),
    "derived_features": ("delete_primary", "invalidate", "rebuild"),
    "annotation": ("delete_primary", "invalidate", "rebuild"),
    "dataset_version": ("invalidate", "rebuild"),
    "split_version": ("invalidate", "rebuild"),
    "experiment_run": ("invalidate", "rerun"),
    "model_artifact": ("invalidate", "retire", "retrain"),
    "evaluation_report": ("invalidate", "reevaluate", "republish", "retract"),
    "public_demo": ("invalidate", "republish", "retract"),
    "withdrawal_tombstone": ("retain",),
}


class WithdrawalPlanningError(GovernanceContractError):
    """Raised when a complete, safe withdrawal dry run cannot be produced."""


def _stable_id(prefix: str, *parts: str) -> str:
    payload = canonical_json_bytes({"parts": parts, "prefix": prefix})
    suffix = hashlib.sha256(payload).hexdigest()[:32]
    return f"{prefix}_{suffix}"


def _validated_inputs(
    request: GovernanceInput,
    inventory: GovernanceInput,
) -> tuple[WithdrawalRequestV1, LineageInventoryV1]:
    try:
        return validate_withdrawal_request(request), validate_lineage_inventory(inventory)
    except GovernanceContractError as error:
        raise WithdrawalPlanningError(str(error)) from error


def _complete_children(
    inventory: LineageInventoryV1,
) -> tuple[dict[str, GovernanceAssetV1], dict[str, tuple[str, ...]]]:
    by_id = {asset.asset_id: asset for asset in inventory.assets}
    if len(by_id) != len(inventory.assets):
        raise WithdrawalPlanningError("lineage inventory contains duplicate asset IDs")

    child_lists: dict[str, list[str]] = {asset_id: [] for asset_id in by_id}
    indegree: dict[str, int] = {}
    for asset in inventory.assets:
        if not asset.logical_uri.startswith("signlab://"):
            raise WithdrawalPlanningError("lineage inventory contains an incomplete locator")
        indegree[asset.asset_id] = len(asset.parent_asset_ids)
        for parent_id in asset.parent_asset_ids:
            if parent_id not in by_id:
                raise WithdrawalPlanningError("lineage inventory contains an orphan parent")
            child_lists[parent_id].append(asset.asset_id)

    # Validate the complete inventory, not merely the request-reachable subgraph.
    ready = [asset_id for asset_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        asset_id = heapq.heappop(ready)
        visited += 1
        for child_id in sorted(child_lists[asset_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, child_id)
    if visited != len(by_id):
        raise WithdrawalPlanningError("lineage inventory contains a cycle")

    children = {asset_id: tuple(sorted(child_ids)) for asset_id, child_ids in child_lists.items()}
    return by_id, children


def _direct_roots(
    request: WithdrawalRequestV1,
    assets: tuple[GovernanceAssetV1, ...],
) -> tuple[str, ...]:
    participant_roots = tuple(
        asset
        for asset in assets
        if asset.asset_kind == "raw_recording" and request.participant_id in asset.participant_ids
    )
    if not participant_roots:
        raise WithdrawalPlanningError("withdrawal participant is unknown to the lineage inventory")

    known_receipts = {receipt_id for asset in participant_roots for receipt_id in asset.receipt_ids}
    if not set(request.receipt_ids).issubset(known_receipts):
        raise WithdrawalPlanningError(
            "withdrawal request references consent not attached to this participant"
        )
    return tuple(sorted(asset.asset_id for asset in participant_roots))


def _descendant_closure(
    roots: tuple[str, ...],
    children: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    affected = set(roots)
    pending = list(roots)
    heapq.heapify(pending)
    while pending:
        asset_id = heapq.heappop(pending)
        for child_id in children[asset_id]:
            if child_id not in affected:
                affected.add(child_id)
                heapq.heappush(pending, child_id)
    return tuple(sorted(affected))


def _planned_actions(asset_kind: AssetKind) -> tuple[str, ...]:
    try:
        actions = _ACTIONS_BY_KIND[asset_kind]
    except KeyError as error:  # pragma: no cover - AssetKind is currently exhaustive.
        raise WithdrawalPlanningError("affected asset kind has no withdrawal action") from error
    requires_invalidation = asset_kind != "withdrawal_tombstone"
    if tuple(sorted(set(actions))) != actions or (
        requires_invalidation and "invalidate" not in actions
    ):
        raise WithdrawalPlanningError("withdrawal action policy is incomplete")
    return actions


def _impact(
    *,
    request: WithdrawalRequestV1,
    inventory: LineageInventoryV1,
    asset: GovernanceAssetV1,
    relationship: str,
) -> WithdrawalImpactV1:
    actions = _planned_actions(asset.asset_kind)
    payload: dict[str, object] = {
        "schema_version": "withdrawal-impact/1",
        "impact_id": _stable_id(
            "impact",
            request.request_sha256,
            inventory.inventory_sha256,
            asset.asset_id,
            relationship,
            *actions,
        ),
        "asset_id": asset.asset_id,
        "logical_uri": asset.logical_uri,
        "asset_sha256": asset.sha256,
        "asset_kind": asset.asset_kind,
        "relationship": relationship,
        "planned_actions": actions,
        "impact_sha256": ZERO_DIGEST,
    }
    payload["impact_sha256"] = withdrawal_impact_digest(payload)
    try:
        return validate_withdrawal_impact(payload)
    except GovernanceContractError as error:
        raise WithdrawalPlanningError("withdrawal impact could not be validated") from error


def plan_withdrawal_dry_run(
    request: GovernanceInput,
    inventory: GovernanceInput,
    *,
    generated_at: str,
) -> WithdrawalReportV1:
    """Plan complete participant withdrawal without mutating data or external state."""

    checked_request, checked_inventory = _validated_inputs(request, inventory)
    if checked_inventory.generated_at < checked_request.effective_at:
        raise WithdrawalPlanningError("lineage inventory predates the effective withdrawal request")
    if generated_at < checked_inventory.generated_at:
        raise WithdrawalPlanningError("withdrawal report cannot predate its lineage inventory")
    if any(asset.created_at > checked_inventory.generated_at for asset in checked_inventory.assets):
        raise WithdrawalPlanningError(
            "lineage inventory contains an asset created after inventory generation"
        )
    by_id, children = _complete_children(checked_inventory)
    roots = _direct_roots(checked_request, checked_inventory.assets)
    affected_ids = _descendant_closure(roots, children)

    # V1 has no completed-action attestations, so lifecycle state cannot prove which
    # individual deletion, invalidation, or rebuild actions already succeeded. Re-plan
    # the complete idempotent action set for every affected node on every retry.
    root_set = set(roots)
    impacts = tuple(
        _impact(
            request=checked_request,
            inventory=checked_inventory,
            asset=by_id[asset_id],
            relationship="direct" if asset_id in root_set else "downstream",
        )
        for asset_id in affected_ids
    )

    impacted_ids = {impact.asset_id for impact in impacts}
    if impacted_ids != set(affected_ids) or any(
        impact.asset_kind != "withdrawal_tombstone" and "invalidate" not in impact.planned_actions
        for impact in impacts
    ):
        raise WithdrawalPlanningError("withdrawal plan omitted an affected descendant")

    # Simulate invalidation in a detached state map. The frozen inventory is never changed.
    simulated_states = {
        asset_id: (
            "retained"
            if asset_id in impacted_ids and asset.asset_kind == "withdrawal_tombstone"
            else "invalidated"
            if asset_id in impacted_ids
            else asset.lifecycle_state
        )
        for asset_id, asset in by_id.items()
    }
    if any(simulated_states[asset_id] == "active" for asset_id in affected_ids):
        raise WithdrawalPlanningError("withdrawal simulation left an affected asset active")

    impact_digests = tuple(impact.impact_sha256 for impact in impacts)
    report_payload: dict[str, object] = {
        "schema_version": "withdrawal-report/1",
        "report_id": _stable_id(
            "report",
            checked_request.request_sha256,
            checked_inventory.inventory_sha256,
            generated_at,
            *impact_digests,
        ),
        "mode": "dry_run",
        "request": checked_request.model_dump(mode="json", round_trip=True),
        "inventory_id": checked_inventory.inventory_id,
        "inventory_sha256": checked_inventory.inventory_sha256,
        "generated_at": generated_at,
        "status": "complete",
        "impacts": [impact.model_dump(mode="json", round_trip=True) for impact in impacts],
        "affected_asset_count": len(impacts),
        "direct_asset_count": len(roots),
        "downstream_asset_count": len(impacts) - len(roots),
        "unresolved_asset_ids": (),
        "report_sha256": ZERO_DIGEST,
    }
    report_payload["report_sha256"] = withdrawal_report_digest(report_payload)
    try:
        return validate_withdrawal_report(report_payload)
    except GovernanceContractError as error:
        raise WithdrawalPlanningError("withdrawal report could not be validated") from error


def withdrawal_report_json_bytes(report: WithdrawalReportV1) -> bytes:
    """Return canonical deterministic JSON bytes for a validated dry-run report."""

    checked = validate_withdrawal_report(report)
    return canonical_json_bytes(checked)


def render_withdrawal_report_markdown(report: WithdrawalReportV1) -> str:
    """Render stable human-review evidence without exposing machine-specific state."""

    checked = validate_withdrawal_report(report)
    lines = [
        "# Withdrawal dry-run report",
        "",
        f"- Report ID: `{checked.report_id}`",
        f"- Request ID: `{checked.request.request_id}`",
        f"- Inventory ID: `{checked.inventory_id}`",
        f"- Generated at: `{checked.generated_at}`",
        f"- Status: `{checked.status}`",
        f"- Affected assets: {checked.affected_asset_count}",
        f"- Direct recording roots: {checked.direct_asset_count}",
        f"- Downstream descendants: {checked.downstream_asset_count}",
        "- Storage mutations performed: `false`",
        "- Simulated non-tombstone state: `invalidated`",
        "- Withdrawal tombstones: `retained`",
        "",
        "| Asset | Kind | Relationship | Planned actions | Logical locator |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        "| "
        f"`{impact.asset_id}` | `{impact.asset_kind}` | `{impact.relationship}` | "
        f"{', '.join(f'`{action}`' for action in impact.planned_actions)} | "
        f"`{impact.logical_uri}` |"
        for impact in checked.impacts
    )
    lines.extend(
        [
            "",
            "This is a read-only plan. Deletion, backup purge, invalidation, rebuilding, "
            "and republication require separately authorized storage adapters and attestations.",
            "",
        ]
    )
    return "\n".join(lines)


# Short aliases retain one obvious application-service entry point for future adapters.
plan_withdrawal = plan_withdrawal_dry_run
render_withdrawal_markdown = render_withdrawal_report_markdown


__all__ = [
    "WithdrawalPlanningError",
    "plan_withdrawal",
    "plan_withdrawal_dry_run",
    "render_withdrawal_markdown",
    "render_withdrawal_report_markdown",
    "withdrawal_report_json_bytes",
]
