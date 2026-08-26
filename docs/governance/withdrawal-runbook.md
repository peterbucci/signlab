# Withdrawal runbook

This runbook is executable from versioned JSON and does not rely on remembered
storage locations or experiment names. It plans actions but does not delete data.

## Inputs

1. A verified complete-withdrawal request containing only pseudonymous IDs.
2. The current consent events and recording grants.
3. A complete lineage inventory containing raw data, derivatives, datasets, splits,
   experiments, models, reports, demos, caches, and backups.
4. An explicit UTC `as-of` time. Wall-clock time is never hidden in evidence.

Do not pass signer IDs or request details as command arguments. They remain inside
access-controlled input files so shell history does not expose them.

## Procedure

1. Verify the request through the identity vault and record only the attestation.
2. Freeze new collection, processing, training, evaluation, release, and restore for
   the affected signer and consent IDs.
3. Export and validate a fresh lineage inventory. Cycles, missing parents, duplicate
   IDs or locators, and unknown stores block the plan. Already pending or invalidated
   nodes do not stop traversal; their active descendants remain affected.
4. Run the deterministic dry run with its explicit `as-of` timestamp.
5. Review every root and descendant. Shared datasets, runs, models, reports, and
   demonstrations remain affected and require rebuilding.
6. Execute future store-specific actions only after adapters and authorization exist.
7. Record deletion, purge, invalidation, retirement, retraction, and rebuild
   attestations. Re-run validation until no affected usable asset or unresolved
   external action remains.
8. Retain the minimal tombstone and test that a restore cannot reintroduce withdrawn
   data.

## Action matrix

| Asset kind | Machine-planned actions | Simulated state |
| --- | --- | --- |
| Raw recording | `delete_primary`, `invalidate`, `revoke_access` | `invalidated` |
| Backup copy | `invalidate`, `purge_backup` | `invalidated` |
| Cache, feature, annotation | `delete_primary`, `invalidate`, `rebuild` | `invalidated` |
| Dataset version, split manifest | `invalidate`, `rebuild` | `invalidated` |
| Experiment run | `invalidate`, `rerun` | `invalidated` |
| Model or bundle | `invalidate`, `retire`, `retrain` | `invalidated` |
| Evaluation report | `invalidate`, `reevaluate`, `republish`, `retract` | `invalidated` |
| Public demo | `invalidate`, `republish`, `retract` | `invalidated` |
| Minimal withdrawal tombstone | `retain` | `retained` |

## Completion standard

A dry run is complete when every affected asset has exactly one impact entry with
its required action set and the simulated graph has no usable affected node. It is
not proof of deletion. Execution is complete only after every external action is
attested, rebuilt assets exclude the withdrawn roots, releases are republished where
needed, backups pass the next purge test, and restore reconciliation succeeds.

Version 1 deliberately re-plans the full action set on every retry because the
inventory does not yet carry authenticated per-action completion records. Storage
adapters must therefore make each listed action safe to repeat. This conservative
retry rule prevents an already invalidated raw root or a partially completed prior
attempt from hiding an active feature, model, backup, report, or public copy.

If public raw media or features were previously redistributed with valid consent,
the report must record the recovery limit rather than claiming copies held by third
parties were deleted.
