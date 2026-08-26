# SignLab withdrawal runbook

This procedure plans actions from versioned pseudonymous JSON. It does not delete
data and must not be described as proof of deletion.

## Required inputs

1. A verified complete-withdrawal request containing only pseudonymous identifiers.
2. The current consent evidence and recording grants.
3. A complete lineage inventory covering raw data, derivatives, datasets, splits,
   runs, models, reports, demonstrations, caches, and backups.
4. An explicit UTC `as-of` time; evidence never relies on hidden wall-clock time.

Keep participant and request identifiers inside access-controlled input files, not
shell arguments or free-text logs.

## Procedure

1. Verify the request through the identity vault and retain only the attestation.
2. Freeze collection, processing, training, evaluation, release, and restore for the
   affected participant, receipts, grants, recordings, and descendants.
3. Validate a fresh lineage inventory. Duplicate IDs or locators, missing parents,
   cycles, and unknown stores block completion. Pending or invalidated nodes remain
   traversal points so active descendants cannot be skipped.
4. Run the deterministic dry run and review every direct root and transitive
   descendant. Shared assets remain affected.
5. Execute store-specific deletion, purge, invalidation, retirement, retraction, and
   rebuild actions only through future approved adapters and authorization.
6. Record attestations, rerun validation, retain the minimal tombstone, and verify
   that restoration cannot reintroduce withdrawn data.

## Completion standard

The dry run is complete when every affected asset has exactly one deterministic
action set and no unresolved affected locator remains. Physical execution is
complete only after every external action is attested, rebuilt assets exclude the
withdrawn roots, releases are republished where required, backups pass their purge
test, and restore reconciliation succeeds. Previously authorized public copies held
by third parties are recorded as a recovery limit rather than falsely reported as
deleted.

Version 1 retries the complete action set because authenticated per-action
completion records do not yet exist. Every future storage adapter must make those
actions safe to repeat.
