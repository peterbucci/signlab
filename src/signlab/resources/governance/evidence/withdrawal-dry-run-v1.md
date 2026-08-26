# Withdrawal dry-run report

- Report ID: `report_b28078bf0b5695a3c2d0bd7bbd05cf8c`
- Request ID: `withdrawal_00000000000000000000000000000001`
- Inventory ID: `inventory_00000000000000000000000000000001`
- Generated at: `2026-08-26T14:00:00Z`
- Status: `complete`
- Affected assets: 12
- Direct recording roots: 1
- Downstream descendants: 11
- Storage mutations performed: `false`
- Simulated non-tombstone state: `invalidated`
- Withdrawal tombstones: `retained`

| Asset | Kind | Relationship | Planned actions | Logical locator |
| --- | --- | --- | --- | --- |
| `asset_00000000000000000000000000000001` | `raw_recording` | `direct` | `delete_primary`, `invalidate`, `revoke_access` | `signlab://store-00000000000000000000000000000001/raw_recording/asset_00000000000000000000000000000001` |
| `asset_00000000000000000000000000000003` | `derived_features` | `downstream` | `delete_primary`, `invalidate`, `rebuild` | `signlab://store-00000000000000000000000000000001/derived_features/asset_00000000000000000000000000000003` |
| `asset_00000000000000000000000000000005` | `annotation` | `downstream` | `delete_primary`, `invalidate`, `rebuild` | `signlab://store-00000000000000000000000000000001/annotation/asset_00000000000000000000000000000005` |
| `asset_00000000000000000000000000000006` | `dataset_version` | `downstream` | `invalidate`, `rebuild` | `signlab://store-00000000000000000000000000000001/dataset_version/asset_00000000000000000000000000000006` |
| `asset_00000000000000000000000000000008` | `split_version` | `downstream` | `invalidate`, `rebuild` | `signlab://store-00000000000000000000000000000001/split_version/asset_00000000000000000000000000000008` |
| `asset_00000000000000000000000000000009` | `experiment_run` | `downstream` | `invalidate`, `rerun` | `signlab://store-00000000000000000000000000000001/experiment_run/asset_00000000000000000000000000000009` |
| `asset_0000000000000000000000000000000a` | `model_artifact` | `downstream` | `invalidate`, `retire`, `retrain` | `signlab://store-00000000000000000000000000000001/model_artifact/asset_0000000000000000000000000000000a` |
| `asset_0000000000000000000000000000000b` | `evaluation_report` | `downstream` | `invalidate`, `reevaluate`, `republish`, `retract` | `signlab://store-00000000000000000000000000000001/evaluation_report/asset_0000000000000000000000000000000b` |
| `asset_0000000000000000000000000000000c` | `public_demo` | `downstream` | `invalidate`, `republish`, `retract` | `signlab://store-00000000000000000000000000000001/public_demo/asset_0000000000000000000000000000000c` |
| `asset_0000000000000000000000000000000f` | `cache` | `downstream` | `delete_primary`, `invalidate`, `rebuild` | `signlab://store-00000000000000000000000000000001/cache/asset_0000000000000000000000000000000f` |
| `asset_00000000000000000000000000000010` | `backup_copy` | `downstream` | `invalidate`, `purge_backup` | `signlab://store-00000000000000000000000000000001/backup_copy/asset_00000000000000000000000000000010` |
| `asset_00000000000000000000000000000011` | `withdrawal_tombstone` | `downstream` | `retain` | `signlab://store-00000000000000000000000000000001/withdrawal_tombstone/asset_00000000000000000000000000000011` |

This is a read-only plan. Deletion, backup purge, invalidation, rebuilding, and republication require separately authorized storage adapters and attestations.
