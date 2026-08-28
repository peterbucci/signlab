# Frozen PopSign smoke split

## Decision

**READY for the first bounded experiment.** The exact immutable 750-attempt landmark
inventory from `#79` was replayed with the fixed active-sign window from `#83`, and
one deterministic offline rule selected 80 feature-ready clips: 50 train, 15
validation, and 15 test.

This is a data-readiness result, not a model-performance result. The local split
manifest contains opaque licensed-source identities and stays outside the public Git
repository; this report contains only aggregate evidence.

## Frozen selection rule

Rule ID:
`first_usable_distinct_signer_by_stable_sample_id_per_split_target/1`

1. Replay every one of the 750 retained landmark Parquets; do not run MediaPipe or
   select new videos.
2. Apply `popsign_longest_detected_hand_episode/1`, then the unchanged quality policy.
3. Treat `pass` and `warning` as equally eligible. Do not quality-rank candidates.
4. Within each official source split and target gesture, sort usable rows only by
   stable opaque sample ID.
5. Walk that order and retain the first row for each unseen signer until reaching 10
   train or 3 validation/test rows.
6. Preserve PopSign's official signer-disjoint split assignments; map source `val` to
   experiment partition `validation`.

This is deliberately not a replay of `#79`'s acceptance-dependent online attempt
loop. The window changed which attempts are usable, so claiming the old loop was
replayed would be false.

## Reproduced source evidence

| Evidence | Value |
| --- | --- |
| Source corpus | `sha256:bd2e552d28792346b7c8e345f8387ebcc52938692cde7f0a316763aa09bdceb9` |
| External dataset | `sha256:3eb0bb1e73cacddf3b59a84d5c946207c7cb973d6526edea8c75fe9138e8669c` |
| Attempt-landmark inventory | `sha256:fa2028b212c342fe273b0afd9101114c157b4822f3cc95860535848d993f9f44` |
| Attempt landmarks | 750 |
| Active-sign window | `popsign_longest_detected_hand_episode/1` |
| Pass | 582 |
| Warning | 111 |
| Quarantine | 46 |
| No safe window | 11 |
| Usable (`pass + warning`) | 693 |
| MediaPipe executions | 0 |

The replay matched the `#83` result exactly. Every retained Parquet was byte-hashed,
schema-checked, semantically decoded, joined to one unique external recording, and
reassessed. No fallback rule, quality exception, threshold change, or feature failure
was used.

## Exact selected membership

| Source split | Gesture | Selected | Distinct signers | Pass | Warning |
| --- | --- | ---: | ---: | ---: | ---: |
| train | `hello` | 10 | 10 | 9 | 1 |
| train | `no` | 10 | 10 | 10 | 0 |
| train | `please` | 10 | 10 | 6 | 4 |
| train | `thank_you` | 10 | 10 | 9 | 1 |
| train | `yes` | 10 | 10 | 10 | 0 |
| validation | `hello` | 3 | 3 | 3 | 0 |
| validation | `no` | 3 | 3 | 3 | 0 |
| validation | `please` | 3 | 3 | 3 | 0 |
| validation | `thank_you` | 3 | 3 | 3 | 0 |
| validation | `yes` | 3 | 3 | 3 | 0 |
| test | `hello` | 3 | 3 | 3 | 0 |
| test | `no` | 3 | 3 | 3 | 0 |
| test | `please` | 3 | 3 | 2 | 1 |
| test | `thank_you` | 3 | 3 | 2 | 1 |
| test | `yes` | 3 | 3 | 3 | 0 |
| **Total** |  | **80** |  | **72** | **8** |

The manifest contains 40 distinct signers. No signer crosses partitions, and no
recording or feature-sequence identity is repeated.

## Reproducibility result

| Evidence | Value |
| --- | --- |
| Frozen split | `sha256:673c777c67e47715127b75f2bf18aaa794a15e1604458d30da164ec7f90ead20` |
| Independent fresh freezes | 2 |
| Files per freeze | 82 (80 features, 1 manifest, 1 aggregate report) |
| Bytes per freeze | 14,887,227 |
| Relative path / SHA-256 / size differences | 0 |
| Reconciled feature bindings | 80 / 80 |

The validator rejects stale split hashes, wrong quotas, noncanonical membership,
within-group signer reuse, cross-partition signers, repeated recording or feature
identities, unsafe/non-content-addressed feature paths, changed feature bytes, and
feature lineage that disagrees with the frozen configuration, quality policy, or
feature plan.

## Scope limit

This split is sufficient to exercise the bounded five-gesture isolated-sign training
and evaluation path. It does not establish participant-study coverage, natural or
continuous signing performance, production generalization, or any accuracy claim.

PopSign ASL v1.0 is provided by the Georgia Institute of Technology and Deaf
Professional Arts Network under CC BY 4.0.
