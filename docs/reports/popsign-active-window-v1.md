# PopSign active-sign window result

## Decision

**READY for quota selection.** The fixed, label-blind active-sign window leaves 693 of
the 750 already-extracted PopSign attempts usable under the unchanged quality policy.
Every split/gesture group has more distinct usable signers than the required 10 train,
3 validation, or 3 test clips.

This result resolves the seven-clip shortfall reported by `#79`. It does not claim model
performance and does not overwrite the immutable `#79` corpus. The exact 80-clip corpus
and leakage-resistant split manifest remain downstream artifacts.

## Frozen rule

Rule ID: `popsign_longest_detected_hand_episode/1`

For each landmark sequence, before any label, split, signer, filename, or quota metadata
is joined:

1. Mark frames with at least one detected hand.
2. Bridge only an internal gap of at most two frames whose neighboring observations are
   at most 100,000 microseconds apart. These are the existing policy's interpolation
   limits.
3. Form contiguous episodes and rank them by observed-hand frame count, quantized palm
   motion, inclusive span, then earliest start.
4. Require at least three observed-hand frames. Otherwise return a coded no-safe-window
   decision.
5. Rebase the selected view to frame and timestamp zero while preserving the original
   inclusive frame bounds and source PTS as provenance.

The same rebased view is passed to both quality assessment and feature derivation. The
quality thresholds and policy resource were not changed.

## Reproducibility evidence

| Evidence | Value |
| --- | --- |
| Source corpus | `sha256:bd2e552d28792346b7c8e345f8387ebcc52938692cde7f0a316763aa09bdceb9` |
| Source summary | `sha256:cfd760a6d73d8850199a25dae9a78827dea8e809b060724cbd9fc53bfa65be41` |
| External dataset | `sha256:3eb0bb1e73cacddf3b59a84d5c946207c7cb973d6526edea8c75fe9138e8669c` |
| Attempt-landmark inventory | `sha256:fa2028b212c342fe273b0afd9101114c157b4822f3cc95860535848d993f9f44` |
| Landmark files / rows / bytes | 750 / 76,520 / 71,358,032 |
| Extraction configuration | `sha256:7343cd8bb724313b4063a3ebd5d7f7470a78b00f2eeda275a15e5f9b2e66e94c` |
| Hand model | `sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Pose model | `sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |
| Unchanged quality policy | `sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545` |
| MediaPipe executions during replay | 0 |

All 750 source Parquets passed exact-byte, Arrow-schema, semantic-digest, row-count, and
recording-identity verification. Window decisions were made from rows alone; manifest
metadata was joined afterward for aggregate quota accounting.

## Before and after

`Usable` means `pass + warning`. A no-window decision is terminal and is not silently
replaced by another rule.

| Split | Gesture | Before usable | After pass | After warning | After quarantine | No window | After usable | Distinct usable signers | Required | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | `hello` | 10 | 96 | 16 | 11 | 1 | 112 | 31 | 10 | ready |
| train | `no` | 8 | 108 | 11 | 3 | 2 | 119 | 31 | 10 | ready |
| train | `please` | 10 | 55 | 16 | 5 | 0 | 71 | 31 | 10 | ready |
| train | `thank_you` | 10 | 72 | 24 | 10 | 0 | 96 | 31 | 10 | ready |
| train | `yes` | 10 | 63 | 10 | 5 | 3 | 73 | 30 | 10 | ready |
| val | `hello` | 1 | 26 | 9 | 3 | 0 | 35 | 8 | 3 | ready |
| val | `no` | 3 | 15 | 3 | 0 | 0 | 18 | 8 | 3 | ready |
| val | `please` | 3 | 20 | 0 | 0 | 1 | 20 | 8 | 3 | ready |
| val | `thank_you` | 3 | 22 | 3 | 1 | 0 | 25 | 8 | 3 | ready |
| val | `yes` | 1 | 32 | 4 | 1 | 0 | 36 | 8 | 3 | ready |
| test | `hello` | 3 | 14 | 1 | 2 | 1 | 15 | 8 | 3 | ready |
| test | `no` | 3 | 11 | 2 | 1 | 0 | 13 | 8 | 3 | ready |
| test | `please` | 3 | 10 | 3 | 2 | 1 | 13 | 7 | 3 | ready |
| test | `thank_you` | 2 | 28 | 7 | 1 | 1 | 35 | 8 | 3 | ready |
| test | `yes` | 3 | 10 | 2 | 1 | 1 | 12 | 7 | 3 | ready |
| **Total** |  | **73** | **582** | **111** | **46** | **11** | **693** |  | **80 selected clips** | **ready** |

The before replay reproduced 36 pass, 37 warning, 140 quarantine, and 537 reject.
The after replay accounts for all 750 attempts as 582 pass, 111 warning, 46 quarantine,
and 11 `no_hand_observations` decisions. No real sequence ended in
`episode_too_short`.

## Fixed visual review

The same 12 local clips selected before windowing for `#81` were reviewed again. They
were not reranked using the new result, and no video, frame, path, or identity is
published.

| Coded observation | Result |
| --- | ---: |
| Selected bounds retain the complete visible labeled motion | 12 / 12 |
| Selected bounds visibly truncate the labeled motion | 0 / 12 |
| Window changed from the full source span | 10 / 12 |
| Multi-episode trailing hand reappearances retained by mistake | 0 / 2 |
| Corrupt or unreadable reviewed video | 0 / 12 |

The excluded frames were quiet preparation, return-to-rest, or later unrelated hand
motion. This visual gate supports the aggregate result but does not assess linguistic
correctness or model suitability.

## Limits

- The replay proves data sufficiency for the bounded five-gesture isolated-sign smoke
  corpus, not a production recognition claim.
- PopSign remains public isolated-sign data; no continuous-sign or natural-use claim is
  made.
- The existing `#79` artifact remains an immutable record of the pre-window run.
- No alternate window, threshold change, per-video exception, new provider, extraction
  run, split generation, feature experiment, or training run was attempted.

PopSign ASL v1.0 is provided by the Georgia Institute of Technology and Deaf
Professional Arts Network under CC BY 4.0.
