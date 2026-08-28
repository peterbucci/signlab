# PopSign hand-coverage shortfall diagnostic

## Decision

The `#79` shortfall is primarily a windowing mismatch, not evidence that PopSign lacks
enough candidate recordings. SignLab currently applies an all-frame hand-coverage rule
to complete, untrimmed isolated-sign videos. Quiet preparation and return-to-rest frames
therefore count against otherwise usable gesture motion.

**Next action:** retain the existing quality thresholds and open a separate, narrowly
scoped story to define a deterministic, label-blind active-sign window for isolated
public clips. Validate that the window preserves the complete visible motion, then
reapply the existing policy to the already-extracted landmarks. Do not add another data
source or lower thresholds until that bounded remediation is evaluated.

This report makes the decision only. It does not change extraction, windowing, quality
policy, features, splits, or training.

## Scope and method

- Reused all 750 landmark Parquets retained by the bounded `#79` run; MediaPipe was not
  run again.
- Verified every Parquet's bytes, semantic content identity, row count, and recording
  identity before assessment.
- Joined all 750 recording identities to the pinned PopSign manifest with zero missing
  or ambiguous matches.
- Recomputed the same pinned quality policy over all 750 sequences.
- Reconciled all 73 accepted sequences to their stored disposition and quality-report
  identity with zero mismatches.
- Compared all 15 split/target groups, then limited detailed analysis to the four groups
  that missed quota.
- Reviewed exactly 12 local videos: for each short group, the accepted clip nearest the
  80% usable boundary, the hand-coverage failure nearest that boundary, and the distinct
  failure nearest the group's unusable median. The review set and source paths remain
  local and are not published.

## Reproducibility evidence

| Evidence | Value |
| --- | --- |
| Source corpus | `sha256:bd2e552d28792346b7c8e345f8387ebcc52938692cde7f0a316763aa09bdceb9` |
| Source summary | `sha256:cfd760a6d73d8850199a25dae9a78827dea8e809b060724cbd9fc53bfa65be41` |
| External dataset | `sha256:3eb0bb1e73cacddf3b59a84d5c946207c7cb973d6526edea8c75fe9138e8669c` |
| Extraction configuration | `sha256:7343cd8bb724313b4063a3ebd5d7f7470a78b00f2eeda275a15e5f9b2e66e94c` |
| Hand model | `sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Pose model | `sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |
| Quality policy | `sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545` |
| Feature plan | `sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6` |
| Attempt-landmark inventory | `sha256:fa2028b212c342fe273b0afd9101114c157b4822f3cc95860535848d993f9f44` |
| Landmark files / rows / bytes | 750 / 76,520 / 71,358,032 |

The attempt-landmark inventory is the domain-separated canonical digest of the sorted
750-item inventory. Each item contains only its semantic digest, exact-byte digest, row
count, and byte size; it contains no local path or source identity.

## All-group comparison

`Accepted` is `pass + warning`; quarantine and reject are not retained.

| Split | Target | Attempted | Accepted | Acceptance | Pass | Warning | Quarantine | Reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `hello` | 124 | 10 | 8.1% | 6 | 4 | 19 | 95 |
| train | `no` | 124 | 8 | 6.5% | 6 | 2 | 19 | 97 |
| train | `please` | 76 | 10 | 13.2% | 4 | 6 | 25 | 41 |
| train | `thank_you` | 106 | 10 | 9.4% | 5 | 5 | 15 | 81 |
| train | `yes` | 81 | 10 | 12.3% | 7 | 3 | 15 | 56 |
| val | `hello` | 38 | 1 | 2.6% | 0 | 1 | 4 | 33 |
| val | `no` | 18 | 3 | 16.7% | 0 | 3 | 3 | 12 |
| val | `please` | 21 | 3 | 14.3% | 1 | 2 | 5 | 13 |
| val | `thank_you` | 26 | 3 | 11.5% | 2 | 1 | 2 | 21 |
| val | `yes` | 37 | 1 | 2.7% | 0 | 1 | 8 | 28 |
| test | `hello` | 18 | 3 | 16.7% | 0 | 3 | 2 | 13 |
| test | `no` | 14 | 3 | 21.4% | 1 | 2 | 2 | 9 |
| test | `please` | 16 | 3 | 18.8% | 1 | 2 | 9 | 4 |
| test | `thank_you` | 37 | 2 | 5.4% | 2 | 0 | 9 | 26 |
| test | `yes` | 14 | 3 | 21.4% | 1 | 2 | 3 | 8 |
| **Total** |  | **750** | **73** | **9.7%** | **36** | **37** | **140** | **537** |

## Dominant failure mechanism

The expected-hand rule warns below 95% all-frame coverage, quarantines below 80%, and
rejects below 50%.

- All 537 rejected clips violated expected-hand coverage.
- 131 of 140 quarantined clips also violated expected-hand coverage.
- Hand coverage therefore explains 668 of 677 quality exclusions (98.7%).
- Within the four short groups, it explains 223 of 224 unusable clips. The remaining
  validation `hello` clip had 100% expected-hand coverage and was quarantined for a long
  track-specific internal hand gap.

The all-group table provides each group's quarantine/reject severity counts. This table
attributes those exclusions to the governing mechanism.

| Short group | Accepted / attempted | Expected-hand exclusions | Other exclusion | Unusable coverage median | Closest hand failure below 80% |
| --- | ---: | ---: | ---: | ---: | ---: |
| train / `no` | 8 / 124 | 116 | 0 | 35.2% | 77.1% |
| val / `hello` | 1 / 38 | 36 | 1 track-specific gap | 33.3% | 73.8% |
| val / `yes` | 1 / 37 | 36 | 0 | 37.4% | 76.8% |
| test / `thank_you` | 2 / 37 | 35 | 0 | 40.6% | 79.6% |

For the 668 hand-coverage exclusions, the median share of missing-hand frames before the
first or after the last detected hand was 100%. Median hand coverage between the first
and last detection was also 100%. As a diagnostic projection only, restricting the
measurement to that span would move 635 clips above the 50% line, 566 above the 80% line,
and 511 above the 95% line. First-to-last detection is not proposed as the production
windowing algorithm; later hand reappearances and multiple episodes require explicit
handling.

## Capped visual review

The visual review confirms mechanism, not prevalence. Prevalence comes from the full
750-sequence replay.

| Coded observation | Result |
| --- | ---: |
| Labeled motion visibly present in the sampled clip | 12 / 12 |
| Failed sample includes quiet preparation or return-to-rest around the motion | 8 / 8 |
| Obvious hand loss, crop, or occlusion during the visible labeled motion | 0 / 8 |
| Corrupt or unreadable sampled video | 0 / 12 |

One borderline validation `hello` sample had a reported internal gap. Its visible wave
was complete; the gap occurred after that motion and before a trailing hand reappearance.
This is why a new window must be deterministic and episode-aware rather than a naive
first-to-last crop.

The review did not assess linguistic correctness, signer identity, or model suitability.
No reviewed video, frame, local path, signer identity, recording identity, or sample
identity is committed.

## Interpretation and limits

- The original all-frame policy is internally consistent: it measures exactly what its
  documentation says it measures. The mismatch is applying that measurement to untrimmed
  isolated-sign videos with inactive edges.
- Lowering the global thresholds would hide the mismatch and could admit true motion-time
  detection failures. The evidence supports changing the measurement window first.
- The active-span projection is counterfactual diagnostic evidence, not a validated
  algorithm and not proof that all projected clips will become usable.
- PopSign remains public isolated-sign data. This report makes no continuous-sign,
  natural-use, participant-data, or model-performance claim.
- Split assignments and signer-separation rules were not changed.

PopSign ASL v1.0 is provided by the Georgia Institute of Technology and Deaf
Professional Arts Network under CC BY 4.0. The bounded source result is recorded in
[`popsign-trainable-smoke-v1.json`](popsign-trainable-smoke-v1.json).
