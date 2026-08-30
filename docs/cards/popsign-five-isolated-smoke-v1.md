# Dataset card: PopSign five-gesture isolated smoke corpus

## Summary

- Status: bounded development smoke corpus
- Source: PopSign ASL v1.0
- Publishers: Georgia Institute of Technology and Deaf Professional Arts Network
- License: CC BY 4.0
- Scope: five isolated SignLab target gestures only
- Size: 80 clips from 40 distinct signers
- Frozen split: `sha256:673c777c67e47715127b75f2bf18aaa794a15e1604458d30da164ec7f90ead20`
- Source corpus: `sha256:bd2e552d28792346b7c8e345f8387ebcc52938692cde7f0a316763aa09bdceb9`
- Test status for candidate nomination: `sealed_not_loaded`

## License and consent boundary

PopSign ASL v1.0 is licensed under CC BY 4.0. Reuse requires attribution and
notice of changes. The provider reports participant consent for the source dataset.
That is provider-reported consent, not direct consent collected by SignLab.

The source contains identifiable human video. SignLab does not redistribute those
videos, frames, signer identities, recording identities, or website preview media in
this repository. The checked public evidence is aggregate and content-addressed.

## Composition

| Partition | Clips | Clips per target |
| --- | ---: | ---: |
| Train | 50 | 10 |
| Validation | 15 | 3 |
| Test | 15 | 3 |
| **Total** | **80** | **16 overall** |

Each of `hello`, `no`, `please`, `thank_you`, and `yes` contributes 16 clips. The
selected set contains 72 `pass` and 8 `warning` clips. No quarantined clip,
rejected clip, repeated recording, or repeated feature sequence is included.

## Split policy

PopSign's official train, validation, and test assignments are preserved, with
source `val` named `validation` in the experiment. Selection is deterministic within
each split and target: usable rows are ordered by stable opaque identity, and the first
row for each unseen signer is retained until the fixed quota is reached.

No signer crosses partitions. The test partition remains sealed for the development
candidate nomination and was not loaded by the #30, #31, or #33 evidence used there.

## Transformations

1. MediaPipe-derived landmarks are verified against the pinned extraction identities.
2. The label-blind `popsign_longest_detected_hand_episode/1` rule selects one active
   episode and rebases it to frame and timestamp zero.
3. The unchanged quality policy assigns pass, warning, quarantine, or reject.
4. Eligible sequences are converted to the frozen 64-frame feature contract.
5. The candidate uses the 126-value hand-local view; body context is not an input.
6. The five reviewed source glosses map to SignLab labels without claiming full
   linguistic equivalence or translation capability.

## Observed limitations

Before active-window selection, missing-hand coverage explained 668 of 677 quality
exclusions in the 750-attempt diagnostic. Missing hands were usually at quiet
preparation or return-to-rest edges, showing a windowing mismatch in untrimmed clips.
The fixed active-window replay produced 693 usable attempts and then the frozen 80-clip
set. This does not prove that hand detection is reliable in natural use.

The corpus is small, balanced by construction, isolated-sign only, and drawn from one
public provider. It does not represent natural `other`, inactive periods, transitions,
continuous signing, capture diversity, or the full signer population.

## Intended use

- Reproduce bounded feature, training, calibration-mechanics, and export experiments.
- Compare development architectures while preserving signer-disjoint partitions.
- Verify pipeline contracts without redistributing identifiable source media.

## Unsupported claims

This card does not support population accuracy, fairness, robustness, natural-use
generalization, continuous-sign recognition, sign-language translation, production
readiness, or direct SignLab participant consent. It does not authorize opening the
sealed test partition for the candidate nomination.

## Checked evidence

- `docs/reports/popsign-frozen-smoke-split-v1.md`
- `docs/reports/popsign-active-window-v1.md`
- `docs/reports/popsign-hand-coverage-diagnostic-v1.md`
- External dataset identity: `sha256:3eb0bb1e73cacddf3b59a84d5c946207c7cb973d6526edea8c75fe9138e8669c`
- Feature plan identity: `sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6`
