# Model card: PopSign hand-local TCN portable-export candidate

## Status and identity

- Candidate status: frozen development checkpoint; nomination requires dossier verification
- Nomination scope: portable export only
- Champion status: none; activation and rollback are blocked because no champion exists
- Metric claim: development only
- Test status: `sealed_not_loaded`
- Architecture: hand-local temporal convolutional network (TCN)
- Input: 64 frames x 126 hand-local features
- Classes: `hello`, `no`, `please`, `thank_you`, `yes`, and constructed `other`
- Parameters: 29,094
- Research checkpoint bytes: 438,146
- Research checkpoint: `sha256:79b69bd30fef1986e8b33cbc39e5102b80a95b332489e6fe2ab7a7992ee8a0fd`
- Verified #31 ledger run: `5848bc5ae6ae40cf9bca46552774bac5`
- Decision policy file: `sha256:edaee1135f914326636695838897e1a8cac12198e81db2d71defff9d970aa21e`

## Inputs and training scope

The model consumes only the hand-local portion of the frozen 64-frame feature
representation. Its five target classes come from the signer-disjoint 80-clip PopSign
smoke corpus. The sixth `other` class is constructed from deterministic transition
fragments; it is not a natural out-of-vocabulary corpus.

- Frozen split: `sha256:673c777c67e47715127b75f2bf18aaa794a15e1604458d30da164ec7f90ead20`
- Source feature plan: `sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6`
- Candidate input plan: `sha256:1c62d2738ce0609168967b675fa0dcd1797f8fbe881cd9b5c775d4e2a83e4a3e`
- Taxonomy: `sha256:c0f6cbddfe43e3a6eb3de01dbbbbc1ceebcb83d50cc197999776f58e3d9ce20d`

PopSign is CC BY 4.0 public data whose provider reports participant consent. SignLab
did not collect direct consent from those participants. Identifiable videos are not
redistributed in this repository.

## #30 supporting architecture evidence

The grouped #30 ablation supports carrying forward the hand-local TCN architecture;
it does **not** evaluate this exact six-class checkpoint. Across three grouped
development folds, the five-class hand-local TCN reported OOF macro-F1 0.834,
balanced accuracy 0.831, fold macro-F1 0.823/0.796/0.877, 29,061 parameters, and
pre-vectorized CPU p50/p95 33.759/41.041 ms on the measured machine.

Those values are supporting architecture metrics from a different fitted model. They
are development evidence, not exact-candidate, test, natural-use, or release metrics.

## #31 exact-candidate constructed evidence

#31 binds the exact checkpoint above to 18 development validation rows: 15 target rows
from 15 unique clips plus three deterministic transition fragments. Temperature 0.050
was selected at the search boundary. The selected inclusive threshold is 0%, so none
of those rows abstains. NLL changed from 0.173056 to 0.000076 and Brier score from
0.084185 to 0.000000 on the same development rows used for fitting and selection.

This proves calibration and rejection mechanics for the exact candidate. The three
`other` rows are constructed, validation is reused, and the 0% threshold is not a
natural-use or production operating point.

## #33 constructed scorer evidence

#33 verifies deterministic replay scoring with constructed intervals and decisions.
The model was not run on video or natural sessions. All #33 rates, latencies, and
bootstrap intervals are scorer-conformance evidence, not checkpoint performance.

## Intended use

- Verify the exact local checkpoint identity and reproducible nomination dossier.
- Export the frozen candidate in #35 and measure portable/native parity later.
- Support bounded development experiments on the documented isolated-sign corpus.

## Known risks and failure modes

- Missing or poorly tracked hands can corrupt the hand-local input.
- Active-window behavior proven on isolated clips may not transfer to live sessions.
- Constructed `other` examples do not cover natural gestures, inactivity, or transitions.
- The small balanced corpus cannot establish population, subgroup, or capture robustness.
- A zero abstention threshold can accept uncertain natural inputs; it is not a
  production policy.

## Release gates still unavailable

No threshold is asserted for unseen-signer release performance, per-class floors,
natural-other calibration, natural continuous false activations, end-to-end latency,
bundle size, native/ONNX parity, or TypeScript/Web parity. All remain blocked/pending.

## Unsupported claims

This card makes no champion, promotion, activation, rollback, production-readiness,
natural-use performance, continuous-sign recognition, translation, fairness, or safety
claim. The sealed test set remains unopened, and no release measurement is inferred
from the #30 architecture metrics or the constructed #31/#33 evidence.
