# Gesture taxonomy and claim boundary

SignLab taxonomy `signlab-five@1.0.0` defines five target gestures and one learned
negative class. The authoritative machine-readable artifact is packaged at
`src/signlab/resources/taxonomies/signlab-five-1.0.0.json`; its validated canonical
content digest is
`sha256:c0f6cbddfe43e3a6eb3de01dbbbbc1ceebcb83d50cc197999776f58e3d9ce20d`.

## Approved product claim

> SignLab is a research prototype for recognizing isolated performances of five
> predefined hand gestures within continuous webcam video. It separates non-target
> events (`other`) from no active event (`inactive`) and uncertain decisions
> (`abstain`); no sign-language or translation capability is claimed.

The English display names describe this project's gesture prompts. They are not a
claim that the forms are validated lexical items in American Sign Language or any
other named language. A named-language claim requires a qualified reviewer, an
identified lexical source entry for every gesture, review scope and date, variant
notes, and an evidence digest. Version 1 rejects every named-language claim; after
human review, support requires a new bounded claim profile and taxonomy release.

## Learned outputs

The output order is immutable for version `1.0.0`:

| Index | Stable ID | Role | Operational definition |
| ---: | --- | --- | --- |
| 0 | `hello` | target | Open hand near the temple or upper face with a short outward or side-to-side wave. |
| 1 | `no` | target | Thumb closes toward extended index and middle fingertips one or more times. |
| 2 | `please` | target | Flat or open palm at the upper chest with a small circular stroke. |
| 3 | `thank_you` | target | Flat or open hand begins at the chin or lower face and moves outward in one stroke. |
| 4 | `yes` | target | Closed fist near the upper torso performs one or more wrist-driven vertical nods. |
| 5 | `other` | learned negative | A complete candidate event confidently outside the five-target vocabulary. |

Each class has positive, clear-negative, ambiguous, and transition-boundary
examples in the machine-readable taxonomy. Those examples—not the English mnemonic
alone—define annotation eligibility.

## Detector state, class, and decision

These concepts belong to different layers and must never share a classifier index:

1. No finalized candidate event exists: the event detector emits `inactive`.
2. A target score meets its acceptance policy: the decision is that target ID.
3. A non-target score meets its acceptance policy: the learned class is `other`.
4. No class can be accepted at the declared risk: the policy emits `abstain`.

`ambiguous` and `ignore` are annotation dispositions, not runtime classes.
Ambiguous events require adjudication before use. Ignore regions require a recorded
reason and are excluded according to the evaluator's predeclared overlap rule.

## Edge-case rules

- Either hand is accepted only after handedness normalization is recorded.
- An incidental second hand is allowed only when it does not contact, occlude, or
  alter the active hand. Coordinated two-hand activity is `other` in this version.
- A clearly incomplete target attempt is `other/partial_target`; insufficient
  visibility is `ambiguous`.
- A complete out-of-vocabulary candidate is `other/oov_gesture`.
- Target transitions do not automatically become training examples. Derived
  `other/transition_fragment` examples are allowed only on train or development
  splits.
- Poor segmentation does not change source truth. Continuous evaluation must still
  record detector misses, truncation, and fragmentation.
- Ignore reasons include consent exclusion, camera setup, third-party presence,
  unresolved annotation conflict, and unusable occlusion.

The broad `other` class retains `other_kind` metadata: `partial_target`,
`transition_fragment`, `oov_gesture`, `incidental_activity`, or
`two_hand_non_target`. These are analysis subtypes, not extra classifier outputs.

## Legacy migration

The historical spelling `thank you` has one explicit import alias to `thank_you`.
Core manifests reject the alias. The legacy token `nothing` is not equivalent to
`other`: it mixed idle footage, transitions, and unrelated activity, so it remains
quarantined for reannotation.

## Contract propagation

Collection, annotation, training, evaluation, model-bundle, and public-copy
contracts must embed the same `TaxonomyRef` (`id`, semantic version, and canonical
SHA-256). Closed JSON Schemas for all six bindings are generated from Pydantic and
shipped with the package. The training binding additionally requires the exact
six-output label map and a positive observed count for every output. Both Python
and standalone JSON Schema reject a missing or empty `other` class.

Validate the packaged artifact with:

```shell
uv run signlab taxonomy validate
uv run signlab taxonomy validate-resources
uv run signlab train validate-taxonomy path/to/training-taxonomy-binding.json
```

Regenerate schemas with `uv run python scripts/generate_taxonomy_schemas.py`. Tests
fail if the generated output, packaged taxonomy, or golden digest drifts. Semantic
changes require a new taxonomy version and artifact; an existing version is never
edited in place. The wire-level schemas enumerate every registered immutable
release, so adding a new taxonomy version does not invalidate manifests that still
reference an older supported release.

## Research basis

- The [National Association of the Deaf](https://nad.org/knowledge-hub/american-sign-language/what-is-american-sign-language/)
  describes ASL as a language with its own grammar whose forms depend on handshape,
  movement, placement, face, and body. English word labels alone are not validation.
- The [ASL Citizen datasheet](https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/)
  documents lexical and regional variation and the involvement of Deaf researchers.
- The [NIST AI Risk Management Framework](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
  calls for explicit intended scope, context, knowledge limits, and limitations.
- Selective classification treats abstention as a reject decision with a measurable
  risk/coverage tradeoff; see [El-Yaniv and Wiener (JMLR, 2010)](https://jmlr.csail.mit.edu/papers/v11/el-yaniv10a.html).
- Portable schemas use [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12).
