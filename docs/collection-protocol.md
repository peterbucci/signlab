# Collection and annotation protocol

- Protocol ID: `signlab-collection-protocol`
- Draft version: `0.1.0`
- Status: pilot draft; not approved for real participant collection
- Taxonomy: `signlab-five@1.0.0`
Taxonomy digest:
`sha256:c0f6cbddfe43e3a6eb3de01dbbbbc1ceebcb83d50cc197999776f58e3d9ce20d`

This protocol makes SignLab collection and temporal annotation repeatable. It covers
the five predefined target gestures and the learned `other` class described by the
[gesture taxonomy](gesture-taxonomy.md). It does not claim that those gestures are
validated words in a named sign language.

This is a draft for a synthetic dry run and the later pilot. The numerical targets
below are engineering starting points, not universal statistical standards. Story
#21 will measure rejection rates and coverage during a real, authorized pilot and
then either revise or freeze version 1.

## Hard stop before real collection

**Do not turn on a camera for a participant while the governance readiness result is
`blocked`.** The current public repository cannot authenticate consent or authorize
collection. A real session requires all of the following outside this protocol:

1. An authorized operator confirms that the exact collection-readiness record is
   `ready`, with every required external approval and storage control resolved.
2. The identity-vault workflow produces an authenticated consent receipt and a
   recording-level grant for the exact purpose and study.
3. The participant receives the approved information, can ask questions, and knows
   that participation is voluntary and may be stopped or withdrawn.
4. The private capture destination, access roles, backup behavior, and deletion path
   match the approved governance configuration.

Never copy names, contact information, signatures, device serials, or the
identity-to-pseudonym mapping into a prompt plan, filename, dataset table, log, Git,
DVC metadata, or experiment tracker. Audio is disabled for every SignLab recording.
See [participant-data governance](governance/README.md) for the authoritative gate.

The committed mock session is intentionally identity-free, uses no person or camera,
and does not represent consent, approval, or production readiness.

## Roles and records

One person may hold multiple roles during a synthetic rehearsal, but the real pilot
records who performed each role using approved operator/reviewer IDs:

- **Session operator:** verifies the gate and checklists, prepares the prompt plan,
  operates capture, records deviations, and never coaches the answer after a prompt.
- **Participant:** chooses whether to continue and may pause, retry, skip, or stop.
- **Annotator:** applies temporal boundaries and dispositions without seeing model
  predictions.
- **Reviewer:** independently checks labels, boundaries, and exclusions.
- **Adjudicator:** resolves recorded disagreements; the original draft remains
  auditable.

Use three distinct record layers:

1. The normalized dataset tables hold stable participant/session/recording identities,
   camera facts, root-recording intervals, labels, dispositions, and review state.
2. The collection sidecar holds protocol version, visit, prompt seed and realized
   order, repetition, condition codes, checklists, technical retries, and deviations.
   The fixture in `tests/fixtures/public/collection/mock-session-plan.json` is an
   evidence format, not yet a production interchange contract. Story #17 owns that
   tool-facing contract.
3. Authenticated consent and identity records stay behind the identity-vault boundary;
   only the approved pseudonymous grant snapshot reaches restricted research data.

Do not overload filenames, device IDs, or annotation reason codes with information
that belongs in the collection sidecar.

## Pilot collection design

The initial pilot design is deliberately small enough to execute and rich enough to
expose leakage and coverage problems:

- Use at least three adult signers after the governance gate becomes ready.
- Give each signer a random pseudonymous ID and collect two visits at least 24 hours
  apart. Tear down or reposition the setup between visits.
- A visit contains separate `isolated` and `continuous` dataset sessions because
  `capture_mode` is a scalar session field.
- In each visit, schedule five isolated attempts for each target: 25 target trials
  per visit and ten per target across the two visits. Retain mistakes and technical
  failures; the pilot report distinguishes scheduled, accepted, and recollected
  counts.
- In each continuous block, include every target at least once and at least 60 total
  seconds of explicit inactivity distributed before, between, and after events.
- Across a signer's two visits, prompt at least two examples of each source hard-
  negative kind: `partial_target`, `oov_gesture`, `incidental_activity`, and
  `two_hand_non_target`.
- Retain natural mistakes, partial attempts, unrelated movement, pauses, and direct
  target-to-target transitions. Do not keep only successful, tightly trimmed clips.
- Cover at least two device configurations, with each configuration used by at least
  two signers. Every declared framing, distance, lighting, and clothing-contrast
  profile must include every target and be used by at least two signers.
- Seek naturally occurring left- and right-hand coverage. Never ask a participant to
  perform unnaturally with a nondominant hand just to fill a cell. Record the gap and
  narrow the eventual claim if coverage remains insufficient.

These are minimum pilot targets, not a promise of dataset sufficiency. The intended
production study is roughly 8–12 signers, at least two separated visits per signer,
and about ten accepted performances per signer and target. Story #21 must replace
those estimates with a documented decision based on pilot evidence before scaling.

### Condition ledger

Before a visit, assign opaque condition codes for these controlled dimensions:

| Dimension | Required record | Pilot variation rule |
| --- | --- | --- |
| Framing | upper body and both hand-travel areas visible, with margin | Use each declared profile for every target. |
| Distance | measured setup profile, not a person's location | Use each profile for every target and at least two signers. |
| Lighting | stable, usable face/hand/body-anchor visibility | Include ordinary variation; reject unsafe or unusable setups. |
| Clothing | coarse contrast profile only | Vary contrast without recording sensitive free text. |
| Handedness | observed `left`, `right`, or `both` per recording | Preserve natural performance; normalization is downstream. |
| Camera | facing, resolution, frame rate, rotation, and mirror state | Record actual facts; do not infer them from a filename. |

Change conditions by block or visit, not by class. Every target appears under each
assigned profile so a label cannot be learned from lighting, clothing, device, or
background. Keep signer, visit, session, and device grouping intact for later
leakage-resistant splits.

### Repeatable pilot profiles and prompt cards

The draft pilot begins with two setup profiles. Measure distance from the camera
lens to the participant's upper-chest plane after the participant is comfortably
positioned; do not record a home or venue location.

| Profile code | Distance and orientation | Lighting and clothing observation | Framing check |
| --- | --- | --- | --- |
| `near_soft_front_landscape` | `0.8 m +/- 0.1 m`; landscape; record actual rotation | Diffuse source within 30 degrees of camera; plain top whose boundary with the hands remains visually clear | Top of head through mid-torso and both complete hand-travel areas remain visible with roughly one hand-width of margin. |
| `medium_diffuse_side_portrait` | `1.2 m +/- 0.1 m`; portrait; record actual rotation | Diffuse source 30–60 degrees to one side; plain top with reduced but still usable hand-boundary contrast | Same body and hand-travel coverage; rotate/reframe without cropping either side. |

`High contrast` means that an operator can follow the hand boundary throughout the
short setup playback; `medium contrast` means the boundary remains visible but has
less separation. Do not record skin color, ethnicity, clothing descriptions, room
descriptions, or other sensitive/free-text attributes. If the distinction cannot be
made consistently, use an `unassessed` profile, record the limitation, and do not
invent a value. Story #20 later defines machine-computable landmark-quality policy.

The prompt display uses the immutable taxonomy text, not an operator paraphrase:

| Stable ID | Card title | Card instruction |
| --- | --- | --- |
| `hello` | Hello | Open hand near the temple or upper face; make one short outward or side-to-side wave. |
| `no` | No | Close the thumb toward the extended index and middle fingertips one or more times. |
| `please` | Please | Place a flat or open palm at the upper chest; make one small circular stroke. |
| `thank_you` | Thank you | Begin with a flat or open hand at the chin or lower face; move outward in one stroke. |
| `yes` | Yes | With a closed fist near the upper torso, make one or more wrist-driven vertical nods. |

Every card also says: “Perform this once at a natural, comfortable speed using the
hand you would normally use. You may pause, skip, retry, or stop.” The title is a
project mnemonic, not a named-language claim. Show all cards during orientation;
during capture show only the current occurrence ID and its card.

## Prepare the prompt plan

Generate the plan once before capture and preserve the realized order even if the
session later aborts. Record a seed, an algorithm/version marker, and the exact
realized order. The realized order is authoritative evidence; a seed without the
matching algorithm is not enough to reconstruct it. The fixture demonstrates these
fields, while Story #17 owns their production contract and generator.

For each isolated session:

1. Create five occurrences of each target ID: `hello`, `no`, `please`, `thank_you`,
   and `yes`.
2. Randomize the balanced list with a recorded seed.
3. Reject an order only if the same target is immediately adjacent to itself; use a
   deterministic balancing procedure, not operator preference.
4. Use a different realized order for the second visit.
5. Assign a stable prompt occurrence ID and repetition number. Never renumber after a
   failure, skip, or retry.

For each continuous session, prepare a block containing all five targets, inactivity,
direct transitions, natural mistakes, partial attempts, and unrelated hand activity.
Randomize target order independently of the isolated block. Prompts describe what to
attempt, not what the annotation must become.

Do not reroll a sequence because a participant made mistakes. A technical failure may
create a linked retry, but the failed occurrence and coded reason remain in the
sidecar. A performance mistake remains source evidence and is annotated according to
what is visible.

## Run a visit

### Before capture

For a real visit, stop unless every consent item below is confirmed through the
authorized workflow:

- the readiness result is `ready`, current, and bound to this policy and storage;
- the authenticated receipt, recording grant, purpose ID, and study ID agree;
- the participant received the current documents and a copy to keep;
- requested uses are no broader than the choices the participant allowed;
- the participant knows how to pause, stop, ask questions, and withdraw;
- the pseudonymous participant ID is active and the identity mapping stays in the
  separate vault; and
- audio capture is disabled.

Then complete the setup checklist:

- clean the lens; stabilize and orient the camera;
- remove third parties and identifying material from the frame;
- verify that the full hand-travel region, upper torso, face-area anchors, and both
  hands when relevant remain visible with margin;
- verify usable, stable lighting without glare or silhouette;
- record the device configuration, camera facing, actual resolution and frame rate,
  rotation, mirror state, distance, framing, lighting, and clothing-contrast codes;
- confirm monotonic timestamps and available private storage; and
- show the participant the five standardized taxonomy prompts and allow questions
  without coaching a particular performance.

The pilot targets 720p at 30 frames per second when the approved device supports it,
but always record actual values. Landmark completeness and missing-frame thresholds
belong to Story #20, not to operator judgment in this protocol.

### During isolated capture

Present one occurrence at a time in the realized order. Allow a neutral rest before
and after the attempt. Do not stop early because the operator thinks the gesture was
wrong. Record one of these outcomes in the sidecar:

- completed as presented;
- participant-requested pause or skip;
- natural mistake retained;
- technical retry required; or
- session stopped.

Retry only for a technical or protocol failure such as camera interruption, prompt
display failure, a third party entering frame, or the hand-travel area leaving the
declared setup. Link the retry to the original occurrence and keep the reason. Never
silently replace an inconvenient example.

### During continuous capture

Record untrimmed sequences. Include:

- neutral rest and ordinary inactivity;
- every target in independently randomized order;
- direct target-to-target transitions;
- partial or aborted target attempts;
- complete out-of-vocabulary gestures;
- incidental activity such as reaching, device interaction, face or clothing touch;
- coordinated two-hand non-target activity; and
- naturally occurring occlusion or off-frame evidence.

Natural transition movement is context, not automatically a source `other` label.
Only a later derived train/development artifact may use
`other/transition_fragment`, as allowed by the taxonomy.

### After each recording

Before leaving the capture workflow:

- play a short portion from the beginning, middle, and end;
- confirm expected duration, readable motion, audio absence, orientation, mirror
  state, timestamps, and recorded resolution/frame rate;
- check for frozen, truncated, corrupt, or unexpectedly missing content without
  inventing downstream landmark-quality thresholds;
- reconcile prompt occurrence IDs, skips, retries, deviations, and recording IDs;
- compute and store the media checksum in the restricted manifest;
- quarantine any recording with a consent, third-party, or storage-boundary problem;
  and
- record accept, quarantine, or recollect—not a silent deletion.

## Temporal annotation rules

All annotation intervals use root-recording microseconds and half-open bounds:
`[start_us, end_us)`. The start is included; the end is the first excluded instant.
Intervals must be non-empty and must remain inside the source recording and any
referenced clip.

Annotate observable evidence rather than the prompt:

- **Static or held form:** start when the class-defining handshape, orientation, and
  body-relative location are first established. End before a defining parameter
  changes or retraction begins.
- **Dynamic form:** start at the first class-defining motion after generic raising or
  pre-shaping. End after the final defining motion or hold and before retraction or
  the next event.
- **Direct transition:** exclude generic travel from both neighboring targets. When
  there is a defensible direction, shape, or parameter change, place the boundary
  there. If no defensible boundary or class can be observed, use `ambiguous` with a
  coded reason and send it to review.
- **Inactivity:** leave it without a candidate annotation. `inactive` is a detector
  state, never a classifier label.

Use the taxonomy's positive, negative, ambiguous, and transition examples for each
target. `abstain` is a runtime decision and never annotation ground truth.

### Dispositions

| Visible evidence | Annotation |
| --- | --- |
| Complete, defensible target | `class_label` with its target `label_id` |
| Complete non-target event | `class_label`, `label_id: other`, and registered `other_kind` |
| Clearly incomplete target attempt | `other/partial_target` |
| Complete out-of-vocabulary gesture | `other/oov_gesture` |
| Unrelated intentional hand activity | `other/incidental_activity` |
| Coordinated two-hand non-target activity | `other/two_hand_non_target` |
| Defining evidence is not reliably visible | `ambiguous` with a coded reason |
| Region is excluded from training and scoring | `ignore` with a coded reason |

Do not create a source `other/transition_fragment` annotation merely because two
targets touch. Do not convert poor detector segmentation into a different source
class.

The initial reason-code allowlist is:

- `unusable_occlusion` or `boundary_unclear` for genuinely ambiguous evidence;
- `consent_exclusion`, `camera_setup`, `third_party_presence`,
  `unresolved_conflict`, or `unusable_occlusion` for ignored regions.

Story #17 must publish the controlled sidecar/reason-code contract before production
import. Free-text explanations belong only in an approved restricted review system,
not in dataset identifiers.

## Review and adjudication

The first pass is `draft` and never eligible for training. A reviewer sees the source
recording, taxonomy, protocol version, and draft annotation but not a model prediction.
The reviewer checks class, `other_kind`, disposition, interval boundaries, reason,
and source IDs.

For the pilot:

- independently review every `ambiguous` and `ignore` row;
- independently review every pilot class row (the future production minimum may be
  a signer/session/label/condition-stratified 20% sample only after pilot evidence);
- record each disagreement rather than overwriting the first pass;
- mark an agreed row `reviewed`; and
- route every disagreement to an adjudicator, who records the final row as
  `adjudicated`.

Only `class_label` rows with `reviewed` or `adjudicated` status are
`eligible_for_training: true`. `draft`, `ambiguous`, and `ignore` rows are always
ineligible. Review status proves workflow completion, not current consent; use still
requires the separate authenticated authorization check.

## Coverage and release check

After each visit, update a coverage ledger by target, signer, visit, dataset session,
device configuration, condition code, observed handedness, and hard-negative kind.
Before declaring the pilot complete, verify:

- every target has the planned isolated repetitions for every signer and visit;
- every target appears in every declared condition profile;
- no target is uniquely associated with one device, background, lighting, clothing,
  distance, prompt position, signer, or visit;
- visits remain separated and grouping IDs can support signer- and session-held-out
  splits;
- continuous recordings include the inactivity and hard-negative plan;
- all skips, retries, quarantine decisions, and protocol deviations are retained;
- all annotations satisfy the dataset contract and independent review rules; and
- consent authorization is checked independently at every permitted use boundary.

Missing coverage is a recorded limitation or a recollection decision. It is never
filled by copying a recording, relabeling a transition, moving the same signer across
evaluation groups, or fabricating authorization.

## Synthetic mock rehearsal

A mock annotation output uses this exact closed `annotations-table/1` shape. The
top-level collection is named `rows`, every listed field is required, and no other
field is allowed:

```json
{
  "schema_version": "annotations-table/1",
  "rows": [
    {
      "annotation_id": "annotation_<32 lowercase hexadecimal characters>",
      "clip_id": null,
      "disposition": "class_label | ambiguous | ignore",
      "eligible_for_training": true,
      "interval": {
        "schema_version": "media-interval/1",
        "start_us": 1000000,
        "end_us": 2000000
      },
      "label_id": "hello | no | please | thank_you | yes | other | null",
      "other_kind": "partial_target | transition_fragment | oov_gesture | incidental_activity | two_hand_non_target | null",
      "participant_id": "participant_<32 lowercase hexadecimal characters>",
      "reason_code": null,
      "review_status": "draft | reviewed | adjudicated",
      "session_id": "session_<32 lowercase hexadecimal characters>",
      "source_recording_id": "recording_<32 lowercase hexadecimal characters>"
    }
  ]
}
```

The union-style strings above document allowed values; an actual row contains one
JSON string or `null`, not the `|` notation. For `ambiguous` and `ignore`, set
`label_id` and `other_kind` to `null`, supply the protocol reason code, and set
eligibility to `false`. For `class_label`, set `reason_code` to `null`; only the
`other` class has a non-null `other_kind`. Use `clip_id: null` for the root-recording
mock observations. Sort rows by unique `annotation_id`.

A new collector can rehearse without a camera or participant:

1. Read this protocol and the gesture taxonomy.
2. Open `tests/fixtures/public/collection/mock-session-plan.json`; verify that it is
   marked fixture-only, synthetic, and governance-blocked.
3. Check the two visit times, separate capture-mode sessions, taxonomy digest,
   condition ledger, realized prompt sequences, repetition counts, and checklists.
4. Treat the mock observations as the visible evidence and apply the disposition,
   boundary, review, and eligibility rules above.
5. Write the result in the exact `annotations-table/1` shape above, sorted by
   `annotation_id`, without adding prompt or reviewer fields to table rows.
6. Compare with
   `tests/fixtures/public/collection/annotations-table-1.mock-session.json` and run:

   ```shell
   uv run pytest --no-cov tests/test_collection_protocol.py
   ```

Passing proves that the synthetic plan and annotations are repeatable and
schema-valid. It does not prove capture quality, participant consent, operator
training, institutional approval, private storage, or production readiness.

The mock plan supplies synthetic independent-review/adjudication outcomes so a new
collector can derive the final `review_status`. It does not preserve real reviewer
identities, first-pass drafts, comments, or disagreement history. Story #17 owns
that auditable production sidecar; the normalized annotation table remains the final
reviewed projection.

## Versioning and deviations

Every collection sidecar records the exact protocol version. Correct spelling,
clarify language, or add non-semantic examples with a patch version. Change
collection quantities, condition assignments, boundary rules, review thresholds, or
eligibility behavior only in a reviewed new version. Never rewrite the meaning of a
version already used by a recording.

Record deviations with the visit, session, affected prompt occurrences or recording,
reason, operator decision, and whether recollection is required. Story #21 owns the
pilot report and the decision to freeze or revise version 1.

## Research basis

The operating rules combine primary-source guidance with conservative engineering
judgment:

- [NIST randomized designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri331.htm)
  motivate preserving randomized run order to avoid confounding conditions with
  experimental order.
- The [IPN Hand paper](https://gibranbenitez.github.io/2021_ICPR_IpnHand.pdf) and
  [project](https://gibranbenitez.github.io/IPN_Hand/) motivate randomized prompts,
  continuous sequences, non-gesture activity, breaks, and direct transitions.
- [EgoGesture](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/EgoGesture.pdf) and
  [HaGRID](https://openaccess.thecvf.com/content/WACV2024/papers/Kapitanov_HaGRID_--_HAnd_Gesture_Recognition_Image_Dataset_WACV_2024_paper.pdf)
  motivate deliberate subject, scene, distance, lighting, and device variation.
- The [DGS Corpus segmentation guidelines](https://www.sign-lang.uni-hamburg.de/dgs-korpus/arbeitspapiere/DGS-Korpus_AP03-2010-01v02_en.pdf)
  motivate written onset, offset, retraction, and transition rules.
- [CVAT's manual QA workflow](https://docs.cvat.ai/docs/qa-analytics/manual-qa/)
  motivates distinct annotation, review, correction, and adjudication steps.
- [HHS OHRP consent checklists](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/checklists/index.html)
  inform the consent-process checklist without determining whether a regulation
  applies to SignLab.
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) motivates documenting
  collection, composition, uses, limitations, and maintenance decisions.

The exact signer counts, repetition counts, visit separation, coverage minima, and
review percentages in this draft are SignLab engineering decisions. The cited sources
do not establish them as universal statistical or regulatory thresholds.
