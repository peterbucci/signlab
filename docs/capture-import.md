# Capture and raw dataset import

SignLab's capture/import boundary represents collection state independently of any
particular camera application or UI. Authorization is a separate, fail-closed
decision. The current public importer exercises the handoff with explicitly
synthetic inputs only, using two main
contracts with deliberately different jobs:

- `collection-sidecar/1` records the workflow state, preallocated opaque IDs,
  recording attempts, consent grants, and annotation review decisions.
- `raw-dataset-manifest/1` is the immutable handoff from import to extraction. It
  binds the accepted media, normalized raw tables, lineage inventory, governance
  policy, taxonomy, and finalized sidecar by digest.

The raw handoff is not `dataset-manifest/2` and does not introduce a
`dataset-manifest/3`. A raw collection does not yet have derived clips or trainable
samples, so forcing it into the sample-bearing dataset contract would invent
evidence. Extraction, quality assessment, and feature construction consume the raw
handoff in later stories and eventually produce the existing normalized dataset
contract.

## Stable workflow identity and byte identity

The `capture-identifier-set/1` helper carries random, opaque workflow identities.
Allocate them before recording so a recording-level consent grant can name the exact
attempt recording it authorizes:

```shell
uv run --locked signlab data allocate-capture-ids capture-ids.json
```

The allocation file is local operational state. Running the command again against
the same file validates it and leaves the assigned IDs unchanged; it does not
silently replace them. Names, prompt text, original filenames, hostnames, and other
identifying values do not belong in IDs.

Allocate a retry into a new file while retaining its existing workflow identity:

```shell
uv run --locked signlab data allocate-capture-ids retry-ids.json \
  --retry-of capture-ids.json
```

This preserves the collection, visit, participant, session, prompt-occurrence,
annotation, and review IDs while assigning new recording, attempt, and source IDs
for the new physical capture.

SHA-256 has a different purpose: it identifies captured bytes and detects duplicate
media. The prompt-occurrence ID remains stable across retries, while every attempt
uses its own preallocated attempt, source, and recording IDs and links directly to
the preceding retry. Reusing a source file under a different recording ID is still
a duplicate; reusing a recording ID for different bytes is a conflict. These
workflow IDs are never regenerated during pause, resume, retry, quarantine, or
idempotent re-import, while content hashes independently prove which bytes were
captured.

## Sidecar workflow

The sidecar is UI-independent. A browser, desktop capture tool, or carefully
prepared fixture may produce it, but all producers must satisfy the same strict
contract. The collection may be `active`, `paused`, or `complete`. Only a complete,
digest-valid sidecar—with a finalization time, no `pending` occurrence, and at least
one `accepted` occurrence—can be imported into a raw dataset. The `fixture_only`
marker describes the input honestly; setting it to `false` is not a readiness or
authorization decision.

Session plans bind each visit and coded condition profile to the reviewed protocol,
the prompt-randomization algorithm/version, a seed digest rather than the seed, the
authoritative realized order, and coded consent/capture checklist results. A plan
must state that the order is authoritative and was not rerolled. Prompt occurrences
also retain their planned ordinal and repetition number rather than inferring either
from filenames. A fixture-only plan marks every consent and capture check
`not_applicable` with the coded `synthetic_no_person_no_camera` reason; fake bytes
never claim that a real-world check passed.

Prompt occurrences retain their preallocated ID while attempts record `accepted`,
`retry`, or `quarantined` outcomes. Each retry directly links to the immediately
preceding attempt; an accepted attempt is terminal and supplies the occurrence's
one recording and exact recording-level consent grant. Retry and quarantine records
carry a coded reason and no consent grant. A `skipped` occurrence has no
capture attempt and carries a coded skip reason; a quarantined occurrence always
retains captured bytes. Pending, skipped, and quarantined occurrences remain
auditable but cannot enter the accepted recording table. An unknown or unverifiable
legacy consent state is quarantined rather than inferred.

Append one captured attempt without hand-editing the sidecar. The writer accepts an
existing active or paused sidecar and preallocated identifier file, computes the
media SHA-256 and byte size itself, derives any direct retry link, recomputes the
sidecar digest, and replaces the sidecar atomically:

```shell
uv run --locked signlab data append-capture-attempt collection-sidecar.json \
  --identifiers capture-ids.json \
  --media synthetic-source/incoming.webm \
  --outcome retry \
  --reason-code camera_interruption \
  --recorded-at 2026-08-26T12:10:00Z \
  --media-type video/webm \
  --duration-us 5000000 \
  --handedness right \
  --mirror-state mirrored \
  --rotation-degrees 0
```

An exact replay validates and leaves the sidecar unchanged. A replay that changes
the bytes or coded metadata for an existing attempt fails without replacing the
sidecar. For a later attempt, first allocate a new retry identifier file with
`--retry-of`, then invoke the writer with that file. A `retry` leaves the occurrence
pending; `quarantined` closes it as quarantined. The sidecar remains active or paused
as supplied.

An `accepted` append forbids a reason code and requires
`--consent-grant recording-consent-grant.json`. The writer strictly validates that
grant and its recording, participant, timestamp, taxonomy, and retention bindings;
it never creates or infers consent. Structural validation is not current consent
authorization, and the fixture-only import boundary below remains unchanged. The
writer deliberately does not author participants, sessions, plans, annotations, or
camera facts, and it does not probe media duration or codecs.

Annotations retain timestamped decisions made in the exact sequence annotator,
reviewer, then—when needed—adjudicator, using opaque actor IDs. One decision projects
to `draft`; matching annotator and reviewer proposals project to `reviewed`; a label,
boundary, disposition, or reason disagreement requires a third decision and projects
its final proposal to `adjudicated`. The complete history remains in the sidecar,
while import emits exactly one normalized annotation row with the derived
participant/session relationship and review status. Every continuous-video proposal
uses a non-empty, half-open `[start_us, end_us)` interval in the accepted recording's
microsecond coordinate system. Draft or unresolved work is not treated as trainable
evidence.

The sidecar and committed output contain only pseudonymous, opaque, or registered
coded values. A separate source map resolves opaque source keys to local files for
one import run:

```json
{
  "source_0123456789abcdef0123456789abcdef": "incoming/capture.webm"
}
```

The map must contain exactly one workspace-relative entry for every attempt source
key, with no missing or additional keys. Then validate and import the sidecar:

```shell
uv run --locked signlab data validate-capture collection-sidecar.json
uv run --locked signlab data import-capture collection-sidecar.json \
  --source-map source-map.json \
  --source-root synthetic-source \
  --output raw-dataset
```

Treat an operational source map as ephemeral input. Never commit one that refers to
participant media, place it in the raw dataset, or copy its source paths into logs.
The repository's checked-in map resolves only declared fake fixture bytes. Import
reports use opaque IDs and coded outcomes. The importer resolves every source beneath
the explicit source root and rejects missing files, directories, symlinks, and path
escapes.

## Atomic, idempotent publication

Import validates the complete sidecar and exact gesture-taxonomy reference before
publishing anything. It then hashes the supplied media, rejects duplicate accepted
bytes and conflicting recording, source, participant/session, consent, taxonomy, or
annotation assignments, builds the participant, session, recording, annotation,
clip, and derived-artifact tables, and verifies their relationships and referenced
bytes. Clips and derived artifacts are empty at this raw boundary.

The importer stages accepted content-addressed media, every nonaccepted attempt's
bytes under a separate opaque quarantine address, the finalized sidecar, lineage
inventory, explicit no-consent quarantine inventory, normalized Parquet tables, and
raw manifest together. The raw manifest is written as the completion marker only
after the other artifacts are durable; the whole staged bundle is then validated and
published atomically. A failure leaves no partial dataset at the requested
destination.

An identical rerun validates the existing output and returns it unchanged. A rerun
that would change the same dataset/version destination fails instead of merging,
duplicating, or overwriting records. Validate a published handoff independently:

```shell
uv run --locked signlab data validate-raw-dataset \
  raw-dataset/raw-dataset-manifest.json \
  --workspace-root raw-dataset
```

This validator reports table, semantic, row-artifact, sidecar, accepted-lineage, and
quarantine-inventory integrity separately. Its consent-authorization result is
`not_checked`, never an inferred positive authorization.

The raw semantic identity is computed from validated logical rows and their required
governance, lineage, taxonomy, and sidecar bindings. It excludes source locations,
workspace locations, and Parquet encoding details. Equivalent imports on Windows
and Linux therefore have the same semantic data identity and manifest identity.

## Authorization boundary and limitations

The repository's executable example is synthetic and fixture-only. The public
importer rejects every sidecar whose `fixture_only` value is not exactly `true`.
This path demonstrates schema validation, workflow state, duplicate detection,
redaction, atomic publication, and deterministic replay; it does not demonstrate
participant consent, institutional approval, production storage, or collection
readiness.

Real participant media remains fail-closed until the governance readiness checklist,
authorized private storage, and an authenticated consent verifier are available.
A structurally valid receipt or grant and a matching hash do not establish current
authorization. Never substitute the synthetic fixture, a caller assertion, or a
successful import for that external decision.

This boundary intentionally does not provide a camera UI, media transcoding,
codec/duration/audio probing, landmark extraction, quality policy, grouped splitting,
feature construction, or a second DVC graph. Source size and SHA-256 are checked
against the bytes, but supplied media type, duration, and camera facts are
structurally validated rather than probed. The later quality and extraction stages
must measure the relevant media properties. The production services continue to fit
behind the single registered DVC stage graph.
