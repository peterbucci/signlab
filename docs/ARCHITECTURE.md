# Architecture

## Core release

```text
Licensed public archives -> external-dataset-manifest/1 --+
Synthetic fixture or future consented media                  |
  + collection-sidecar/1 -> raw-dataset-manifest/1 ----------+
  -> reviewed extraction input bridge
  -> MediaPipe Tasks extraction
     - image-space and world hand landmarks
     - selected body anchors
     - timestamps, validity masks, handedness, confidence
  -> quality policy -> dataset-manifest/2 + frozen split manifest
  -> versioned feature representations
  -> train -> calibrate -> evaluate -> continuous replay benchmark
  -> immutable ONNX model bundle

Research surface: Python CLI, DVC, MLflow, Parquet
Public surface: React/Vite, Web Worker, MediaPipe Tasks, ONNX Runtime Web
```

The root DVC graph is generated from the typed SignLab stage registry. Its current
stages produce synthetic receipts only: they prove the intended boundaries and DVC
wiring, not production ingestion, extraction, quality, splitting, or features. Later
stage stories provide importable services while preserving one registered graph;
the raw importer and MediaPipe extractor now exist independently of their still-
synthetic DVC receipts. Public Git contains only the synthetic fixture lock. When
Story #19 creates the first approved production version, participant-data pointers
and production lock history will live in a separate protected metadata repository.
See [data versioning](data-versioning.md).

## Static browser shell

`apps/web/` is a self-contained React, TypeScript, and Vite application. Its current
release is a responsive static shell with overview, live demo, replay, results,
methodology, feedback, privacy, and limitations routes. It needs no FastAPI process,
database, user account, runtime token, local filesystem path, analytics endpoint, or
development API. Hash-based navigation works on a root domain and on an arbitrary
configured subpath without server-side route rewrites.

The feature routes describe their intended behavior but deliberately do not imitate
unfinished features. This shell requests no camera permission, makes no runtime data
request, loads no model, reads no replay input, and stores no feedback. Those
capabilities belong to later stories and retain their own evidence gates.

The intended bundle-loading flow is manifest-first:

```text
Static model-bundle files
  -> load and validate manifest + schema versions
  -> verify every declared asset checksum
  -> start landmark and inference workers from verified bytes
  -> expose readiness or a clear failure to the route UI
```

The immutable bundle, workers, MediaPipe Tasks, ONNX Runtime Web, cache policy, and
Python/TypeScript parity fixtures are not part of the shell. No browser model is
usable until those later gates are complete.

The [licensed external-data boundary](external-datasets.md) is intentionally
separate from participant ingest. It registers source, license, attribution,
sensitivity, use limitations, a reviewed label mapping, and an offline acquisition
plan. Its PopSign adapter imports only operator-downloaded official archives and
publishes `external-dataset-manifest/1`; it never downloads human media. The
manifest preserves source split and exact byte lineage with source-namespaced opaque
signer identities, but it contains no SignLab consent object. Story #74 owns the
later bridge into the common extraction path; Story #73 does not create features or
trainable samples.

## State model

- **Inactive:** no candidate event is currently present; owned by the candidate-event detector.
- **Other:** a candidate event occurred but is not one of the target gestures; learned negative class.
- **Abstain:** evidence is insufficient to name a target; calibrated decision policy.

The candidate-event detector, classifier, calibrator, and decision policy are
versioned and evaluated separately.

## Data contracts

The [pipeline-contract baseline](contracts.md) gives datasets, grouped splits,
preprocessing plans, resolved configurations, terminal runs, and research models a
single digest-bound provenance chain. New contract identities use domain-separated
RFC 8785 canonical JSON and SHA-256. Physical storage locations remain outside the
identity boundary: documents carry only normalized workspace-relative paths or
logical `signlab://` URIs plus exact content digests.

Every reader dispatches on an exact `schema_version` and rejects unknown versions;
validation never performs an implicit migration. JSON Schema covers portable
structure, while application validators prove sample coverage, group isolation,
preprocessing-schema continuity, terminal run state, and cross-document identity.

Every collection, annotation, training, evaluation, bundle, and public-copy
contract embeds the same immutable taxonomy reference: `signlab-five@1.0.0` plus
the canonical SHA-256. The [taxonomy](gesture-taxonomy.md) is authoritative for
label order, event boundaries, negative examples, legacy aliases, and the public
claim. Downstream contracts may add fields but cannot reinterpret its identifiers.

The [capture/import boundary](capture-import.md) separates a UI-independent
`collection-sidecar/1` from the immutable `raw-dataset-manifest/1` handed to
extraction. The sidecar owns prompt order, condition assignments, stable opaque
workflow IDs, attempts, and review decisions; source paths live only in an ephemeral
operator map. The raw manifest reuses the normalized participant, session,
recording, clip, annotation, and derived-artifact tables, with the two derived-media
types—clips and derived artifacts—empty before extraction. It binds the finalized
sidecar, governance policy, lineage inventory, and taxonomy without pretending that
raw recordings are trainable samples.

The [landmark-extraction boundary](landmark-extraction.md) consumes that raw
manifest without creating sample, label, feature, quality, or split identity. Its
canonical Python batch pins `mediapipe==1.0.1`, PyAV `18.1.0`, CPU/VIDEO execution,
all task thresholds, the tracker, and exact Hand Landmarker Full and Pose Landmarker
Lite bytes. The future browser path pins `@mediapipe/tasks-vision@1.0.1` and the same
two `.task` hashes. Model assets remain external and are loaded only from verified
local buffers; extraction never auto-downloads them.

Each source frame preserves PTS, rational time base, relative microseconds, and the
strictly increasing MediaPipe millisecond timestamp derived from them. Output keeps
two stable hand slots with 21 image/world points, detector order, handedness and
score; six ordered shoulder/elbow/wrist anchors; and explicit absent/invalid masks.
Source mirror and rotation facts are recorded but never silently normalized. A
semantic frames-table digest and an exact Parquet-byte digest answer different
reproducibility questions, and the extraction manifest binds both to raw data,
configuration, model assets, and derived-artifact lineage.

This is a raw diagnostic boundary only. The [landmark-quality boundary](landmark-quality.md)
then recomputes elapsed-time timing and resampling evidence plus gap, pose,
confidence, discontinuity, and suspected-swap findings from those exact immutable
rows. It publishes only a canonical
report manifest: raw Parquet is never rewritten, short-gap interpolation remains a
declared plan rather than a feature tensor, and no sample or split identity is created.
The [portable feature boundary](landmark-representations.md) now derives three
independently selectable, fixed-shape representations from those exact rows and
quality decisions. It preserves stable hand slots, masks missing body context,
fits optional normalization on explicitly identified training inputs only, and
uses all upstream semantic hashes in a content-addressed cache. Licensed-corpus
execution and publication remain Story #74 rather than being hidden inside the
transform library. Private participant extraction additionally requires
authenticated authorization for the consent scope's `derived_features` field and
runs in a network-isolated environment after task assets have been acquired. Public
extraction and quality fixtures are synthetic and do not relax that gate.

The sample-bearing [dataset manifest](dataset-manifests.md) includes stable
identifiers for clips, participants, sessions, source recordings, devices, camera
facts, handedness, mirroring, consent, and checksums. Derived samples inherit the
source recording and split. Six normalized tables use explicit Arrow schemas and
Parquet storage while retaining storage-independent semantic hashes. V2 row
artifacts use hash-derived logical paths whose only filename is the opaque artifact
ID; dataset locators therefore cannot carry hostnames, participant names, or
free-text labels.

## Participant-data boundary

The identity vault is a separate encrypted administrative system. It alone holds
names, contacts, signatures, completed consent forms, and the mapping to random
pseudonymous signer IDs. Git, DVC, MLflow, manifests, logs, datasets, runs, and
artifacts must never contain that identity data.

Restricted research records snapshot both a validated consent receipt and a
recording-level grant, including opaque purpose and study IDs. Missing, expired,
withdrawn, incompatible, or broader-later consent fails closed. Reuse under a new
study ID also requires the explicit same-purpose-future-research choice, and a
different purpose ID is never authorized by the original receipt. A positive
authorization decision additionally requires an authenticated verifier for the
complete receipt, recording grant, and consent-event-log tuple; hashes and caller
assertions alone do not establish authenticity. The legacy export remains
`consent_status: unknown` and is quarantined from training, evaluation,
demonstrations, and publication.

The raw importer enforces that same boundary. Its public fixture is explicitly
synthetic; structurally valid sidecar data is not evidence that a real collection is
authorized. Real media cannot cross the boundary until readiness, private storage,
and authenticated consent verification are supplied by their external owners.

Every consent-bound participant asset uses a `signlab://` logical locator and
participates in the governance lineage graph. Captured retry or quarantine bytes
without consent evidence cannot truthfully use that contract; a separate strict
quarantine inventory retains their content-addressed locations, pseudonymous
participant/recording identities, coded reasons, and explicit absent-consent status
for withdrawal discovery. A deterministic withdrawal planner computes direct roots
and every downstream descendant of consent-bound assets, including shared datasets,
runs, models, reports, demos, caches, and backups. It emits a dry-run plan only;
external deletion requires future authorized store adapters and attestations.

The model bundle will eventually contain:

```text
model.onnx
labels.json
input-schema.json
preprocess.json
segmenter.json
decision-policy.json
calibration.json
manifest.json
model-card.md
golden/
```

## Optional platform

Only after the static release is complete, a local React Studio may call FastAPI
to start durable Prefect jobs and query MLflow/DuckDB. PostgreSQL and S3-compatible
storage are introduced only after local SQLite/artifacts become limiting.

The optional platform must reuse the same Python functions and contracts; it must
not become a second implementation of the pipeline.
