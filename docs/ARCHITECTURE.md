# Architecture

## Core release

```text
Raw private videos
  -> versioned sample manifest and frozen split manifest
  -> MediaPipe Tasks extraction
     - image-space and world hand landmarks
     - selected body anchors
     - timestamps, validity masks, handedness, confidence
  -> versioned feature representations
  -> train -> calibrate -> evaluate -> continuous replay benchmark
  -> immutable ONNX model bundle

Research surface: Python CLI, DVC, MLflow, Parquet
Public surface: React/Vite, Web Worker, MediaPipe Tasks, ONNX Runtime Web
```

The root DVC graph is generated from the typed SignLab stage registry and ends at
the split/feature data boundary. A future Prefect flow schedules those same services
or DVC targets; it cannot define a second DAG. Public Git contains only the synthetic
fixture lock. Participant-data pointers and production lock history live in a
separate protected metadata repository. See [data versioning](data-versioning.md).

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

The [dataset manifest](dataset-manifests.md) includes stable identifiers for clips,
participants, sessions, source recordings, devices, capture conditions,
handedness, mirroring, consent, and checksums. Derived samples inherit the source
recording and split. Six normalized tables use explicit Arrow schemas and Parquet
storage while retaining storage-independent semantic hashes. V2 row artifacts use
hash-derived logical paths whose only filename is the opaque artifact ID; dataset
locators therefore cannot carry hostnames, participant names, or free-text labels.

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

Every participant-derived asset uses a `signlab://` logical locator and participates
in a complete lineage graph. A deterministic withdrawal planner computes direct
roots and every downstream descendant, including shared datasets, runs, models,
reports, demos, caches, and backups. It emits a dry-run plan only; external deletion
requires future authorized store adapters and attestations.

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
