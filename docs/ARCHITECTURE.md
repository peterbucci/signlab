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

## State model

- **Inactive:** no candidate event is currently present; owned by the candidate-event detector.
- **Other:** a candidate event occurred but is not one of the target gestures; learned negative class.
- **Abstain:** evidence is insufficient to name a target; calibrated decision policy.

The candidate-event detector, classifier, calibrator, and decision policy are
versioned and evaluated separately.

## Data contracts

Every collection, annotation, training, evaluation, bundle, and public-copy
contract embeds the same immutable taxonomy reference: `signlab-five@1.0.0` plus
the canonical SHA-256. The [taxonomy](gesture-taxonomy.md) is authoritative for
label order, event boundaries, negative examples, legacy aliases, and the public
claim. Downstream contracts may add fields but cannot reinterpret its identifiers.

The sample manifest will include stable identifiers for clips, signers, sessions,
source recordings, devices, capture conditions, handedness, mirroring, consent,
and checksums. Derived samples inherit the source recording and split.

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
