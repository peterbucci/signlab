# Roadmap

## Phase 0 — Preserve and audit

Archive the legacy repository, inventory runs and promoted models, export live
feedback, reproduce reported results, and publish an honest baseline report with
known limitations.

## Phase 1 — Reproducible data foundation

Define consent-aware manifests, DVC versioning, grouped split generation, a shared
MediaPipe Tasks extractor, raw landmark/timestamp/mask storage, quality validation,
and a pilot multi-signer collection protocol.

## Phase 2 — Trustworthy ML core

Implement a simple baseline, GRU and TCN; evaluate representation ablations;
collect hard negatives; calibrate abstention; implement continuous event matching;
and report signer/session-level uncertainty and runtime metrics through MLflow.

## Phase 3 — Portable inference

Define the model-bundle schema, export ONNX, and validate Python-to-ONNX,
Python-to-TypeScript feature, decision-policy, and continuous-replay parity.

## Phase 4 — Browser and portfolio release

Build a static React application with worker-based camera/replay inference, local
feedback, curated results, model/data cards, accessibility guidance, performance
budgets, CI, deployment, and a short demonstration video.

## Phase 5 — Optional research platform

If a real interactive workflow justifies it, add FastAPI, Prefect, SSE progress,
DuckDB error analysis, PostgreSQL/object storage, and custom Studio features that
MLflow does not already provide.

## Milestone gates

- Do not tune architectures before split-leakage tests pass.
- Do not publish accuracy before signer/session-held-out evaluation exists.
- Do not build the browser demo before an immutable bundle and parity fixtures exist.
- Do not build the optional platform before the static public release is complete.
