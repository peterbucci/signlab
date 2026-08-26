# SignLab

SignLab is a reproducible research and deployment project for a small-vocabulary,
continuous hand-gesture recognition system. It replaces an earlier Streamlit
prototype with a UI-independent ML pipeline, leakage-resistant evaluation, and
a privacy-preserving browser demo.

## Project goals

- Evaluate on unseen signers and sessions rather than random clip splits.
- Preserve local hand shape, body-relative placement, timing, and detection quality.
- Model `inactive`, learned `other`, and calibrated `abstain` as separate states.
- Measure continuous-video event accuracy, false activations, and end-to-end latency.
- Export an immutable ONNX model bundle with Python/TypeScript parity tests.
- Publish a zero-install React demo that performs webcam inference on-device.

## Initial architecture

```text
Private research data
  raw videos -> versioned manifest -> MediaPipe Tasks extraction
             -> feature variants -> train/calibrate/evaluate -> model bundle

Local research tools
  Python CLI + DVC + MLflow + Parquet

Public product
  React/Vite -> MediaPipe Web Worker -> ONNX Runtime Web
```

The public demo is intentionally independent of the optional training platform.
FastAPI, Prefect, PostgreSQL, object storage, and a custom experiment Studio are
deferred until a concrete workflow requires them.

## Research questions

1. Does body-relative context improve signer-held-out generalization?
2. How much do learned hard negatives and calibrated abstention reduce false activations?
3. Which temporal model is sufficient for this dataset: a simple baseline, GRU, or TCN?
4. Does exported browser inference preserve the behavior of the research implementation?

## Roadmap

The delivery plan is organized into seven phases:

0. Preserve and audit the legacy school project.
1. Build the reproducible, consent-aware data foundation.
2. Establish trustworthy baselines and continuous evaluation.
3. Export and validate a portable inference bundle.
4. Build and validate the static browser demo.
5. Publish the portfolio release, evidence, and tutorials.
6. Add an interactive research platform only if justified.

See [docs/ROADMAP.md](docs/ROADMAP.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
and the [SignLab Roadmap](https://github.com/users/peterbucci/projects/5) for the
implementation backlog.

## Repository status

This repository is in project-initialization. The first milestone is a reproducible
legacy audit and a frozen dataset/experiment contract; no headline model result is
considered valid until those foundations are complete.

## Data and privacy

Raw participant video and contributed feedback are not committed to Git. Private
data is tracked through DVC pointers and a controlled remote. Public fixtures must
be synthetic, explicitly consented, or separately licensed.

## License

Code is licensed under the MIT License. Dataset and model artifacts may have
separate terms documented in their data and model cards.
