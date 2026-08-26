# SignLab

SignLab is a research prototype for recognizing isolated performances of five
predefined hand gestures within continuous webcam video. It separates non-target
events (`other`) from no active event (`inactive`) and uncertain decisions
(`abstain`); no sign-language or translation capability is claimed.

The project replaces an earlier Streamlit prototype with a UI-independent ML
pipeline, leakage-resistant evaluation, and a privacy-preserving browser demo.

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

## Developer quickstart

Prerequisites are Git and `uv` 0.12.6; `uv` installs the pinned CPython 3.12.14
runtime, so a system Python installation is not required.

```shell
git clone https://github.com/peterbucci/signlab.git
cd signlab
uv python install
uv sync --locked --all-groups
uv run signlab --help
uv run signlab doctor check
```

The locked environment contains every development tool. Run the same gates as CI:

```shell
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_repository_hygiene.py
uv build --no-build-isolation
uv run python scripts/verify_distribution.py dist
```

See [docs/development.md](docs/development.md) for directory ownership, dependency
updates, generated-file policy, and the Python-version decision.

The [gesture taxonomy and claim boundary](docs/gesture-taxonomy.md) define the
versioned six-output classifier vocabulary and the evidence required before making
any named-language claim.

The [participant-data governance baseline](docs/governance/README.md) provides the
blank consent form, machine-readable policy, collection-readiness gate, and an
executable withdrawal dry run. Real participant collection remains blocked until
the documented human, institutional, legal, storage, and contact checks are
resolved outside this public repository.

The [legacy audit](docs/legacy-audit.md) and
[portable evidence export](docs/legacy-export.md) document what was retained from
the school project, why it is development-only, and how to validate it without any
legacy application dependencies.

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

The immutable legacy audit, reproducible developer foundation, sanitized legacy
evidence export, versioned gesture taxonomy, and fail-closed participant-data
governance baseline are established. The remaining core pipeline contracts are
still under construction; no headline model result is considered valid until the
grouped evaluation foundations are complete.

## Data and privacy

Raw participant video and contributed feedback are not committed to Git. Names,
contacts, signatures, and the identity-to-pseudonym mapping belong only in a
separate encrypted identity vault; research systems use random pseudonymous IDs.
The legacy quarantine is currently local-only and explicitly not a durable backup;
a later private-DVC story will place approved data behind a controlled remote.
Public fixtures must be synthetic, explicitly consented, or separately licensed.

## License

Code is licensed under the MIT License. Dataset and model artifacts may have
separate terms documented in their data and model cards.
