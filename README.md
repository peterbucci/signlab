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
Authorized source media
  licensed public archives -> external-dataset-manifest/1 --+
  future consented video  -> raw-dataset-manifest/1 ----------+-> extraction
                                                              -> feature variants
                                                              -> train/evaluate
                                                              -> model bundle

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
uv sync --locked --all-groups --all-extras
uv run signlab --help
uv run signlab doctor check
```

After that locked install, run the complete synthetic reference experiment with:

```shell
uv run --locked --no-sync signlab train reference-experiment
```

It writes a verified research artifact pack and local MLflow evidence only under the
ignored `runs/` directory. Its scores prove reproducible pipeline mechanics on
no-person synthetic data; they do not measure real gesture quality, signer
generalization, or sign-language performance, and they do not replace the separately
evaluated PopSign browser candidate.

The locked environment contains every development tool. Run the same gates as CI:

```shell
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_repository_hygiene.py
uv run python scripts/generate_feature_resources.py --check
uv run python scripts/generate_feature_goldens.py --check
uv build --no-build-isolation
uv run python scripts/verify_distribution.py dist
```

The static browser shell is a separate locked application and does not require the
Python process to run:

```shell
cd apps/web
npm ci --no-audit --no-fund
npm run dev
```

See [docs/development.md](docs/development.md) for directory ownership, dependency
updates, generated-file policy, and the Python-version decision.

The [DVC data-versioning guide](docs/data-versioning.md) defines the fixture-only
stage scaffold, local clean-room transport proof, the future Story #19 protected
private-data gate, and tracker-neutral reproduction metadata that a later
experiment-tracking story records.

The [gesture taxonomy and claim boundary](docs/gesture-taxonomy.md) define the
versioned six-output classifier vocabulary and the evidence required before making
any named-language claim.

The [participant-data governance baseline](docs/governance/README.md) provides the
blank consent form, machine-readable policy, collection-readiness gate, and an
executable withdrawal dry run. Real participant collection remains blocked until
the documented human, institutional, legal, storage, and contact checks are
resolved outside this public repository.

The [collection and annotation protocol](docs/collection-protocol.md) defines the
draft pilot design, randomized prompt procedure, capture checklists, temporal
boundary rules, review workflow, and a no-camera synthetic rehearsal. It does not
authorize real collection; the governance readiness gate remains authoritative.

The [capture and raw dataset import guide](docs/capture-import.md) defines the
UI-independent collection sidecar, stable opaque workflow IDs, retry and quarantine
behavior, reviewed annotation projection, and the atomic `raw-dataset-manifest/1`
handoff. Its executable example is synthetic only; real media remains fail-closed
without readiness approval, private storage, and authenticated consent verification.

The [licensed external-dataset guide](docs/external-datasets.md) defines a separate
offline PopSign ASL v1.0 registry, five-target label selection, deterministic
acquisition plan, hostile-tar import boundary, and `external-dataset-manifest/1`.
CC BY 4.0 authorization is recorded without being misrepresented as SignLab
participant consent; the source contains identifiable people and remains outside
Git.

The [version-pinned landmark extraction guide](docs/landmark-extraction.md) defines
the exact MediaPipe/PyAV runtime and model bytes, source-timestamp preservation,
two-hand tracking, raw hand/body observation masks, deterministic Parquet evidence,
and the network-isolated private-data boundary. The subsequent
[landmark quality policy](docs/landmark-quality.md) assesses those immutable rows
with elapsed-time timing and resampling evidence, explicit gap and continuity
findings, deterministic triage, and a report-only manifest. The subsequent
[portable landmark representation](docs/landmark-representations.md) stage now
materializes hand-local, body-relative, and combined fixed-shape features with
train-only fitted statistics, complete cache identities, and synthetic goldens.

The [versioned pipeline contracts](docs/contracts.md) give datasets, grouped
splits, preprocessing plans, resolved configurations, terminal runs, and research
models portable RFC 8785 identities with explicit compatibility checks.

The [dataset manifest and Parquet contract](docs/dataset-manifests.md) defines the
normalized participant, session, recording, clip, annotation, and derived-artifact
tables, their lineage and consent checks, and the distinction between semantic
dataset identity and exact Parquet bytes.

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
1. Build the reproducible, license- and consent-aware data foundation.
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
governance baseline are established. A UI-independent capture sidecar and atomic raw
importer now produce a validated `raw-dataset-manifest/1` handoff from explicitly
synthetic inputs. A separate offline licensed-data boundary registers PopSign ASL
v1.0, freezes the reviewed five-target mapping and 15-archive plan, and imports only
explicitly acknowledged local archives into `external-dataset-manifest/1`. The
sample-bearing dataset contract retains six normalized,
lineage-preserving Parquet tables and the published v1 reader. A pinned MediaPipe
Tasks boundary now produces schema-validated, lineage-bound landmark Parquet with
separate semantic and exact-byte evidence; public examples remain synthetic. The
quality boundary records deterministic timing, gap, pose, continuity, and triage
evidence without rewriting extraction. A pure portable-feature service now derives
three exact representation variants, optional geometry and elapsed-time kinematics,
masked training-only statistics, and content-addressed cache objects. Licensed
PopSign execution now produces one deterministic, leakage-checked 80-sample public
smoke split from the verified retained landmark inventory without rerunning MediaPipe.
A static React/Vite app now exposes the public routes without a backend and runs its
verified MediaPipe and ONNX paths in workers. Its live route provides consent-first
camera controls and on-device event results; model bundles can be cached and rolled
back, deterministic post-landmark replay is tested, and explicitly saved feedback
stays local until a separate export. The exact PopSign candidate remains an
unpublished development checkpoint, not a configured public release. A minimal local
MLflow ledger can now log, query, and byte-verify one
portable baseline result without a server or custom dashboard. The first frozen
signer-disjoint benchmark now compares majority, seeded stratified-random, and
train-only multinomial-logistic references from one checked-in configuration. The
selected logistic reference reached 0.725 test macro-F1 and 0.733 balanced accuracy
on the deliberately small 15-clip final test; the full
[reference report](docs/reports/popsign-reference-baselines-v1.md) publishes the
per-class failures, CPU latency, exact identities, and limits rather than treating
this smoke result as a product claim. A bounded
[Keras/ONNX compatibility run](docs/reports/popsign-legacy-gru-compatibility-v1.md)
now proves that the recovered two-layer GRU can train on the current 64-by-134 feature
contract and preserve all 15 validation outputs through fixed-shape ONNX Runtime CPU
inference; its perfect tiny-set validation score is explicitly not a model-quality
claim. A subsequent fixed
[GRU/TCN feasibility run](docs/reports/popsign-sequence-baselines-v1.md) now confirms
that one small forward GRU and one similarly sized causal residual TCN can train,
checkpoint, reload, and run on the same sequence contract under one shared protocol.
All four bounded checkpoints reloaded successfully; test features stayed sealed, and
the 15-clip validation observations do not select a model-quality winner.
The completed signer-grouped ablation supports carrying the hand-local TCN design
forward, without treating its development metrics as results for a particular
checkpoint. A separate six-class checkpoint now has bounded constructed calibration
and continuous-replay scorer evidence. Its
[dataset card](docs/cards/popsign-five-isolated-smoke-v1.md),
[model card](docs/cards/popsign-tcn-portable-export-candidate-v1.md), and
[nomination report](docs/reports/popsign-tcn-portable-export-nomination-v1.json)
freeze the exact evidence and nominate it for portable export only. No champion or
natural-use, test, release, or production claim is made.
The public DVC graph remains fixture-only by design. The remaining release limits
are the candidate's unresolved redistribution status, sealed locked test, missing
natural continuous-use evidence, and not-yet-deployed public site.

## Data and privacy

Raw participant video and contributed feedback are not committed to Git. Names,
contacts, signatures, and the identity-to-pseudonym mapping belong only in a
separate encrypted identity vault; research systems use random pseudonymous IDs.
Approved data belongs behind a controlled DVC remote whose pointers and lock history
live in a separate access-controlled metadata repository. The public repository
contains only explicitly synthetic DVC metadata. Public fixtures must be synthetic,
explicitly consented, or separately licensed. Licensed human media stays ignored
and content-addressed locally unless a reviewed redistribution decision explicitly
permits a particular artifact. Dataset-license authorization never substitutes for
SignLab participant consent.

## License

Code is licensed under the MIT License. Dataset and model artifacts may have
separate terms documented in their data and model cards.
