# Developer foundation

## Ownership and boundaries

| Path | Owner | Responsibility |
| --- | --- | --- |
| `src/signlab/` | Python application | Importable domain, application, and adapter code; no private data |
| `apps/web/` | Public browser application | Static React/Vite shell and, after later review gates, on-device inference |
| `tests/` | Verification | Fast unit/integration tests and tiny public fixtures |
| `docs/` | Evidence and decisions | Architecture, protocols, cards, audits, and tutorials |
| `configs/` | Portable inputs | Reviewed versioned configuration; never resolved local state |
| `examples/` | Learning surface | Small demonstrations using public or synthetic inputs |
| `scripts/` | Repository automation | Thin maintenance, verification, and release adapters |
| `.github/` | Delivery controls | Review ownership, dependency updates, and CI policy |
| `data/`, `artifacts/`, `models/`, `runs/` | External/private storage | Never committed; referenced through approved DVC metadata outside public Git |
| `dvc.yaml`, `dvc.lock`, `.dvc/` | Public synthetic proof | Generated stage graph, fixture-only lock, and safe tracked defaults; never private pointers or remotes |

The project owner is the default reviewer through `.github/CODEOWNERS`. A later
story may split ownership when the Python pipeline and browser application have
independent maintainers.

## Runtime decision

SignLab uses standard-GIL CPython 3.12.14 and accepts only the 3.12 minor line.
Python 3.12 is the conservative compatibility intersection for the planned MediaPipe
Tasks, PyTorch, ONNX Runtime, OpenCV, MLflow, and DVC stack on Windows and Linux.
Python 3.13 becomes supported only after the complete native smoke matrix passes;
installability alone is not an upgrade criterion.

`uv` 0.12.6 owns interpreter installation, dependency resolution, the virtual
environment, and the cross-platform lock. Hatchling is pinned as both the build
backend and a locked development dependency; builds disable isolation so they use
that reviewed version. Change dependencies through `uv add` or `uv remove`, then
review and commit `pyproject.toml` and `uv.lock` together. Never hand-edit `uv.lock`
or maintain a parallel requirements file.

The native landmark boundary is an optional `extraction` extra containing exact
`mediapipe==1.0.1` and `av==18.1.0` pins. Core contract, governance, dataset, and CLI
imports must remain usable without loading either native package. The developer
workflow installs all extras so Linux and Windows exercise that boundary; consumers
that do not extract video can install the base wheel. The future browser runtime is
separately pinned to `@mediapipe/tasks-vision@1.0.1` by the extraction contract.

The browser application is an independent npm project under `apps/web/`. Node.js
24.20.0 LTS and npm 11.19.0 are pinned through `.node-version`, `package.json`, and the
committed npm lockfile. It does not share dependencies or runtime state with the
Python research pipeline.

## Canonical workflow

```shell
uv python install
uv sync --locked --all-groups --all-extras
uv run pre-commit install
uv run signlab --help
uv run python scripts/generate_dvc_pipeline.py --check
uv run python scripts/generate_feature_resources.py --check
uv run python scripts/generate_feature_goldens.py --check
uv run dvc repro --force --no-run-cache
```

## Static browser application

Install and run the browser shell without starting Python or an application server:

```shell
cd apps/web
npm ci --no-audit --no-fund
npm run dev
```

Before a pull request, run the web checks from that same directory:

```shell
npm run format:check
npm run lint
npm run typecheck
npm test
npm run build
npm run build:subpath
```

The two builds prove the static files can be published at a root domain or a
configured `/signlab/` subpath. Hash-based routes keep every page usable on a plain
static host without rewrite rules. Story #38 provides the responsive, honest shell
only: it does not request camera access, load a model, process replay inputs, save
feedback, or call a backend.

Before a pull request:

```shell
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_repository_hygiene.py
uv run python scripts/generate_dvc_pipeline.py --check
uv run python scripts/generate_feature_resources.py --check
uv run python scripts/generate_feature_goldens.py --check
uv build --no-build-isolation
uv run python scripts/verify_distribution.py dist
```

CI repeats the locked install and checks on clean Linux and Windows runners. Both
platform jobs run the fixture-only DVC clean-room proof and retain its phase report;
the remaining quality, test, package, CLI, and checkout-cleanliness checks follow the
workflow definition. A separate job scans complete pull-request history for secrets.

DVC and its S3 transport are exact locked dependencies in the `reproducibility`
group, not dependencies of the published wheel. PyYAML is a runtime dependency only
because the installed SignLab snapshot reader extracts registered stage entries from
`dvc.lock`. After changing the typed stage registry, regenerate `dvc.yaml`, run
`dvc repro --force --no-run-cache`, and review the resulting lock change. The fixture
scope and separate Story #19 authorized private-data gate are in
[data-versioning.md](data-versioning.md).

## Generated and private state

Pydantic models in `src/signlab/contracts/` are authoritative. Reviewable taxonomy
JSON Schemas under `src/signlab/resources/schemas/` and governance resources under
`src/signlab/resources/governance/` are generated package data. After a taxonomy-
model change, run:

```shell
uv run python scripts/generate_taxonomy_schemas.py
```

After a governance-contract, policy, or synthetic lineage-scenario change, run:

```shell
uv run python scripts/generate_governance_resources.py
```

After a pipeline-contract or coherent synthetic contract-chain change, run:

```shell
uv run python scripts/generate_contract_resources.py
uv run signlab contracts validate-resources
```

After changing a licensed external-dataset contract, source registry, or reviewed
label selection, run:

```shell
uv run python scripts/generate_external_dataset_resources.py
uv run signlab data validate-resources
```

That generator owns only reviewable JSON metadata and schemas. It never downloads
or packages public-dataset video or tar archives.

After changing an extraction contract, default config, model lock, or landmark
Arrow schema, run:

```shell
uv run python scripts/generate_extraction_resources.py
```

That generator owns the three extraction JSON Schemas, the default CPU/VIDEO
configuration, the MediaPipe task-model provenance lock, and the Arrow schema
snapshot. The lock records exact external model sizes and SHA-256 values; the
multi-megabyte `.task` files are never generated, downloaded, or committed. Test
fixtures may script detector results but must not substitute fake task bytes for the
registered production hashes.

After changing a landmark-quality contract or the packaged pilot policy, run:

```shell
uv run python scripts/generate_quality_resources.py
```

That generator owns the quality policy and report JSON Schemas plus the packaged
default policy. Quality outputs are report manifests, not public fixtures or feature
artifacts; real sequence reports remain consent-bound private research metadata.

After changing a portable-feature contract or a default representation plan, run:

```shell
uv run python scripts/generate_feature_resources.py
uv run python scripts/generate_feature_goldens.py
```

The resource generator owns the four feature JSON Schemas and the hand-local,
body-relative, and combined default plans. The golden generator owns the
synthetic cross-runtime expected value arrays and hashes. Neither processes public
or participant media, fits statistics, populates the feature cache, or modifies DVC
state.

All generators are deterministic. The test suite, `signlab contracts
validate-resources`, `signlab data validate-resources`, and `signlab governance
evidence-check` fail when generated resources drift from the Pydantic source of
truth, the frozen compatibility corpus, or the committed synthetic dry-run evidence.

Generated governance JSON Schemas enforce structure and the semantic constraints
that JSON Schema can express. They do not replace application validation for
canonical hashes, consent lifecycle, timestamp ordering, lineage ancestry, or
cross-document compatibility; producers and consumers must run both layers.

Tests compare every committed schema byte-for-byte at the JSON-document level and
verify that the wheel contains both schemas and immutable taxonomy instances. A
semantic change to a published taxonomy requires a new versioned artifact and a new
golden digest, not an in-place edit.

The six pipeline-contract examples form a coherent synthetic dataset-to-model
chain. Their RFC 8785 hashes are published compatibility goldens. An incompatible
change requires a new instance schema version, schema `$id`, example, and explicit
migration; never rewrite a retained v1 resource or silently change its digest.

Never place completed consent forms, identity-vault exports, raw recordings, or
participant-derived fixtures in this repository. Governance examples and evidence
must use the reserved synthetic identifiers generated by the governance resource
script. Extraction fixtures must also be explicitly synthetic. Real landmark
Parquet is participant-derived data covered by the consent scope's
`derived_features` permission and belongs only in approved private storage.

The Git ignore rules cover local environments, caches, coverage, builds, DVC cache,
tracking databases, datasets, videos, features, checkpoints, and model bundles. The
repository guard independently rejects ignored high-risk path families, artifact
extensions, files larger than 1 MiB, CRLF text, high-confidence secrets, and absolute
machine paths. This defense remains active even if an ignore rule is accidentally
bypassed.
