# Developer foundation

## Ownership and boundaries

| Path | Owner | Responsibility |
| --- | --- | --- |
| `src/signlab/` | Python application | Importable domain, application, and adapter code; no private data |
| `tests/` | Verification | Fast unit/integration tests and tiny public fixtures |
| `docs/` | Evidence and decisions | Architecture, protocols, cards, audits, and tutorials |
| `configs/` | Portable inputs | Reviewed versioned configuration; never resolved local state |
| `examples/` | Learning surface | Small demonstrations using public or synthetic inputs |
| `scripts/` | Repository automation | Thin maintenance, verification, and release adapters |
| `.github/` | Delivery controls | Review ownership, dependency updates, and CI policy |
| `data/`, `artifacts/`, `models/`, `runs/` | External/private storage | Never committed; later referenced through approved DVC metadata |

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
environment, and the cross-platform lock. Change dependencies through `uv add` or
`uv remove`, then review and commit `pyproject.toml` and `uv.lock` together. Never
hand-edit `uv.lock` or maintain a parallel requirements file.

## Canonical workflow

```shell
uv python install
uv sync --locked --all-groups
uv run pre-commit install
uv run signlab --help
```

Before a pull request:

```shell
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_repository_hygiene.py
uv build
uv run python scripts/verify_distribution.py dist
```

CI repeats the locked install and checks on clean Linux and Windows runners. The
Linux job performs the full quality suite; Windows repeats the tests, package build,
clean-wheel install, CLI smoke test, and checkout-cleanliness proof. A separate job
scans complete pull-request history for secrets.

## Generated and private state

The Git ignore rules cover local environments, caches, coverage, builds, DVC cache,
tracking databases, datasets, videos, features, checkpoints, and model bundles. The
repository guard independently rejects ignored high-risk path families, artifact
extensions, files larger than 1 MiB, CRLF text, high-confidence secrets, and absolute
machine paths. This defense remains active even if an ignore rule is accidentally
bypassed.
