# Contributing

## Workflow

1. Start from a GitHub issue with explicit acceptance criteria.
2. Keep raw participant data and credentials outside Git.
3. Use a focused feature branch and reference the issue in the pull request.
4. Add tests for contracts, split invariants, preprocessing, or runtime parity as applicable.
5. Update model/data documentation whenever a result or intended use changes.

Start from the locked environment and run the canonical commands in
[docs/development.md](docs/development.md). Add dependencies with `uv add` or
`uv add --dev`; do not edit `uv.lock` or create a parallel requirements file.

CLI modules parse inputs and delegate to importable services. They must not contain
training, extraction, evaluation, or export implementations. Examples and tests may
use only synthetic, explicitly consented, or separately licensed data.

## Definition of done

A story is complete when its acceptance criteria pass, evidence is attached, relevant
documentation is current, and the change can be reproduced from versioned inputs.

Experimental results must include the dataset version, split version, resolved
configuration, seed, Git commit, environment, and exact evaluation protocol.

CI is the source of truth. A change is not complete when it relies on ignored local
state, rewrites the lockfile during setup, leaves the checkout dirty, or bypasses the
secret, artifact-size, package-content, or machine-path guards.
