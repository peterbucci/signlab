# Contributing

## Workflow

1. Start from a GitHub issue with explicit acceptance criteria.
2. Keep raw participant data and credentials outside Git.
3. Use a focused feature branch and reference the issue in the pull request.
4. Add tests for contracts, split invariants, preprocessing, or runtime parity as applicable.
5. Update model/data documentation whenever a result or intended use changes.

## Definition of done

A story is complete when its acceptance criteria pass, evidence is attached, relevant
documentation is current, and the change can be reproduced from versioned inputs.

Experimental results must include the dataset version, split version, resolved
configuration, seed, Git commit, environment, and exact evaluation protocol.
