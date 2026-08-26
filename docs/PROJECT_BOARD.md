# Project board

The GitHub Project is the delivery source of truth. Work is organized by phase,
priority, status, and effort; repository labels identify technical areas.

## Workflow

- **Backlog:** accepted work that is not yet ready to start.
- **Ready:** scoped, dependency-ready, and acceptance criteria are testable.
- **In progress:** actively owned work; keep work-in-progress deliberately small.
- **In review:** implementation and evidence are ready for review.
- **Done:** acceptance criteria and the repository definition of done are satisfied.

## Priority

- **P0:** required to protect validity, privacy, or the critical path.
- **P1:** required for the first credible public release.
- **P2:** valuable follow-up after the first release.
- **P3:** explicitly deferred or optional platform work.

## Effort

- **S:** usually a focused change with limited integration risk.
- **M:** several coordinated changes or a bounded experiment.
- **L:** cross-cutting work that should be decomposed before implementation.

## Phase gates

The board intentionally prevents platform infrastructure from competing with the
scientific and browser-release critical path. Optional-platform stories remain P3
until the static public release is complete and a concrete workflow justifies them.
