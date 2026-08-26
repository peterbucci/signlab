# Security policy

## Reporting a vulnerability

Use the repository's **Security** tab to submit a private vulnerability report. Do
not post credentials, participant data, exploit details, or private file locations in
a public issue. If private reporting is unavailable, open a minimal issue requesting
a private contact channel without including sensitive details.

Reports should identify the affected version, expected and observed behavior, and a
minimal reproduction that contains no real participant data or active credential.

DVC remote URLs, `.dvc/config.local`, participant-data pointers, production lock
entries, remote listings, and private content hashes must also stay out of public
issues and pull requests. Use environment credential chains and the local-only setup
described in [docs/data-versioning.md](docs/data-versioning.md); never paste cloud
credentials into DVC commands or configuration.

## Supported versions

SignLab is pre-release software. Security fixes are applied to the current `main`
branch until the first tagged release establishes a longer support policy.
