# SignLab data-governance policy

Policy version `1.0.0` is the machine-enforced baseline for participant data. The
packaged JSON policy fixes the reviewed policy values. Generated JSON Schemas are
the structural interchange contract plus selected portable invariants; the strict
Pydantic/application validators remain authoritative for hashes, graph closure,
time ordering, consent lifecycle, and other cross-record semantic rules. Both layers
must pass. This policy is deliberately stricter than the minimum behavior of many
storage systems.

## Purpose and limits

Participant data may be used only for the bounded SignLab claim and only within the
scope recorded on both the consent receipt and each recording grant. A later
receipt may narrow or replace permission but cannot retroactively expand an earlier
recording's scope.

The project does not claim universal legal compliance. Required jurisdictional and
institutional review remains a collection gate.

## Identifiers and filenames

- The identity vault generates a random 128-bit signer ID and retains the only
  mapping to identity.
- Research stores receive pseudonymous consent, signer, recording, and asset IDs.
- A recording filename contains its recording ID only—never a signer ID, name,
  contact, date of birth, initials, handle, or free-text participant value.
- Names, contact details, signatures, filled forms, and identity mappings never
  enter Git, DVC, MLflow, manifests, logs, datasets, experiments, models, or reports.
- Hashes and pseudonyms are linkable data and are protected accordingly.

## Roles and access

| Role | Identity vault | Restricted research | Release candidates | Public |
| --- | --- | --- | --- | --- |
| Data steward | Required duties only | Approved governance operations | Review | Read |
| Approved researcher | No | Least privilege | Propose | Read |
| Release reviewer | No | Provenance/scope evidence only | Approve or reject | Read |
| Public user | No | No | No | Read |

Access must be authenticated, least-privilege, reviewed, and removed when no longer
needed. Service providers require an approved safeguard and deletion arrangement.
Secrets and identity-vault references never appear in repository configuration.

## Scope enforcement

Every recording snapshots the receipt's scope and its canonical digest. Training or
internal evaluation requires feature-derivation permission. Public model weights
require training permission. Raw-media distribution requires raw capture and its
specific distribution choice. Feature redistribution requires feature derivation.
Only aggregate evaluation results may be public. No permission is inferred from a
missing field, prior participation, or a broader later receipt.

Every receipt and recording grant also binds an opaque `purpose_id` and `study_id`.
A use under a different purpose ID is denied. A later study with the same approved
purpose ID is denied unless both receipt and recording snapshots explicitly allow
same-purpose future research. The event log binds the complete receipt digest, and
positive authorization requires an authenticated verifier for the complete
receipt, recording grant, and event-log tuple. `completeness_attested`, an integrity
hash, or an attestation hash is not by itself proof that evidence came from an
authoritative store.

## Retention and backup

- Raw and derived participant data: at most 730 days from capture, or earlier when
  purpose completes, consent expires, or withdrawal takes effect.
- Verified withdrawal freezes new use immediately.
- Impact inventory target: five business days.
- Primary deletion/invalidation/rebuild target: 30 calendar days.
- Backup purge: next tested rotation, no later than 30 additional calendar days.
- Signed consent evidence follows the separately approved institutional or legal
  rule in the identity vault; no universal term is hard-coded here.
- A restore remains isolated until withdrawal tombstones are reapplied and all
  restored assets are reconciled against current consent state.

Backups are part of the lineage inventory. A local delete does not prove cloud,
snapshot, removable-media, or provider sanitization. Each store needs a tested,
documented destruction or cryptographic-erasure procedure appropriate to its media.

## Withdrawal and deletion

The project retains a minimal pseudonymous tombstone to prevent accidental
re-import. Raw recordings, features, caches, and ordinary working copies are
deleted; backup copies are purged; datasets and splits are invalidated and rebuilt;
runs are invalidated and rerun; models are retired and rebuilt; reports and demos
are retracted or republished. Shared descendants are affected even when they also
contain other participants.

The committed planner is read-only. It proves deterministic closure over a supplied,
validated inventory; it does not prove that the inventory covers an unregistered
external store and never claims external deletion occurred. Execution requires
future adapters, an authoritative store inventory, and store-specific attestations.
Planning remains resumable when a root or descendant is already pending or
invalidated: traversal continues through it so later active descendants cannot be
missed. Because version 1 does not model per-action completion attestations, a retry
emits the complete safe-to-repeat action set for every affected asset rather than
guessing which external side effects finished.

## Incident and review

Suspected disclosure freezes affected processing, preserves non-sensitive audit
evidence, invokes the approved incident process, and obtains required institutional
or legal advice. Policy, access, service providers, restores, and deletion tests are
reviewed when systems or project context change and at the approved periodic
interval.
