# SignLab data-governance policy

Policy version 1.0.0 is the machine-enforced engineering baseline for participant
data. It is not legal advice or a claim of universal compliance. Real collection
remains blocked until the collection-readiness record documents all required
human, institutional, legal, contact, storage, and deletion decisions.

## Purpose and prohibited uses

Participant data may be used only for the bounded SignLab five-gesture research
claim and only within the scope recorded on both the consent receipt and each
recording grant. Missing permission fails closed. Audio collection, participation
by minors, identity inference, commercial sale, surveillance, and participant-level
public ranking are prohibited in version 1.

## Data zones and identifiers

- Public: blank forms, policy, schemas, synthetic fixtures, and aggregate evidence.
- Identity vault: names, contacts, signatures, completed forms, and the sole mapping
  from identity to a random 128-bit pseudonymous participant ID.
- Restricted research: pseudonymous receipts, recordings, derived features,
  manifests, lineage, experiments, and models.
- Release candidate: aggregate reports, models, and expressly permitted media or
  features; private until provenance and scope review passes.

Names, contacts, signatures, filled forms, and identity mappings never enter Git,
DVC, MLflow, filenames, manifests, logs, datasets, experiments, models, or reports.
Recording filenames contain only recording IDs. Pseudonyms, landmarks, embeddings,
and integrity hashes remain linkable restricted data; hashing is not anonymization
or encryption.

## Access, storage, and backups

The data steward alone administers the identity vault. Approved researchers receive
authenticated least-privilege research access. Model operators and release reviewers
receive only the minimum approved evidence. Access is logged, reviewed, and removed
when no longer needed. Service providers require approved safeguard and deletion
arrangements. Secrets and machine paths do not enter repository configuration.

Restricted storage and backups must be encrypted. Raw-media backup use, location,
rotation, restoration, and sanitization must be approved and tested before capture.
A restored copy stays isolated until current consent and withdrawal tombstones are
reapplied and every restored asset is reconciled.

## Scope and retention

Each recording snapshots the receipt's canonical scope and digest; a later receipt
cannot retroactively broaden it. Training and evaluation require derived-feature
permission. Participant-media demonstrations require raw-media retention and the
specific public-demonstration choice. Public model weights require training and the
model-weight redistribution choice. Raw media, derived features, and aggregate
evaluation results each require their specific redistribution choice.

Raw and derived participant data is retained for at most 730 days from capture, or
less when the approved purpose completes, consent expires, or withdrawal takes
effect. Verified withdrawal freezes new use immediately. The impact inventory target
is five business days; primary action is 30 calendar days; backup purge occurs by
the next tested rotation no later than 30 additional calendar days. Signed consent
evidence follows the separately approved identity-vault rule.

## Withdrawal, deletion, and incidents

The project retains a minimal pseudonymous tombstone to prevent re-import. Raw
recordings, features, caches, and working copies are deleted; backups are purged;
datasets and splits are invalidated and rebuilt; runs are invalidated and rerun;
models are retired and rebuilt; reports and demos are retracted or republished.
Shared descendants remain affected. The packaged planner is read-only and does not
claim external deletion occurred; authorized store adapters and attestations are
required for execution. Pending or invalidated nodes never stop descendant
traversal. Until authenticated per-action completion records exist, retries emit
the full safe-to-repeat action set for every affected asset.

Suspected disclosure freezes affected processing, preserves non-sensitive audit
evidence, and invokes the approved incident and escalation process. Policy, access,
providers, restores, retention, and deletion tests are reviewed whenever the system
or project context changes and at the approved periodic interval.
