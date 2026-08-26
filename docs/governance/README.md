# Participant-data governance

This directory defines SignLab's engineering privacy baseline. It is not legal
advice and does not claim that the Common Rule, a biometric-privacy statute, or any
other specific law applies. Project affiliation, funding, publication plans,
participant location, operator location, and institutional rules determine the
review that is required.

## Collection status

**Real participant collection is blocked.** The version-1 readiness contract cannot
authorize collection; it records only a fail-closed checklist while no authenticated
approval verifier exists. The repository contains a blank form, machine-readable
policy, schemas, and synthetic tests only. Before real capture, a later reviewed
contract and verifier must bind the approvals below to the exact policy and storage
configuration. The project owner must resolve and obtain any required approval for:

1. Adults-only participation, or a separately designed guardian/minor protocol.
2. Every participant state and country and the operator's jurisdiction.
3. Whether the work is independent or connected to a school, employer, sponsor,
   research program, funding source, or intended publication.
4. The written IRB, ethics, institutional privacy, or legal determination that is
   required, if any. The project owner must not self-declare an institutional
   exemption.
5. The identity-vault operator; restricted-data and backup locations; encryption
   and key owner; access roles; and a tested deletion path for each store.
6. The real study and withdrawal contacts printed on participant copies.
7. The approved retention period and each permitted publication scope.

If Illinois participants or another biometric-law jurisdiction may be involved,
obtain qualified legal review before capture. Unknown answers keep collection
blocked; they are never converted into permissive defaults.

## Data boundary

| Class | Examples | Location and access |
| --- | --- | --- |
| Public | Blank form, policy, schemas, synthetic fixtures, aggregate evidence | Git and the public package |
| Identity vault | Names, contacts, signatures, filled forms, ID mapping | Separate encrypted store; data steward only |
| Restricted research | Pseudonymous receipts, video, landmarks, manifests, lineage | Approved private storage; approved research roles |
| Release candidate | Aggregate reports, models, selected media or features | Private until automated scope and provenance review passes |

Pseudonymous does not mean anonymous. Landmarks and embeddings remain restricted.
Hashes provide integrity, not anonymization or encryption. Audio, minors, identity
inference, sale, and participant-level public results are prohibited in version 1.

The form choices map to required machine fields as follows:

| Participant choice | Consent-scope field |
| --- | --- |
| Raw video capture | `raw_media_capture` |
| Raw video retention | `raw_media_retention` |
| Derived hand/body features | `derived_features` |
| Model training | `model_training` |
| Internal model evaluation | `model_evaluation` |
| Curated public demonstration | `public_demonstration` |
| Raw-recording redistribution | `raw_media_redistribution` |
| Derived-feature redistribution | `derived_features_redistribution` |
| Aggregate evaluation-result publication | `evaluation_results_redistribution` |
| Model-weight publication | `model_weights_redistribution` |
| Same-purpose future research | `same_purpose_future_research` |

`research_use` is the required purpose-level ceiling; the individual choices above
cannot broaden it. `withdrawal_supported` is always true. Prohibited-use fields are
always false, and a missing or false permission never authorizes the use.

Receipts and recording grants also bind opaque `purpose_id` and `study_id` values.
The purpose ID must match every later use. A different study ID additionally
requires the same-purpose-future-research choice; it never converts a different
purpose into an allowed one.

The historical Story #8 quarantine has `consent_status: unknown`. It cannot enter
training, evaluation, a demonstration, or a public release through absence of an
objection. It requires valid new authorization for the historical material or must
be deleted and replaced with newly consented collection.

## Executable evidence

The `signlab governance` commands validate the policy, consent receipts,
recording-level grants, lineage inventories, and deterministic withdrawal dry runs.
The committed synthetic scenario contains two signers whose recordings flow into a
shared dataset, run, model, report, and demo; withdrawing either signer therefore
invalidates and rebuilds the shared descendants.

The public validation commands prove internal structure, document registration,
hash integrity, scope compatibility, and lifecycle consistency. They do not
authenticate the identity-vault attestations. The event log binds the complete
receipt digest, and code may return a positive consent authorization only when an
external authenticated verifier accepts the complete receipt, recording grant, and
event-log tuple. Version 1 provides no such production verifier and therefore
cannot authorize real collection or use on its own.

From a locked checkout, reproduce the public evidence with:

```shell
uv run signlab governance evidence-check
uv run signlab governance validate-consent src/signlab/resources/governance/examples/consent-receipt.example.json
uv run signlab governance validate-recording src/signlab/resources/governance/examples/consent-receipt.example.json src/signlab/resources/governance/examples/recording-consent-grant.example.json src/signlab/resources/governance/examples/consent-event-log.example.json
uv run signlab governance validate-withdrawal src/signlab/resources/governance/examples/withdrawal-request.example.json src/signlab/resources/governance/examples/lineage-inventory.example.json src/signlab/resources/governance/evidence/withdrawal-dry-run-v1.json
```

`uv run signlab governance readiness-check` intentionally exits nonzero while real
collection remains blocked. `withdrawal-dry-run` requires an explicit `--as-of`
timestamp and a new `--output` path, refuses to overwrite evidence, and performs no
storage mutation. Participant and request IDs stay inside the JSON inputs rather
than entering shell history.

The synthetic proof traces one direct recording through 11 descendants. It includes
a shared dataset and split, an experiment, model, evaluation, public demonstration,
cache, enumerated backup copy, and retained anti-reimport tombstone. The second
participant's private branch remains untouched. This proves closure over the
committed synthetic inventory, not discovery of stores omitted from an inventory.

See [data-governance-policy.md](data-governance-policy.md) for storage and retention
rules and [withdrawal-runbook.md](withdrawal-runbook.md) for the exact response
procedure.

## Research basis

The design uses conservative elements from these authoritative sources without
asserting their applicability:

- [45 CFR 46.116](https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-A/part-46/subpart-A/section-46.116)
  describes understandable, voluntary consent; concise key information; purpose,
  procedures, risk, confidentiality, future use, contacts, and withdrawal.
- [HHS OHRP's consent-form checklist](https://www.hhs.gov/ohrp/consent-form-check-list.html)
  emphasizes everyday language, participant perspective, focused sections, and
  review by an unfamiliar reader.
- [HHS electronic-consent guidance](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/use-electronic-informed-consent-questions-and-answers/index.html)
  treats electronic consent as a process with questions, a participant copy, and
  investigator responsibility—not merely a checkbox.
- [Illinois 740 ILCS 14/15](https://my.ilga.gov/legislation/ilcs/fulltext?DocName=074000140K15)
  is a useful conservative reference for written notice/release, specific purpose
  and term, retention/destruction, disclosure restrictions, and reasonable care.
- [New York GBS § 899-bb](https://www.nysenate.gov/legislation/laws/GBS/899-BB)
  describes administrative, technical, physical, service-provider, testing, and
  disposal safeguards where applicable.
- [NIST Privacy Framework 1.0](https://www.nist.gov/document/nist-privacy-frameworkv10pdf)
  supports lifecycle governance, data processing transparency, roles, and
  privacy-risk management.
- [NIST SP 800-88 Rev. 2](https://csrc.nist.gov/pubs/sp/800/88/r2/final) is the
  current NIST media-sanitization reference; ordinary file deletion is not assumed
  to sanitize every storage medium or backup.
