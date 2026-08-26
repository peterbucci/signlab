# SignLab participant consent form

> DRAFT TEMPLATE — NOT APPROVED FOR REAL COLLECTION
>
> Every `{{PLACEHOLDER}}` must be resolved and the collection-readiness record must
> be approved before a camera is turned on for a participant.

Form version: 1.0.0

## Key information

You are being invited to contribute video to a research prototype.
SignLab recognizes isolated performances of five predefined hand gestures in
continuous webcam video. It is not a sign-language translator or a complete
accessibility system.

Participation is voluntary. You may say no or stop at any time without penalty.
The main risks are being recognizable in video, accidental disclosure, possible
reidentification from derived hand/body measurements, and discomfort or
embarrassment about a recording or model result. There is no guaranteed direct
benefit to you.

Before deciding, you may ask questions and take time to consider. You will receive
a copy of the completed form.

Study contact: `{{STUDY_CONTACT}}`

Questions about your rights or withdrawal: `{{WITHDRAWAL_CONTACT}}`

Project affiliation and review determination: `{{PROJECT_AFFILIATION_AND_REVIEW}}`

## What participation involves

If you agree, a researcher will ask you to perform predefined hand-gesture prompts
and non-target hand activity in front of a camera. A session is expected to take
`{{SESSION_DURATION}}`. The camera may capture your hands, face, body, clothing,
voice-free background activity, and surroundings. Audio is disabled and must not
be recorded.

Software may derive hand landmarks, body anchors, motion features, quality signals,
embeddings, predictions, and evaluation records. These derived values are
pseudonymous but are not guaranteed anonymous; geometry or learned representations
may remain identifying or biometric in some contexts.

The project will not use the recordings for face recognition, identity inference,
surveillance, sale, advertising, or participant-level public ranking.

## Separate choices

Initial each choice. Saying no to one optional use does not change your other
choices. The research record stores these choices per recording.

### Internal research

- [ ] I allow raw video capture for the described SignLab sessions.
- [ ] I allow approved raw video to be retained for the period stated below. If I
      decline, permitted features may be derived during capture and the raw video
      must then be deleted.
- [ ] I allow hand/body features to be derived from approved recordings.
- [ ] I allow approved derived features to be used for model training.
- [ ] I allow approved derived features to be used for internal evaluation.

### Public or third-party distribution

- [ ] I allow selected raw-video excerpts in a curated public demonstration.
- [ ] I allow raw recordings to be redistributed as a dataset.
- [ ] I allow derived features to be redistributed as a dataset.
- [ ] I allow only aggregate, non-participant-level evaluation results to be public.
- [ ] I allow trained model weights to be public. Models may retain information
      learned from the approved training data even when they do not contain video.

### Future use

- [ ] I allow the approved data to be reused only for future research with the same
      bounded five-gesture purpose.

No box may be preselected. A denied or blank choice means that use is not allowed.
Raw-media and derived-feature redistribution are off by default.

## Storage, access, retention, and backups

Your name, contact information, signature, completed form, and the mapping to your
pseudonymous signer ID remain in a separate encrypted identity vault operated by
`{{IDENTITY_VAULT_OPERATOR}}`. They never appear in filenames, datasets, manifests,
experiment logs, model bundles, Git, DVC, or MLflow. Research files use random
pseudonymous IDs; recording filenames contain only recording IDs.

Restricted research data is stored at `{{RESEARCH_STORAGE_DESCRIPTION}}` and may be
accessed only by approved roles listed in the data-governance policy. Backups are
stored and rotated as described at `{{BACKUP_DESCRIPTION}}`. Integrity hashes do not
anonymize or encrypt data.

Unless an earlier purpose-completion, expiration, or withdrawal date applies, raw
video and derived participant data will be retained for no more than 24 months from
capture. The approved rule for this collection is
`{{RETENTION_END_DATE_OR_RULE}}`. Signed consent evidence follows the separately
approved institutional or legal retention rule in the identity vault.

## Withdrawal

You may request complete withdrawal through `{{WITHDRAWAL_CONTACT}}`. After the
request is verified, SignLab will freeze new use, trace your recordings through
derived features, datasets, splits, runs, models, reports, demonstrations, caches,
and backups, then delete, invalidate, retire, retract, or rebuild affected assets.
The target is an impact report within five business days, primary action within 30
calendar days, and backup purge by the next tested rotation, no later than another
30 days.

A minimal pseudonymous tombstone may be retained to prevent accidental re-import;
it contains IDs, hashes, and deletion attestations, not your name or contact details.
Material already redistributed publicly may not be fully recoverable from people
who downloaded it. Any such limit must match the choices above and will be explained
when the request is processed.

## Confidentiality and limits

Reasonable administrative, technical, and physical safeguards will be used, but no
storage or transfer is risk-free. Required disclosure under applicable law or a
valid legal process may be outside the project’s control. This form does not ask you
to waive legal rights or release anyone from responsibility for negligence.

Jurisdictions and any required institutional, privacy, ethics, or legal review:
`{{JURISDICTIONS_AND_REQUIRED_REVIEW}}`.

## Agreement and copy

By signing, you confirm that you are at least 18 years old, had an opportunity to
ask questions, received understandable answers, made each choice freely, and will
receive a copy of this form.

Participant name: ______________________________________________

Participant signature: _________________________________________

Date and time: _________________________________________________

Collector role (no name in research records): ___________________

Collector signature: ___________________________________________

Copy provided: [ ] Yes

The completed form remains only in the identity vault. The research system records
its SHA-256, version, consent scope, pseudonymous IDs, and attestations.
