# Legacy audit

## Snapshot

- Git commit/tag:
- Dataset inventory and checksum:
- Number of signers and sessions:
- Run inventory:
- Promoted models:
- Live-feedback attempts:

## Reported results

Record the original split, sample counts, exact preprocessing, model configuration,
seed, checkpoint-selection rule, and integer confusion matrix—not only percentages.

## Known limitations

- Random clip-level rather than signer/session-grouped evaluation.
- Small test set where a single prediction materially changes rankings.
- Legacy `nothing`/label-map mismatch.
- Potential training/live preprocessing drift.
- First-detected-hand extraction and repeated last-valid frames.
- Historically mislabeled `legacy_dense_temporal` model.

## Preservation outputs

- Run index and summary table.
- Promoted model bundles and hashes.
- Exported feedback/replay corpus.
- Preprocessing plans and label maps.
- Reproduction notes and environment record.
