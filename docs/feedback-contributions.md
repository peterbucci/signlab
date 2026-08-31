# Feedback contribution boundary

SignLab keeps event corrections in the browser unless the user deliberately downloads
a contribution package. Saving locally and exporting are separate consent decisions.
The export action creates a file on the user's device; it does not upload or transmit
anything.

## Browser workflow

1. Open **Feedback** and inspect the valid record count, model-bundle versions, fields,
   local consent scope, and landmark count.
2. Correct or delete records before exporting. Damaged records are never included.
3. Leave landmark export off unless derived coordinates are intentionally required.
4. Grant the separate export consent for manual research review, then download the
   `.signlab-feedback.json` file.

The package contains strict `signlab-feedback-record/1` copies, a readable manifest,
the exact serialized record payload, and its SHA-256 digest. Excluding landmarks
creates privacy-filtered copies and does not change the records stored in IndexedDB.
Raw video, audio, free text, device details, and browser fingerprints are never part
of the package.

## Quarantine import

Treat downloaded packages as untrusted private data. Do not commit them to Git.
Validate and quarantine one package with:

```powershell
uv run signlab data import-feedback-package <downloaded-package.signlab-feedback.json>
```

The command verifies versions, consent, exact field allowlists, summaries, duplicate
record IDs, landmark declarations, and the payload digest before writing anything. It
then stores the exact package bytes and a sanitized receipt under the ignored,
content-addressed `data/private/feedback-quarantine` directory. Re-importing identical
package bytes is rejected.

Every receipt says `trainable: false`. Human review, cross-package deduplication, a
split decision, and a new DVC dataset version are still required. Story #120 owns that
later decision; this workflow has no promotion or training path.
