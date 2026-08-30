# Constructed transition calibration check

This is a development-only mechanics check, not a model-quality or deployment claim.

- Evidence: `constructed_transition_calibration_conformance`
- Test partition: `sealed_not_loaded`
- Validation rows: 18 (15 source + 3 constructed)
- Temperature: 0.050
- Policy: `selected`; threshold: 0%
- NLL before/after: 0.173056 / 0.000076
- Brier before/after: 0.084185 / 0.000000
- Verified ledger run: `5848bc5ae6ae40cf9bca46552774bac5`

## Limitations

- The 18 validation rows contain only 15 unique source clips.
- The three other examples are deterministic transition fragments, not natural out-of-vocabulary signs.
- The temperature search selected its 0.050 lower boundary.
- The selected 0% threshold abstains on none of these 18 rows; it is not a real-use operating point.
- Validation fit the temperature and selected the threshold, so all metrics are development diagnostics.
- No session, capture-condition, continuous-signing, false-activations-per-hour, event-recall, promotion, or deployment claim is supported.
- Test features were never requested.
