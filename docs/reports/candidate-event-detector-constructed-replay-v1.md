# Candidate-event detector constructed replay v1

> Technical conformance evidence only; no natural-use performance claim.

- Evidence kind: `constructed_replay_conformance`
- Metric claim: `none`
- Test status: `sealed_not_loaded`
- Corpus status: `not_applicable_constructed_fixture`
- Split status: `not_applicable_constructed_fixture`
- Configuration: `sha256:0443badf68d34347a00096682cf049b6f49b5253c12e47bf61b068a597aa162d`
- Fixture: `sha256:ca32799c140c1b432b109d094376d0f1e3857814c3c92aab5fcea64ed31032be`

## Result

| Measure | Result |
| --- | ---: |
| Expected events | 2 |
| Emitted events | 2 |
| Matched events | 2 |
| Fixture recall | 1.000 |
| Missed events | 0 |
| Fragmented truth events | 0 |
| Duplicate proposals | 0 |
| Extra proposals | 0 |
| False candidates | 0 |
| Mean absolute onset error | 0.000 ms |
| Mean absolute offset error | 0.000 ms |

## Constructed scenario coverage

- `idle_no_hand`
- `static_hand`
- `quick_gesture`
- `long_hold`
- `motion_resume_during_finalizing`
- `back_to_back_after_cooldown`
- `two_frame_gap`
- `gap_exceeded`

## Limits

The fixture contains invented numeric observations, not video, landmarks, portable
features, public-corpus samples, or participant data. It proves deterministic state
transitions, boundary handling, gap tolerance, and duplicate suppression. It does not
tune thresholds or estimate natural-session recall, false activations per hour,
generalization, classification quality, or continuous-sign recognition.
