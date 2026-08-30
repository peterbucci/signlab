# PopSign GRU/TCN feasibility v1

> Engineering feasibility only. Fifteen validation clips selected the best epochs and supplied these descriptive results; neither model is a winner.

## Reproducibility identity

- Source commit: `e8efd2fc881e766e12ccc8ff44220e8fc59bc72a` (dirty: `false`)
- Configuration: `sha256:49818b3d5eb4bce0bb449c412c2dca5973fc5215c431d45514c7ff5b3c7725f3`
- Frozen split: `sha256:673c777c67e47715127b75f2bf18aaa794a15e1604458d30da164ec7f90ead20`
- Feature plan: `sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6`
- Seed: `20260828`
- Opened features: 50 train and 15 validation; test features were never requested.

## Side-by-side observations

| Model | Params | Epochs / best | Validation loss | Accuracy | Macro-F1 | CPU p50 / p95 ms | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 26,741 | 30 / 30 | 0.632 | 0.800 | 0.781 | 391.006 / 435.683 | 3 |
| TCN | 29,317 | 30 / 22 | 0.232 | 0.933 | 0.931 | 39.036 / 52.960 | 1 |

## Training and checkpoint evidence

| Model | Fit seconds | Checkpoint | Bytes | SHA-256 | Reloaded |
| --- | ---: | --- | ---: | --- | --- |
| GRU | 10.429 | best | 349,819 | `sha256:453e70eb11f82cda8192519e1bf3a40ddf14ee1659be009abfa1e84cae8e9088` | yes |
| GRU | 10.429 | last | 349,819 | `sha256:453e70eb11f82cda8192519e1bf3a40ddf14ee1659be009abfa1e84cae8e9088` | yes |
| TCN | 13.115 | best | 440,824 | `sha256:f4eeaf677be99f34d58f1d7bac30d46cdf7af03647f622c89db7086d661284a6` | yes |
| TCN | 13.115 | last | 440,824 | `sha256:0cb51acce8cf32c23dbc31ceb1adc1be97b2fa9ac724ebab8bc1941cdd0d6fbc` | yes |

## Per-class validation counts

| Model | Class | Support | Predicted | Correct | Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| GRU | hello | 3 | 3 | 3 | 0 |
| GRU | no | 3 | 5 | 3 | 0 |
| GRU | please | 3 | 2 | 2 | 1 |
| GRU | thank_you | 3 | 4 | 3 | 0 |
| GRU | yes | 3 | 1 | 1 | 2 |
| TCN | hello | 3 | 3 | 3 | 0 |
| TCN | no | 3 | 4 | 3 | 0 |
| TCN | please | 3 | 3 | 3 | 0 |
| TCN | thank_you | 3 | 3 | 3 | 0 |
| TCN | yes | 3 | 2 | 2 | 1 |

## Concrete validation failures

### GRU

| Sample | Actual | Predicted | Quality | Uncalibrated max score |
| --- | --- | --- | --- | ---: |
| sample_009 | please | thank_you | pass | 0.393 |
| sample_014 | yes | no | pass | 0.554 |
| sample_015 | yes | no | pass | 0.457 |

### TCN

| Sample | Actual | Predicted | Quality | Uncalibrated max score |
| --- | --- | --- | --- | ---: |
| sample_014 | yes | no | pass | 0.840 |

Each model produced one best and one actual-last `.keras` checkpoint; all four were reloaded and exercised.
Checkpoint byte sizes include training state and are not deployment bundle sizes.

## What this does and does not show

- Both fixed graphs can train, checkpoint, reload, and run on the current 64 x 134 feature contract.
- Validation was used twice--early-stopping selection and reporting--so the numbers are optimistic and descriptive.
- Both models average all 64 steps, including neutral zero padding; this experiment does not test mask-aware inputs.
- This does not establish generalization, production or real-time performance, calibration, robustness, fairness, continuous signing, or broader sign-language recognition.
- Architecture evidence across seeds or folds belongs to the next evaluation story, not this run.

## Reproduce

```shell
uv run --locked --extra experiments signlab train sequence-baselines configs/experiments/popsign-sequence-baselines-v1.json --corpus-root <frozen-split-root> --external-manifest <external-dataset-manifest.json> --output-root runs/popsign-sequence-baselines-v1
```
