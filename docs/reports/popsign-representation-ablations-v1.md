# PopSign grouped development ablation v1

> Development evidence only; no model winner or test result.

- Development-fold identity: `sha256:c30bea0617d8b409b0be3d0283e503e73904d4999327014b8d88917aaf01fd1f`
- Verified local MLflow run: `c0d7dedf5a244f3ead2d5a2f48d49415`
- Test status: `sealed_not_loaded`

| Model | View | OOF macro-F1 | Balanced accuracy | Fold macro-F1 | Params | CPU p50/p95 ms |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| LOGISTIC | hand local | 0.792 | 0.785 | 0.776, 0.861, 0.740 | 40,325 | 0.215/0.339 |
| LOGISTIC | body relative | 0.280 | 0.277 | 0.395, 0.210, 0.210 | 2,565 | 0.210/0.400 |
| LOGISTIC | combined | 0.792 | 0.785 | 0.776, 0.861, 0.740 | 42,885 | 0.270/0.469 |
| GRU | combined | 0.620 | 0.677 | 0.583, 0.647, 0.620 | 26,741 | 319.884/346.765 |
| TCN | hand local | 0.834 | 0.831 | 0.823, 0.796, 0.877 | 29,061 | 33.759/41.041 |
| TCN | combined | 0.769 | 0.769 | 0.708, 0.743, 0.835 | 29,317 | 35.556/43.336 |

## Pre-registered comparisons

- **Architecture:** pooled delta +0.150; folds +0.125, +0.096, +0.215; supported for carry forward.
- **Body Context:** pooled delta -0.065; folds -0.115, -0.052, -0.042; unsupported.

## Limitations

- This five-gesture smoke corpus does not estimate population performance.
- Three grouped folds are too few for a formal confidence interval.
- Story #28 informed this development experiment; it is not confirmatory.
- Test features were never requested.
- Carry-forward decisions are not model selection or promotion.
- Latency is machine-specific prevectorized CPU time.

Sanitized per-signer counts and concrete out-of-fold errors are in the evidence artifacts.
