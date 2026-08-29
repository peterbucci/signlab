# PopSign reference baselines v1

> Exploratory smoke benchmark only. These numbers are not product-performance or sign-language claims.

## Reproducibility identity

- Source commit: `bd03aac9fa76e8d169462b7bfcb05d208d276f93` (dirty: `false`)
- Configuration: `sha256:23a12193f2be198d560d1df447c2cb5b5dfba13c26966a1236867d2e203317c7`
- Frozen split: `sha256:673c777c67e47715127b75f2bf18aaa794a15e1604458d30da164ec7f90ead20`
- Source corpus: `sha256:bd2e552d28792346b7c8e345f8387ebcc52938692cde7f0a316763aa09bdceb9`
- Feature plan: `sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6`
- Seed: `20260828`

The signer-disjoint split contains 50 training, 15 validation, and 15 final-test clips across five targets.
Test features were opened only after the single logistic-regression choice was fixed on validation macro-F1.

## Results

| Model | Validation macro-F1 | Test macro-F1 | Test balanced accuracy | Parameters | CPU p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.067 | 0.067 | 0.200 | 0 | 0.007 / 0.008 |
| Stratified Random | 0.139 | 0.187 | 0.200 | 0 | 0.214 / 0.297 |
| Logistic Regression | 0.931 | 0.725 | 0.733 | 42,885 | 0.084 / 0.183 |

The predeclared logistic choice was `C=0.1` from `[0.1, 1.0]`; ties resolve to the smaller value and the selected train-only fit was not refit.

## Selected-model test errors

| Class | Support | Errors |
| --- | ---: | ---: |
| hello | 3 | 2 |
| no | 3 | 0 |
| please | 3 | 1 |
| thank_you | 3 | 1 |
| yes | 3 | 0 |

| Quality disposition | Support | Errors |
| --- | ---: | ---: |
| pass | 13 | 3 |
| warning | 2 | 1 |

Signer grouping: 7 held-out signers; 4 had at least one error; the maximum was 1 error for one signer. No source identifier is published.

## Limitations

- This is a 50-train, 15-validation, 15-test smoke benchmark with three test examples per class.
- Logistic regression fits 42,885 coefficients and intercepts from only 50 training examples.
- Results use one signer-disjoint split and one seed; they do not estimate uncertainty across splits or seeds.
- Only two test examples have a warning disposition, and each test signer contributes few examples.
- The benchmark covers five isolated targets only: it provides no evidence for other, inactive, abstention, continuous events, or sign-language translation.
- Latency measures pre-vectorized single-example CPU prediction and varies by machine; extraction and JSON loading are excluded.

## Reproduce

```shell
uv run --locked --extra experiments signlab train reference-baselines configs/experiments/popsign-reference-baselines-v1.json --corpus-root <frozen-split-root> --external-manifest <external-dataset-manifest.json> --output-root runs/popsign-reference-baselines-v1
```
