# PopSign public-data vertical slice

This report proves that one bounded, licensed public corpus ran through the existing
SignLab extraction, quality, and `combined-64` feature pipeline.

- Candidate clips: 4339
- Split/target groups: 15
- Selected usable clips: 7
- Groups without a usable clip: 8
- Corpus SHA-256: `sha256:83a2058cd4c4d4bd67cb08a8d75a04f1b106121522f8e00bd1887f0d4ebf5fdc`

## Selected clips

| Source split | Count |
| --- | ---: |
| test | 1 |
| train | 3 |
| val | 3 |

| Target | Count |
| --- | ---: |
| hello | 1 |
| no | 2 |
| please | 2 |
| thank_you | 1 |
| yes | 1 |

## Coded exclusions

| Reason | Count |
| --- | ---: |
| `quality.quarantine` | 17 |
| `quality.reject` | 36 |
| `selection.attempt_limit` | 1991 |
| `selection.not_needed_after_accepted` | 2288 |

## Reproducibility identities

- External dataset: `sha256:3eb0bb1e73cacddf3b59a84d5c946207c7cb973d6526edea8c75fe9138e8669c`
- Extraction configuration: `sha256:7343cd8bb724313b4063a3ebd5d7f7470a78b00f2eeda275a15e5f9b2e66e94c`
- Hand model: `sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- Pose model: `sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`
- Quality policy: `sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545`
- Feature plan: `sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6`
- Source orientation basis: `reviewed_popsign_v1_official_samples_upright_with_readable_unmirrored_scene_text`

## License and limits

PopSign ASL v1.0, Georgia Institute of Technology and Deaf Professional Arts Network, licensed under CC BY 4.0.

License: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

**Limitation:** Public isolated-sign data only; no participant, continuous-sign, or natural-use claim.
