# Single-run 50k–100k trajectory audit

Source: GitHub Actions run **41876472562**, artifact `late-horizon-trajectory` (`late-horizon-checkpoints.csv`, `late-horizon-summary.csv`), commit `0c3c4d9`.

## Protocol and integrity

Experimental NumPy PCC; five selected dataset-seed cases: Wine seeds 37,67,83, Iris seeds 11,37. The unmodified propagation proceeds to **100,000 iterations** without early stopping. A read-only callback records dominance on initially unlabeled nodes every **5,000 iterations** within this single continuous run, while a separate 50k replay verifies its exact 50k predictions. All 50k and 100k checkpoint predictions matched independent capped runs; this validates the observational approach for these cases. Neither cap establishes convergence. The cases were selected for unusual late changes and are not a random representative sample.

| Dataset | Seed | Unlabeled | Nodes changed after 50k at any sampled checkpoint | Changed then returned to 50k by 100k | Adjacent-checkpoint flips 50k–100k | 50k–100k agreement | Accuracy 50k | Accuracy 100k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wine | 37 | 172 | 69 | 36 | 105 | 80.81% | 87.21% | 74.42% |
| Wine | 67 | 172 | 14 | 0 | 16 | 91.86% | 47.67% | 41.86% |
| Wine | 83 | 172 | 47 | 1 | 54 | 73.26% | 84.30% | 58.72% |
| Iris | 11 | 144 | 3 | 0 | 3 | 97.92% | 54.17% | 56.25% |
| Iris | 37 | 144 | 4 | 0 | 6 | 97.22% | 88.89% | 86.11% |

These are **sampled** flips: unseen changes between checkpoints may make actual totals higher.

## Abrupt changes after apparent stability

- **Wine seed 37:** Zero sampled label changes from 50k through 60k, then just 4 changes over 60k→75k. At 75k→80k **20 nodes** change; 80k→85k **15**; 85k→90k **18**; 90k→95k **17**; 95k→100k **31**. Accuracy falls 88.37% (75k) to 58.72% (95k) then rebounds to 74.42% (100k). P95 per-node dominance change reaches 0.972, 0.649, 0.914, 0.950, 1.000 over these successive 5k intervals. The average top-two dominance margin stays relatively high (about 0.85–0.90) despite label changes.
- **Wine seed 83:** 15 changed labels at 60k→65k and 20 at 65k→70k. Accuracy declines from 84.30% (50k) to 62.79% (70k), finishing at 58.72% (100k). P95 dominance change at 65k→70k is 0.946 while mean top-two margin is 0.905.
- **Wine seed 67:** Label changes are comparatively gradual after 50k (0–3 changes per 5k, except 50k→55k has 3). Accuracy declines from 47.67% to 41.86%.
- **Iris seeds 11 and 37:** much fewer changes, but neither is mathematically converged.

The large mean dominance margins are calculated **across all initially unlabeled nodes** and therefore do **not** establish that the *flipping nodes* were confident; a per-flip margin audit is needed.

## Implications

1. Global unchanged labels over consecutive checkpoints is not evidence of eventual stability: Wine seed 37 exhibits a quiet window followed by a large late regime change.
2. Even an apparently high *global average* top-two dominance margin may conceal abrupt transitions. Investigate pre/post-change margins of affected nodes, class transition counts, and narrower time resolution around Wine seeds 37 and 83. Avoid inferring cause without these diagnostics.
3. In-loop hybrid stopping is not ready for production; no merge to `master`. Results support studying dynamics and deciding when useful predictions occur, not blindly optimizing agreement with arbitrarily longer capped runs.
