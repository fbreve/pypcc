# Global label-stability counterfactual: 5k checkpoint sensitivity

Source: GitHub Actions run 41636598901, artifact `global-label-stopping/global-label-stopping-5k.csv`, commit `055ab09`. All seven workflows for the commit passed.

Protocol: Iris and Breast Cancer, seeds 11,23,37, two labeled nodes per class, 10-NN graph, NumPy PCC without early stopping. Independently replayed checkpoints 10k,15k,...,50k. Global stopping condition: *all* initially unlabeled nodes unchanged for 1, 2 or 3 successive sampled intervals. 50k is a capped reference, not a convergence guarantee. Six dataset-seed runs per patience; total reference iterations 300,000.

| Stable intervals | Early triggers / 6 | Hypothetical iterations saved | Mean agreement with 50k | Mean accuracy delta (stopped minus 50k) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 90,000 (30.0%) | 99.4213% | +0.5787 percentage points |
| 2 | 3 | 65,000 (21.7%) | 99.5370% | +0.4630 percentage points |
| 3 | 2 | 15,000 (5.0%) | 100% | 0 |

All early triggers were on Iris, none on Breast Cancer. Iris seeds 11,23,37 triggered for patience=1 at 20k,15k,25k respectively; patience=2 at 25k,20k,40k; patience=3 only seeds 23,37 at 40k,45k. Different predictions from 50k sometimes improved observed accuracy, but this small and selected sample does not show a generally superior stopping policy.

Replay overhead (mean per dataset-seed): 5k pilot Iris 21.80 s and Breast Cancer 14.90 s; previous 10k-checkpoint experiment Iris 11.88 s and Breast Cancer 8.14 s. These timings reflect repeated *independent capped replays* on CI, **not** runtime of a true early-stopping implementation and **not** the expected runtime savings. Repeated checkpoints can make the experimental audit substantially more expensive than a single propagation.

**Interpretation:** More frequent sampled checkpoints create earlier global trigger opportunities on Iris, but none on Breast Cancer. The current all-node rule is dataset-dependent and does not warrant production integration. A next step is a prospective in-loop stopping criterion with measured wall-clock runtime, compared against no early stopping on identical seeds/graphs, without modifying production code. Test first on a small controlled pilot before expanding the matrix.
