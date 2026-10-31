# Breast Cancer late-flip sampling resolution

Source: GitHub Actions run 41160416380, artifact `breast-cancer-late-flips-1k`, commit `d32dee0`. Four independent PCC executions per checkpoint are **not** assumed: each checkpoint is a separate replay with the same seed, and there are five seeds total (11, 23, 37, 41, 53).

Protocol: breast_cancer, two labeled samples per class, 10-NN graph, NumPy experimental backend, no early stopping. Compare checkpoints from 40,000 through 50,000 iterations, using 2,000-step and 1,000-step sampling. Aggregate counts across five seeds, each with 565 initially unlabeled nodes.

| Metric | Every 2,000 | Every 1,000 |
| --- | ---: | ---: |
| Nodes changing at least once | 128 | 133 |
| Nodes changing more than once | 37 | 48 |
| Adjacent sampled label changes | 169 | 197 |
| Nodes returning to initial class | 33 | 38 |
| Nodes different at endpoints | 95 | 95 |

At 1,000-step resolution, seed-level counts (changed at least once / changed > once / returns / endpoint changes):

- Seed 11: 41 / 8 / 6 / 35
- Seed 23: 41 / 19 / 17 / 24
- Seed 37: 16 / 8 / 6 / 10
- Seed 41: 14 / 6 / 4 / 10
- Seed 53: 21 / 7 / 5 / 16

Finer sampling exposes five additional nodes that changed and returned and eleven additional nodes with multiple sampled transitions. It cannot establish the total number of within-checkpoint transitions. This supports retaining the stopping research as experimental; no new stopping criterion should be promoted solely from these results.
