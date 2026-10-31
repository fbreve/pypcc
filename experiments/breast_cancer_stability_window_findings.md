# Retrospective sampled stability windows: Breast Cancer

Source: GitHub Actions run 41205100701, artifact `breast-cancer-late-flips-1k/breast-cancer-late-stability.csv`, commit `c947bae`.

Protocol: Breast Cancer, five seeds (11,23,37,41,53), 10-NN graph, two labeled examples per class, experimental NumPy backend without early stopping. Checkpoints independently replayed at 40k,41k,...,50k iterations. A window is stable when a node retains its argmax class across w adjacent checkpoint intervals; a subsequent change is any later sampled checkpoint with a different class, within the observed horizon.

| Consecutive stable intervals | Stable node-windows | Later-changing node-windows | Fraction | Distinct nodes with at least one false-stable window, summed across seeds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25254 | 665 | 2.633% | 123 |
| 2 | 22320 | 517 | 2.316% | 108 |
| 3 | 19427 | 398 | 2.049% | 98 |
| 4 | 16577 | 298 | 1.798% | 83 |
| 5 | 13756 | 215 | 1.563% | 67 |

For w=5, distinct false-stable nodes by seed: 11→23, 23→19, 37→9, 41→5, 53→11.

**Limitations:** Overlapping node-windows are dependent, and the observation horizon ends at 50k. Increasing w reduces the number of eligible windows and the amount of future observation available, so the declining fractions are **not** an unbiased estimate of the effect of a longer stopping patience. The longest sampled pre-change streak (summed per seed) is not a useful pooled statistic; do not interpret it as an observed global maximum. Neither these rates nor checkpoint stability prove convergence. Future analysis should hold the prediction horizon fixed across patience settings, ideally evaluate a real global stop trigger, and assess accuracy/runtime against a longer reference. No production change is justified yet.
