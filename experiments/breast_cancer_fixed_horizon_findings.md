# Breast Cancer stability patience at matched decision checkpoints

Source: GitHub Actions run 41207912273, artifact `breast-cancer-late-flips-1k/breast-cancer-fixed-horizon.csv`, commit `f0ba121`. All six workflows for this commit succeeded, including the experimental real-data pilot.

Protocol: Breast Cancer, five seeds, 10-NN, two labeled examples per class, NumPy experimental backend without early stopping. Checkpoints every 1,000 iterations from 40k through 50k. All patience settings (1–5 consecutive unchanged sampled intervals) are assessed at the **same** decision checkpoints (45k,46k,47k,48k,49k), against a fixed one-checkpoint (1k iteration) future horizon. Counts aggregate across five seeds and five decision checkpoints. These are overlapping per-node windows, not independent trials.

| Stable checks | Apparently stable node-windows | Changed at next checkpoint | Conditional sampled change fraction |
| ---: | ---: | ---: | ---: |
| 1 | 14035 | 84 | 0.5985% |
| 2 | 13957 | 81 | 0.5804% |
| 3 | 13879 | 76 | 0.5476% |
| 4 | 13814 | 72 | 0.5212% |
| 5 | 13756 | 67 | 0.4871% |

Compared with one check, five checks reduce the observed conditional next-checkpoint change fraction by about 18.6% (not the 41% suggested by unequal-horizon windows). Longer patience filters some at-risk node-windows, but **67 still changed** despite five unchanged checks. Importantly, these per-node conditional rates cannot be used directly as a global stopping failure probability: a real stopping criterion considers all unlabeled nodes simultaneously, and patience also changes the stop time and compute cost. The next experiment should replay a prospective global stopping trigger against longer reference trajectories, with accuracy and iteration savings evaluated on held-out dataset/seed combinations. No production criterion is justified by this diagnostic alone.
