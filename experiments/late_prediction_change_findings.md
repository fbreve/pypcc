# Late prediction changes versus dominance margins

Source: GitHub Actions run 41101104586, artifact `late-prediction-changes`, commit `f329894`.

Protocol: four real datasets, five seeds each, 10-NN, two labeled examples per class, NumPy experimental backend, early stopping disabled. Each cap restarts the simulation from the same seed. Margins are the difference between the largest and second-largest dominance values at the earlier cap, on initially unlabeled nodes. The prediction is the argmax of dominance.

| Interval | Changed predictions | Initially unlabeled nodes | Changed fraction | Mean starting margin among changed nodes | Fraction of changed nodes with starting margin < 0.01 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10k to 40k | 1345 | 13290 | 10.12% | 0.2690 | 50.86% |
| 40k to 50k | 278 | 13290 | 2.09% | 0.3514 | 25.54% |

## 40k to 50k by dataset

| Dataset | Changed / unlabeled | Fraction | Mean starting margin among changed nodes | Fraction changed with starting margin < 0.01 |
| --- | ---: | ---: | ---: | ---: |
| iris | 2 / 720 | 0.28% | 1.0000 | 0% |
| wine | 15 / 860 | 1.74% | 0.7332 | 6.67% |
| breast_cancer | 95 / 2825 | 3.36% | 0.6192 | 11.58% |
| digits | 166 / 8885 | 1.87% | 0.1559 | 35.54% |

Interpretation: late changes are not limited to near-ties; a large starting margin is not a guarantee that the predicted class remains fixed. The diagnostic samples only endpoints, so it does **not** distinguish repeated oscillation from a single transition or measure within-interval trajectories. This does not justify a new stopping criterion or promotion to production. Next investigate finer-grained snapshots or transitions, while keeping computational cost bounded.
