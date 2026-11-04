# Prospective in-loop global label-stability stopping: first runtime pilot

Source: GitHub Actions run 41709172930, artifact `inloop-global-stopping/inloop-global-stopping.csv`, commit `b75e670`. All eight workflows passed.

Protocol: NumPy experimental backend, Iris and Breast Cancer, seeds 11/23/37, two labeled examples per class, 10-NN, cap 50k. The actual propagation loop stops after 1–3 consecutive unchanged global predictions at 1k or 5k iteration intervals. Compare wall-clock times and predictions against a separate same-seed run without early stopping. Reference is capped at 50k, not a proof of convergence. Times are single CI measurements, not repeated timing distributions.

| Dataset | Interval | Patience | Early stops / 3 | Total iterations saved / 150k | Mean stopped time (s) | Mean reference time (s) | Mean agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Iris | 1000 | 1 | 3 | 135000 | 0.533 | 5.335 | 97.222% |
| Iris | 1000 | 2 | 3 | 129000 | 0.746 | 5.335 | 97.917% |
| Iris | 1000 | 3 | 3 | 126000 | 0.855 | 5.335 | 97.917% |
| Iris | 5000 | 1 | 3 | 95000 | 1.954 | 5.335 | 98.843% |
| Iris | 5000 | 2 | 3 | 70000 | 2.844 | 5.335 | 99.074% |
| Iris | 5000 | 3 | 2 | 35000 | 4.085 | 5.335 | 99.769% |
| Breast Cancer | 1000 | 1 | 2 | 15000 | 3.295 | 3.647 | 97.876% |
| Breast Cancer | 1000 | 2 | 0 | 0 | 3.665 | 3.647 | 100% |
| Breast Cancer | 1000 | 3 | 0 | 0 | 3.660 | 3.647 | 100% |
| Breast Cancer | 5000 | 1 | 0 | 0 | 3.661 | 3.647 | 100% |
| Breast Cancer | 5000 | 2 | 0 | 0 | 3.657 | 3.647 | 100% |
| Breast Cancer | 5000 | 3 | 0 | 0 | 3.656 | 3.647 | 100% |

Critical examples: Iris seed 37, interval 1000, patience 1–3, stopped at 8k/9k/10k with accuracy delta -1.3889 percentage points vs 50k. Breast Cancer seed 11, interval 1000, patience 1, stopped at 46k and agreed with 50k on only 95.575% of unlabeled nodes. Breast Cancer seed 37 stopped at 39k with accuracy delta -0.1770 percentage points. A shorter runtime is not evidence of safe convergence. On runs reaching the cap, check overhead was small but timings are noisy and not statistically conclusive.

**Methodological caution:** The tracker checks after each full interval and counts consecutive stable *sampled* label vectors, not convergence of dominance values. It can stop before later flips. A 50k reference is only an endpoint comparison. The results are not suitable for production; retain the experimental branch and design a prospective, robust evaluation including longer references, more seeds, multiple datasets, and a safety criterion that accounts for dynamics rather than label agreement alone. No master merge.
