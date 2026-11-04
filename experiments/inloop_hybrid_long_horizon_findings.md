# Hybrid in-loop stopping: 50k vs 100k reference sensitivity

Source: GitHub Actions run **41866310696**, artifact `inloop-hybrid-long-horizon/inloop-hybrid-long-horizon.csv`, commit `9e2f947`. Workflow and PCC regression both passed.

## Protocol

Experimental NumPy PCC, datasets Iris and Wine, seeds 11,23,37,67,83; two labeled samples per class; 10-NN graph. Compare *independent same-seed* no-stop runs at 50k and 100k, and prospective hybrid stop capped at 100k. Hybrid: P95 dominance-change tolerance 0.01, labels unchanged on all initially unlabeled nodes, sampled every 1k iterations, two consecutive stable checks. Metrics below are for initially unlabeled nodes. A 100k cap is **not** proof of convergence. Timings are a single GitHub Actions observation for each condition.

| Dataset | Hybrid early stops | Hybrid iterations saved / 500k | Mean runtime reduction vs 100k | Mean 50k–100k reference agreement | Mean hybrid–100k agreement | Accuracy at 50k | Accuracy at 100k | Hybrid accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Iris | 3/5 | 193k (38.6%) | 38.0% | 97.92% | 99.03% | 72.78% | 73.47% | 73.61% |
| Wine | 4/5 | 311k (62.2%) | 61.7% | 87.33% | 87.91% | 78.49% | 68.72% | 76.98% |
| **Overall** | **7/10** | **504k (50.4%)** | **49.9%** | **92.62%** | **93.47%** | — | — | — |

Accuracy changes versus 100k must **not** be framed as improved convergence or stopping reliability; Wine often became *less accurate* as the capped run continued.

## Individual runs

| Dataset | Seed | Hybrid stop | Iterations saved | Agreement 50k–100k | Agreement hybrid–100k | Hybrid accuracy minus 100k (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Iris | 11 | 49k | 51k | 97.92% | 97.92% | -2.08 |
| Iris | 23 | 25k | 75k | 100% | 100% | 0 |
| Iris | 37 | 33k | 67k | 97.22% | 97.22% | +2.78 |
| Iris | 67 | 100k (no stop) | 0 | 95.83% | 100% | 0 |
| Iris | 83 | 100k (no stop) | 0 | 98.61% | 100% | 0 |
| Wine | 11 | 40k | 60k | 94.77% | 94.77% | +1.74 |
| Wine | 23 | 16k | 84k | 95.93% | 95.35% | +2.33 |
| Wine | 37 | 22k | 78k | 80.81% | 79.07% | +11.05 |
| Wine | 67 | 100k (no stop) | 0 | 91.86% | 100% | 0 |
| Wine | 83 | 11k | 89k | 73.26% | 70.35% | +26.16 |

Wine seed 83 illustrates the issue: 50k reference accuracy 84.30% and 100k 58.72%; 50k–100k predictions agree on 73.26% of unlabeled nodes. The hybrid stopped at 11k with accuracy 84.88%, agreeing with 100k on 70.35%. This supports **persistent nonconvergence or drift within the sampled horizon**, but does not by itself establish the cause, whether transient oscillation, a longer-term regime transition, or algorithm dynamics.

## Interpretation / next steps

1. Do not treat either 50k or 100k as a ground-truth converged endpoint. Label-only and hybrid stop criteria remain experimental and unsuitable for `master`.
2. Examine **within-run intermediate checkpoints**, at least for problematic Wine seeds 37,67,83 and Iris seeds 11,37, rather than repeatedly replaying independent trajectories; this will reveal the *timing* and *reversibility* of label changes, plus dominance margins.
3. Compare class changes and accuracy over time using a callback that observes state without modifying the RNG or model; verify that the final snapshot equals the independently capped result. Keep callbacks and diagnostics in the experimental backend only.
4. Only after understanding long-horizon drift consider new stopping rules; refining patience/tolerance on these same seeds risks overfitting.

The latest legacy real-data pilot was cancelled; the targeted long-horizon workflow and regression checks passed.
