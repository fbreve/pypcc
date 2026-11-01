# Counterfactual global label-stability stopping results

GitHub Actions run 41323326987, artifact `global-label-stopping/global-label-stopping.csv`, commit `490e87c`. The experimental global stopping workflow passed.

Protocol: iris, wine, breast_cancer, digits × seeds 11,23,37,41,53 (20 dataset-seed runs), 10-NN, two labeled nodes per class, NumPy PCC without early stopping. Independently replayed capped predictions at 10k,20k,30k,40k,50k. At each checkpoint, a counterfactual global trigger requires *all* initially unlabeled nodes to retain the same predicted class over 1, 2, or 3 successive sampled intervals. The final 50k snapshot is a capped reference, **not** a convergence guarantee. These are hypothetical iteration savings, not measured wall-clock savings.

| Patience | Early triggers / 20 | Total iterations saved / 1,000,000 | Mean reference agreement | Mean accuracy delta (stopped minus 50k) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 70,000 (7%) | 99.9653% | +0.0347 percentage points |
| 2 | 1 | 20,000 (2%) | 100% | 0 |
| 3 | 1 | 10,000 (1%) | 100% | 0 |

All early triggers occurred on **Iris**; none occurred on Wine, Breast Cancer or Digits. At patience=1, Iris seed 23 triggered at 20k and differed from its 50k reference on one initially unlabeled node (agreement 99.3056% for that run), with +0.6944 percentage points accuracy. Iris seed 37 triggered at 40k with identical endpoint predictions; Iris seed 41 triggered at 20k with identical endpoint predictions. At patience=2 and 3 only Iris seed 41 triggered, at 30k and 40k respectively, with identical endpoint predictions.

**Interpretation:** The global all-node label-stability trigger is too conservative to provide meaningful savings on three of four datasets under this sparse 10k-checkpoint protocol. Accuracy agreement at 50k does not establish convergence. Further experiments should test a prospective global criterion at a finer observation cadence and measure actual runtime and accuracy; do not merge a stopping criterion based on these findings.
