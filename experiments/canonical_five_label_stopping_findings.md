# Canonical undirected PCC: five labels per class, 100k stopping sensitivity

## Provenance and integrity

Predeclared protocol: [`canonical_five_label_stopping_protocol.md`](canonical_five_label_stopping_protocol.md). GitHub Actions run **42016191620**, source commit `b064d70`; all four dataset jobs **passed**. PCC regression on this commit, run **42016191758**, also **passed**. Artifacts:

| Dataset | GitHub artifact ID |
| --- | ---: |
| Iris | 13409891503 |
| Wine | 13409308356 |
| Breast Cancer | 13410855192 |
| Digits | 13410485945 |

Graphs: unmodified production `pcc_graph.build_knn_graph`, k=10, union-symmetric, degrees verified, same graph within policy comparisons. Five labeled observations/class, new seeds 3109, 3203, 3301; sequential experimental NumPy PCC. Three conditions per seed: uninterrupted 100k reference, label-change stability every 1k with patience **3** and patience **10**. All **12/12** checkpoint-instrumented references exactly matched independent same-seed, same-graph **no-callback** references; all graph reciprocal-arc fractions exactly 1.0.

## Overall result, 12 paired cases

| Measure | Label stability patience 3 | Label stability patience 10 |
| --- | ---: | ---: |
| Early stops | **9/12** | **9/12** |
| Mean iterations saved | **60.42%** | **42.42%** |
| Mean observed wall time saved | **60.45%** | **42.51%** |
| Average agreement with 100k | **99.56%** | **99.63%** |
| Lowest individual agreement | 98.77% | 98.77% |
| Mean accuracy change vs 100k | **−0.301 pp** | **−0.248 pp** |
| Worst accuracy loss vs 100k | **−1.227 pp** | **−1.227 pp** |

Differences are descriptive of these specific seeds and the fixed 100k reference, **not** confidence intervals or evidence of true convergence. A single wall-time run per case is not a statistically valid hardware benchmark.

## Dataset-specific observations

| Dataset | P3 early stops /3, mean wall-time saving, mean Δ accuracy | P10 early stops /3, mean wall-time saving, mean Δ accuracy |
| --- | --- | --- |
| Iris | **3/3**, 90.71%, −0.247 pp | **3/3**, 83.69%, −0.247 pp |
| Wine | **3/3**, 81.80%, −0.613 pp | **3/3**, 47.44%, −0.613 pp |
| Breast Cancer | **0/3**, −0.25%, 0.000 pp | **0/3**, −0.15%, 0.000 pp |
| Digits | **3/3**, 69.54%, −0.343 pp | **3/3**, 39.07%, −0.134 pp |

On **Iris**, patience 3 and patience 10 made *the same final predictions within every seed*, even though patience 10 performed additional iterations. On **Wine**, both policies had the same mean accuracy delta but different individual outcomes. For Wine seed **3203**, patience 10 stopped at **78k** yet still lost **1.227 pp**, equal to patience 3 stopping at **19k**. For Wine seed **3301**, patience 3 stopped at **5k** without net accuracy change whereas patience 10 stopped at **19k** and lost 0.613 pp. These two cases demonstrate that higher patience is not guaranteed to improve classification accuracy or reference agreement.

On **Digits**, all P10 trajectories stopped at **57k, 46k and 80k**. P10 average accuracy change was −0.134 pp versus P3 −0.343 pp; P10 used more steps and saved an average 39.07% observed wall time. The respective P3 endpoints were 33k, 32k and 27k.

**Breast Cancer** did not stop under either rule in all three seeds; monitoring caused a small average time penalty. No configuration is a universal speedup.

## 50k → 100k reference drift

| Dataset | Mean prediction changes per trajectory | Largest changes | Mean change in accuracy, 100k minus 50k |
| --- | ---: | ---: | ---: |
| Iris | 1.00 | 2 | +0.741 pp |
| Wine | 2.33 | 3 | +0.613 pp |
| Breast Cancer | 10.33 | 13 | +0.775 pp |
| Digits | 4.33 | 6 | +0.191 pp |

A 100k reference remains a capped trajectory rather than proof of convergence. Accuracy is calculated only on initially unlabeled nodes; their numbers differ across datasets and from prior two-label-per-class cohorts.

## Interpretation and next control

Raising initial supervision from two to five labeled points/class does **not** establish that patience 10 is uniformly superior. It improves mean label agreement only marginally relative to patience 3 while sacrificing roughly 18 pp mean wall-time savings, and the worst observed loss remained 1.227 pp.

Most importantly, **all previous canonical stopping studies compared experimental label-change stopping with no-stop capped references but not with the existing production potential-based stopping default**. The next properly controlled experiment should explicitly compare **PyPCC's production NumPy `stop_criterion='potential'`, default `es_chk=2000`** against candidate label patience 10 and a full no-stop control on new canonical graph seeds, while verifying no-stop candidate versus production numerical/prediction parity. When installing an externally built graph via `set_graph`, pass `k_nn=10` to preserve the production build_graph stopping calibration. Do not merge candidate stopping to `master` without demonstrating advantage against this actual baseline.
