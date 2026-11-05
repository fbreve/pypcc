# Prospective canonical PCC: actual production potential stopping versus label stability

## Provenance and integrity

Preregistered protocol: [`canonical_default_stop_comparison_protocol.md`](canonical_default_stop_comparison_protocol.md), prior to observing these new results. Executable: [`audit_canonical_production_stop_baseline.py`](audit_canonical_production_stop_baseline.py). GitHub Actions **42018129345**: 4/4 dataset matrix jobs passed. PCC regression **42018129015** passed. Source commit: `6585b2c`.

| Dataset | Artifact ID |
| --- | ---: |
| Iris | 13409713979 |
| Wine | 13409564304 |
| Breast Cancer | 13410796456 |
| Digits | 13410198824 |

All 12 dataset-seed cases (seeds **3407, 3511, 3607**) were run with production `pcc_graph.build_knn_graph`, symmetrized undirected union 10-NN, StandardScaler and two labeled examples/class; 100k no-stop reference; sequential NumPy only. The PyPCC production model used `set_graph(...,k_nn=10)` to preserve the calibration of the original `build_graph(k_nn=10)` default potential criterion, `es_chk=2000`. Experimental label stopping used 1k checks and patience **10**.

**Mandatory no-stop prediction parity gate passed in all 12/12 cases**: the same-seed, same-graph 100k predictions from the candidate experimental NumPy propagator exactly matched the production NumPy no-stop predictions. This validates final outputs at 100k but does **not** prove bitwise equality of all intermediate states or of different backends. All instrumented/stopped runs completed successfully and reported valid iterations.

## Results

| Stopping policy | Early stops /12 | Mean actual wall-time saving vs production 100k | Mean iteration saving | Average agreement with production 100k | Worst agreement | Mean accuracy difference vs production 100k | Worst accuracy loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Existing production potential** | **8/12** | **48.70%** | **49.48%** | **98.76%** | **94.44%** | **+0.179 pp** | **−1.389 pp** |
| **Experimental label stability P10** | **8/12** | **38.36%** | **38.92%** | **99.36%** | **94.44%** | **+0.391 pp** | **−0.581 pp** |

These mean accuracy differences are **not absolute errors**: they can cancel gains and losses. The notable minimum agreement of 94.44% occurred on Iris seed 3607 for **both** policies, which had identical predictions there yet were +4.167 pp more accurate than the 100k reference. This also illustrates that the capped 100k reference is not a ground-truth optimum. Actual wall time is based on one GitHub Actions observation per policy per case; do not infer reliable variance or speedups of Cython/Numba from these timings.

### Dataset-level breakdown

| Dataset | Potential stops /3; mean saved wall time; mean agreement | Label P10 stops /3; mean saved wall time; mean agreement |
| --- | --- | --- |
| Iris | **3/3**, **81.46%**, 97.22% | **3/3**, **67.99%**, 98.15% |
| Wine | **3/3**, **80.86%**, 98.84% | **3/3**, **63.75%**, 99.42% |
| Breast Cancer | **2/3**, **33.22%**, 99.00% | **0/3**, **−1.34%**, 100.00% |
| Digits | **0/3**, **−0.74%**, 100.00% | **2/3**, **23.02%**, 99.87% |

### Concrete tradeoffs

- **Iris seed 3407:** production potential stopped at **13,541** steps and lost **1.389 pp** against 100k, whereas label P10 stopped at **38k** with **identical 100k predictions** and 61.5% measured wall-time saving.
- **Iris seed 3607:** production potential stopped at 19,911 steps and label P10 at 35k; *both produced the same predictions* and had +4.167 pp accuracy versus the later 100k reference. Greater runtime is not necessarily closer to truth.
- **Wine seed 3511:** potential stopped at 19,281 steps, losing **1.163 pp**, whereas label P10 at 32k lost **0.581 pp**.
- **Wine seed 3607:** potential stopped at 16,921 steps and lost 0.581 pp, label P10 at 53k improved 0.581 pp over 100k (the difference between the two stopped policies was **1.163 pp**).
- **Breast Cancer:** potential stopped in 2/3 cases, with outcome change −0.708 pp in seed 3407 and **+1.239 pp** in seed 3511; label P10 never stopped, so perfect 100k agreement here is **not** an actual early-stopping success.
- **Digits:** potential stopped in **0/3** cases; label P10 stopped in **2/3** (51k and 80k). For seed 3511 its stopped labels differed on 6 of 1,777 initially unlabeled nodes (agreement 99.662%) yet had the same aggregate accuracy as 100k; seed 3607 changed one prediction (agreement 99.944%, −0.056 pp).

## Interpretation and recommendation

The potential rule already in PyPCC is **competitive**: the experimental label-stability P10 has not demonstrated unconditional superiority. The potential criterion yielded more average wall-time savings across these 12 cases (~10.3 percentage points), but P10 had somewhat higher fidelity to the 100k labels, a smaller observed worst negative accuracy delta, and a **meaningful advantage in Digits** where potential never triggered. Neither is a clear universal choice.

**Before production adoption**: compare both stop rules against 100k on additional **independently selected seeds, datasets, and different supervision fractions**; measure accelerated-backend Cython and Numba behavior or prove stop-rule parity across backends; include robustness to longer horizons and actual time overhead. Consider a predeclared hybrid selection policy only after reproducible evidence, not post-hoc tuning on these 12 cases. The current experiment does **not** justify modifying production defaults, particle strength, reset rules or `master`.
