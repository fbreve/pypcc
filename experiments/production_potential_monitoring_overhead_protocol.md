# Frozen audit: read-only potential-monitoring overhead in real PCC backends

**This protocol is frozen before observing CI outcomes. Experimental benchmark only.** Based on the complete held-out stopping comparison in draft PR #174, which did **not** justify a direct replacement of the production potential-stopping criterion with label P10. No production kernels, default parameters, graph algorithms, RNG or `master` are altered.

## Question

How much runtime overhead is caused **only by checking the existing production potential metric every 10 steps**, independently of the compute saved when early stopping actually terminates propagation? Previous single-run Digits comparisons reported substantial overhead for accelerated backends, but compared an ordinary stop-enabled configuration to a 100k no-stop reference and were not designed to estimate timing variability.

## Frozen cohort and paired contrast

- Datasets: `iris`, `wine`, `breast_cancer`, `digits`; same production **undirected union-symmetrized kNN=10** and standardized sklearn features, with two true labeled examples per class, verified by `canonical_prepare`.
- **Fresh seeds: 6101, 6203, 6301, 6407** (four per dataset), each used in **three actual backends** `numpy`, `numba`, `cython`. Verify no silent fallback by checking the production callable name.
- Common cap: **20,000 steps**. For each dataset × seed × backend, start a new fit twice from identical graph, labeled nodes and seed:
  - **off:** `early_stop=False, es_chk=2000` (no metric checks).
  - **on, cannot stop:** `early_stop=True, es_chk=2000000`, which increases the production `stop_max` threshold above the maximum number of 10-step potential checks in 20k iterations. This leaves the **same production metric check every 10 steps** in place, but prevents termination. Verify calculated `stop_max > cap//10 + 1` and actual executed iterations = 20k.
- **Mandatory full-state parity:** compare predictions, particle locations, strengths, distance table, node dominance and normalized ownership, requiring **exact `np.array_equal` for every array**. If any state differs between on and off, fail the test before interpreting timing: monitoring must be observational. This gate also defends against unintended differences in RNG draw consumption.
- Balance order of on/off across the four seeds (**off-first, on-first, off-first, on-first**) to partially mitigate runner drift. Prepare new model/graph installation for each run; time only `fit_predict` (same allocation/postprocessing work in both conditions). CPU runners and one trial per case remain noisy.
- Before any timed Numba run, separately initialize both on/off modes for that dataset and memory-layout signature with **seed 29 and a 25-step cap** outside timing to exclude JIT compilation, including the Digits 10-class layout. Cython compiled during dependency installation.
- **Total:** 4 datasets × 4 seeds × 3 backends × 2 modes = **96 timed fits**, **48 state-parity gates**. CSV per dataset, all 24 rows (12 matched pairs) retained. A dependent summary job requires all 4 raw artifacts, validates all 96 records and 48 pairs, and reports per-backend and per-dataset means/medians/min/max of `100 * (time_on/time_off - 1)` plus `100 * (time_on-time_off)/time_on`. Small negative timing differences are possible from noise and **must not be discarded**. State parity is the actual hard pass/fail criterion, not a minimum estimated speed loss.
- CI matrix: 4 dataset jobs; a final mandatory aggregate summary job. Environment versions archived.

## Interpretation limitations

This does **not** benchmark default stopping end-to-end or its savings; it isolates **metric checking cost while prohibiting stopping**. An artificially increased `es_chk` is a measurement technique, not a recommended production setting. Measured ratios combine instrumented C-level or NumPy metric work and its impact on caches, and the timing interval includes `fit_predict` initialization/postprocessing. One runner observation per paired seed is **not** a robust microbenchmark and must not support a universal speedup claim. If material overhead repeats and arrays match bitwise, any future implementation change must be attempted on a **separate experimental branch** with exact output/stop-iteration parity across seeds, classes and graph sizes before consideration for production. Do not change the production `early_stop` semantics or thresholds based solely on timings.
