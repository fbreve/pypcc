# Frozen experiment: Fortran-friendly Cython potential-max scan

**Protocol frozen before optimized code outcomes.** Based on draft benchmark PR #175, which measured **+50.24% potential-monitoring overhead on canonical 10-class Digits with actual production Cython** when all trajectories were forced to the same 20k iterations. All work is in an isolated draft experiment. **No changes to `master`, default monitor schedule/thresholds, graph algorithms, Numba, NumPy or particle motion.**

## Hypothesis

The original Cython `_pcc_calc_mmpot` loops over nodes `i` and then classes `j`, reading `dominance[i,j]` strided by node count in the **Fortran-contiguous** dominance matrix used by production Cython. An alternate calculator can loop over class columns first, maintaining the same maximum for each node in an O(n_nodes) scratch vector, then sum node maxima **in the original i order**. This may improve cache locality, especially with 10 classes. It must preserve the exact double-valued result and stop decisions; sums must remain in original row order.

## Candidate implementation (opt-in only)

- In existing `pcc_step.pyx`, retain `_pcc_calc_mmpot` **unchanged** and add a private experimental columnwise helper, accepting a scratch 1D memoryview; create its scratch NumPy array **once** per propagation when the new trailing optional `experimental_columnwise_mmpot=False` flag is true and monitoring is enabled. Default `False` retains the original path.
- At each same existing every-10-step potential check, select between the original function and columnwise helper; no other change in the loop, RNG, update cadence, potential threshold, `stalled_checks`, or `stop_reason`.
- The columnwise helper must use `>` as in original row-max search, and sum the final per-node maxima in **increasing i order** to avoid reordering floating additions. Input matrices are finite under the normal validated dynamics.
- Test opt-in calls through an experimental subclass adapter that overrides `_get_backend_fn`; do not modify public `pcc.py` or production defaults.

## Frozen validation cohort

- Actual **production Cython** only, canonical production undirected union-symmetrized 10-NN on standardized Iris, Wine, Breast Cancer, Digits; **two labeled nodes per class**. Check reciprocal graph/unique neighbors/no loops via `canonical_prepare`.
- **New seeds 6503, 6607, 6701, 6803** (four each). Each graph/seed is run in two policies, each independently with original Cython and opt-in columnwise calculator:
  1. **Forced fixed work:** 20,000 iterations, `early_stop=True`, `es_chk=2000000`. Calculated `stop_max` structurally exceeds the number of checks, so neither run stops early. Measures monitor overhead in actual Cython propagation.
  2. **Actual production stopping:** 100,000 iteration cap, `early_stop=True`, `es_chk=2000` (original default). Tests the actual stop event.
- Each original/columnwise pair must match **exactly** across (a) full `propagation_info` dictionary: iterations/checks/stop reason/stop_max/stalled checks, and (b) all six arrays: final predictions, particle positions, strengths, distances, node dominance and normalized ownership. Strict `np.array_equal`, no tolerance or label-only parity. Any failure aborts the job; no speed interpretation.
- Balance order across four seeds (original-first, alternate-first, original-first, alternate-first). Measure only `fit_predict` time with fresh model each run.
- **4 × 4 × 2 × 2 = 64 timed fits, 32 exact parity gates**, one CSV per dataset; dependent summary job validates every case and includes raw per-case timing ratios; no removal of outliers.
- Single GitHub runner trial per condition is noisy. Include mean/median range of speedups separately by dataset/policy. Do not present statistical equivalence or a deployment speedup.

## Precommitted decision rule

**Parity first.** If any state/stop iteration changes, reject the candidate, regardless of speed. If all 32 pass, candidate merits a more serious replication **only if** the median total-runtime improvement for *Digits forced workload* is **at least 10%** with **no dataset's forced-workload median total-runtime regression worse than 5%**. Otherwise discard this columnwise approach. Even passing this exploratory rule does **not authorize master merge**; require repeated hardware/runner measurements, robust regression coverage (additional shapes/seed/c), memory safety review, and standalone Cython extension testing. The existing kernel remains the default throughout.
