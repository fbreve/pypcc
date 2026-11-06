# Frozen experiment: incremental per-node maxima for the original global potential

**Registered before measuring this candidate. Isolated draft branch based on the completed monitoring-overhead PR #175. No changes to `master` or production stopping defaults.**

## Previously tested approaches / novelty check

Prior draft #44, #65, #82–#87 fused the *local* dominance maximum used for a particle's movement with a dominance update. They did **not** cache the mean-of-row-maxima metric used for global stopping; in those tests `_pcc_calc_mmpot` continued to scan the whole dominance matrix. Draft #176 tried **column-major full rescanning** of the global potential and was rejected after making Digits Cython approximately 41.49% slower despite strict parity.

The new candidate has a different mechanism: the production Cython particle loop **already calculates `max_dom` on each visited node after any dominance mutation** to decide movement. When a visited node is unlabeled, update an experimental `row_max[node]` cache from that exact value. At every original 10-step stopping checkpoint, sum cached maxima in **strictly increasing node order** to compute mean maximum dominance. This changes global metric check complexity from `O(N * C)` to `O(N)` but adds at most one scalar cache write per successful unlabeled particle visit. It **does not maintain a running global floating sum**, whose changed addition order could alter stop decisions.

## Frozen implementation plan

- Keep production `_pcc_calc_mmpot` unchanged. Add opt-in trailing keyword flags `experimental_rowmax_cache=False` and `experimental_verify_rowmax_cache=False` to experimental `pcc_step.pcc_propagate`. Both default to `False` and preserve legacy callers.
- Initialize the vector of cached maxima from the validated starting dominance matrix using the **same `>` comparisons and class order** as the original function. Only allocate when `early_stop` and the experimental flag are enabled.
- The existing particle update already obtains the row maximum as `max_dom` after dominance mutations; at the same point update the cache only if the destination node is **unlabeled**. Labeled rows do not change dominance. Skipped/invalid moves leave cache unchanged. Preserve every original update, RNG call, and movement decision.
- Compute the global check by summing the cached maxima **in original increasing node-index order** with the same division by `n_nodes` and same stopping cadence, thresholds and `>` comparisons. Do not alter the default path. Add an extra **unmeasured opt-in verification** that compares the cache scalar to a fresh original full-scan scalar at **every check** and fails with its iteration number on any mismatch; this verification is never used in timing.
- No NumPy/Numba alteration, no public `pcc.py` changes: an experiment-only subclass overrides backend dispatch with an adapter that passes the opt-in flag to compiled Cython.

## Independent, frozen comparisons

- Canonical standardized Iris, Wine, Breast Cancer, Digits; production **undirected union-symmetrized 10-NN**, exactly two labels/class, and fresh seeds **6907, 7001, 7103, 7207**.
- Four dataset jobs. For each dataset/seed, first run the **unmeasured 2,000-iteration candidate with full-scan verification of the cached metric at every 10-step checkpoint**, forced not to stop (`es_chk=2000000`). This must match the original full scalar on every visited checkpoint. This audit verifies *all observed potential values*, not merely final states.
- In each dataset/seed, run the same original Cython backend and the experimental cache in two regimes (fresh model and same graph/labels/seed for each run):
  1. **Forced fixed-work:** 20,000 iterations, `early_stop=True`, `es_chk=2000000` (number of checks cannot reach `stop_max`). Isolate overhead and metric cost.
  2. **Actual production potential:** 100,000 cap, default `early_stop=True`, `es_chk=2000`. Do not force stopping. Check exact stop iteration and all metadata.
- For each of the resulting **32 legacy/candidate pairs**: hard require strict `np.array_equal` across final predictions, particle positions, strengths, distance tables, dominance and normalized ownership **AND identical full `propagation_info` dictionary**. Require actual Cython backend, no fallback, and 20k exact cap in fixed-work mode.
- Alternate original-first/opt-in-first order across the four seeds, measure `fit_predict` only. **64 timed fits + 16 untimed potential-trace verification runs**. All raw timing outcomes (including losses) retained.
- Four CI jobs archive raw CSV and installed package versions; aggregate job validates **all 64 policy rows / 32 parity pairs / 16 independent metric-trace parity controls**. No exclusions.

## Precommitted exploratory decision rule

Hard reject for **any** metric value mismatch, full-state mismatch, or stop metadata mismatch. If all pass, call it **worth replicating** only when the median *whole-fit* saving on the 20k **Digits** forced-workload is **at least 10%**, and no dataset's 20k median regresses more than **5%**. This is a limited exploratory go/no-go criterion, not proof of a true speedup. Even if favorable, no production merge before repeat-run and cross-platform benchmarks, broader graph shapes and class counts, zero/near-zero potential changes and other numerical edge cases. **One observation per condition on shared GitHub runners cannot establish a generalizable performance gain.**

### Caveat on terminology

The cache is incrementally refreshed **per node**; the mean is still reduced in canonical order every 10 steps. This is deliberate: a true O(1) running sum of maxima could differ by floating-point nonassociativity and must not be substituted without a new, explicitly separate experiment.
