# Frozen protocol: NumPy-only continuous single-run trajectory observer (V1)

Pre-registered **before testing candidate code**. Branch `experiment/numpy-continuous-trajectory-v1` starts from current merged `master` (Cython #178 potential cache, #180 finite checks); historical PR #18 remains unmerged. This experiment touches **only NumPy's propagation entry point**, an isolated observer module, tests and an experiment-only CI workflow. No Cython, Numba, PCC graph, movement rule, random sampling, early-stop criterion or production default changes.

## Why this differs from PR #181

PR #181 uses repeated fresh seeded fits to reach milestones; it does **not** prove they reproduce an actual **single uninterrupted observed run**. Here one NumPy fit really runs once from iteration zero, optionally recording bounded full-state snapshots at predetermined milestones, and we compare the entire observed run with a genuinely unobserved fresh fit from the current master implementation. This is a read-only, opt-in **`observer=None` trailing keyword on `pcc_propagate_numpy`**, not an export format or a new stopping criterion. The normal public `PCC.fit_predict` API remains unmodified and never passes an observer.

## Exact change under test

- At entry to NumPy propagation **after validation and seed installation**, call observer at effective iteration 0 with a read-only view of mutable arrays and NumPy's current MT19937 RNG state.
- After each normal particle iteration and its original stopping test, invoke observer only if provided; it decides whether the current iteration is a selected milestone. On exiting the original loop, send exactly one `final=True` notification with original `propagation_info` from the **same** iteration/stop counter; no extra particle step, no extra random draw, and no altered check order. Exactly one physical snapshot per effective iteration (final flag may upgrade a scheduled row).
- Experimental collector strictly accepts only increasing iteration counts (with a permitted final upgrade of last iteration), copies all mutable low-level arrays, and records the entire NumPy RNG-state fingerprint (not just `np.random.seed`). Bounded predefined schedule **0,1,11,101,501,2001,6001**, a maximum of **9 snapshots and 48 MiB** per fit, including final if early termination creates an extra milestone. The observer must not modify particle state or call any RNG.
- `observer=None` must not affect production outputs or RNG; callbacks must be invoked only for opt-in calls. No new high-level `fit_predict` parameter or public diagnostic file format.
- Test an **unsupported observer failure** with a callback that raises at first entry; no propagation iteration may occur; this is a private experimental API, not a promised transaction-safe mutation contract.

## Frozen rigorous evaluation

- Canonical production **undirected union-symmetrized kNN=10** on standardized Iris, Wine, Breast Cancer and Digits, exactly 2 labeled examples/class selected by a separate `default_rng`. All adjacency rows reciprocal with no self/duplicates. NumPy backend must be actually selected, not silent fallback.
- Completely new diagnostic seeds **12007,12101,12203**. Compare ordinary `ParticleCompetitionAndCooperation(impl="numpy").fit_predict` with an instrumented subclass using the same NumPy function with `observer=` and `return_info=True` (the wrapper retains the exact backend name to preserve initialization layout).
- Three named policies, all sharing the same settings `p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0`:
  1. **no_stop**: `max_iter=6001, early_stop=False, es_chk=2000`.
  2. **real_potential**: `max_iter=6001, early_stop=True, es_chk=2000` (the actual standard potential check and threshold, no forced stop).
  3. **aggressive_potential**: `max_iter=6001, early_stop=True, es_chk=10` (tests actual stops near threshold; *not* a production recommendation).
- For every dataset × seed × policy, execute two **fresh** fits (unobserved baseline versus continuous observer), **72 fits / 36 strict matched pairs**. Alternate order across seeds/policies. Both runs must produce **exactly identical six-array final state**: classifications, particle positions, strengths, distances, dominance and *normalized* ownership, plus the **entire native `propagation_info` dictionary**. Require equality of full RNG state using all `np.random.get_state()` fields and also the next **16 NumPy RNG outputs** after each fit, observed without introducing any draws into the measured fit. Include actual stop iteration and reason; require no_stop to finish at cap.
- The continuous observer must record the configured milestones it actually visits and the final state at the true effective stopping iteration; for checkpoints that lie later than the actual stop it must not invent rows. Its last snapshot's six low-level arrays and full final RNG state must match the state captured by the instrumented propagation immediately after stopping. The initial snapshot must be the initialized state (no random step yet).
- Each fit's collector has a fixed **48 MiB** byte budget and **9** observation cap, with no dynamic growth past the cap. Preserve **all 36** full final paired comparisons and raw per-snapshot metrics/digests in one CSV per dataset. The collector captures *low-level raw ownership before normalization* and tracks the full native particle state; final high-level parity compares normalized ownership. No cross-backend bitwise equality assumptions.
- Four GitHub Actions Linux matrix dataset jobs, with mandatory aggregate validation of **36 paired comparisons / 36 observed traces**, exact 72 fits and all expected dataset/seed/policy keys. A completed green baseline regression `python -m unittest discover` on Linux/Windows Python 3.10/3.12 is also required. All raw results, source and pinned environment versions archived. No omitted anomalous cases.
- Collect both observed and unobserved full-fit runtime as *exploratory overhead* (do not infer production speedup or guarantee from one hosted runner). Predeclare exploratory threshold: median observer-on overhead must not exceed **50%** for any dataset's no_stop cohort, otherwise reduce sampling/memory or reject instrumentation scope; correctness remains absolute regardless of speed. Compare full-fit time only.

## Do not claim

- Do not call checkpoint net label differences **all flips**, and do not infer convergence from a quiet interval; the observer samples discrete moments, not every iteration.
- No hidden Cython/Numba RNG state is captured; this is **NumPy only**.
- This is not yet the PR #18 16-milestone binary snapshot format, and does not authorize that PR's unexecuted historical pilots.
- Keep this work a **draft PR**, never auto-merge into `master` even on green CI. Replicate on Windows and test default-off standalone native performance separately before considering production integration.
