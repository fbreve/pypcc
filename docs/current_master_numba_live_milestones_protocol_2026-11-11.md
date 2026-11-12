# Modern-master Numba-native live milestone feasibility — 2026-11-11

**Synthetic-only, opt-in scientific experiment. Not a production algorithm
change, historical PR #18 migration, or authorization for C2/G1 real fits.**
This research branch is based on immutable production
[`ea69f690208620d5563c8efece6b2e04369f52c0`](https://github.com/fbreve/pypcc/commit/ea69f690208620d5563c8efece6b2e04369f52c0).
It is independent of the NumPy [#253](https://github.com/fbreve/pypcc/pull/253)
and terminal-only [#252](https://github.com/fbreve/pypcc/pull/252).

## Observation point and numerical fidelity

`experiments.live_numba_milestones.observe_numba_sequential` creates a
**separate compiled Numba (`@njit`) outer loop**. It calls the exact current
`pcc_numba._pcc_step_numba_sequential` particle movement function, without
reimplementation, patching or a Python callback; the unchanged
`_pcc_calc_mmpot_sequential` computes the metric on iterations 1, 11, 21,
etc. The original strict `>` plateau reset and `stop_cnt > stop_max`
break conditions are retained. **No change to Numba's RNG calls or
seeding**: the same original `_seed_numba_rng(seed)` runs once before
the compiled loop. No calls to Python random/NumPy random are added in
the native loop or recorder.

The new `@njit` snapshot copier reads, after each completed iteration and
the ordinary early-stop check, the current `dominance`, live
**pre-normalization** `owndeg`, positions, strengths and `uint8`
distance table. Boundary zero is a separately allocated initial-state
snapshot. A stopped run omits all later scheduled snapshots; its
`effective_iterations` and stop reason come from production's
`pcc_metrics.propagation_info`.

The complete potential storage of all requested snapshots is
**allocated before seeding**, with a strict **64 MiB** cap, a maximum of
32 ordered integer milestones and no implicit truncation or resampling.
Only snapshots actually reached are exposed as array views flagged
read-only. These views refer to *private preallocated snapshot arrays*,
not to live model arrays; read-only flags are an API safety convention,
not cryptographic authentication. `reserved_bytes` represents the
entire preallocated potential snapshot storage and `bytes_copied`
only the reached data (different under early stop).

## Synthetic validation checklist

The permanent `tests/test_live_numba_milestones.py` compares against
**unmodified production Numba propagation** under matched seed 71:

- Every requested milestone, including initial state and zero iterations,
  versus a separate production run stopped at exactly the same iteration,
  with exact mutable array values (no tolerance or backend comparison).
- Final positions, strengths, distance table, dominance, unnormalized
  owndeg, scratch state, actual number of iterations and stop diagnostics.
- **Native Numba RNG state**, inspected through Numba's testing helper,
  and the *next generated Numba random number*, not merely Python
  NumPy global RNG; checks that global NumPy RNG is also unaffected.
- A no-motion graph with early stopping and correctly absent post-stop
  observations, invalid schedule/budget rejection *before native RNG
  state mutation*, and snapshot array nonaliasing.
- Public `fit_predict` parity under test-local instance dispatch, with
  returned labels, final node/particle states, **normalized** owndeg and
  native RNG continuation. This covers two-class synthetic fit only.
  Independent tests also check inherited native RNG with `seed=None`
  (by restoring the exact Numba MT state), plus `deltap=0.33/0.35`
  and pure-random `p_grd=0.0` movement. Larger-class C-contiguous
  Numba dominance and other graph structures will need separate targeted
  coverage before any blanket claim.

Do not infer identity across NumPy, Numba and Cython: they have distinct
random-number streams, numeric layouts and native execution paths.

## Performance and follow-on scientific gates

1. This preliminary gate requires an exact-HEAD Linux/Windows × Python
   3.10/3.12 run of the complete PCC regression, including installed
   Cython, NumPy and Numba. JIT compilation/warmup must not be counted
   in steady-state CPU medians.
2. An independent opt-in Numba complete-`fit_predict` CPU pilot is
   provided by `experiments/bench_live_numba_fullfit.py` and the **branch-
   scoped** `.github/workflows/research-numba-live-fullfit-cpu.yml`.
   It measures **five AB/BA CPU pairs** on both Windows and Linux at
   500 synthetic symmetric-ring nodes, 24 particles, 1,200,000 iterations,
   ten snapshots and fixed seed 71, after **separate JIT warmups**.
   Every paired result is rejected unless predictions, full final fit
   state and *native Numba* RNG SHA match exactly. Preserve raw samples,
   separate wall times, source SHA and artifacts. The earlier 150,000-
   iteration pilot finished but its 0.28s Windows CPU values showed coarse
   15.625ms timer steps; it is a **superseded timing design**, not evidence
   of zero overhead. Distinguish snapshot reserved/copied bytes from
   compiled-loop CPU overhead. No real data or acceptance claim follows
   from synthetic timings alone.
3. Continue with additional graph/topology, multi-class C-contiguous
   dominance, parameter and malformed-input scenarios, including the
   low-level early-stop/seed-none contracts already covered on synthetic
   two-class cases. Do not generalize those characterized cases to all
   datasets or automatically treat backend-specific floating differences
   as defects.
4. Native Cython remains separate: current specialized cached-rowmax and
   legacy kernels plus their **local xorshift** RNG cannot be replaced
   by an October 2026 wrapper. Independent within-backend RNG and
   stopping-equivalence proofs are required for its future observer.
5. Reuse *none* of the historical PR #18 B/C1/C2-R1 or G1-R1 preflights.
   Only an explicitly approved **new source-, environment- and
   dataset-specific execution preflight SHA** could authorize later
   real-data C2 work.

**Disposition:** stand-alone synthetic research draft; no production
merge, no real datasets, no speedup/zero-cost claims. The deliberately
**dense padded `neib_list`** remains untouched; do not replace it with
sparse CSR without full-fit CPU evidence and exact within-backend
semantics/RNG equivalence.
