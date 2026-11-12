# Current-master NumPy **live** milestone feasibility — 2026-11-11

**Research-only and opt-in.** This experiment establishes a first exact-state
observation boundary **inside a real current NumPy propagation call** while
reusing the *unchanged* production particle-step function. It does not claim
that the legacy C2/C2-R1 scientific protocol or its dataset preflights can be
executed on the current code.

## What runs and what does not

The separate entrypoint
`experiments.live_numpy_milestones.observe_numpy_sequential` accepts the
ordinary NumPy propagator buffers, scalar parameters, stopping controls, and
an explicit strictly increasing schedule of **up to 32** iteration indices.
Each boundary makes independent, read-only, C-contiguous copies of:

- full `dominance`, unnormalized live `owndeg`;
- particle positions, strengths and `uint8` distance table.

These are the actual in-memory arrays **after** complete particle updates at
iteration `i` (and after evaluating the ordinary early-stop check for that
iteration). Index zero means the initial state before the first step. No
snapshots are taken for unreachable milestones after early stopping. The
reported `effective_iterations`, `stalled_checks` and `stop_reason`
are taken from the same `pcc_metrics.propagation_info` function used by
production; unlike a post-fit cap-only reader, the actual iteration count
is known here.

The implementation imports the **current** production
`pcc_numpy._pcc_step_numpy_sequential_unchecked` and calls it directly,
with precisely the same arguments and order, once per iteration. Only the
small *outer* loop (the mean-max potential early-stop cadence and break)
is replicated in this research module. No particle-step source,
RNG routine, numeric parameter, early-stop condition, graph constructor,
particle ordering, default fit API, or Numba/Cython compiled loop is changed.
The experiment **does not** alter production `pcc.py`, nor does it add a
persistent production instrumentation flag.

This module only supports **sequential NumPy**, not synchronous mode or
Numba/Cython. It does not inherit the canonical Cython rowmax-cache logic.
Do not use the adapter to claim full C2 three-backend parity or cross-backend
bitwise equivalence.

## Before any RNG/state consumption

Validations match the production sequential NumPy propagator and occur
before seeding. Additional research-only checks require an explicit,
strictly ordered schedule with integer indices in `0..max_iter`, maximum
observations 1..32, and a **hard 64 MiB combined cap** for all planned
array copies. The entire maximum potential number of retained bytes is
precomputed from input array sizes; if the schedule cannot fit, the
adapter refuses the run **before** any random draw or state mutation.
No automatic downsampling or opportunistic truncation occurs.

The observer itself uses no callback, numpy.random, `np.random.seed`, or
dynamic instrumentation of the particle step. The only seed operation is
the production-compatible single `np.random.seed(seed)` before the loop.
All snapshots remain independent arrays; the read-only flag prevents
normal writes through the returned references, but is **not** a security
boundary or an archival integrity contract.

## Deterministic synthetic tests

`tests/test_live_numpy_milestones.py` builds a small symmetric undirected
ring, with an intentionally **dense padded** neighbor matrix. It compares
the observed call to the **unmodified production**
`pcc_numpy.pcc_propagate_numpy` with the same seed and parameters:

- exact final mutable states, stop reason, effective iterations and
  subsequent NumPy RNG state **plus next random draw**;
- **every milestone independently** against a fresh ordinary baseline
  run truncated to the same number of iterations (zero included);
- an early-stop case with no active edges, halting at a known bound and
  correctly omitting later requested snapshots;
- no-fit zero iterations, strictly enforced memory/schedule/input bounds,
  rejection before RNG or state changes, copy ownership and nonaliasing.

Within-backend byte-for-byte floating values are required, not a tolerance
selected after observing data. Tests do **not** require equal RNG streams
across NumPy, Numba and Cython.

The existing 4-platform GitHub Actions matrix
(Linux/Windows × Python3.10/3.12) is the initial reproducibility gate;
these tests are automatically included in regular unittest discovery.
No assertion of CPU improvement or acceptable observation overhead
is made from CI success.

## Scientific and performance boundaries

This is a **genuine mid-fit observation** *for NumPy alone*, but it does
not furnish all previous C2 details or a signed archival bundle, and its
`owndeg` is intentionally the live **pre-normalization** matrix rather
than the normalized output returned by `fit_predict`. No ground-truth
labels or evaluation metrics feed into movement. The whole-state
matrices are synthetic and small; no real datasets or private fixture
stores are touched.

Before adding native adapters:

1. Quantify complete propagation/fit CPU overhead from the **same source
   and environment**, with both no-observation and observation schedules,
   separating array-copy/export overhead. Do **not** justify CPU changes
   from single-step timings or compare unlike backend RNG trajectories.
2. Preserve the **current** Cython cached row-max and legacy kernels,
   Numba nopython execution, continuation RNG, and exact early-stop
   scheduling. Never transplant PR #18's obsolete native kernel text.
3. Verify full state, actual stop iteration, hidden RNG continuation
   and all prefix milestones separately within **each** backend before
   claiming a three-backend observer.
4. Keep previous experimental preflights fixed to the original source,
   binary and data; issue #251 requires separate source-specific,
   explicitly approved new preflight SHA before any real C2 research fit.

**Disposition:** experimental draft only. The frozen PR #18 remains
open and unmerged. Current production mathematical behavior is unchanged.
