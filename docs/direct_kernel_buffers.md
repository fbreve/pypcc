# Direct kernel calls: buffer ownership and aliasing

The supported high-level entry point is
`ParticleCompetitionAndCooperation.fit_predict()`. It prepares the working
arrays used by the selected backend. The rules below apply when bypassing that
entry point and supplying arrays directly to the propagation functions:

- `pcc_numpy.pcc_step_numpy` / `pcc_numpy.pcc_propagate_numpy`
- `pcc_numba.pcc_step_numba` / `pcc_numba.pcc_propagate_numba`
- `pcc_step.pcc_propagate` (Cython)

## Caller responsibilities

Provide **independent, non-overlapping storage** for mutable state and scratch
arrays. This includes views into a common allocation: two arrays can have
different Python identities yet overlap in memory. In particular, do not alias:

- State arrays: `dominance`, `owndeg`, `part_curnode`,
  `part_label`, `part_strength`, and `dist_table`.
- Active scratch arrays in Numba: `dom_row`, `dom_list`, `dist_list`,
  `prob`, and `slices`; in Cython: `prob` and `slices`.

The NumPy sequential step ignores the six supplied scratch arrays. The
`reduc` parameter is legacy and is not read or written by any of the three
sequential kernels in the audited revision. This distinction describes current
implementation behavior, **not** permission to rely on aliasing of unused
parameters in future versions.

Keep read-only inputs such as graph neighbor arrays, `labels`, and
`dist_weights` independent of arrays the kernel writes. `zerovec` is a
legacy input with no active scientific reads in these sequential kernels. A buffer intended
to be read-only must not share memory with mutable state or scratch space.

Direct propagation updates mutable particle/node state in place. Scratch
arrays may be overwritten during execution; their contents after a call are
not a stable public result. The caller owns the lifetime and allocation of
arrays supplied to these functions.

Some backends allocate optional scratch arrays internally. This does not
make it safe to pass overlapping arrays for other parameters.

## Validation and consequences

Existing input validation checks selected shapes, lengths, numeric values,
and distance-table ranges. **It does not guarantee non-overlapping memory.**
Overlapping arrays may silently change particle state, dominance, and
predictions without raising an exception. Not every overlapping pair showed
a difference in the diagnostic tests: for example, complete `prob`/`slices`
aliasing did not diverge, while partially overlapping views did in Numba and
Cython. Such observations are not guarantees for arbitrary layouts or inputs. Behavior can differ between NumPy,
Numba, and Cython. A call returning normally is not evidence that its result
is valid.

For custom direct-kernel integrations, allocate separate buffers and avoid
overlapping NumPy views. Check suspected overlaps before calling the kernel
(e.g. with `np.shares_memory(a, b)`), outside the iteration-critical path.
`np.may_share_memory` is a conservative screening check and can report
possible overlap when none exists.

These are **preconditions for direct calls**, not new runtime checks or
changes to the algorithm. A diagnostic audit found reproducible alias-induced
state/prediction differences in all three backends using artificially
overlapping buffers (432 paired control/alias comparisons). Fifteen observed
ordinary `fit_predict` calls had independent kernel arguments; no public-API
aliasing path was reproduced. That observation is not a proof that every conceivable
public-API interaction is alias-free.
