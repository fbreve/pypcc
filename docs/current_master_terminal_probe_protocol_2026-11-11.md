# Experimental current-master terminal-state audit — 2026-11-11

**Scope:** low-risk, separately reviewable **synthetic-only building block**
for [scientific reentry issue #251](https://github.com/fbreve/pypcc/issues/251).
This is **not** Stage B, C1, C2, C2-R1, G1-R1, a migration of frozen
historical studies, or a validated convergence trajectory tool.

`experiments.terminal_state_probe.capture_terminal_state(model)` is a
strictly **opt-in post-fit** observer. It is called **only after** a normal,
unmodified `ParticleCompetitionAndCooperation.fit_predict`. It makes
bounded independent read-only copies of the **terminal** dominance matrix,
internal node assignments, particle positions/strengths/homes/classes,
mapped known labels and postprocessed normalized `owndeg`. Including the
terminal `uint8` distance matrix is optional and explicitly budgeted.
Copies are canonical little-endian C-contiguous arrays; hashes are SHA256
of their **raw values**, not a reproducible NPY file or a versioned archival
format. All bytes are pre-budgeted and unsupported/missing arrays or an
unfitted model are rejected **before** copying.

The observer calls **no propagator**, performs **no global or Numba/Cython
random draw**, introduces **no per-iteration hook**, accepts no dataset,
changes no default and touches no PCC model attribute. It never claims
that the number of executed iterations is `max_iter` when early stopping
might intervene: the returned `iteration_cap` is merely the configured
maximum and `actual_iterations_known` is always **False**. There are no
trajectory milestones, true visits, causal evidence, RNG checkpoints,
intermediate particle states or resumable fits.

## Testing gate

In `tests/test_terminal_state_probe.py`, small symmetric undirected
synthetic ring graphs are fitted with an explicit seed separately for
**each** enabled sequential NumPy/Numba/Cython backend, not requiring
cross-backend bitwise identity. Tests check:

1. End-state snapshot arrays exactly equal the source arrays without
   sharing memory; copies are read-only and C contiguous.
2. Repeating the same backend with the same seed yields identical end
   predictions and all captured-array hashes, including a zero-iteration
   path and postprocessed state.
3. Reading the snapshot does not alter the **Python NumPy global RNG
   state**, although the preceding fit itself may have consumed RNG.
   This is not yet a complete independent probe of the Numba internal
   RNG nor Cython's hidden native seed state; those require stronger
   separate source-aware within-backend continuation tests before a
   *live milestone observer* is proposed.
4. A strict pre-copy byte budget, unfitted model/invalid type rejection,
   and isolation from future modifications to source model arrays.

The full existing cross-platform regression suite is the required
first integration check: **Linux and Windows, Python 3.10/3.12**. Native
compiled instrumented hooks **do not exist** in this branch, so a speed
claim about mid-fit observation is not warranted. It is safe to claim
only that **ordinary fits execute the exact unchanged baseline code**,
because this branch adds no import to `pcc.py` and does not patch any
backend/compiled kernel.

## Next independent gates (NOT authorized by this code)

- Investigate whether the old reviewed local `.cache` B/C1/C2 assets and
  native binary still exist; never infer data existence from the Git
  research branch or rerun a historical task without its exact approved
  source-specific preflight SHA.
- Design a limited **opt-in live observation boundary** for *current*
  Cython/Numba/NumPy separately, retaining cached Cython rowmax logic,
  original RNG state and CPU-optimized dense padded `neib_list`.
  Verify full state/stop/subsequent RNG parity with unobserved runs,
  including true intermediate milestones; compare full-fit CPU overhead.
- Only after new technical/scientific approval and an exact execution
  preflight SHA could a newly preregistered real-data study run. A
  terminal snapshot cannot reconstruct a C2 trajectory after the fact.
- Keep PR #18 OPEN/DRAFT/UNMERGED and preserve its frozen source and
  historical test/result manifests. Never transplant its October
  numeric hotloop patches into this newer `master`.

**Status:** exploratory interface and synthetic tests only, draft PR
pending exact-head CI. No performance, algorithmic or statistical
conclusions follow from this probe.
