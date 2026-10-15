# Convergence trajectories: diagnostic protocol (PR #18)

Status: Stage A NumPy, Numba and Cython instrumentation implemented; historical experiments remain pending. No historical measurements are changed.

## Research questions

1. At what iteration does PageBlocks unlabeled accuracy deteriorate relative to the fixed 10,000-iteration baseline, and how do class assignments and dominance margins evolve?
2. How does Spambase improve over the same propagation trajectory?
3. How sensitive are Banknote and Sonar trajectories to the frozen historical label seeds?
4. Do dominance stability, class-change counts and existing early-stopping checks correlate with useful accuracy endpoints? Correlation is descriptive, not proof of a safe stopping rule.

## Frozen comparability

Use the historical fixture construction, preprocessing, ordered k-NN graph, three labels per class, label seeds 10–19 and original backend-specific RNG policies. Parameters: p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2, es_chk=2000; iteration cap 500,000. Do not alter the PR #16 fixed-iteration or PR #17 early-stopping archives. Never compare unequal-work elapsed times as intrinsic kernel speedups.

## Instrumentation contract

- Diagnostic mode is strictly opt-in, disabled by default and isolated from the normal fit path.
- Capture observations at a configurable set of iteration milestones, including iteration 0 when meaningful, 10,000, existing stopping-check boundaries and the final iteration.
- For each observation record dataset, label seed, backend, iteration, effective stop reason if final, unlabeled accuracy (diagnostic only), class-change count versus previous observation, dominance margin/stability summaries, state or assignment fingerprints and elapsed time separately from uninstrumented benchmarks.
- Do not feed unlabeled ground-truth accuracy into propagation, convergence decisions, hyperparameter selection or early-stopping logic. Label it explicitly as retrospective transductive evaluation.
- Preserve backend-specific numerical/RNG semantics; do not silently replace the native backend with NumPy or reset/reseed between milestones. Prefer an explicit opt-in observation hook at a safe iteration boundary over repeated fresh fits that change trajectories. If a backend cannot expose a safe hook without perturbation, document the limitation and validate an alternative before collecting data.
- Write portable machine-readable CSV/JSON with protocol, source commit, dependencies, fixture fingerprints, seed policy, checkpoint schedule and checksum manifest. Raw results and plots are generated artifacts, not committed by default.

## Validation gates before real experiments

1. Tiny synthetic fixture: observed vs unobserved run has identical final predictions, effective iteration count, stopping reason and state fingerprints for each available backend.
2. Milestone snapshots do not change RNG draws, loop order, dominance update, backend selection or early-stop semantics.
3. Repeated diagnostic runs with the same frozen fixture/backend are deterministic in classification and state hashes; timing is reported as descriptive overhead, not a performance benchmark.
4. Test edge cases: zero changes, ties, missing classes, capped vs genuinely early-stopped fits, invalid milestone schedule and partial/corrupt output.
5. CI tests must be small and portable. No expensive historical campaign in CI.

## Execution gates

Stage A: implementation and synthetic validation only.
Stage B: small PageBlocks/Spambase pilot, with bounded work and explicit cost estimate.
Stage C: Banknote/Sonar and ten-seed replication after reviewing pilot results.
Stage D: descriptive plots and comparison with PR #16/#17 endpoints; no algorithm changes or stopping-rule recommendation without separate evidence.

## Stage A implementation and limits

NumPy, Numba and Cython observations are now available.
Neither production fit defaults nor ordinary benchmark runners change.
All observations come from one fit and one continuous propagation loop.

### Interface and provenance

`observe_fit` still returns `(predictions, collector, metrics)`. It requires an
explicit sequential `impl="numpy"`, `impl="numba"` or `impl="cython"`, rejects fallback and fewer
than two labeled classes, wraps only that model's dispatch, and restores dispatch
on failure. Truth is mapped from public labels to internal class indices; absent
truth classes count as incorrect. Accuracy is retrospective and cannot enter the
propagation or stopping calculations. Assignments are stored as internal indices.

```python
import numpy as np
from pcc import ParticleCompetitionAndCooperation as PCC
from experiments.convergence_trajectories import observe_fit, metadata, write_output

neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
degrees = np.full(4, 2)
labels = np.array([10, -1, -1, 20])
truth = np.array([10, 10, 20, 20])
model = PCC(impl="numba")  # "numpy" and "cython" use the same diagnostic interface
model.set_graph(neighbors, degrees)
predictions, trace, metrics = observe_fit(
    model, labels, truth=truth, propagation_seed=19,
    milestones=(0, 3, 7, 10000), max_iter=35, early_stop=True)
provenance = metadata(dataset="synthetic-ring", label_seed=0, collector=trace)
write_output(".cache/synthetic-trajectory", trace, provenance)
```

Execution facts are captured at the propagation boundary: actual graph arrays,
public labels and optional truth (canonical int64 hashes), backend, class mapping,
particle/node counts, effective fit parameters including defaults, k_nn, and the
computed stop threshold. Source commit/dirty status, normalized source/protocol
hashes, environment and dependency versions are captured before fitting. Label
seed is explicitly a **caller-declared fixture annotation**, not proof of how
labels were chosen. The infrastructure cannot reconstruct a label selection RNG
from an array. Missing arrays or inconsistent claims are errors, not guesses.

An explicit `propagation_seed` seeds only the chosen backend once before the
single propagation call. Omitting it preserves caller RNG state and records an
unknown seed with a caller-state policy. No seed resets occur at observations.
Legacy `metadata(arrays=..., parameters=..., propagation_seed=...)` is retained:
all three inputs neighbors/degrees/labels, and truth if evaluated, are required;
provided parameters are completed from production defaults and checked against
execution at export. A legacy externally seeded NumPy claim is accepted only if
the complete captured initial RNG state matches `RandomState(seed)` exactly.
An externally seeded Numba state cannot be verified through a public state API,
so its numeric seed must remain unknown unless passed to `observe_fit` explicitly.
For Cython, omit `propagation_seed` or pass 123456789: it always uses its original
fixed per-call xorshift seed, recorded as `cython-fixed-seed`. Any other seed is
rejected rather than ignored. The example's seed 19 is only valid for NumPy/Numba.

### Frequency, resources and compatibility

Milestones are strictly increasing, nonnegative int64 Python integers. Final
observations are unconditional and deduplicated. Both interfaces now default to
`include_checks=False`: only milestones and final are retained. This intentional
change to the diagnostic default prevents a 500,000-step run from retaining
50,000 full assignment vectors automatically. Production defaults are unchanged.
Set `include_checks=True` explicitly for 1, 11, 21, ... check-boundary sampling
(also sampled for capped runs with early stopping disabled).

Default limits are 128 observations and 64 MiB estimated working/export storage.
`observe_fit` checks the entire possible schedule and native buffer allocation
before seeding or propagation, even when early stopping might finish sooner.
Dense schedules exceeding a limit fail; they are never silently thinned. Limits
can be increased explicitly with `max_observations` and `max_bytes`. A standalone
NumPy `Trajectory` checks limits at capture time because it does not know the cap.

For S snapshots, N nodes, C classes and P particles, the native buffer uses exactly
`S * (16*N*C + 16*P + N*P + 16) + 16` bytes for Numba; Cython adds `4*S` bytes
for native RNG states. It retains dominance and raw owndeg,
positions, strengths and uint8 distances, then releases the buffer after Python
postprocessing. Shared conservative accounting adds
`S * (128*N + 16384) + 128*N*C` bytes for rows, scratch and export. This is a
planning estimate, not an OS RSS guarantee. Export also checks actual combined
JSON/CSV UTF-8 size against `max_bytes`. Source/provenance, memory allocation,
postprocessing and export can dominate small-graph timing.

The public NumPy observer callback and return tuple are preserved. Custom NumPy
observers are trusted code receiving live arrays and must neither mutate state
nor draw RNG. Low-level standalone collectors still work for in-memory analysis,
but export requires `observe_fit` execution binding; unverified legacy bundles
are not silently upgraded. New bundles carry schema v2; v1 reads fail explicitly.
Previously written v2 NumPy/Numba bundles with the original source-file inventory
remain readable. Cython v2 bundles require fingerprints for its pyx source and
the shared buffer helper, and include `cython_rng` among the state hashes.

### Native observation mechanism and timing

The private nopython propagation kernel accepts an optional compiled recorder and
fixed-size typed array buffers. Its default None specialization has no recorder.
The observed specialization calls the same step kernel in the same original loop,
uses the same mmpot arithmetic and break condition, and copies selected boundary
states. Terminal recording occurs after the original loop, including zero caps
and early-stop/cap collisions. No object mode, loop restart, step replay, private
RNG API or extra random draw is used. Native snapshots are processed into the same
row schema after propagation, with truth only accessible to Python diagnostics.

Cython keeps the public `pcc_propagate` signature/defaults and forwards to one
shared private implementation. An optional `_TrajectoryBuffer` copies selected
states inside the same `nogil` loop. Its step kernel, update order, mmpot arithmetic,
stopping check and fixed seed are unchanged. No Python callbacks run between
milestones. Both native adapters validate schedules, capacity, dtype, layout,
freshness and exact live-state dimensions before entering native propagation.
The Cython implementation also validates buffers independently at its private
entry point before unchecked copying. Non-diagnostic validation remains unchanged.

The initial Numba boundary regression exposed that its low-level allocator relied
on the fit helper to validate schedules/count limits. A test failed before the fix.
Shared validation now rejects malformed schedules, over-limit allocations and
short supplied buffers before a single step or RNG draw. The ordinary Numba
propagation kernel is untouched in this continuation.

State hashes cover assignments, dominance, particle positions/strengths/distances
and raw owndeg before public-fit normalization, using canonical little-endian
C-order data including dtype and shape. NumPy additionally fingerprints its RNG
state. Numba explicitly marks RNG fingerprints as not captured; tests establish
continuation parity through the next ten native draws. No NumPy RNG hash is
misrepresented as a native RNG hash.
Cython records the actual uint32 xorshift state at every selected boundary.
A private test-only probe exports terminal state plus the next ten uint32 draws
from a COPY of that state. This proves continuation parity without changing the
live RNG or exposing a new persistent-RNG public API. Public Cython fits still
initialize once per call and discard their local RNG state when finished.

NumPy rows retain elapsed observation time. Native intermediate elapsed time is
null: a portable nopython wall clock is unavailable without additional intrinsics.
The Cython recorder uses this same intermediate-null timing convention.
The final native row records whole-loop elapsed time. Metadata separates native
propagation and Python postprocessing, and records whether a new JIT specialization
compiled during propagation. All timing is excluded from trace determinism claims.
Changes and stability compare successive observations, not every intervening step;
reversed changes between snapshots are invisible. The first observation has null
change/stability values. Ties use argmax; labeled nodes remain fixed. Empty
unlabeled sets or missing truth yield null accuracy.

### Bundle validation

JSON contains shared provenance and rows; CSV repeats dataset, label seed and
backend and uses JSON cells for assignments/hashes. Export validates binding,
schema and budget before creating a new output directory. A checksum manifest is
written last. `read_output` verifies the manifest and both files, then validates
required fields/types, finite summaries, class mapping, assignment hashes/change
counts, strict iteration order and exact schedule coverage through the final
iteration, seed policy, effective parameters/stop metrics, resource estimates,
terminal observations and CSV/JSON agreement. A rechecksummed malformed bundle is
still rejected. Checksums are integrity checks, not signatures or proof of origin.
No checkpoint/restart facility or historical campaign runner is added.

### Synthetic validation and overhead

Focused tests cover all three backends' observed/unobserved prediction/state/stop
parity, continued native RNG draws, deterministic repeated traces excluding time,
branching multiclass graphs, multiple seeds, strength/exponent parameters, labeled
node preservation, valid distances/positions, finite dominance/simplex invariants,
zero caps, isolated graphs, ties, early-stop/cap collisions, truth independence,
metadata mismatch, bounded schedules and rechecksummed corrupt output. Small
unobserved prefix runs are test oracles for native snapshots only, never the
instrumentation mechanism. Tests assert the native recorder/kernel compile only
in nopython mode. Cython tests additionally compare terminal RNG state and ten
copied continuation draws against unobserved propagation, with an independent
xorshift oracle and prefix snapshot/RNG checks. Exact milestone stops, zero caps,
one-slot buffers, last-slot final deduplication, byte/count limits, malformed
schedules and undersized external buffers have explicit tests for both natives.

Optional overhead measurement is independent of ordinary benchmarks:

```text
.review-venv/Scripts/python.exe -m experiments.trajectory_overhead --output .cache/native-trajectory-overhead.json
```

This uses 64 synthetic nodes, 3 classes, 6 particles, degree 4, 20,000 capped steps,
5 repetitions with Numba seeds 19-23 and the fixed Cython seed 123456789,
alternating execution order and excluded JIT warmup. Any compilation during a
measured repetition is an error. Every pair verifies predictions, state and stop
metrics; Numba continuation draws are also checked in the probe. Cython RNG parity
is tested separately so probing draws are excluded from its timing. Use
`--backends numba` or `--backends cython` to restrict this optional probe.
Local Windows/Python 3.13.13, NumPy 2.5.4, Numba 0.68.0 median measurements:

| Backend / mode | Observations | Native propagation | Complete fit | Snapshot processing | Export + read validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Numba ordinary | 0 | 10.08 ms | 10.37 ms | n/a | n/a |
| Numba sparse | 3 | 12.99 ms | 70.61 ms | 0.53 ms | 21.21 ms |
| Numba dense (every 200) | 101 | 13.52 ms | 84.13 ms | 12.98 ms | 39.64 ms |
| Cython ordinary | 0 | 5.99 ms | 6.26 ms | n/a | n/a |
| Cython sparse | 3 | 6.27 ms | 63.99 ms | 0.53 ms | 21.99 ms |
| Cython dense (every 200) | 101 | 6.35 ms | 76.00 ms | 13.07 ms | 42.46 ms |

Native overhead was about 29%/34% for Numba and 5%/6% for Cython on this tiny
fixture. These are instrumentation overhead measurements, not general backend
benchmarks or historical results. The Numba native timer surrounds its compiled
kernel call, while the Cython native timer surrounds its extension entry point
including native-entry validation. Ordinary Numba timing includes its wrapper's
input validation; the diagnostic ledger also measures its buffer/input validation
separately. This small boundary difference is not interpreted as a speedup.
Export/read time is additional to complete-fit time. Native and Python snapshot
postprocessing times are reported separately.

`collector.phase_timings` now contains a per-run timing ledger outside the native
loop. It measures preflight, source metadata, model initialization, diagnostic
setup, native buffer validation, propagation, snapshot processing, model
finalization and a residual; nested source measurements split Git commit/status
queries, dependency versions, source hashing and platform lookup. It is not part
of deterministic trace content or the v2 bundle schema. Raw probe output records
these phases, samples, effective seed policies and the probe source checksum.

The large Numba native/complete-fit gap is attributable to measured provenance
capture. Across the five sparse repetitions, **mean** complete fit was 71.27 ms:
12.91 ms native, 56.97 ms source metadata, 0.53 ms snapshot processing and 0.86 ms
for the other measured phases/residual. Source metadata included 22.76 ms for
`git rev-parse`, 26.60 ms for `git status`, 5.71 ms reading dependency versions and
1.89 ms hashing source/protocol files. Thus source capture explains about 98% of
the time outside native propagation for this fixture. Cython shows the same fixed
provenance cost. No provenance cache or production performance change is added.
Means are used for this additive attribution; table medians need not add up.
Every per-run ledger balances; the largest native-probe residual was 0.101 ms.

Sparse/dense Numba buffers used 10,720/360,384 bytes; Cython used 10,732/360,788
bytes including RNG snapshots. Example bundles were about 12/271 KB for Numba and
13/289 KB for Cython (content/timing/source identifiers affect serialized size).
Raw output stays in .cache and is not a published artifact.

Validation commands (the CI matrix remains Windows/Linux Python 3.10/3.12):

```text
.review-venv/Scripts/python.exe -m unittest tests.test_numba_trajectories tests.test_cython_trajectories tests.test_propagation_metrics tests.test_sequential_safety tests.test_convergence_trajectories tests.test_trajectory_infrastructure -v
.review-venv/Scripts/python.exe -m unittest discover -s tests -v
```

Local results: all 35 focused tests passed with all three backends available.
The full suite completed 181 tests successfully, with two packaging archive checks
skipped locally because `PCC_DIST_DIR` was unset; CI supplies built archives.
The final Cython source was compiled in an isolated wheel using a workspace copy
after a Windows lock prevented regeneration of the checkout's C file. That exact
wheel's extension was used for the final tests and timing probe. Installing the
wheel into a separate target and running `tools/check_installation.py --require-all`
from outside the checkout passed all three backend smoke tests. Generated C build
output is excluded from the commit; isolated builds regenerate it from pyx.

All three backends now pass the small synthetic trajectory gates. Remaining
limits are full-state snapshot costs, observations missing changes that reverse
between milestones, native intermediate timestamps unavailable, Numba RNG-state
fingerprints unavailable, and no historical-data validation in this stage.
Next: review Stage A evidence and propose a bounded PageBlocks/Spambase Stage B
pilot with explicit work, milestone/storage and time budgets. No historical pilot
is executed or authorized by these synthetic results. PR #18 stays draft and
unmerged; PR #16/#17 archives and Release assets are unchanged.
