# Convergence trajectories: diagnostic protocol (PR #18)

Status: Stage A NumPy and Numba instrumentation implemented; Cython instrumentation and historical experiments remain pending. No historical measurements are changed.

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

NumPy and Numba observations are now available. Cython instrumentation remains
pending. Neither production fit defaults nor ordinary benchmark runners change.
All observations come from one fit and one continuous propagation loop.

### Interface and provenance

`observe_fit` still returns `(predictions, collector, metrics)`. It requires an
explicit sequential `impl="numpy"` or `impl="numba"`, rejects fallback and fewer
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
model = PCC(impl="numba")  # "numpy" uses the same diagnostic interface
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
`S * (16*N*C + 16*P + N*P + 16) + 16` bytes. It retains dominance and raw owndeg,
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

### Native observation mechanism and timing

The private nopython propagation kernel accepts an optional compiled recorder and
fixed-size typed array buffers. Its default None specialization has no recorder.
The observed specialization calls the same step kernel in the same original loop,
uses the same mmpot arithmetic and break condition, and copies selected boundary
states. Terminal recording occurs after the original loop, including zero caps
and early-stop/cap collisions. No object mode, loop restart, step replay, private
RNG API or extra random draw is used. Native snapshots are processed into the same
row schema after propagation, with truth only accessible to Python diagnostics.

State hashes cover assignments, dominance, particle positions/strengths/distances
and raw owndeg before public-fit normalization, using canonical little-endian
C-order data including dtype and shape. NumPy additionally fingerprints its RNG
state. Numba explicitly marks RNG fingerprints as not captured; tests establish
continuation parity through the next ten native draws. No NumPy RNG hash is
misrepresented as a native RNG hash.

NumPy rows retain elapsed observation time. Native intermediate elapsed time is
null: a portable nopython wall clock is unavailable without additional intrinsics.
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

Focused tests cover NumPy and Numba observed/unobserved prediction/state/stop
parity, continued native RNG draws, deterministic repeated traces excluding time,
branching multiclass graphs, multiple seeds, strength/exponent parameters, labeled
node preservation, valid distances/positions, finite dominance/simplex invariants,
zero caps, isolated graphs, ties, early-stop/cap collisions, truth independence,
metadata mismatch, bounded schedules and rechecksummed corrupt output. Small
unobserved prefix runs are test oracles for native snapshots only, never the
instrumentation mechanism. Tests assert the native recorder/kernel compile only
in nopython mode. Existing Cython regressions run; its implementation is unchanged.

Optional overhead measurement is independent of ordinary benchmarks:

```text
.review-venv/Scripts/python.exe -m experiments.trajectory_overhead --output .cache/trajectory-overhead.json
```

This uses 64 synthetic nodes, 3 classes, 6 particles, degree 4, 20,000 capped steps,
5 repetitions with seeds 19-23, alternating execution order and excluded JIT warmup.
Every pair verifies predictions, state, stop metrics and subsequent RNG draws.
Local Windows/Python 3.13.13, NumPy 2.5.4, Numba 0.68.0 median measurements:

| Mode | Observations | Native loop | Complete fit | Export + read validation |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 0 | 9.91 ms | 10.17 ms | n/a |
| Sparse | 3 | 12.92 ms | 70.84 ms | 22.19 ms |
| Dense (every 200) | 101 | 13.11 ms | 81.75 ms | 40.20 ms |

Native overhead was about 30%/32% on this tiny graph; complete diagnostic fit
ratios were 6.97x/8.04x, with provenance subprocesses included. These small timings
are descriptive overhead, not general performance claims. Sparse/dense native
buffers used 10,720/360,384 bytes; estimated peaks were 109,024/2,867,136 bytes;
example bundles were approximately 11/271 KB. Raw local probe output stays in
.cache and is not a historical measurement or published artifact.

Validation commands (the CI matrix remains Windows/Linux Python 3.10/3.12):

```text
.review-venv/Scripts/python.exe -m unittest tests.test_convergence_trajectories tests.test_trajectory_infrastructure tests.test_numba_trajectories tests.test_propagation_metrics tests.test_sequential_safety -v
.review-venv/Scripts/python.exe -m unittest discover -s tests -v
```

Local results: all 29 focused tests passed with all three backends available.
The full suite completed 175 tests successfully, with two packaging archive checks
skipped locally because `PCC_DIST_DIR` was unset; CI supplies built archives.

Next: review the bounded native interface and validate a separate Cython observer
before claiming all-backend trajectory parity or requesting a bounded Stage B
pilot. No historical fits, PR #16/#17 archive changes or releases are included.
