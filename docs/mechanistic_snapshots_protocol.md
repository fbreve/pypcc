# Mechanistic observations, profile A — protocol and binary contract v1

Scope: optional full dominance, particle positions/strengths and static particle
identity. No exposure, acceptance, occupancy or flip counters; no changes to the
NumPy propagation loop or either native propagation loop. Production PCC defaults
and rules are unchanged. PR #18 remains draft and unmerged.

## Interface and meaning

The existing `observe_fit` interface/return tuple and v2 summary remain available.
The new keyword-only `snapshot_config=None` disables full snapshots by default.
Use `SnapshotConfig` explicitly to retain them. This is a research interface in
`experiments`, not a new production `fit_predict` option.

```python
from experiments.convergence_trajectories import observe_fit, metadata
from experiments.mechanistic_snapshots import (
    SnapshotConfig, write_snapshots, read_snapshots,
)

# model already has a graph, with an explicit sequential backend.
predictions, trace, metrics = observe_fit(
    model, labels, truth=truth, propagation_seed=seed,
    milestones=(0, 10, 100, 1000, 10000, 100000),
    max_iter=100000, early_stop=False, max_observations=32,
    snapshot_config=SnapshotConfig(),
)
write_snapshots(new_directory, trace,
    metadata(dataset="example", label_seed=10, collector=trace))
with read_snapshots(new_directory) as bundle:
    dominance = bundle["arrays"]["dominance"]  # read-only memory map
    # Analyze while the context is open; copy explicitly if needed afterward.
```

Truth is used only for retrospective summary evaluation. It never enters PCC
decisions. NumPy/Numba seed policies remain native/caller-specific; Cython requires
its fixed seed 123456789 (or None, resolved to that seed). No reseeding between
milestones, no propagation replay, no Python callback/object mode in native loops.

All snapshots refer to the existing safe boundary after the sequential iteration
(or initialization at zero); final observation is deduplicated. Early termination
can produce fewer observations than the reserved schedule. Static particle order
is ascending labeled node index, matching PCC initialization; particle IDs are the
original zero-based indices. Particle classes are internal indices, resolved by
the summary's `class_index_to_label`; homes and node IDs are graph row indices.
There is no node/particle reordering. Source row identity beyond graph indices must
be resolved through the pinned fixture/source-index mapping.

NumPy retains preallocated independent copies at selected observations. Native
backends retain references to **only** the already allocated dominance, positions,
strengths and iteration arrays, not the entire native tuple. The native own/label-
distance buffers may be released after postprocessing. No extra full-state copy is
introduced in the native propagation path; the new option uses those arrays during
the existing Python postprocessing. These are observations, not resumable PCC/RNG
checkpoints. Intermediate events and visits are not reconstructible.

## Files and schema

`pypcc-mechanistic-snapshots-v1` creates an exclusive new directory:

| File | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| iterations.npy | S | little-endian int64 | Actual strictly ordered observations |
| dominance.npy | S × N × C | little-endian float64 | PCC dominance, no normalization/casting |
| positions.npy | S × P | little-endian int64 | Current particle node at the observation boundary |
| strengths.npy | S × P | little-endian float64 | Current strength |
| node_ids.npy | N | little-endian int64 | 0…N−1, graph row identity |
| particle_ids.npy | P | little-endian int64 | 0…P−1, initialization order |
| particle_home_nodes.npy | P | little-endian int64 | Original labeled node for each particle |
| particle_classes.npy | P | little-endian int64 | Static internal class index |
| mechanism.json | — | UTF-8 JSON | Protocol, new module source hash, shapes/dtypes/order, per-array hashes, actual/scheduled milestones, limits/estimates |
| summary/trajectory.json, summary/trajectory.csv, summary/manifest.json | — | existing v2 | Unchanged summary and existing execution provenance |
| manifest.json | — | checksummed envelope | Final completion marker and exact file SHA256 inventory |

Arrays use uncompressed **NPY v1.0, C order, no pickle**. No ZIP compression or
duplicated full-state JSON/CSV. Per-array hashes use the existing canonical
shape/dtype/little-endian-value hash; file hashes additionally cover headers.
Reader validates the exact NPY version, dimensions, dtype, layout and file length
before mapping, rejecting object arrays, trailing data and malformed descriptors.
It checks particle/home/class identity against the input-label fingerprint, node
order, dominance invariants and labeled anchors, actual iteration ordering, and
each snapshot against the corresponding dominance/position/strength fingerprint
and assignments in the v2 summary. Checksums prove consistency, not authenticity.

`read_snapshots` maps arrays read-only and returns a context manager. Close it
promptly, particularly before deleting/moving files on Windows. `max_export_bytes`
and `max_memory_bytes` on the reader default to 64 MiB and can be tightened; larger
limits require explicit caller choice. Readers do not compare the current checkout
with the recorded source commit, so completed bundles remain portable.

## Budgets and publication

`SnapshotConfig(max_bytes=64 MiB, max_memory_bytes=64 MiB,
max_export_bytes=64 MiB)` supplies independent limits for snapshot arrays, estimated
diagnostic memory including legacy native buffers/workspace, and the entire enriched
bundle including the v2 summary. `max_observations` still bounds the schedule; no
automatic thinning, truncation or per-iteration retention. Budgets are checked
before fit/seed consumption and again during export/validation. Metadata has a
1 MiB workspace limit; dataset context is at most 256 UTF-8 bytes in this profile.

For planned S, raw arrays require `S*(8*N*C+16*P+8) + 8*N+24*P` bytes. Native
buffers are already included in the existing Stage A estimate. The full-snapshot
memory estimate adds static identity, independent NumPy snapshot arrays where
applicable, and a conservative hash/validation workspace of twice the largest
array plus 1 MiB. This is **not total process RSS**: model/fixtures/libraries/JIT,
allocator and filesystem page-cache behavior are additional. Export estimates
include bounded JSON/CSV overhead; the actual on-disk bytes including headers and
manifest are also enforced before publication. Integer limits reject bool/float.

Files are written/fsynced, their semantics/checksums are validated, then a final
manifest is atomically renamed into place. The destination was exclusively created
by this writer and existing destinations are refused. No root manifest means
incomplete; a temporary publication file is not a completion marker. Readers never
fall back to `summary/` when enriched output is incomplete. A separately requested
v2 read of `summary/` is only a summary read, not validation of the full bundle.
Failure leaves partial artifacts for audit; no automatic overwrite/cleanup/refit.
Manifest publication completed before a process exit is a recoverable complete
bundle and must be reconciled without propagation. Publication is not a claim of
power-loss immunity on every filesystem; Windows lacks portable directory fsync.

Existing Stage B/C1 bundles and manifests are immutable and receive no sidecars.
Source changes invalidate old **execution** preflights; do not resume those stores
with new code. Read-only frozen adapters validate their artifacts without confusing
the old source identity with permission to execute them again.

## Exploratory pilot — separate approval required

PageBlocks and Spambase; label seeds **10 and 13**; NumPy/Numba/Cython; 12 new
diagnostic fits. This intentionally revisits fixtures with known outcomes to obtain
previously unexported values, in a new store. It is not completion of C1 or blind
validation, and never overwrites old results. C2 is excluded.

Frozen parameters/RNG and k_nn=10/three labels per class are retained. Cap 100k,
early stopping disabled. Milestones: **0, 10, 25, 50, 100, 250, 500, 1000, 2000,
5000, 10000, 20000, 40000, 60000, 80000, 100000**, deduplicated final. Reserve at
most 32 observations, 64 MiB per each configured array/memory/export budget. Actual
preallocation is the sixteen-entry schedule, not automatic allocation of 32 slots.

Seed 10 is a known reference; seed 13 includes the Spambase/NumPy global-accuracy
exception. Selection is outcome-informed and exploratory. Primary analyses:
confidence of persistent/new errors by node, spatial propagation of incorrect class
territory, relationship to local graph purity/degree and labeled origins, and sampled
strength evolution. Truth-based homophily/error analysis remains offline. Compare
global accuracy with balanced accuracy/recall, keeping their different objectives
explicit. No stopping rule is fitted/adopted by this implementation. Dominance by
class is not individual particle territory, and milestone positions are not visits.

Each new common milestone must reproduce B/C1 state hashes, assignments and
retrospective accuracy for its existing reference. Do not compare `class_changes`
or dominance-delta summaries at 1k: their preceding observation is now 500 rather
than zero, so those summaries legitimately have different interval definitions.
The preflight pins cached raw/processed/ordered-graph/labels, prior B/C1 bundles and
analysis, all runtime source/environment/native binary identities, and PR16/17
compatibility. Frozen algorithm sources and execution environment must match the
references for this pilot's exact-prefix gate.

Preflight (no downloads or fits; output directory must be new):

```powershell
$py = '.review-venv/Scripts/python.exe'
$fixed = '.cache/performance/windows-fixed-full-c9f512c-01/fixed-10000'
& $py -m experiments.mechanistic_pilot preflight --output .cache/mechanistic-pilot-preflight --fixture-cache "$fixed/fixtures" --timing-summary "$fixed/summary.json" --historical16 $fixed --historical17 .cache/performance/windows-full-early-stopping-01 --stage-b .cache/convergence-stage-b-pilot --stage-c1 .cache/convergence-stage-c1 --analysis .cache/convergence-joint-analysis
```

Review the exact report SHA before execution. Timing uses the existing PR16 rate
projection (3× maximum per-iteration rate, +10 s/fit, +60 s/process), not a measured
mechanistic overhead or guaranteed duration. Loops, complete fit, postprocessing,
export and validation should be measured separately if approved later.

After explicit approval, the commands would be:

```powershell
& $py -m experiments.mechanistic_pilot init --preflight .cache/mechanistic-pilot-preflight --root .cache/mechanistic-pilot
& $py -m experiments.mechanistic_pilot run --root .cache/mechanistic-pilot --reviewed-preflight <APPROVED_SHA> --max-runs 1
& $py -m experiments.mechanistic_pilot status --root .cache/mechanistic-pilot
```

Whole-fit resumption reuses durable locks, hash-chain ledger, checksummed plans and
results. Completed outputs are validated/skipped; valid interrupted publications
reconcile without refit. Partial fits need explicit `--retry-task`/`--reason` and
preserve old attempts. No automatic retries, no particle/RNG checkpoint resumption.
`recover-lock` retains the existing local dead-owner/token/reason checks. Use one
store per reviewed pilot, not alternate roots to repeat completed work. No real fit
is authorized by the mere existence of this protocol or preflight.

## Verification and inference limits

Synthetic tests compare full/legacy/unobserved execution within each backend,
predictions, arrays, stop iteration/reason and subsequent RNG behavior; Cython uses
its native copy-only continuation probe. Repeated traces exclude timing. Tests also
cover zero iterations, exact milestone early stop, final deduplication, exact/invalid
budgets, malformed schedules, preserved particle order/public label mapping,
interrupted publication, corruption even after rechecksumming, reader limits and
resume/duplicate prevention. No real datasets are used in CI.

Equality across different backend RNG trajectories is not required or manufactured.
Portable arrays must retain float64 values; instrumentation must not introduce
float32, renormalization, fast-math, exposure counters or native-loop changes.
Subsequent causal or stopping validation requires reserved units and separate
protocols; the two deliberately selected label seeds provide no such validation.
