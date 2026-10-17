# Stage B: bounded convergence pilot

Infrastructure only. The real pilot must wait for review of its exact preflight
report. PR #18 remains draft and unmerged. No historical propagation is run by
`preflight`, `init` or `status`; these commands never download data.

## Fixed specification

- PageBlocks and Spambase; label seeds 10 and 11.
- NumPy, Numba and Cython, one completed fit per combination (12 fits).
- Maximum 100,000 iterations, early stopping disabled, no real-data warmup fits.
- Milestones: 0, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000,
  100000. The final milestone is deduplicated, giving ten observations per fit.
- Frozen parameters: p_grd=0.5, delta_v=0.1, deltap=1, dexp=2, es_chk=2000,
  k_nn=10 and three labels per class. Parameters come from the existing frozen
  protocol. Label selection uses its existing default_rng policy; these seeds
  were previously examined and are not independent validation.
- NumPy/Numba propagation seed 0, once before each continuous fit; Cython's
  existing fixed seed 123456789. No resets or independent fits at milestones.
- 128 observations and 64 MiB estimated diagnostic memory/export storage per
  fit. Full schedule/resource preflight is mandatory; no silent thinning.

## Preflight and review

Use the same Python environment and built native extension as Stage A. The
repository's original historical fixture adapter, preprocessing and ordered graph
construction are reused. `--fixture-cache` reads existing fixture directories
without changing them; `--historical-cache` instead prepares from already-cached
raw sources, checks the frozen preprocessing/graph hashes, and never downloads.
Raw, processed, ordered-graph and seed-specific label fingerprints must match the
frozen inventory. There is no mismatch override.

For the existing verified local PR #16 fixture cache:

```powershell
python -m experiments.convergence_pilot preflight --fixture-cache .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000/fixtures --timing-summary .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000/summary.json --output .cache/convergence-stage-b-preflight
```

Alternatively, replace `--fixture-cache ...` with
`--historical-cache .cache/stopping_patience/sources` to rebuild using the frozen
preparation adapter. Preflight requires a new output directory and creates
independently sealed copies of the four fixtures plus `report.json`. Existing
caches, benchmark runners, archives and Release assets are read-only inputs.
Outputs inside this checkout must be under `.cache`; an external output directory
can also be specified. Generated reports, fixtures and trajectories are not
committed to source control.

Review the report envelope's `sha256` and `payload`. It reports:

- Graph shapes/degrees, node/features/class/particle counts and ten observations.
- Native buffer bytes, conservative diagnostic peak estimates, fixture array
  bytes and whether each backend fits the limits. A reduced three-milestone
  suggestion is emitted only if the proposed schedule exceeds a limit; execution
  refuses a non-fitting preflight rather than applying that suggestion silently.
- Twelve tasks, 1.2 million iteration cap, particle-update cap, no execution begun.
- Historical timing input path/hash and a per-task cost projection. Rates must
  match the exact seed-specific fixture and fixed 10,000-iteration PR #16 groups.
  The planning scenario is three times each historical maximum per-iteration rate
  at 100,000 steps, plus ten seconds per fit and sixty seconds per process startup.
  It is not a measured pilot, a performance comparison or a hard timeout.

Source/dependency/hardware differences, graph-state timing changes, audits, JIT,
startup and export can exceed those allowances. Additional resumed processes can
repeat compilation/startup costs. Snapshot estimates exclude libraries, allocator
overhead, resident fixtures and JIT from process RSS. Bundles accumulate on disk;
the per-fit export limit permits up to 768 MiB for twelve completed bundles, plus
fixture copies and retained interrupted attempts.

Preflight pins code commit/source hashes, frozen manifests, dependency versions,
LLVM dependency, host/processor/thread environment and native binary checksum.
Regenerate into a new directory if these change before review. Resume requires
the original verified preflight directory at its original path and the same
execution identity. CSV/JSON bundles remain portable independent of that store.

## Execution commands (only after preflight review)

`init` creates a new pilot store but runs no fits. `run` requires the exact
reviewed report SHA explicitly on every invocation; a digest is an operator
attestation of review, not automatic approval or a signature. The agent preparing
this infrastructure does not invoke real `run` before review.

```powershell
python -m experiments.convergence_pilot init --preflight .cache/convergence-stage-b-preflight --root .cache/convergence-stage-b-pilot
python -m experiments.convergence_pilot status --root .cache/convergence-stage-b-pilot
python -m experiments.convergence_pilot run --root .cache/convergence-stage-b-pilot --reviewed-preflight REVIEWED_SHA256 --max-runs 1
```

After inspecting that first result, resume with the same command and a suitable
`--max-runs` (default 12). Counts are bounded to 0..12 launches per invocation;
already completed tasks never repeat. A full uninterrupted pilot has exactly one
fit for each of its twelve combinations. No historical stopping campaign or
ordinary benchmark is invoked. Results use the Stage A v2 CSV/JSON schema.

## Resume, interruption and corruption

The store reuses existing exclusive-lock, fsynced record, checksum and hash-chain
primitives without modifying the benchmark controller. It identifies a pilot
store read-only before taking a writer lock, refusing older benchmark campaigns.
The immutable plan binds the reviewed preflight and all twelve task IDs.

Resumption is **between complete fits**, not within a trajectory. An intent is
durably recorded before each fit. Complete bundles are validated/fsynced before
publishing an exclusive result record, then a completion event. On restart:

- Valid completed results are revalidated and skipped.
- A valid bundle left before result/completion publication is reconciled without
  another fit. A published result missing its completion event is also reconciled.
- A fit interrupted before a valid manifest needs explicit retry authorization;
  it restarts that entire fit with the original seed and retains the prior attempt.
  Retries add work beyond the nominal budget and are never automatic.
- Invalid checksums/schema/provenance, frozen fixture drift, duplicate completion
  or unknown records stop execution. Corruption is not interpreted as permission
  to rerun. Interrupted artifacts are fingerprinted when retry is authorized and
  remain immutable afterward.

```powershell
python -m experiments.convergence_pilot run --root .cache/convergence-stage-b-pilot --reviewed-preflight REVIEWED_SHA256 --retry-task t0000 --reason "Reviewed interruption before complete bundle" --max-runs 1
```

A process kill can leave `writer.lock`. Inspect its envelope, then use the exact
owner token and reason. Recovery refuses live owners, other hosts, or an
unprovably inactive process; never delete the lock manually.

```powershell
python -m experiments.convergence_pilot recover-lock --root .cache/convergence-stage-b-pilot --token DEAD_OWNER_TOKEN --reason "Verified local owner exited"
```

## Synthetic validation and limits

```powershell
python -m experiments.convergence_pilot preflight --synthetic --iterations 7 --output .cache/stage-b-synthetic-preflight
python -m experiments.convergence_pilot init --preflight .cache/stage-b-synthetic-preflight --root .cache/stage-b-synthetic-run
python -m experiments.convergence_pilot run --root .cache/stage-b-synthetic-run
python -m unittest tests.test_convergence_pilot -v
python -m unittest discover -s tests -v
```

Synthetic mode uses two tiny blob datasets and both seeds/all three backends.
Tests cover all twelve combinations, read-only dry-run, immutable specifications,
resume/skip/duplicate prevention, corruption and rehashed graph-pin rejection,
old-store protection, review/environment gates, interrupted fits with retained
partial output, reconciliation at bundle/result/completion boundaries, and a real
child process exit with stale-lock recovery. CI loads no historical data.

Stage A limits remain: changes reversing between observations are invisible,
native intermediate wall times and Numba RNG-state fingerprints are unavailable,
and resource estimates are not OS memory or runtime guarantees. The pilot provides
descriptive trajectories, not a new stopping rule, backend speed claim or broad
scientific replication. Real execution remains pending review.
