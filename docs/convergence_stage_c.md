# Stage C: replication and exploratory generalization

PR #18 stays draft and unmerged. Infrastructure/preflight only until the maintainer
reviews each exact report SHA. No historical fits, downloads, warmups or production
defaults are changed. Stage B and PR #16/#17 inputs are read-only dependencies.

## Scope

| Part | Datasets | New label seeds | Backends | New fits | Iteration cap total |
| --- | --- | --- | --- | --- | --- |
| C1 | PageBlocks, Spambase | 12–19 | NumPy, Numba, Cython | 48 | 4,800,000 |
| C2 | Banknote, Sonar | 10, 11 | NumPy, Numba, Cython | 12 | 1,200,000 |

C1 completes the frozen seeds 10–19 by retaining Stage B's twelve existing fits;
it never schedules PageBlocks/Spambase seeds 10/11. C2 is exploratory. Both use
100,000 iterations per continuous native propagation, early stopping disabled,
and milestones 0, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000, 100000.
The final milestone is deduplicated (ten observations). k_nn=10, three labels per
class, p_grd=.5, delta_v=.1, deltap=1, dexp=2, es_chk=2000 remain frozen.
NumPy/Numba seed 0 once per fit, Cython native fixed seed 123456789. Backend RNG
trajectories are not assumed identical. Truth is retrospective evaluation only.

## Audit and preflight

`experiments.convergence_stage_c` reuses the Stage B transaction/observation path
without modifying its source. An offline, explicitly read-only adapter audits the
completed Stage B independently of today's Git commit/environment: it validates
the old pinned plan/report, sealed fixtures, all bundles and ledger, and independently
recalculates accuracy, class changes and label preservation for 120 observations.
It also validates the analysis manifest and its binding to the twelve source
bundles. This is provenance validation, not permission to resume old runs using
a different environment. The adapter refuses execution.

Each new preflight seals its own fixture copies, Stage B store/preflight/analysis
inventories, historical compatibility inputs, new source hashes, native binary,
host/environment and exact specification. Real preparation requires the existing
fixture cache, fixed-10k timing summary, and PR #16/#17 local results. All raw,
processed, ordered-graph and seed-label fingerprints must match the frozen inventory.
All six historical repetitions must have matching fixtures/parameters/effective
seeds. PR #17 results are checked against their checksummed immutable plan.
This does not assert new state/RNG equality at unobserved iterations. PR #17 has
early stopping enabled and cap 500k; comparisons of endpoints remain unequal-work.
Historical source/NumPy/Python versions can differ, as documented for Stage B.

PowerShell preparation (no fits; output directories must be new):

```powershell
$py = '.review-venv/Scripts/python.exe'
$fixed = '.cache/performance/windows-fixed-full-c9f512c-01/fixed-10000'
$early = '.cache/performance/windows-full-early-stopping-01'
$common = @('--fixture-cache', "$fixed/fixtures", '--timing-summary', "$fixed/summary.json", '--historical16', $fixed, '--historical17', $early, '--stage-b', '.cache/convergence-stage-b-pilot', '--analysis', '.cache/convergence-stage-b-analysis')
& $py -m experiments.convergence_stage_c preflight --part c1 --output .cache/convergence-stage-c1-preflight @common
& $py -m experiments.convergence_stage_c preflight --part c2 --output .cache/convergence-stage-c2-preflight @common
```

Review the envelope `sha256` and payload in each `report.json`. Reports include
graph dimensions/degrees, particles, fixture and source hashes, per-backend native
snapshot and estimated peak bytes, observation counts, accumulated export limit,
iteration/particle-update budgets and wall-clock projections. Per run: at most 128
observations and 64 MiB estimated diagnostic storage. C1 export ceilings sum to
3 GiB, C2 to 768 MiB, excluding fixture copies/interrupted attempts. These ceilings
are conservative budgets, not predictions of actual serialized size or RSS.

Timing is a planning scenario: three times PR #16's maximum fixed-10k per-iteration
rate extrapolated to 100k, plus 10 seconds/run and 60 seconds/process. It is not a
new benchmark, upper runtime guarantee or timeout. Full fixture/dependency audits,
JIT, changed versions/hardware/load, propagation state and repeated process startup
may exceed it. Reports pin their timing input and recompute the projection on resume.

## Commands after explicit approval of each report

Do not substitute the report SHA merely because it exists: supplying it attests
that the operator reviewed and approved that specific report. C1 and C2 approval
are separate. Use one store per part; do not clone/reinitialize a campaign in another
root to repeat work. Deduplication is within a store, with fixed disjoint case sets
across C1/C2/Stage B; it is not a machine-wide registry of arbitrary copied stores.

```powershell
& $py -m experiments.convergence_stage_c init --preflight .cache/convergence-stage-c1-preflight --root .cache/convergence-stage-c1
& $py -m experiments.convergence_stage_c run --root .cache/convergence-stage-c1 --reviewed-preflight <APPROVED_C1_SHA>
& $py -m experiments.convergence_stage_c status --root .cache/convergence-stage-c1

& $py -m experiments.convergence_stage_c init --preflight .cache/convergence-stage-c2-preflight --root .cache/convergence-stage-c2
& $py -m experiments.convergence_stage_c run --root .cache/convergence-stage-c2 --reviewed-preflight <APPROVED_C2_SHA>
& $py -m experiments.convergence_stage_c status --root .cache/convergence-stage-c2
```

`--max-runs N` bounds how many additional independent fits the invocation launches.
C1 internally uses transactions of at most twelve fits while continuing each fit's
native loop without milestone restarts. The default finishes the part's remaining
tasks. Completed bundles are checked and skipped; bundles interrupted during
publication reconcile without refits. Partially executed fits require explicit
`--retry-task tNNNN --reason '...'`, preserving old artifacts and restarting the
whole fit with its original seed. No automatic retries or particle/RNG checkpoints.
Stale local locks use `recover-lock --root ... --token ... --reason ...`, with the
same dead-owner checks as Stage B. Unknown outputs, corruption, changed dependencies,
duplicate completion or metadata/environment drift stop execution.

Stage B/analysis dependencies must remain unchanged and available at their sealed
paths during C runs. Do not regenerate its analysis while C is executing. New code
commits/environment changes require fresh preflights and approval; they do not
authorize migration or refitting completed campaigns. Portable CSV/JSON bundles
remain readable independently of execution environment. Output roots cannot overlap
protected input roots. All generated results remain outside tracked source files.

## Interpretation limits

The additional frozen seeds already have examined historical endpoints. Stage C is
descriptive trajectory replication/generalization, not blind confirmatory validation
of a stopping indicator. Do not infer correctness from rising dominance margins or
state stability. Any new dense schedule, propagation seeds, stopping rule or reserved
validation set needs a separate protocol and preflight; C1 preserves Stage B's exact
schedule for comparability. Banknote/Sonar use that same schedule/cap initially.
