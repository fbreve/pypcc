# Reproducible Python performance benchmarking

The optional `benchmarks/` harness compares sequential NumPy, Numba and Cython.
It does not change production modules, public APIs, stopping rules, defaults,
package versions or frozen scientific experiments. No new performance claims
or replacement README numbers accompany this infrastructure.

## Installation and quick start

Use a checkout and a separate environment as described in the README:

```console
python -m pip install -e ".[numba]"
python -m benchmarks.run --dry-run
```

Cython must already be compiled. There is no build or backend fallback during
benchmarking. Requesting an unavailable backend fails explicitly. Individual
backends can be selected with `--backends numpy`, for example.
Run commands from the repository root with normal Python, not `python -O`.
The harness is a repository tool and is not installed as part of the runtime
package. No new runtime dependency is required.

The default action is a dry-run. Execution requires `--execute` and a **new**
output directory. All examples write ignored local results under `.cache/`.

Small validation (six repetitions, five iterations, all three backends):

```console
python -m benchmarks.run --datasets iris --iterations 5 --repeats 6 --dry-run
python -m benchmarks.run --datasets iris --iterations 5 --repeats 6 --execute --output .cache/performance/small-01
```

Medium fixed-budget experiment (review the plan before executing):

```console
python -m benchmarks.run --datasets wine digits --iterations 1000 --dry-run
python -m benchmarks.run --datasets wine digits --iterations 1000 --execute --output .cache/performance/medium-01
```

Early-stopping experiment with the original criterion and a configurable cap:

```console
python -m benchmarks.run --datasets wine --mode early_stopping --iterations 30000 --dry-run
python -m benchmarks.run --datasets wine --mode early_stopping --iterations 30000 --execute --output .cache/performance/stopping-01
```

These commands describe future experiments; only small validation is part of
this infrastructure change. A low cap can intentionally produce entirely
cap-limited early-stopping results.

## Datasets and preprocessing

| CLI name | Samples | Features | Classes | Source |
| --- | ---: | ---: | ---: | --- |
| `iris` | 150 | 4 | 3 | scikit-learn built-in |
| `wine` | 178 | 13 | 3 | scikit-learn built-in |
| `breast_cancer` | 569 | 30 | 2 | scikit-learn built-in Breast Cancer Wisconsin |
| `digits` | 1797 | 64 | 10 | scikit-learn built-in |
| `mnist_5000` | 5000 | 784 | 10 | deterministic subset of OpenML `mnist_784`, version 1 |
| `mnist_10000` | 10000 | 784 | 10 | same cached full source |

The default selects the four built-in datasets, 100 iterations and six
repetitions per case/backend. Loading and conversion to float64 are recorded
separately from preprocessing. `StandardScaler` fits the **complete selected
transductive dataset**, including unlabeled samples. Constant columns remain
in place. There is no imputation, row removal or feature selection.

Each case builds one symmetric Euclidean graph with the unchanged production
builder, `k_nn=10`, and graph construction `n_jobs=1`. The exact ordered neighbor
lists, degrees, raw/scaled features, truth and label mask are saved once and
shared across all backends. Propagation is sequential; synchronous NumPy is
an algorithmically different experiment and is excluded.

By default, `default_rng(label_seed)` chooses three nodes without replacement
per sorted class. Every class is represented and initially has unlabeled
samples. `--label-seeds 0 1 2` makes distinct cases, each summarized separately.
`--labels-per-class` changes the particle count. `--labeled-fraction` instead
selects one node per class, then uniformly selects remaining nodes without
replacement to reach `max(classes, floor(nodes * fraction))`. The fraction
policy guarantees labeled class coverage, but does not guarantee unlabeled
coverage of every class. Policies that label every node are rejected.

### Explicit MNIST preparation

Only this command downloads data; regular execution and tests never do:

```console
python -m benchmarks.run --prepare-mnist .cache/performance/mnist-source
```

It creates a new cache directory, downloads OpenML `mnist_784` version 1,
and stores the complete 70,000-row numeric dataset as `mnist.npz` with a
`source.json` containing source/version, OpenML ID/URL and file SHA-256.
Preparation may require substantial download time and memory. Both subsets
reuse this cache. Selection uses prefixes of a full `default_rng(data_seed)`
permutation, so the 5000 subset is contained in the 10000 subset when seeds
match. `--data-seed` defaults to 0. Exact original row indices are saved.
Each subset fits its own scaler and builds its own graph.

Large configuration, explicitly opt-in:

```console
python -m benchmarks.run --datasets mnist_5000 mnist_10000 --mnist-cache .cache/performance/mnist-source --iterations 100 --dry-run
python -m benchmarks.run --datasets mnist_5000 mnist_10000 --mnist-cache .cache/performance/mnist-source --iterations 100 --allow-expensive --execute --output .cache/performance/mnist-01
```

Missing cache files fail with a preparation hint and no download. Add
`--skip-missing-optional` to record and skip unavailable MNIST cases. A corrupt
cache fails validation rather than silently changing the input. If every case
is skipped, completion explicitly reports `no_measurements`.

### Synthetic scalability

`--synthetic` adds these deterministic `make_blobs` cases:

| Name | Nodes | Features | Classes | Label policy | k-NN |
| --- | ---: | ---: | ---: | --- | ---: |
| `synthetic_small` | 150 | 4 | 3 | 3/class | 10 |
| `synthetic_medium` | 500 | 16 | 5 | 3/class | 10 |
| `synthetic_particles` | 1000 | 32 | 10 | fraction 0.1 | 10 |

Synthetic presets use generation seed 0, cluster SD 1, center box [-10, 10]
and shuffled samples. Custom cases use `--synthetic-config cases.json`:

```json
[
  {"name": "nodes2000", "nodes": 2000, "features": 16, "classes": 5,
   "k_nn": 10, "labels_per_class": 3, "data_seed": 42},
  {"name": "particles2000", "nodes": 2000, "features": 16, "classes": 5,
   "k_nn": 20, "labeled_fraction": 0.1, "data_seed": 42}
]
```

Use `--datasets` with no following names to select only synthetic cases:

```console
python -m benchmarks.run --datasets --synthetic-config cases.json --iterations 100 --dry-run
```

Set dimensions, class count, label count/fraction and neighborhood size
independently through custom cases. Use identical generation seeds and vary
one dimension at a time for interpretable scalability experiments. CLI graph
and labeling options apply to real datasets; synthetic case JSON/presets
explicitly carry their own parameters. Label seeds apply to both.

## Modes, randomness and repetitions

Fixed iteration is the primary performance comparison: every backend performs
exactly the supplied budget with `early_stop=False`. Early stopping uses the
existing criterion with `es_chk=2000`; no counter, check phase or comparison
is changed. All runs use `p_grd=0.5`, `delta_v=0.1`, `deltap=1`, `dexp=2`.
`set_graph` explicitly receives the construction `k_nn` so patience does not
silently use maximum graph degree. Parameters are saved in the manifest.

Each repetition creates a new model and fresh particle, dominance, distance
and buffer state. Seed reset is outside the fit timer. NumPy uses its global
RandomState, Numba uses its independent NumPy RNG seeded through a small
`njit` helper, and Cython retains its internal fixed xorshift32 seed
`123456789`. `--propagation-seed` controls supported generators (default 0).
Matching numbers do not imply matching draws or trajectories. Repetitions
within one backend must reproduce state hashes and stopping metrics; no
cross-backend bitwise-equality requirement is imposed.

Backend order rotates by case index and repetition. Six repetitions are
default (`--repeats`), giving two complete three-repetition rotation cycles
for NumPy, Numba and Cython. Each backend occupies every execution position
exactly twice per case. Custom repetition counts remain supported, but counts
that are not multiples of the selected backend count need not balance positions.
Summaries are
per fixture/backend/mode/budget, with median, Q1, Q3, IQR, minimum and maximum.
Quartiles use NumPy's `method="linear"`. There is no across-dataset pooling.
Timing variation within a case is distinct from variation across label masks,
datasets and machines. Keep different environments in separate result sets.

Early-stopping records include effective iterations, check count, stalled
checks, stopping reason, cap reached, and whether termination occurred before
the cap. A stopping event on the final iteration can have reason `early_stop`
and `reached_max_iter=True` simultaneously. Reports retain this distinction.
Cap fraction is computed from actual counts. Time per iteration is propagation
time divided by actual iterations; changing state/branch distributions means
this is descriptive throughput, not a convergence-independent speed guarantee.

## Timing and compilation

All execution times use `time.perf_counter`. Boundaries are:

| Field | Scope |
| --- | --- |
| `dataset_load_s` | Load/verify source, select subset, convert arrays |
| `preprocessing_and_labels_s` | Full-data scaling and label mask preparation |
| `graph_build_s` | Production graph construction |
| `initialization_and_dispatch_s` | Fit entry to propagation entry: label mapping, state/buffer allocation, dispatch |
| `propagation_s` | Existing backend call, including validation and final metric packaging |
| `postprocessing_s` | Propagation exit to fit return: predictions, mapping and own-degree normalization |
| `fit_predict_s` | Complete fit, excluding separately prepared graph |

Model construction, graph copying, backend identity verification, RNG seeding,
state hashing, invariant checks, output writing and progress printing are
outside fit timing. Initialization cannot be isolated further without more
intrusive instrumentation, so it is explicitly reported with dispatch.
Data preparation is measured once per fixture, not repeatedly.

The harness wraps only the instance's backend-dispatch boundary and requests
existing `return_info=True` metrics. It never wraps individual steps or
recomputes stopping statistics. The metric packaging and a constant amount
of boundary timer overhead remain included, especially relevant for tiny runs.

Two fresh two-iteration fits warm each case/backend/mode before measurement.
The first and warm calls are saved separately. First-call latency can include
Numba compilation, validation, loading and execution; it is not a compiler-only
measurement. The RNG seed helper's first-call latency is recorded separately.
The `first` label means the first warm-up of that case, not necessarily a cold
process invocation: later cases can reuse compiled signatures and loaded code.
Measured Numba fits must retain the already compiled signature set. Native
Cython is built before execution, never inside a measured fit.

`--build-info build.json` accepts a user-supplied record such as:

```json
{"compiler": "MSVC 14.44", "flags": ["/O2"],
 "command": "python -m pip install -e .[numba]",
 "build_log_sha256": "<actual recorded build log hash>"}
```

Preserve the actual installation log separately. Build provenance is marked
unavailable unless supplied; the Python interpreter compiler string does not
identify the extension compiler. Existing platform-specific build flags remain
unchanged. No compilation time is attributed to steady-state performance.

## Output and environment schema

Every output directory belongs to one run and is created exclusively. Existing
directories, fixture directories and files are refused, including empty ones.
No historical result file is used as an output target.

| File | Schema version 1 contents |
| --- | --- |
| `manifest.json` | UTC timestamp, command/config, repository SHA/dirty status, source hashes, CPU/RAM/OS/interpreter, dependency versions, binary hash, thread environment/affinity, optional build record, RNG/timing/warm-up policies |
| `fixtures/<case_id>/fixture.json` | Dataset source, case dimensions/seeds, preprocessing/graph policy, class coverage, array and fixture hashes, preparation timings |
| `fixtures/<case_id>/arrays.npz` | Named `raw`, `x`, `truth`, `labels`, `neighbors`, `degrees`, `source_indices`, `scaler_mean`, `scaler_scale` arrays; load with `allow_pickle=False` |
| `measurements.jsonl` | One execution per line: case/fixture/backend identity, repetition/order, timings, seed, actual stopping metrics, accuracy and final state hashes |
| `warmups.jsonl` | Same execution fields, marked `first`/`warm`; excluded from summaries |
| `seed_helper.json` | Numba seed helper first-call latency, when used |
| `summary.json` | Per-case/backend/mode/budget timing/iteration medians and IQRs, stop-reason counts, cap fraction |
| `completion.json` | Completed execution count, fixture identities, skipped cases and status |
| `failure.json` | Failure description and completed count; prior JSONL records remain available |

Array hashes use SHA-256 over JSON shape/dtype descriptors and canonical
little-endian C-order bytes. Fixture identity hashes the sorted mapping of
named array hashes. It excludes timings and includes raw/scaled data, graph,
labels and source indices. NPZ container byte identity is not required.
Exact saved arrays are the strongest cross-version input reference; regenerating
them can differ with library releases or nearest-neighbor tie handling.

Unavailable environment metadata is null. Thread environment variables are
recorded, not overridden; graph construction uses `n_jobs=1`. CPU affinity and
power policy are not controlled. Background work, CPU frequency/temperature,
allocator/cache state and OS scheduling can affect measurements. Warm-process
measurements do not establish cold-cache performance. Synthetic scalability
characterizes these generated graphs, not every real-world topology.

## Workload planning and validation

Dry-run lists case count, backend executions, six default repetitions,
warm-ups, per-case iteration caps, total capped iterations, particle-update
caps, distance-table sizes and an `N^2 * features` graph-work upper proxy.
It performs no loading, downloads, graph construction or compilation.
These are workload bounds, **not wall-time predictions**; graph algorithms,
degree/class counts, backend branches and early stopping affect actual cost.

Execution requires `--allow-expensive` when any case exceeds 2000 nodes or
the graph-work proxy exceeds one billion, or total measured particle updates
exceed 20 million. These conservative gates do not guarantee short runs.
MNIST preparation/loading may itself require hundreds of MB and does not fit
in the distance-table estimate. Review dry-run, estimate wall time from an
approved small pilot on the same machine, and obtain approval before expensive
experiments. The harness never launches a pilot automatically.

The small Iris command above plans one case, 18 measurements, six warm-ups,
90 measured propagation iterations and 810 particle updates. The default
MNIST-10000 plan has one case, 18 measurements, six warm-ups and a 100-iteration
cap: 1800 capped measured iterations and 54000 particle updates. It requires
explicit opt-in even at that conservative budget. The two-subset MNIST example
plans 36 measurements, 12 warm-ups, 3600 capped measured iterations and 108000
particle updates. The four-dataset default plans 72 measurements, 24 warm-ups,
7200 capped measured iterations and 97200 particle updates.

```console
python -m unittest -v tests.test_performance_benchmarks
python -m unittest discover -s tests -v
```

Tests use small graphs and tiny iteration budgets. No external data download
or performance campaign runs in CI. Existing CI requires all three backends;
local optional-backend tests can omit unavailable backends, so check availability
before claiming cross-backend validation.

## Historical provenance and future MATLAB/MEX integration

README tables are historical i9-14900K/128 GB measurements with Python
3.12.12, NumPy 2.4.4, Numba 0.64.0, Cython 3.2.4 and MATLAB R2025b.
Python aggregates are preserved in `benchmark.py` and were last refreshed in
commit `bd0690d` (2026-05-10). Their source workload used full-data scaling,
10% globally selected labels (floor rounding), split seeds 0..99/0..9,
500000 iteration caps, 100 early-stopping/10 fixed-budget runs, and one full
untimed fit per backend/configuration. The timer covered `fit_predict`,
excluding graph construction. Means/population SD were printed, but individual
records and actual early-stop iteration counts were not retained. NumPy/Numba
propagation was not explicitly seeded; fallback was not rejected.

The linked MATLAB repository's `benchmark.m` at commit
`f13c4ac5c2cc925bedf9fa5983af742502243a4f` preserves matching MATLAB aggregates.
It times graph construction inside `pcc`, uses split seeds 1..N, guarantees
one labeled node/class and rounds the 10% label count (18 Wine labels versus
17 Python labels). It uses standardized Euclidean distance, removes three
zero Digits columns, and discards no warm-up. Pure MATLAB checks stopping at
iterations 10,20,...; MEX/Python check 1,11,... . MATLAB `rng(r)` does not seed
the MEX C RNG unless the separate `useseed` option is enabled. Matching graph
geometry is not proof of identical ordered neighbors or random trajectories.
Historical compiler/binary provenance and full environment details are absent.

The newer `docs/benchmark_backends.py` records use an i7-14700K and different
software, three labels/class, seeds 0..4 and 10000 fixed iterations. Convergence
follow-ups use 10000/30000 caps and actual metrics. They cannot reproduce the
README protocol. The original synchronous NumPy table timed full fits while
its current script times instrumented propagation. Scientific observer/I/O
timings are not comparable steady-state performance results. All historical
tables, manifests, protocols and artifacts remain preserved.

Future MATLAB integration can load named fixture arrays from a neutral MAT
export and convert zero-based graph indices to one-based, `-1` unlabeled to
zero, and classes to MATLAB's positive contiguous labels. Preserve degrees,
ordered neighbors and exact label masks; use the existing `Xgraph` input to
exclude graph building from fit timing. Canonical hashes must be checked after
conversion back to the neutral representation. Preserve each backend's RNG
and check phase rather than forcing equal trajectories. Effective MATLAB/MEX
iterations require an appropriate observer or untimed audit replay; they cannot
be inferred from runtime. Timing boundaries, compilation/warm-up, binary/build
provenance and index/label types must be reconciled explicitly. MATLAB/MEX
execution and any changes to those algorithms are outside this PR.
