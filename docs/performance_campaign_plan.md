# Proposed Python performance campaign (PR #16, phase 1)

Status: dataset selection and experimental design approved; preparation refinements
and two pilot stages prepared for review. Performance measurements remain unexecuted.
Full campaigns and README result updates still require separate approval.
No campaign measurements are supplied by this PR. Historical Python and
MATLAB/MEX results, scientific manifests, protocols and release assets remain intact.

## Frozen 20-dataset inventory

Dimensions are after the original preprocessing, with no subsampling. OpenML
versions are all 1. The names below identify the original datasets; CLI aliases
are spelling conveniences, not replacements. In particular, `diabetes` is
OpenML Pima Indians diabetes (37), not scikit-learn's regression dataset.

| Historical name | CLI name | Source | Nodes | Processed features | Classes |
| --- | --- | --- | ---: | ---: | ---: |
| Wine | `wine` | load_wine | 178 | 13 | 3 |
| Digits | `digits` | load_digits | 1797 | 64 | 10 |
| Iris | `iris` | load_iris | 150 | 4 | 3 |
| BreastCancer | `breast_cancer` | load_breast_cancer | 569 | 30 | 2 |
| BalanceScale | `balancescale` | OpenML 11, v1 | 625 | 4 | 3 |
| GermanCredit | `germancredit` | OpenML 31, v1 | 1000 | 61 | 2 |
| Diabetes | `diabetes` | OpenML 37, v1 | 768 | 8 | 2 |
| Sonar | `sonar` | OpenML 40, v1 | 208 | 60 | 2 |
| Glass | `glass` | OpenML 41, v1 | 214 | 9 | 6 |
| Vehicle | `vehicle` | OpenML 54, v1 | 846 | 18 | 4 |
| Ionosphere | `ionosphere` | OpenML 59, v1 | 351 | 34 | 2 |
| PenDigits | `pendigits` | OpenML 32, v1 | 10992 | 16 | 10 |
| Satimage | `satimage` | OpenML 182, v1 | 6430 | 36 | 6 |
| Spambase | `spambase` | OpenML 44, v1 | 4601 | 57 | 2 |
| PageBlocks | `pageblocks` | OpenML 30, v1 | 5473 | 10 | 5 |
| Banknote | `banknote` | OpenML 1462, v1 | 1372 | 4 | 2 |
| BloodTransfusion | `bloodtransfusion` | OpenML 1464, v1 | 748 | 4 | 2 |
| Yeast | `yeast` | OpenML 181, v1 | 1484 | 8 | 10 |
| Segment | `segment` | OpenML 36, v1 | 2310 | 19 | 7 |
| QSARBiodeg | `qsarbiodeg` | OpenML 1494, v1 | 1055 | 41 | 2 |

## Matrix for review

Use three labels per class, `k_nn=10`, full transductive scaling, and the original
PCC parameters (`p_grd=0.5`, `delta_v=0.1`, `deltap=1`, `dexp=2`, `es_chk=2000`).
All three sequential backends use the same saved dataset, graph and labels.
Six repetitions give two balanced backend-order rotation cycles per fixture.
Use PR #8 label seeds 10-19 as ten distinct fixtures, summarized separately.
Propagation seed 0 remains the PR #15 setting; Cython retains its fixed internal
seed. Seeds do not imply matching cross-backend trajectories.

Recommended stages, each requiring review of the preceding cost assessment:

| Stage | Datasets / label seeds | Mode and budget | Measurements | Warm-ups | Measured iteration cap | Particle-update cap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Pilot stage 1 | Wine, Digits, GermanCredit; seeds 10, 11 | fixed 100 | 108 | 36 | 10,800 | 162,000 |
| Pilot stage 2 | same datasets and seeds | fixed 1,000 | 108 | 36 | 108,000 | 1,620,000 |
| Primary fixed campaign | all 20, seeds 10-19 | fixed 10,000 | 3,600 | 1,200 | 36,000,000 | 459,000,000 |
| Initial stopping campaign | all 20, seeds 10-19 | early stopping, cap 100,000 | 3,600 | 1,200 | 360,000,000 | 4,590,000,000 |
| Optional PR #8 cap comparison | all 20, seeds 10-19 | early stopping, cap 500,000 | 3,600 | 1,200 | 1,800,000,000 | 22,950,000,000 |

Do not run every row automatically. Start with approved small pilots; an
additional cached large-data pilot (for example PenDigits at 100 iterations)
needs explicit review even at this low budget. Stage 1 checks correctness, reset state and timing boundaries. Stage 2 repeats
the exact same fixtures at 1,000 iterations for more useful throughput estimates.
Each stage includes six repetitions and all three backends, with balanced order. Use 10,000 fixed iterations as the
initial propagation comparison budget, rather than copying the expensive
500,000 uninterrupted research trajectories. For stopping, start at 100,000
and report actual cap fractions. Review a separate 500,000-cap run if censoring
prevents interpretation; never silently combine budgets or retune stopping.
Each mode/budget belongs to a new output directory and remains separately summarized.

MNIST 5,000 and 10,000 are excluded from every historical-group row. Plan them
separately, with seed 10, six repetitions and 100 fixed iterations initially:
36 measurements, 12 warm-ups, 3,600 capped iterations and 108,000 particle
updates. Downloads and execution require separate explicit opt-in. Their data
seed 0 selects deterministic nested subsets; they do not reproduce PR #8 data.

## Computational cost and estimation limits

These counts are reproducible workload estimates, not measured performance.
The combined fixed and initial stopping campaigns plan 7,200 measured fits,
2,400 warm-ups, at most 396 million measured propagation iterations and
5.049 billion particle updates. Every warm-up uses two iterations; the two
campaigns add 4,800 warm-up iterations and 61,200 particle updates. Early
stopping may reduce work substantially; no savings are assumed here.

The 20 datasets have 85 classes in total. A single graph preparation per dataset
has a combined `N^2 * processed_features` proxy of 5,406,433,612; this is not an
operation count or an allocated dense distance matrix. The current harness
now prepares each dataset/graph once for its consecutive label seeds: the full
ten-seed command builds 20 graphs rather than 200. Separate modes and pilot
stages are separate processes and repeat preparation. The in-memory cache holds
only the current dataset, invalidates on data seed, dimensions, graph k-NN or
source cache path changes, and never reuses particle/backend state. Saved exact
fixtures allow cross-backend verification but are not a persistent disk cache.

The largest uint8 particle distance table is PenDigits at 329,760 bytes; this
excludes graph storage, backend buffers, dominance, feature arrays and source
parsing. Dry-run also reports raw+scaled float64 feature storage and dominance
storage per case, not peak RAM. Frozen OpenML raw files total 4,961,182 bytes,
excluding metadata and parsing overhead. MNIST preparation uses a much larger
full 70,000-row source and is independently opt-in.

Wall-clock estimates are unavailable until approved pilots run on the target
machine. After a pilot, use per-fixture/backend median `propagation_s_per_iteration`
to estimate budget times six repetitions; add median initialization/postprocessing,
actual warm-up costs and preparation timings. Compare Q1/Q3 throughput estimates
as a range, not a confidence interval. Extrapolation across budgets is tentative:
greedy branch costs, state distributions, CPU behavior and stopping can change.
Do not extrapolate one dataset's rate to all 20. Untested datasets/backends remain
unestimated. Keep graph preparation and propagation estimates separate and retain
actual measurements beside any later estimates. No estimated values belong in
README performance tables.

## Read-only plans and future pilot commands

All commands here are dry-runs; they load no datasets, build no graphs, compile
no kernels and make no network requests:

```console
python -m benchmarks.run --pilot-stage 100 --dry-run
python -m benchmarks.run --pilot-stage 1000 --dry-run
python -m benchmarks.run --historical --label-seeds 10 11 12 13 14 15 16 17 18 19 --iterations 10000 --dry-run
python -m benchmarks.run --historical --label-seeds 10 11 12 13 14 15 16 17 18 19 --mode early_stopping --iterations 100000 --dry-run
python -m benchmarks.run --datasets mnist_5000 mnist_10000 --label-seeds 10 --iterations 100 --dry-run
```

After approval, supply the verified cache, replace `--dry-run` with `--execute`
and a new `--output` directory. Expensive plans also require `--allow-expensive`.
The default action remains dry-run. Cache preparation is a distinct explicit
command documented in [the harness guide](performance_benchmarking.md).
Missing historical data fail; the 20-dataset group is never silently reduced.

## Differences from the PR #8 protocol

Dataset identities, dimensions, preprocessing, 3/class labels, sorted class
selection, seeds 10-19, graph construction and PCC parameters are retained.
The group shorthand uses seed 10 for planning convenience; the complete matrix
explicitly selects all ten historical seeds. Ordinary PR #15 CLI defaults stay
unchanged. Raw, processed-data, ordered-graph and primary historical label fingerprints
are checked exactly against frozen provenance and mismatches fail. A custom
`k_nn` has no applicable historical graph reference and is explicitly marked as
such. There is no mismatch override: investigate data, library versions and
ordering before measuring on another environment. This validates shared inputs,
not equality of backend floating-point trajectories.

PR #8 studied uninterrupted 500,000-iteration Numba trajectories with observer
sampling, hypothetical patience multipliers and scientific accuracy outcomes.
This campaign times the unmodified production NumPy/Numba/Cython kernels, with
fixed budgets or actual original early stopping, without scientific observers,
trajectory archives or checkpoints. It uses six reset timing repetitions per
fixture rather than treating repetitions as independent trajectories. PR #15
warm-ups are two fresh two-iteration calls per backend/case/mode, versus PR #8's
101-iteration feasibility warm-up. The proposed 100,000 stopping cap is shorter
than PR #8; the optional 500,000 comparison is kept separate. No patience
multipliers, retrospective best budgets or newly independent seed validation
are implied. RNG policies differ from the historical study and are recorded.

## Phase 1 validation

Automated tests use synthetic ARFF bytes and small built-ins: exact inventory,
pinned versions, dimensions, historical labels, nominal missing-category encoding,
numeric median imputation, scaling, deterministic fixtures, cache failures,
network prohibition, historical-only selection and workload arithmetic. Existing
small backend tests exercise sequential invariants and both stopping modes.
No external data downloads or full benchmark campaigns run in CI.

Local cache-only inspection verified all 20 raw, processed-data and ordered
graph fingerprints, dimensions and class identities against the frozen inventory.
No mismatch occurred: NumPy 2.4.6, SciPy 1.18.1 and scikit-learn 1.9.1 match the
frozen reference environment. Exact ordered graph matches also exclude neighbor
ordering differences in this audit. The report is an ignored local artifact at
`.cache/performance/fingerprint-audit-refined.json`; regenerate it on the target
machine with the audit command below. This is dataset
validation, not propagation timing or a repeated historical trajectory audit.
The repository's Windows/Linux, Python 3.10/3.12 CI remains the required final
check; inspect the exact final commit before claiming completion.


## Preparation reuse and timing report

For consecutive label seeds, the first fixture keeps the original loading,
preprocessing-plus-labeling, and graph timing boundaries. Later fixtures use the
same read-only source/scaled data, truth, scaler arrays and ordered graph with a
fresh deterministic label mask. Their `dataset_load_s` and `graph_build_s` are
zero because that work was not repeated; `preprocessing_and_labels_s` measures
only their label selection. `preparation.source_case_id` references the first
fixture. Array hashes and fixture hashes remain exactly equal to independent
preparation for every seed. Per-fixture NPZ files are still self-contained.

`preparation.json` reports per-fixture charges and totals, including
`dataset_preparation_s = dataset_load_s + preprocessing_and_labels_s`, with graph
construction in `graph_build_s` separately. The historical loader still includes
encoding, imputation and reference scaling in loading; the saved fixture's scaler
remains in the original preprocessing boundary. Reuse does not move preparation
into any fit timer, duplicate its cost across seeds, or substitute lookup time
for graph construction. Save/hash/cache bookkeeping remains outside those timers.

`summary.json` separately reports median/IQR initialization and dispatch,
propagation, postprocessing, total fit and time per actual iteration, per
fixture/backend/mode/budget. Initialization continues to include dispatch, as
in PR #15; no new probes split the production implementation. Preparation is
charged once per dataset/group plus per-seed label generation, not per timing
repetition or backend. Do not add the same preparation total to every backend's
fit time when estimating campaign duration.

## Target machine preflight and pilot stages

Activate the intended benchmarking environment first and verify all three
backends are installed and Cython is compiled. A Windows Conda environment needs
its activated `Library/bin` DLL search path: the initial unactivated local shell
failed in threadpoolctl with Windows error `0xc06d007f` when graph construction
selected a brute-force distance kernel. Supplying the environment's normal DLL
path resolved the error; no library, neighbor algorithm, thread policy or data
was changed. The complete audit then matched all frozen fingerprints.

Validate cached inputs without downloads, propagation or backend compilation:

```console
python -m benchmarks.validate_inventory --historical-cache .cache/stopping_patience/sources --output .cache/performance/target-input-audit-01.json
```

This deliberately constructs one graph per dataset. The optional report file
must be new and its parent must exist; failures are recorded and the command
exits unsuccessfully. It includes actual/expected raw, processed and graph
fingerprints and current/frozen dependency versions. A mismatch or environment
error must be resolved before execution; do not infer a cause from the hash alone.

After review, the two pilot stages are ready to execute separately on the target
machine. Each preset fixes the three representative datasets, label seeds 10 and
11, three labels/class, `k_nn=10`, all three backends, six repetitions and fixed
mode. Conflicting preset options fail; ordinary CLI remains available for custom
plans. Output directories must be new:

```console
python -m benchmarks.run --pilot-stage 100 --historical-cache .cache/stopping_patience/sources --execute --output .cache/performance/pilot-100-01
python -m benchmarks.run --pilot-stage 1000 --historical-cache .cache/stopping_patience/sources --execute --output .cache/performance/pilot-1000-01
```

These are future execution commands, not runs completed in this PR. Review
stage 1 correctness, repeatable states and timing separation before executing
stage 2. Together they plan 216 measurements, 72 warm-ups, 118,800 measured
iterations and 1,782,000 particle updates. Neither preset downloads data nor
requires the conservative expensive-work opt-in. Wine covers a small numerical
dataset, Digits a higher-dimensional 10-class dataset, and GermanCredit a mixed
categorical source. They do not estimate PenDigits or all other datasets reliably.
The full 20-dataset propagation campaign and README updates remain unexecuted.


## Approved targeted permutation follow-up

Wine, Digits, Satimage and PageBlocks; seeds 10 and 11; all three backends;
10,000 fixed iterations; three labels/class and unchanged PCC parameters.
Three fresh consecutive sessions each use six repetitions covering all six
backend permutations exactly once per fixture. Explicit `--order-policy
permutations`, recorded `--order-seed` and `--session-id` distinguish this from
the original campaign's unchanged default rotation. There are 144 measured
fits/session (432 total), 48 warm-ups/session, and four graph preparations/session.

The maintainer explicitly replaced the distinct-period requirement with
consecutive automatic sessions without artificial delays. Actual start/end,
inter-session gaps and process/environment provenance must be reported; the
results cannot demonstrate stability across different machine-use periods or
create independent observations simply by labeling them sessions. Preserve
all records and the original 3,600-fit campaign. Compare native paired ratios
by fixture, session, permutation, direction, and NumPy position, especially
Satimage and PageBlocks. No early-stopping campaign or README table change is
included in this authorization.
