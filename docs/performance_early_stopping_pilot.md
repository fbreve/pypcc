# Python early-stopping calibration pilot

Based on merged PR #16 (`ceb7d22`). The maintainer reviewed the cap-work estimate
and explicitly authorized the original pilot, including reasonable overhead.
The original agent-selected 30-minute gate is superseded; execution has no runtime
threshold. Existing
fixed measurements, README tables, frozen scientific artifacts and the published
Release remain unchanged.

The requested matrix is Wine, Digits, GermanCredit and Satimage, label seeds
10/11, NumPy/Numba/Cython, three measured repetitions: eight fixtures, 24
fixture/backend groups, 72 fits, 48 two-iteration warm-ups and four graphs.
The measured cap is 36 million iterations / 567 million particle updates;
warm-ups add 96 iterations / 1,512 updates.

## Pre-execution runtime estimate

The [machine-readable estimate (archived)](performance_early_stopping_archiving.md)
retains all 24 group estimates. Reproduce it without loading data or fitting:

```console
python tools/estimate_early_stopping_pilot.py --baseline .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000
```

Baseline: PR #16's original 10,000-fixed-iteration measurements on this Windows
machine, measured source `c9f512c2da576a4095e9bf0192133ee34896f494`.
For each matching fixture/backend, multiply propagation median (and separately
Q1/Q3/maximum) by 50 iterations-budget units and three repetitions. Add three
median initialization/postprocessing charges per group, matching historical
warm-up fit costs, preparation charges and seed-helper latency once.

| Accounted cost | Seconds |
| --- | ---: |
| Median cap-work propagation | 2385.857 |
| Initialization/postprocessing | 0.023 |
| Historical warm-up fits | 0.757 |
| Historical seed helper | 0.169 |
| Dataset/label/graph preparation | 0.324 |
| Total median projection | 2387.130 |
| Q1–Q3 projection | 2249.926–2433.898 |
| Projection using historical maxima | 2499.931 |

The median is **39 min 47 s**, with a descriptive throughput range of
**37 min 30 s–40 min 34 s**. No early-stop savings are assumed. These are
linear cap-work projections, not runtime bounds or confidence intervals.
Startup, graph copies, hashes, invariant checking, artifact writing and new
stopping-check costs are unestimated. State distributions and machine load can
change; historical first calls do not predict fresh-process compilation exactly.

The original preparation deferred execution at an agent-selected 30-minute
threshold. The maintainer subsequently approved the approximately 40-minute
workload and reasonable overhead. The original estimate JSON remains an immutable
pre-execution snapshot, including that historical gate decision; the estimator
now reports costs without enforcing a runtime threshold. Neither the cap nor
the matrix was reduced.

## Prepared execution and correctness protocol

The existing PR #16 harness supports the complete requested pilot:

```console
python -m benchmarks.run --datasets wine digits germancredit satimage --label-seeds 10 11 --backends numpy numba cython --repeats 3 --mode early_stopping --iterations 500000 --historical-cache .cache/stopping_patience/sources --dry-run
```

The approved execution uses the same options with
`--allow-expensive --execute --output NEW_DIRECTORY` in place of `--dry-run`.
Use the activated historical environment and prebuilt Cython; no fallback or
download. Keep all raw measurements, outliers, warm-ups, self-contained fixture
arrays, manifest, source/binary hashes and completion/failure records.

Reuse frozen inventory/preprocessing/ordered graph fingerprint validation,
three labels/class, full transductive scaling, symmetric Euclidean `k_nn=10`,
`n_jobs=1`, and historical parameters `p_grd=.5`, `delta_v=.1`, `deltap=1`,
`dexp=2`, `es_chk=2000`. RNG resets are NumPy RandomState seed 0, independent
Numba RNG seed 0, and Cython's fixed xorshift32 seed 123456789. Repetitions are
fresh states and must match hashes, iterations and stop reason within backend;
cross-backend trajectories need not match.

The unchanged harness checks backend identity, no measured Numba compilation,
finite bounded normalized dominance, frozen labeled-node dominance/predictions,
valid particle positions/strengths, uint8 distance tables/home distances, stopping
schedule/check counts, and cap/reason consistency outside fit timers. Preserve
actual iterations, stop reason, propagation and total fit times, unlabeled
accuracy and all state hashes. Report both reaching the cap and reason `max_iter`:
an early-stop event at the cap is distinct from a censored non-stopping run.

Three rotating repetitions balance positions once but cover only three of six
backend permutations. This pilot assesses calibration, not independent
trajectories or a universal backend winner.

## Completed pilot results

All **72 measured fits and 48 warm-ups** completed. All 24 fixture/backend
groups reproduced their own state hashes, stopping metrics and accuracy across
three fresh repetitions. Every measured run terminated with reason `early_stop`,
strictly before 500,000 iterations: **cap-hit rate 0/72 (0%); censored rate
0/72 (0%)**, also 0/3 for every table row below. No result or outlier was removed.
Satimage seed 11 NumPy stopped at 499,941, just **59 iterations before the cap**;
this remains a successful stop rather than a censored observation.

Each timing cell is **median / IQR [minimum–maximum] in seconds**, using linear
quartiles across three timing repetitions of one fixture/backend. Iterations,
reason and accuracy are identical within each row's repetitions. Timing IQRs do
not estimate uncertainty in convergence or independent random trajectories.

| Dataset | Seed | Backend | Iterations | Unlabeled accuracy % | Propagation seconds | Total-fit seconds |
| --- | ---: | --- | ---: | ---: | --- | --- |
| wine | 10 | numpy | 9,701 | 95.8580 | 0.924052 / 0.004965 [0.919394–0.929324] | 0.924166 / 0.004983 [0.919496–0.929463] |
| wine | 10 | numba | 11,221 | 94.0828 | 0.006180 / 0.001579 [0.006178–0.009337] | 0.006335 / 0.001675 [0.006299–0.009649] |
| wine | 10 | cython | 10,611 | 94.6746 | 0.004397 / 0.000210 [0.004326–0.004746] | 0.004867 / 0.000215 [0.004457–0.004887] |
| wine | 11 | numpy | 13,221 | 96.4497 | 1.264096 / 0.011620 [1.258942–1.282182] | 1.264211 / 0.011618 [1.259054–1.282290] |
| wine | 11 | numba | 8,911 | 97.0414 | 0.004843 / 0.000028 [0.004835–0.004892] | 0.004990 / 0.000009 [0.004977–0.004994] |
| wine | 11 | cython | 7,321 | 96.4497 | 0.003002 / 0.000015 [0.002994–0.003024] | 0.003134 / 0.000027 [0.003094–0.003148] |
| digits | 10 | numpy | 81,941 | 89.4737 | 26.574449 / 0.021734 [26.574423–26.617892] | 26.574761 / 0.021741 [26.574735–26.618218] |
| digits | 10 | numba | 87,691 | 89.1907 | 0.179738 / 0.004231 [0.175626–0.184088] | 0.180091 / 0.004232 [0.175994–0.184458] |
| digits | 10 | cython | 85,051 | 89.3605 | 0.214397 / 0.000436 [0.213709–0.214581] | 0.214728 / 0.000324 [0.214304–0.214953] |
| digits | 11 | numpy | 92,381 | 90.3792 | 30.126595 / 0.099509 [29.991181–30.190200] | 30.126891 / 0.099507 [29.991496–30.190510] |
| digits | 11 | numba | 134,341 | 90.3226 | 0.276080 / 0.002563 [0.274660–0.279786] | 0.276699 / 0.002557 [0.275032–0.280145] |
| digits | 11 | cython | 93,821 | 89.4737 | 0.239019 / 0.003218 [0.234752–0.241187] | 0.239426 / 0.003246 [0.235114–0.241605] |
| germancredit | 10 | numpy | 157,941 | 51.1066 | 10.443432 / 0.133203 [10.385149–10.651554] | 10.443601 / 0.133203 [10.385320–10.651726] |
| germancredit | 10 | numba | 148,881 | 55.3320 | 0.069445 / 0.002141 [0.068736–0.073017] | 0.069670 / 0.002126 [0.068959–0.073210] |
| germancredit | 10 | cython | 106,611 | 52.5151 | 0.041009 / 0.001030 [0.039343–0.041403] | 0.041233 / 0.001052 [0.039534–0.041637] |
| germancredit | 11 | numpy | 152,781 | 52.8169 | 10.224319 / 0.024213 [10.223777–10.272202] | 10.224498 / 0.024049 [10.224281–10.272378] |
| germancredit | 11 | numba | 83,171 | 52.8169 | 0.040934 / 0.001350 [0.039310–0.042010] | 0.041470 / 0.001344 [0.039554–0.042241] |
| germancredit | 11 | cython | 84,481 | 55.0302 | 0.032749 / 0.000411 [0.032549–0.033371] | 0.032980 / 0.000391 [0.032785–0.033567] |
| satimage | 10 | numpy | 386,091 | 77.1366 | 105.588488 / 1.373971 [103.830341–106.578284] | 105.589352 / 1.374095 [103.831136–106.579327] |
| satimage | 10 | numba | 439,901 | 77.3706 | 0.999623 / 0.082135 [0.965166–1.129437] | 1.000387 / 0.082019 [0.966048–1.130086] |
| satimage | 10 | cython | 489,151 | 76.3880 | 2.365591 / 0.177946 [2.214988–2.570880] | 2.366224 / 0.177844 [2.215976–2.571664] |
| satimage | 11 | numpy | 499,941 | 81.4566 | 138.781403 / 3.843740 [135.231892–142.919372] | 138.782219 / 3.843968 [135.232867–142.920804] |
| satimage | 11 | numba | 455,411 | 81.7374 | 1.239228 / 0.084813 [1.109395–1.279021] | 1.239787 / 0.084858 [1.110172–1.279888] |
| satimage | 11 | cython | 307,411 | 81.6594 | 1.557358 / 0.279684 [1.233720–1.793089] | 1.557997 / 0.280024 [1.234357–1.794405] |

Accuracy spread across backends, in percentage points, is 1.7751/0.5917 for
Wine seeds 10/11; 0.2830/0.9055 for Digits; 4.2254/2.2133 for GermanCredit;
and 0.9825/0.2807 for Satimage. Different RNG streams, final states and stopping
iterations are expected. These outcomes establish within-backend repeatability
and valid states, not equivalent trajectories or an accuracy winner. For example,
GermanCredit seed 10 Numba exceeds NumPy by 4.2254 points, while Wine seed 10
NumPy exceeds Numba by 1.7751 points. Native times include different amounts of
propagation; their ratios are time-to-stopping observations, not equal-work kernel
speed comparisons. The small timing sample and rotated order limit close native
comparisons; no run was discarded for timing variability.

## Runtime, correctness and provenance

Native harness process: **2026-10-13 19:49:56.938419–20:06:40.037304 UTC**
(16:49:56–17:06:40 in America/Sao_Paulo), **1,003.098590 s (16 min 43 s)**.
Measured propagation totals 993.580695 s; measured fits total 993.610011 s.
The process duration includes startup, warm-ups, preparation, hashing/checking,
artifact writing and lightweight monitoring; excludes launcher's preservation
hashing, report analysis and tests. It used 11,843,952 measured iterations,
32.90% of the 36-million cap. Wall time was 42.02% of the original 2,387.129632 s
accounted projection, **23 min 04 s shorter (57.98% reduction)**. Boundaries differ:
the projection excluded some overhead, so this is an observed planning comparison,
not a calibrated speedup. Early stopping changed workload and state distributions.

Measured source **`cd309b70b9be413a0a7f33cf71ba1ff25396a7de`**, clean at launch.
The propagation modules, fixture/runner/historical source hashes and Cython binary
match the historical baseline. Environment: native Windows, Intel Core i7-14700K,
Python 3.12.15, NumPy 2.4.6, Numba 0.68.0, Cython 3.2.4, SciPy 1.18.1 and
scikit-learn 1.9.1, existing `pypcc-dev` environment with its `Library/bin` DLL
path. The Cython binary SHA-256 remains
`e29083201d16ae30b1470f7abd878ec922c2ba645bff77aa50522a49b39b92a0`;
original extension compiler/build provenance remains unavailable. Frequency,
scheduling, affinity and foreground load were not controlled. External resource
checks every 15 s stopped only for severe RAM/disk shortage (below 2 GiB free);
none occurred. There was no elapsed-time limit, drift cutoff or outlier filter.

The unchanged harness passed all dominance, labeled-node, position, strength,
distance-table, backend-identity, Numba-signature and stopping-schedule checks on
every execution. Offline audit rehashed every saved fixture array, recomputed
fixture hashes, matched every fixture to PR #16 and its frozen fingerprints,
checked exact record inventory/order/seed identities, within-backend states and
accuracy, stopping/censoring fields and timing sums, and reproduced `summary.json`
exactly. **519 original fixed/sensitivity artifact files** passed byte-integrity
preservation checks. This is a new pilot audit, not a rerun of historical trajectories.

The [raw measured records (archived)](performance_early_stopping_archiving.md),
[warm-ups (archived)](performance_early_stopping_archiving.md), manifest,
all fixture metadata, execution/log/monitor records, [CSV table (archived)](performance_early_stopping_archiving.md)
and [offline analysis](benchmarks/early_stopping_pilot/results/summary.json)
are committed. Complete local artifacts, including all eight self-contained NPZ
fixtures, preservation snapshot and exact measured-source tar, remain at
`.cache/performance/windows-early-stopping-pilot-cd309b7-01/`; the committed
[artifact checksum index (archived)](performance_early_stopping_archiving.md)
identifies every local file. NPZ datasets/source tar are retained locally and are
not committed or externally published. No Release was created or updated.
Checksums cover original local bytes; Git's normal Windows text conversion may
normalize CRLF in committed copies, which do not replace the original artifacts.

Reproduce the offline audit and table without fitting (use a new output path):

```console
python -m tools.analyze_early_stopping_pilot --root .cache/performance/windows-early-stopping-pilot-cd309b7-01 --baseline .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000 --output NEW_ANALYSIS_DIRECTORY
```

## Full-campaign estimate and recommendation

Proposed full protocol: all 20 frozen datasets, label seeds 10–19, all three
sequential backends, unchanged 500,000 cap and stopping criterion, six reset
timing repetitions using every backend permutation (`--order-policy permutations`
and a recorded order seed), and two two-iteration warm-ups per fixture/backend.
This is 3,600 fits, 1,200 warm-ups and 20 reused graph preparations, with at most
1.8 billion measured iterations / 22.95 billion particle updates. Warm-ups add
2,400 iterations / 30,600 updates. Per-fixture/backend summaries remain separate.

| Planning scenario | Seconds | Hours |
| --- | ---: | ---: |
| Historical Q1 cap work | 102,000.784 | 28.33 |
| Historical median cap work | 108,665.390 | 30.18 |
| Historical Q3 cap work | 116,257.373 | 32.29 |
| Historical maximum-rate cap work | 126,794.746 | 35.22 |
| Four tested datasets: extrapolated pilot behavior | 3,312.004 | 0.92 |
| Other 16: historical median cap propagation | 82,855.922 | 23.02 |
| Hybrid: tested behavior plus untested caps and overhead | 86,171.771 | 23.94 |
| Conservative cap planning with 50% headroom | 190,192.118 | 52.83 |

Historical cap scenarios sum **each of all 600 fixture/backend groups'** fixed
10,000 propagation statistic × 50 × six repetitions, plus 3.844604 s historical
initialization/postprocessing, warm-up, preparation and helper charges. Q1/Q3
are descriptive throughput scenarios, not confidence bounds. The hybrid uses
pilot medians × ten for the four tested datasets (six versus three repetitions,
ten versus two label seeds), while retaining cap work for the untested 16.
This two-mask extrapolation is explicitly tentative; it does not predict those
other seeds' convergence. The conservative scenario applies **50% planning
headroom** to the larger of historical maximum-rate cap work or historical
untested cap work plus caps at the pilot's maximum observed per-iteration rates
for the tested four. Headroom is a cost scenario, **not an execution threshold or
runtime guarantee**. New stopping-scan costs for untested datasets, compilation,
bookkeeping, state distributions and machine activity remain unestimated.

**Recommendation:** preserve the proposed 500,000 cap and original criterion,
but do not authorize the entire campaign solely from this four-dataset pilot.
Plan for roughly 30 hours of historical median cap work, with about **53 hours**
in the conservative scenario. First review a separately authorized calibration
of remaining workloads, prioritizing PenDigits, PageBlocks and other large or
high-particle cases, to estimate stopping-scan overhead and censoring. The near-cap
Satimage result shows why zero censoring on two masks does not guarantee zero
censoring on all ten. Report `early_stop`, `max_iter`, cap hits and censored
accuracy separately; never infer convergence for a capped run. Use six backend
permutations for the full campaign, retain all measurements, and avoid treating
reset timing repeats as independent trajectories. Do not tune stopping from
accuracy, drop slow/outlying runs or silently combine different caps. Full
execution, README performance edits, Release publication and merging remain
outside this authorization.

## Validation status

Offline audit passed, including negative checks rejecting tampered stopping counts
and repeated-state hashes; the complete local artifact checksum index verified.
Local suite: **124 tests, 121 passed, two archive metadata checks skipped, one
installed-metadata error**. The isolated `python -I` subprocess cannot find an
installed `pcc_step` distribution in the existing benchmarking environment;
ordinary execution resolves checkout metadata and the prebuilt source-tree
Cython binary. This is an installation-validation limitation, not a pilot
failure. All 15 benchmark tests and all backend/algorithm regressions passed.
No package reinstall or binary rebuild was performed to mask that limitation.
Final-commit CI must independently validate installed distributions on the
Windows/Linux, Python 3.10/3.12 matrix. `git diff --check` passed.
