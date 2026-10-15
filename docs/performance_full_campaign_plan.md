# Full early-stopping campaign preparation

Prepared and validated; **no historical fits executed for this stage**. PR #17
remains a draft. Calibrations, fixed results, README tables and Release are
unchanged. Historical stage execution requires separate maintainer authorization.

## Immutable design and first stage

The optional controller freezes 200 fixtures and 3,600 measured tasks at creation:
20 historical datasets, seeds 10–19, three backends, six repetitions. Every fixture
uses all six backend permutations; every backend occupies each position twice.
Original `benchmarks.run.execution_order` supplies the schedule with `order_seed=0`
without consuming propagation RNG. The [specification fingerprint and initial
schedule (archived)](performance_early_stopping_archiving.md) pin the design.

Seed-major dataset order: Wine, Digits, Iris, BreastCancer, BalanceScale,
GermanCredit, Diabetes, Sonar, Glass, Vehicle, Ionosphere, PenDigits, Satimage,
Spambase, PageBlocks, Banknote, BloodTransfusion, Yeast, Segment, QSARBiodeg.
Seed 10 comprises IDs `t0000`–`t0359`: 20 fixtures, 60 groups, 360 measurements
and 120 two-iteration warm-up fits in an uninterrupted session. Full-plan IDs
are `t0000`–`t3599`. Stage selection filters existing tasks, never regenerates
them. Later seeds require explicit selection and all earlier stages complete.

Parameters remain `p_grd=0.5`, `delta_v=0.1`, `deltap=1.0`, `dexp=2.0`,
`es_chk=2000`, cap 500,000. Backend-specific deterministic initialization,
frozen preprocessing/labels/ordered graphs, original invariant checks, backend
identity checks and timing boundaries are preserved. Repetitions must reproduce
their own backend's state hashes, iterations, reason and accuracy; cross-backend
equality is not required. Cap outcomes remain valid censored observations.

## Exact commands — historical execution pending

Use the tested installed NumPy/Numba/Cython environment, existing frozen source
cache and a new root. This command has **not** been executed:

```console
python -m benchmarks.resumable --action start --full-campaign --stage-seed 10 --historical-cache .cache/stopping_patience/sources --root .cache/performance/windows-full-early-stopping-01 --execute
```

Read-only inspection: `python -m benchmarks.resumable --full-campaign`.
Add `--max-tasks 18` to finish one complete fixture at a time, or choose another
complete-fit allowance. Pausing inside a permutation preserves the schedule.
Resume the saved complete plan, explicitly selecting only the authorized seed:

```console
python -m benchmarks.resumable --action status --root .cache/performance/windows-full-early-stopping-01
python -m benchmarks.resumable --action resume --stage-seed 10 --root .cache/performance/windows-full-early-stopping-01 --execute
```

Keep the source checkout, extension and environment available across stages;
strict identity binding rejects code/dependency/binary changes. `--max-tasks`
limits task count, never the iteration cap or runtime.

## Runtime and resources

The [offline estimate (archived)](performance_early_stopping_archiving.md)
uses all 60 seed-10 calibration groups and preserved PR #16 timing data.
`tools/estimate_seed10_stage.py` reproduces it without PCC execution. Four
datasets supply three repetitions; sixteen supply one. This is a forecast,
not a confidence interval.

| Seed-10 scenario | Runtime |
| --- | ---: |
| Observed stopping behavior, six repeats | 58 min 35 s |
| All fits capped, observed propagation rates | 2 h 28 min 34 s |
| All fits capped, historical median rates | 2 h 59 min 13 s |
| All fits capped, historical maximum rates | 3 h 24 min 29 s |
| Conservative cap planning, maximum scenario × 1.5 | 5 h 6 min 43 s |

Allow roughly one hour plus startup, checkpoint/audit I/O and interruption costs.
These are scenarios, not cutoffs or guaranteed bounds. Observed behavior projects
58,429,704 measured iterations and 36 censored fits (10%); cap work is
180,000,000 iterations / 2,295,000,000 particle updates. Permutation order,
contention and temperature can affect rates.

Execution is sequential: one writer, one fit at a time. Use local durable storage
supporting exclusive hard-link publication and same-volume fixture renames.
Seed-10 arrays occupy about 5.5 MiB compressed / 29.8 MiB decoded; the full
200-fixture proxy is 54.8 MiB compressed / 298.1 MiB decoded. Later masks reuse
independently validated seed-10 data/graphs but retain their own sealed fixtures.
Largest dominance/distance arrays are about 0.84/0.31 MiB. Python, Numba/LLVM,
transient graph copies and JSON provenance add costs. Provision an 8 GiB RAM
machine with at least 2 GiB free RAM and 2 GiB free local disk: prudent allowances,
not measured minimums. Avoid sleep and concurrent benchmarks during execution.

## Recovery and validation

Independent plan pins, exclusive canonical publication, writer locks, durable
checksummed events, complete-fit checkpoints and explicit reasoned retries remain.
A result published before its completion event is reconciled without repetition.
Ambiguous attempts retain evidence and block execution pending explicit retry.
Restarted sessions warm only pending backend groups and preserve previous warm-ups.
Derived summaries can be explicitly rebuilt into new generations; invalid
authoritative data cannot be bypassed.

Summaries separate global and per-seed completion: seed 10 complete is still a
globally partial 360/3,600 campaign. Groups report expected six repetitions,
timing medians/IQRs/ranges and censoring. Singleton partial groups have null IQR.

- All 22 recovery tests and seven new matrix tests passed, covering subprocess
  crashes, corruption, locking, repeatability rejection, source binding,
  summary recovery, stage barriers and mid-permutation resume.
- The [optional mock validation (archived)](performance_early_stopping_archiving.md)
  completed 3,600 unique tasks / 600 groups incrementally with stable plan bytes,
  preserved completed results and no terminal-stage repetition. It took 253.3 s;
  storage-only stage completion ranged about 10–36 s. Mock data are placeholders,
  not scientific timing/accuracy measurements; expensive historical fits stay out of CI.
- The [real-backend synthetic validation (archived)](performance_early_stopping_archiving.md)
  paused after five fits, resumed thirteen, then resumed the complete stage with
  zero additional fits. All six permutations and within-backend state/iteration/
  accuracy checks passed. NumPy stopped early; Numba/Cython were censored. This
  30-node test uses `es_chk=10`, cap 5,000; later synthetic seeds and all historical
  stage tasks remain unexecuted. Canonical artifacts remain locally retained.

Remaining risks: machine-load sensitivity, interruption requiring explicit retry,
and increasing audit work as immutable records/summary generations accumulate.
The full mock validates matrix/storage scale, not historical wall time or power-loss
durability. Windows directory fsync remains platform-dependent as described in
the [controller guide](resumable_benchmarking.md). Retain original artifacts and
backups; do not edit a live campaign to resolve an identity mismatch.

Recommendation: authorize seed 10 alone; review its six-repetition timings,
correctness/censoring and measured checkpoint overhead before any further seed.
