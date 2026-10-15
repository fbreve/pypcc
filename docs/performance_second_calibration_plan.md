# Remaining historical datasets: second calibration plan

The maintainer approved this protocol and cost, conditional on recovery tests
and small synthetic validation. The [optional controller](resumable_benchmarking.md)
is now implemented. The [completed 48-fit results](performance_second_calibration_results.md)
report actual outcomes and revised full-campaign costs; this document preserves
the pre-execution estimate and design. PR #17 remains a draft. The first pilot, fixed results, README
tables, frozen artifacts and published Release remain unchanged.

The proposed matrix excludes Wine, Digits, GermanCredit and Satimage: 16
historical datasets, label seed 10, three sequential backends, exactly one
measured fit per fixture/backend, original early stopping with `es_chk=2000`
and a 500,000 cap. Preserve three labels/class, `k_nn=10`, `n_jobs=1`, historical
preprocessing and `p_grd=.5`, `delta_v=.1`, `deltap=1`, `dexp=2`. RNG resets remain
NumPy seed 0, independent Numba seed 0, Cython xorshift32 seed 123456789.

There are **48 measured fits**, **96 two-iteration warm-ups**, **16 graph
preparations**, at most **24 million measured iterations / 288 million particle
updates**, plus 192 warm-up iterations / 2,304 updates. A single fit supplies raw
timing and accuracy, not a meaningful timing IQR or within-backend repeatability
estimate. Record all six final state hashes for provenance; do not run a second
measurement merely to check equality. Existing first-pilot repetition validation
does not establish repeatability for these untested fixtures.

## Offline cost estimate

The [48 group estimates and exact task order (archived)](performance_early_stopping_archiving.md)
come from original PR #16 seed-10 fixed 10,000-iteration groups, each with six
timing repetitions. Multiply each backend's propagation statistic by 50; add
matching median fit initialization/postprocessing, actual historical warm-up
fits, once-per-dataset preparation and once-per-process seed-helper latency.
Original input-file SHA-256 values are recorded. No dataset has been loaded.

```console
python tools/estimate_second_calibration.py --baseline .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000
```

| Cap-work scenario | Seconds | Duration |
| --- | ---: | --- |
| Historical Q1 rates plus charges | 1272.853 | 21 min 13 s |
| Historical median rates plus charges | 1400.406 | **23 min 20 s** |
| Historical Q3 rates plus charges | 1516.591 | 25 min 17 s |
| Historical maximum rates plus charges | 1638.373 | 27 min 18 s |
| Conservative: maximum-rate scenario × 1.5 | 2457.560 | **40 min 58 s** |

Historical charges total 1.160379 s: fit overhead 0.013673, warm-up fits 0.049192,
preparation 0.928176 and seed helper 0.169338. Historical warm-ups mostly reused
compiled signatures; fresh compilation/startup costs are unestimated. No early
stopping savings are assumed for any of these 16 datasets. Q1/Q3 describe
historical throughput variation, not confidence bounds. The 50% headroom is a
planning scenario, not a runtime guarantee or execution threshold. Early-stopping
scans, state distributions, hashing, checkpoint flushes and machine activity can
change rates. Additional process starts on resume require new warm-ups and extra
time, recorded separately; uninterrupted-run charges do not cover those restarts.

## Workload ranking and proposed order

The table below ranks **aggregate propagation cap cost**, excluding common
charges. Native fixed-throughput projections exclude new early-stopping scan
costs and should not be treated as measured early-stop times.

| Dataset | NumPy cap s | Numba cap s | Cython cap s | Total median cap s | Total maximum-rate cap s |
| --- | ---: | ---: | ---: | ---: | ---: |
| pendigits | 232.578 | 1.055 | 0.986 | 234.620 | 264.519 |
| yeast | 216.782 | 0.967 | 0.959 | 218.708 | 243.957 |
| glass | 142.977 | 0.529 | 0.507 | 144.013 | 156.103 |
| segment | 117.643 | 0.589 | 0.533 | 118.765 | 127.016 |
| pageblocks | 102.896 | 0.523 | 0.502 | 103.921 | 121.215 |
| vehicle | 93.115 | 0.366 | 0.356 | 93.837 | 103.923 |
| iris | 63.982 | 0.340 | 0.204 | 64.527 | 78.171 |
| balancescale | 59.455 | 0.259 | 0.190 | 59.904 | 84.719 |
| banknote | 52.395 | 0.189 | 0.121 | 52.706 | 61.331 |
| diabetes | 51.936 | 0.250 | 0.132 | 52.317 | 66.113 |
| bloodtransfusion | 47.812 | 0.197 | 0.120 | 48.129 | 55.183 |
| breast_cancer | 47.669 | 0.230 | 0.141 | 48.040 | 59.925 |
| ionosphere | 45.931 | 0.285 | 0.213 | 46.429 | 62.224 |
| spambase | 39.905 | 0.225 | 0.185 | 40.314 | 58.852 |
| sonar | 39.256 | 0.202 | 0.130 | 39.587 | 55.548 |
| qsarbiodeg | 33.016 | 0.265 | 0.148 | 33.429 | 38.413 |


Start with **PenDigits, then PageBlocks**, since their larger graphs make them
useful risk-first checks. PenDigits has 10,992 nodes / 10 classes / 30 particles;
PageBlocks has 5,473 nodes / five classes / 15 particles. Their graph preparation,
dominance scans and larger state tables are not explained by fixed kernel timing
alone. PenDigits is the largest projected cap workload; PageBlocks ranks fifth.
Yeast and Glass also merit attention despite smaller graphs because of their
class/particle counts and observed fixed-throughput cost.

Then run descending aggregate median cap cost: **Yeast, Glass, Segment, Vehicle,
Iris, BalanceScale, Banknote, Diabetes, BloodTransfusion, BreastCancer,
Ionosphere, Spambase, Sonar, QSARBiodeg**.

Within each dataset use **Numba → Cython → NumPy**, serially, to collect native
stopping/scan information before the more expensive NumPy fit. This is a frozen
risk-first calibration schedule, not a balanced performance comparison. Record
positions and process/session boundaries; use six complete permutations for a
later full campaign. Do not change this order based on emerging accuracy or
silently drop a capped/slow backend. No artificial runtime threshold is proposed.

## Durable checkpoints and resume contract

The existing `benchmarks.run` creates a new exclusive output directory, flushes
JSONL rows but does not `fsync` them, and rejects existing outputs. It has no
validated resume path; invoking it again cannot safely resume this matrix.
The required extension is an optional benchmark controller that reuses
`FixturePreparer`, `save_fixture`, `expected_backend`, `seed_backend`, `measure`
and existing timing/statistics logic. Production loops, stopping, RNG and public
APIs stay unchanged. The controller now implements this design, with independently
pinned plans and explicit recovery of derived summary generations.

Checkpoint at **fixture and complete-fit boundaries**, outside measured timers.
There is no in-flight particle/RNG serialization or mid-fit continuation. This
does not alter the frozen scientific reproduction or add historical checkpoints.

1. **Freeze identity before measurement.** Create a new campaign root exclusively.
   Save immutable `plan.json` with the exact 48 task IDs/order/parameters, fixture
   policy, source and binary hashes, environment/dependency versions and RNG
   identity. Validate its checksum on every resume. A task identity combines the
   campaign/plan hash, fixture hash, backend, measurement index and effective RNG
   policy. Record each resume's provenance/session identity separately. Changed
   parameters, source, binary, library versions or hardware identity fail strict
   resume validation; create a separately identified campaign instead. Changing
   load/frequency cannot be fully prevented: record sessions rather than implying
   identical machine state.
2. **Validate and commit one fixture per dataset.** Reuse historical loading,
   preprocessing, ordered graph and label fingerprint checks. Save arrays and
   metadata under a temporary fixture directory on the same filesystem; flush
   and `fsync` files, rehash and validate them, then rename to a previously absent
   final path. Treat rename/flush failures as errors. On resume, rehash all saved
   arrays and recompute fixture identity before reuse; never silently rebuild a
   mismatched or corrupted fixture. Retain every partial file for inspection.
3. **Allow one writer.** Acquire an exclusive campaign lock before any mutation.
   Record process/session identity. A stale lock must be reviewed and explicitly
   recovered; never automatically steal it solely because time has elapsed.
   Reads can report status, but another executor must fail while the lock exists.
4. **Start durably.** Before each measured call, create an immutable started event
   with a unique attempt ID, task ID, session, timestamps and saved warm-up
   references. Flush/`fsync` it before entering `measure`. Seed and create fresh
   state exactly as today. Each resumed process warms every pending fixture/backend
   it will execute with two fresh two-iteration fits; preserve those extra warm-up
   and helper records under that session, outside measurements. Completed measured
   tasks are not warmed or repeated merely because the campaign resumes.
5. **Commit a validated result.** After `measure` returns and all original
   invariants pass, serialize raw timings, metrics, accuracy, six state hashes,
   provenance and fixture/task/attempt identities into a temporary result file.
   Include a checksum envelope, reject nonfinite/invalid fields, flush/`fsync`,
   then publish to a previously absent canonical per-task path while holding the
   lock. Never use replacement semantics against a completed result. Persist a
   completion event only after the result is durable. Filesystem durability varies
   on Windows: check and surface all flush/rename errors, and validate content on
   restart rather than promising protection from every power/storage failure.
6. **Reconcile strictly on resume.** Verify the plan, environment, fixtures and
   checksum/schema/identity/stopping semantics of every canonical result, not just
   an index or `completion.json`. Valid results are authoritative and skipped even
   if a crash preceded their completion event; append an explicit reconciliation
   event. A start without a valid canonical result is `interrupted/ambiguous`, not
   pending: do not automatically rerun or assume it never finished. Preserve raw
   partial files/logs, require an explicit retry task and recorded reason, and
   use a new attempt ID. Duplicate valid results, mismatched hashes, unknown task
   IDs, corrupt events/results or inconsistent attempt identities stop recovery
   for investigation. Never delete, overwrite or silently ignore evidence.
7. **Keep the result ledger authoritative.** Derived JSONL/CSV/status/summary
   views are rebuildable from the verified canonical results. Generate each
   derived snapshot at a new generation path with checksums and atomic publication;
   preserve old generations. Do not append directly to the authoritative result
   stream. Record every failure, interruption and explicit retry. `max_iter`
   results are completed measurements with censored status, never failed tasks
   to rerun. An `early_stop` event at the cap is reported separately from censoring.
8. **Complete only the exact matrix.** Finalize when all 48 unique task IDs have
   one validated canonical result, with no unresolved attempts or duplicates.
   Preserve all warm-ups/raw timings/outliers. Report per-task iterations,
   stopping reason, cap status, accuracy and state hashes, plus total preparation,
   warm-up, propagation, fit and process/session elapsed costs. Do not manufacture
   medians/IQRs from one measurement or pool different environments.

## Required implementation and verification before authorization

Add the isolated optional controller and result store; do not retrofit destructive
resume behavior into fixed-campaign output directories. Provide read-only
`--status`/resume validation that performs no fits, and explicit start/resume/retry
actions requiring execution opt-in. Existing benchmark dry-run/default behavior
and the first pilot remain unchanged. Use newly named local output roots; no
downloads, Release publication or automatic full-campaign continuation.

Before executing, test crash boundaries (before start, after start, during result
write, after durable result but before completion event), partial fixture writes,
corrupt hashes/metadata/stopping metrics, protocol/environment mismatches,
duplicate results, concurrent and stale locks, explicit retry history and
completion inventory. Verify successful and capped tasks are never repeated,
and that resumed fresh-process warm-ups do not become measured fits. Use small
deterministic fixtures/fake measure calls in these tests, then the existing
three-backend invariant tests; no long calibration should run in CI.

The offline estimator is implemented and tested for exact workload arithmetic,
inventory/order and refusal of missing, duplicate or inappropriate historical
inputs. The controller has 21 passing recovery tests plus small real-backend
end-to-end validation. The maintainer authorized the 48 historical fits after
these checks pass; no full campaign is authorized.
