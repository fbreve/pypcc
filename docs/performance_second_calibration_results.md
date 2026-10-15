# Second early-stopping calibration: completed results

The approved resumable calibration completed **48 measured fits, 96 warm-ups,
16 fixtures and one session**, with exactly 48 started attempts and **no retries,
interruptions, recovery actions or discarded outliers**. All 21 recovery tests,
50 relevant benchmark tests, small real-backend resume validation and all four
[implementation CI jobs](https://github.com/fbreve/pypcc/actions/runs/38468877765)
passed before historical execution.

Measured source: **`4ccca3da21698909c07de3d31112c7c02f2b398c`**, clean at launch.
The optional controller, commands and durability limits are documented in the
[controller guide](resumable_benchmarking.md); the immutable pre-execution
[estimate and protocol](performance_second_calibration_plan.md) remain separate.
Production PCC source and the existing harness match the first pilot exactly.

The frozen matrix uses seed 10, three labels/class, historical preprocessing,
full transductive scaling and symmetric ordered Euclidean `k_nn=10`, `n_jobs=1`.
PCC parameters remain `p_grd=.5`, `delta_v=.1`, `deltap=1`, `dexp=2`, `es_chk=2000`,
early stopping enabled and a 500,000 cap. The exact order is PenDigits, PageBlocks,
Yeast, Glass, Segment, Vehicle, Iris, BalanceScale, Banknote, Diabetes,
BloodTransfusion, BreastCancer, Ionosphere, Spambase, Sonar, QSARBiodeg; within
every fixture Numba → Cython → NumPy. NumPy/Numba reset their independent RNGs to
0; Cython retains xorshift32 seed 123456789. Backend trajectories need not match.

## Actual outcomes and timing

**42 early stops; six capped/censored fits (12.5%)**. The censored cases are all
three PenDigits backends, PageBlocks Cython, and Spambase Cython/NumPy. They are
valid completed measurements and were never rerun. No early-stop event occurred
at the cap. PageBlocks Numba stopped at 494,901 and Spambase Numba at 499,171;
their successful stops retain their small cap margins.

Every row below is **one measured fit**, not a median. There is no within-group
timing IQR/range or measured terminal-state repeatability estimate. All six state
hashes are retained. Two fresh warm-ups preceded each measured fit; warm-up hashes
matched within backend. Accuracy is on unlabeled nodes, including capped states.

| Dataset | Backend | Iterations | Reason | Unlabeled accuracy % | Propagation s | Total fit s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| pendigits | numba | 500,000 | max_iter | 84.8477 | 1.933145 | 1.934404 |
| pendigits | cython | 500,000 | max_iter | 89.9015 | 6.475664 | 6.476860 |
| pendigits | numpy | 500,000 | max_iter | 89.2355 | 164.188551 | 164.189701 |
| pageblocks | numba | 494,901 | early_stop | 27.5009 | 0.712067 | 0.712639 |
| pageblocks | cython | 500,000 | max_iter | 24.7527 | 1.345434 | 1.346034 |
| pageblocks | numpy | 418,541 | early_stop | 31.8981 | 67.291958 | 67.292517 |
| yeast | numba | 100,931 | early_stop | 35.9697 | 0.203009 | 0.203419 |
| yeast | cython | 107,111 | early_stop | 37.7579 | 0.280381 | 0.280808 |
| yeast | numpy | 101,801 | early_stop | 37.1389 | 32.136035 | 32.136415 |
| glass | numba | 5,971 | early_stop | 35.7143 | 0.006276 | 0.006475 |
| glass | cython | 7,681 | early_stop | 40.3061 | 0.007397 | 0.007833 |
| glass | numpy | 8,671 | early_stop | 38.7755 | 1.625849 | 1.626057 |
| segment | numba | 277,401 | early_stop | 73.7003 | 0.572444 | 0.572907 |
| segment | cython | 181,601 | early_stop | 74.0061 | 0.379338 | 0.379813 |
| segment | numpy | 284,941 | early_stop | 73.9624 | 69.162263 | 69.162779 |
| vehicle | numba | 93,311 | early_stop | 50.5995 | 0.077641 | 0.078051 |
| vehicle | cython | 70,961 | early_stop | 50.7194 | 0.069250 | 0.069633 |
| vehicle | numpy | 39,101 | early_stop | 49.8801 | 5.262913 | 5.264102 |
| iris | numba | 8,191 | early_stop | 95.0355 | 0.006684 | 0.006990 |
| iris | cython | 10,441 | early_stop | 94.3262 | 0.004811 | 0.005110 |
| iris | numpy | 10,811 | early_stop | 95.0355 | 1.212011 | 1.212218 |
| balancescale | numba | 34,391 | early_stop | 59.2532 | 0.019880 | 0.020161 |
| balancescale | cython | 60,911 | early_stop | 60.5519 | 0.030884 | 0.031111 |
| balancescale | numpy | 48,041 | early_stop | 60.2273 | 5.041651 | 5.041948 |
| banknote | numba | 301,481 | early_stop | 79.6486 | 0.153985 | 0.154349 |
| banknote | cython | 227,291 | early_stop | 74.6706 | 0.093044 | 0.093436 |
| banknote | numpy | 234,871 | early_stop | 71.0835 | 16.008565 | 16.008841 |
| diabetes | numba | 73,971 | early_stop | 64.6982 | 0.038260 | 0.038706 |
| diabetes | cython | 51,271 | early_stop | 64.0420 | 0.019015 | 0.019335 |
| diabetes | numpy | 82,731 | early_stop | 64.5669 | 5.691962 | 5.692467 |
| bloodtransfusion | numba | 77,991 | early_stop | 55.6604 | 0.033958 | 0.034282 |
| bloodtransfusion | cython | 80,981 | early_stop | 64.4205 | 0.033635 | 0.033914 |
| bloodtransfusion | numpy | 103,731 | early_stop | 60.5121 | 7.268302 | 7.268575 |
| breast_cancer | numba | 35,031 | early_stop | 91.6519 | 0.020131 | 0.020393 |
| breast_cancer | cython | 47,141 | early_stop | 94.1385 | 0.018673 | 0.018924 |
| breast_cancer | numpy | 38,161 | early_stop | 92.7176 | 2.801691 | 2.801975 |
| ionosphere | numba | 27,511 | early_stop | 73.0435 | 0.015876 | 0.016166 |
| ionosphere | cython | 38,111 | early_stop | 72.4638 | 0.011482 | 0.011700 |
| ionosphere | numpy | 41,901 | early_stop | 68.9855 | 3.169607 | 3.169837 |
| spambase | numba | 499,171 | early_stop | 85.7671 | 0.368913 | 0.369433 |
| spambase | cython | 500,000 | max_iter | 84.1567 | 0.449775 | 0.450297 |
| spambase | numpy | 500,000 | max_iter | 79.1295 | 35.290080 | 35.290420 |
| sonar | numba | 21,111 | early_stop | 71.2871 | 0.012622 | 0.012899 |
| sonar | cython | 11,231 | early_stop | 69.8020 | 0.003247 | 0.003425 |
| sonar | numpy | 16,441 | early_stop | 70.7921 | 1.156785 | 1.156967 |
| qsarbiodeg | numba | 138,691 | early_stop | 63.5844 | 0.065261 | 0.065561 |
| qsarbiodeg | cython | 106,191 | early_stop | 67.3975 | 0.039974 | 0.040336 |
| qsarbiodeg | numpy | 102,771 | early_stop | 63.4890 | 7.216725 | 7.217035 |

Largest accuracy spreads are BloodTransfusion **8.7601 percentage points**,
Banknote **8.5652**, PageBlocks **7.1455**, Spambase **6.6376** and PenDigits
**5.0538**. These fixtures have distinct random trajectories and stopping counts;
several states are censored. Valid invariants do not imply equal classification
outcomes. This one-mask calibration cannot establish an accuracy winner or a
universal native-backend winner, and its fixed backend order is not balanced.
Low accuracies (including PageBlocks) are retained, not used to tune stopping,
change graph construction or discard observations.

## Runtime and original cost estimate

Native controller process ran **2026-10-13 21:18:34.297891–21:26:00.454819 UTC**
(18:18:34–18:26:00 in America/Sao_Paulo): **446.157108 s (7 min 26 s)**.
The single recorded execution session lasted 444.279019 s; process startup precedes
it. Measured propagation totals **438.027105 s**, measured fits **438.047257 s**,
and warm-up fits **0.807019 s**. There were **7,723,492 actual measured iterations**,
32.18% of the 24-million cap. The 23 min 20 s median cap projection was exceeded
downward by **15 min 54 s (68.14% less wall time)**; actual wall time is 18.15% of
the conservative 40 min 58 s scenario. This comparison mixes accounted-cost
projection and observed process duration; it is not a calibrated kernel speedup.
Stopping and state-dependent throughput both change workload.

Process time includes startup, preparation, warm-ups, fit checks, fsync/checkpoint
bookkeeping, summary generation and lightweight monitoring. It excludes launcher
preservation hashing, subsequent audit/reporting and tests. No runtime threshold
or automatic retry was used. External 15-second resource checks found no severe
RAM/disk shortage. Frequency, affinity and foreground activity were not controlled.

The same native Windows/pypcc-dev environment and binary were used: Python
3.12.15, NumPy 2.4.6, Numba 0.68.0, Cython 3.2.4, SciPy 1.18.1, scikit-learn
1.9.1, Intel Core i7-14700K. The original extension compiler/build log remains
unavailable. Cython binary SHA-256 is unchanged:
`e29083201d16ae30b1470f7abd878ec922c2ba645bff77aa50522a49b39b92a0`.

## Correctness, durability and preserved artifacts

All original `measure` checks passed: backend identity/no fallback, no measured
Numba compilation, finite bounded normalized dominance, preservation of labeled
dominance/predictions, valid particle positions/strengths, uint8 distance-table
shape/home distances, and exact original stopping schedule/patience/metrics.
The controller validated per-task identities, ordering, seed policy, timings and
all six final hashes before publishing each immutable result. No production loop
or state update was changed for checkpointing.

The offline audit revalidated the independent plan pin, source/environment/binary
identity, event hash chain, sessions, every warm-up, every canonical result and
summary generation. It rehashed all 16 NPZ fixtures, recomputed historical raw,
scaled-data, ordered-graph and label fingerprints, and matched their complete array
and fixture hashes to PR #16. The exact 48-task inventory is complete and no
attempt is ambiguous or awaiting reconciliation. **552 prior fixed/sensitivity/
first-pilot files** matched their preservation snapshot byte-for-byte.

The [flat raw measurements](benchmarks/early_stopping_second_calibration/results/measurements.jsonl),
[CSV (archived)](performance_early_stopping_archiving.md),
[audit/projections](benchmarks/early_stopping_second_calibration/results/summary.json)
and canonical plan/pin/events/results/sessions/warm-ups/fixture metadata/summary
records under `results/raw/` are committed, alongside execution/resource logs.
The full original root remains at
`.cache/performance/windows-second-calibration-4ccca3d-01/`, including all NPZs,
staging files, preservation snapshot, exact runtime source bytes and committed
source tar. The committed [local artifact index (archived)](performance_early_stopping_archiving.md)
covers every original local file. Git text conversion may normalize CRLF in
committed copies; they do not replace original local bytes. No NPZ/source tar is
committed and no Release asset was published or updated.
Final review additionally hardened event validation against unknown retry-task
IDs, with a regression test. The measured source remains the unchanged `4ccca3d`
snapshot. Offline analysis validates those archived source bytes and the measured
binary; live resume still refuses any source/environment revision.
All 22 final recovery tests and another small real-backend end-to-end validation
passed after that hardening. No historical fit was rerun.

Reproduce the offline audit without any fits, using a new output path:

```console
python -m tools.analyze_second_calibration --root .cache/performance/windows-second-calibration-4ccca3d-01/campaign --baseline .cache/performance/windows-fixed-full-c9f512c-01/fixed-10000 --execution .cache/performance/windows-second-calibration-4ccca3d-01/execution.json --output NEW_ANALYSIS_DIRECTORY
```

Read-only status and resume validation remain available through the controller.
A completed-campaign resume runs no fits or warm-ups. Ambiguous attempts always
require explicit retry authorization; no such retry was necessary here.

## Revised full-campaign estimates

The proposed full protocol remains 20 frozen datasets × ten label masks × three
backends × six complete backend permutations: **3,600 measured fits, 1,200
two-iteration warm-ups and 20 graph preparations**, original criterion and
500,000 cap. No full campaign has been executed or authorized here.

| Scenario | Seconds | Hours |
| --- | ---: | ---: |
| Observed stopping behavior, including capped work | 36,221.484 | 10.06 |
| Cap work at observed median rates | 89,233.474 | 24.79 |
| Cap work at observed maximum rates | 89,498.134 | 24.86 |
| Original historical median cap scenario | 108,665.390 | 30.18 |
| Original historical maximum cap scenario | 126,794.746 | 35.22 |
| Conservative cap scenario with 50% headroom | 190,192.118 | 52.83 |

The stopping-based forecast multiplies the second calibration's summed
propagation times by 60 (ten masks × six timing repetitions), and the first
pilot's summed per-group medians by 30 (two observed masks, extrapolated to ten,
× six repetitions). Censored fits remain charged at their caps; their future
convergence is not inferred. Observed rate-cap scenarios replace actual counts
with 500,000 at each group's observed per-iteration rate, using first-pilot medians
or maximum observed rates separately. For the 16 single-fit groups, median and
maximum rates coincide; they are not an observed variability estimate.

To compare with prior planning, scenarios add the original **3.844604 s**
historical initialization/postprocessing, warm-up, preparation/helper charges.
The fit-only stopping projection is separately retained in analysis JSON.
The conservative scenario is 1.5 × the larger historical maximum-rate or observed
maximum-rate cap projection, hence unchanged at **52.83 hours**. This is planning
headroom, not an execution cutoff or runtime guarantee. Actual checkpoint,
startup/restart and new warm-up costs require allowance; rates can change with
state, machine activity and longer sessions.

**Recommendation:** use **10.06 hours** as a tentative stopping-based forecast,
retain about **24.8 hours** of observed-rate cap work and the **52.83-hour**
conservative scenario for approval. The calibration establishes implementation
readiness and provides rates for all datasets, but only one mask for 16 datasets
and two masks for four. It cannot guarantee the other masks' convergence, cap
fractions, accuracy or timing. PenDigits is censored across all backends; PageBlocks
and Spambase have mixed outcomes, and Satimage from the first pilot was near cap.
Keep the criterion and cap unchanged, preserve censored observations, use six
balanced backend permutations, and execute the full campaign only after explicit
maintainer authorization. README performance tables, frozen scientific artifacts,
fixed results, Release and PR merge state remain unchanged. PR #17 stays a draft.
