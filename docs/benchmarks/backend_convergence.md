# Windows backend convergence benchmark

## Protocol and stopping audit

This benchmark preserves the PR #4 cases: Wine (178 nodes) and Digits (1797 nodes), full transductive StandardScaler normalization, saved 10-nearest-neighbor graphs, three initial labels per class and seeds 0?4. Data, graph and label fingerprints match the previous artifacts. Each seed has one measured run per configuration: 40 runs. Two short warmups precede each dataset/backend; execution order rotates cyclically. Parameters remain `p_grd=0.5`, `delta_v=0.1`, `deltap=1`, `dexp=2`, `es_chk=2000`; only `early_stop=True` changes from the previous protocol, with the same `max_iter=10000`.

NumPy synchronous is an experimental variant with different update dynamics. It is not algorithmically equivalent to the sequential implementations.

All backends check mean maximum dominance after actual iterations **1, 11, 21, ...**. NumPy uses array reductions; Numba/Cython use serial reductions. Strict improvement resets the stalled-check counter; otherwise it increments. Stopping requires `stop_cnt > stop_max`. None of these operations or comparisons changed.

`stop_max = max(1, round((nodes / (particles * k_nn)) * round(es_chk * 0.1)))`. The parameter `es_chk` contributes to the threshold; it is not the interval between checks.

| Dataset | Particles | stop_max | Earliest possible early stop |
|---|---:|---:|---:|
| Wine | 9 | 396 | 3971 iterations |
| Digits | 30 | 1198 | 11991 iterations |

The initial checkpoint improves from zero; a complete subsequent plateau needs `stop_max+1` further checks. **Digits cannot stop early within the requested 10000-iteration budget.** All Digits runs reach the cap; this does not demonstrate convergence. No threshold was adjusted to force a stop.

Approved minimal observability appends `return_info=False` to low-level propagation functions. Defaults still return `None`. Opting in reports effective iterations, checks, reason, cap status, threshold and stalled checks. Kernels report final counters without callbacks or tracing; checks derive from the unchanged schedule, rather than being mistaken for iterations. The high-level `fit_predict` return is unchanged.

## Results

Mean ? sample SD (`ddof=1`, five seeds). Accuracy uses only unlabeled nodes. Speedup is the ratio of mean propagation times to NumPy sequential on the same dataset.

| Dataset | Configuration | Propagation (s) | Actual iterations | Accuracy (%) | Speedup | Early / capped |
|---|---|---:|---:|---:|---:|---:|
| Wine | numpy_sequential | 0.935594 ? 0.113648 | 9596.2 ? 902.9 | 95.503 ? 0.324 | 1.00? | 1 / 4 |
| Wine | numba_sequential | 0.004332 ? 0.000829 | 7794.8 ? 1554.5 | 95.148 ? 1.349 | 215.96? | 4 / 1 |
| Wine | cython_sequential | 0.004035 ? 0.000937 | 9142.6 ? 1254.3 | 95.148 ? 1.138 | 231.85? | 3 / 2 |
| Wine | numpy_synchronous_experimental | 0.862754 ? 0.150998 | 9256.4 ? 1641.0 | 95.621 ? 1.296 | 1.08? | 2 / 3 |
| Digits | numpy_sequential | 3.147908 ? 0.009665 | 10000.0 ? 0.0 | 88.115 ? 3.014 | 1.00? | 0 / 5 |
| Digits | numba_sequential | 0.020464 ? 0.000728 | 10000.0 ? 0.0 | 87.991 ? 2.016 | 153.83? | 0 / 5 |
| Digits | cython_sequential | 0.025880 ? 0.001130 | 10000.0 ? 0.0 | 88.727 ? 1.831 | 121.63? | 0 / 5 |
| Digits | numpy_synchronous_experimental | 1.035305 ? 0.015113 | 10000.0 ? 0.0 | 88.308 ? 2.482 | 3.04? | 0 / 5 |

Paired comparisons match seeds and labels. Accuracy differences are percentage points; paired speedups are the mean ? SD of the five individual time ratios. Prediction differences count all nodes; labeled predictions are preserved.

| Dataset | Configuration vs NumPy sequential | Accuracy difference (pp) | Paired speedup | Different predictions |
|---|---|---:|---:|---:|
| Wine | numba_sequential | -0.355 ? 1.227 | 225.68 ? 63.99 | 2.6 ? 1.1 |
| Wine | cython_sequential | -0.355 ? 0.990 | 241.76 ? 63.78 | 2.6 ? 0.9 |
| Wine | numpy_synchronous_experimental | +0.118 ? 1.138 | 1.13 ? 0.33 | 1.4 ? 1.1 |
| Digits | numba_sequential | -0.125 ? 3.214 | 153.98 ? 5.44 | 102.6 ? 47.0 |
| Digits | cython_sequential | +0.611 ? 3.840 | 121.82 ? 5.38 | 109.6 ? 62.1 |
| Digits | numpy_synchronous_experimental | +0.192 ? 4.010 | 3.04 ? 0.04 | 103.6 ? 60.1 |

## Individual stopping results

Wine cells: iterations / checks / reason. Digits: 10000 iterations / 1000 checks / cap for every configuration and seed. Individual timing and accuracy are in the CSV.

| Seed | numpy_sequential | numba_sequential | cython_sequential | numpy_synchronous_experimental |
|---|---|---|---|---|
| 0 | 7981 / 799 / early_stop | 10000 / 1000 / max_iter | 10000 / 1000 / max_iter | 10000 / 1000 / max_iter |
| 1 | 10000 / 1000 / max_iter | 8161 / 817 / early_stop | 9041 / 905 / early_stop | 10000 / 1000 / max_iter |
| 2 | 10000 / 1000 / max_iter | 7981 / 799 / early_stop | 9661 / 967 / early_stop | 10000 / 1000 / max_iter |
| 3 | 10000 / 1000 / max_iter | 7061 / 707 / early_stop | 7011 / 702 / early_stop | 9961 / 997 / early_stop |
| 4 | 10000 / 1000 / max_iter | 5771 / 578 / early_stop | 10000 / 1000 / max_iter | 6321 / 633 / early_stop |

## Timing and methodological limitations

Shared loading, scaling and graph construction are measured once per dataset, separately from per-fit preparation. Preparation includes model/graph attachment and particle initialization; propagation includes the backend call and optional metrics-return overhead; prediction postprocessing is separate. RNG seeding is outside measured times. Mean preparation is approximately 0.10?0.11 ms for Wine and 0.21 ms for Digits; detailed SDs and shared setup times are recorded.

The first two-iteration Numba propagation took 0.736058 s, including compilation/loading; the separate seed-helper first call took 0.179604 s. Warmups are recorded separately; measured runs added no Numba signatures. Cython was regenerated and built natively with MSVC.

Cython resets its RNG to 123456789 on each call, as before. NumPy and Numba receive seeds 0?4 but have independent RNG implementations. Identical labels and seed numbers do not imply identical random trajectories. Different classifications or stopping times alone do not establish a semantic defect; reduction rounding can also matter under strict comparisons.

Wine speedups mix execution cost with different iteration counts. The summary additionally records ratios of mean microseconds per iteration: approximately 175? Numba, 221? Cython and 1.04? synchronous. Digits has equal iteration counts and is a cleaner timing comparison. NumPy sequential particle loops and repeated dominance scans at stopping checkpoints are plausible costs; no profiling or optimization was performed. Five seeds with one timing each do not isolate repeated-run noise or establish an accuracy advantage. This stopping rule is not a guarantee of stable classifications.

All 40 executions passed stopping-accounting and state-invariant checks. All 30 capped executions reproduced previous fixed-budget prediction hashes exactly. Previous artifacts remain untouched. Deterministic tests cover plateau stopping, exhausted and zero budgets, stopping exactly at the cap, and unchanged states/subsequent RNG with metrics enabled or disabled. All 44 tests passed on Windows across NumPy, Numba and native Cython; CI runs them on Python 3.10 and 3.12.

## Reproduction and artifacts

Windows 11, Intel Core i7-14700K; Conda `pypcc-dev`, Python 3.12.15, NumPy 2.4.6, Numba 0.68.0, Cython 3.2.4, SciPy 1.18.1, scikit-learn 1.9.1, setuptools 84.0.0. JSON records backend source commit and native binary hash.

From the repository root, initialize the installed MSVC amd64 developer shell and regenerate/rebuild Cython in `pypcc-dev`:

```powershell
& 'C:\Users\fbrev\miniforge3\Scripts\conda.exe' run -n pypcc-dev --no-capture-output python -c "import sys,runpy,functools,Cython.Build; Cython.Build.cythonize=functools.partial(Cython.Build.cythonize,force=True); sys.argv=['setup.py','build_ext','--inplace','--force']; runpy.run_path('setup.py',run_name='__main__')"
& 'C:\Users\fbrev\miniforge3\Scripts\conda.exe' run -n pypcc-dev --no-capture-output python docs/benchmark_backend_convergence.py --prepare-only
& 'C:\Users\fbrev\miniforge3\Scripts\conda.exe' run -n pypcc-dev --no-capture-output python docs/benchmark_backend_convergence.py --output-dir docs/benchmarks/reproduction
```

That output directory preserves both prior and committed results. The script checks fingerprints, native Windows Cython and actual backend identity, rejecting fallback. Run without Python `-O`, which disables audit assertions.

- [Script](../benchmark_backend_convergence.py)
- [Individual runs](backend_convergence.csv)
- [Summary](backend_convergence_summary.csv)
- [Warmups](backend_convergence_warmups.csv)
- [Protocol and environment](backend_convergence.json)
- [Previous benchmark](backend_comparison.md)

## Second configuration: 30000-iteration cap

The follow-up retains `es_chk=2000`, all other parameters, both datasets, all four configurations and seeds 0-4. It runs with the same native Windows `pypcc-dev` environment and backend source as the 10000-iteration benchmark. New files are isolated under `convergence_30000/`; the first run's CSV, JSON, summary and warmups are unchanged. Two warmups and cyclic execution order are retained. There are 40 additional measured runs.

**Wine: all 20 runs stop early. Digits: all 20 runs hit 30000 iterations; no early stop was observed.** The earliest mathematically possible stop (11991) does not guarantee an actual stop. At the final Digits checkpoint each stalled counter remains below the required 1199 checks; the existing strict-improvement rule continues to reset it. This result does not establish convergence and no criterion was adjusted to force it. The plateau regression demonstrates that all backends can stop at iteration 11991 under the Digits threshold when the score really remains constant.

Mean +/- sample SD, five seeds. Accuracy is on unlabeled nodes. Speedup is relative to NumPy sequential within this new configuration.

| Dataset | Configuration | Propagation (s) | Actual iterations | Checks | Accuracy (%) | Speedup | Early / capped |
|---|---|---:|---:|---:|---:|---:|---:|
| Wine | numpy_sequential | 1.172835 +/- 0.276136 | 12437.0 +/- 2839.1 | 1244.6 +/- 283.9 | 95.621 +/- 0.529 | 1.00x | 5 / 0 |
| Wine | numba_sequential | 0.004522 +/- 0.001359 | 8239.0 +/- 2419.1 | 824.8 +/- 241.9 | 95.030 +/- 1.227 | 259.36x | 5 / 0 |
| Wine | cython_sequential | 0.004710 +/- 0.002182 | 11435.0 +/- 5541.2 | 1144.4 +/- 554.1 | 95.148 +/- 1.138 | 249.03x | 5 / 0 |
| Wine | numpy_synchronous_experimental | 0.974112 +/- 0.295393 | 10683.0 +/- 3158.1 | 1069.2 +/- 315.8 | 95.858 +/- 1.183 | 1.20x | 5 / 0 |
| Digits | numpy_sequential | 9.491899 +/- 0.052575 | 30000.0 +/- 0.0 | 3000.0 +/- 0.0 | 88.795 +/- 2.909 | 1.00x | 0 / 5 |
| Digits | numba_sequential | 0.064292 +/- 0.004207 | 30000.0 +/- 0.0 | 3000.0 +/- 0.0 | 88.761 +/- 2.032 | 147.64x | 0 / 5 |
| Digits | cython_sequential | 0.075584 +/- 0.002426 | 30000.0 +/- 0.0 | 3000.0 +/- 0.0 | 89.327 +/- 1.737 | 125.58x | 0 / 5 |
| Digits | numpy_synchronous_experimental | 3.104606 +/- 0.035721 | 30000.0 +/- 0.0 | 3000.0 +/- 0.0 | 89.111 +/- 2.403 | 3.06x | 0 / 5 |

### Cost per iteration and changes from 10000

Microseconds per actual iteration are computed per run, then averaged. Speedup per iteration is the ratio of those means. Time ratio compares new mean propagation time to the old mean; accuracy change is paired by seed, in percentage points. These are separate benchmark sessions, so timing changes also include run-to-run noise.

| Dataset | Configuration | us/iteration, 10000 | us/iteration, 30000 | Per-iteration speedup, 30000 | Time ratio 30000/10000 | Paired accuracy change (pp) |
|---|---|---:|---:|---:|---:|---:|
| Wine | numpy_sequential | 97.265 +/- 3.107 | 94.184 +/- 0.998 | 1.00x | 1.254 | +0.118 +/- 0.265 |
| Wine | numba_sequential | 0.556 +/- 0.013 | 0.548 +/- 0.006 | 171.76x | 1.044 | -0.118 +/- 0.265 |
| Wine | cython_sequential | 0.439 +/- 0.064 | 0.414 +/- 0.014 | 227.61x | 1.167 | +0.000 +/- 0.000 |
| Wine | numpy_synchronous_experimental | 93.252 +/- 0.919 | 90.982 +/- 1.078 | 1.04x | 1.129 | +0.237 +/- 0.324 |
| Digits | numpy_sequential | 314.791 +/- 0.967 | 316.397 +/- 1.752 | 1.00x | 3.015 | +0.679 +/- 0.280 |
| Digits | numba_sequential | 2.046 +/- 0.073 | 2.143 +/- 0.140 | 147.64x | 3.142 | +0.770 +/- 0.163 |
| Digits | cython_sequential | 2.588 +/- 0.113 | 2.519 +/- 0.081 | 125.58x | 2.920 | +0.600 +/- 0.298 |
| Digits | numpy_synchronous_experimental | 103.530 +/- 1.511 | 103.487 +/- 1.191 | 3.06x | 2.999 | +0.804 +/- 0.441 |

Wine speedups based on total time remain confounded by different stopping iterations; the per-iteration comparison isolates work volume more clearly. Digits performs 30000 iterations for every backend and roughly triples propagation time versus the 10000 cap. More iterations do not imply uniform per-seed accuracy improvements or equivalent trajectories. NumPy synchronous remains an experimental reference.

### Individual stopping outcomes at 30000

Wine cells: iterations / checks / reason. Digits has 30000 / 3000 / max_iter in all 20 runs; exact final stalled counters, timing and accuracy are in the CSV.

| Seed | NumPy sequential | Numba | Cython | NumPy synchronous (experimental) |
|---|---|---|---|---|
| 0 | 7981 / 799 / early_stop | 12221 / 1223 / early_stop | 10371 / 1038 / early_stop | 10011 / 1002 / early_stop |
| 1 | 12831 / 1284 / early_stop | 8161 / 817 / early_stop | 9041 / 905 / early_stop | 14851 / 1486 / early_stop |
| 2 | 15891 / 1590 / early_stop | 7981 / 799 / early_stop | 9661 / 967 / early_stop | 12271 / 1228 / early_stop |
| 3 | 12981 / 1299 / early_stop | 7061 / 707 / early_stop | 7011 / 702 / early_stop | 9961 / 997 / early_stop |
| 4 | 12501 / 1251 / early_stop | 5771 / 578 / early_stop | 21091 / 2110 / early_stop | 6321 / 633 / early_stop |

### Compatibility verification and reproduction

All 44 tests passed again on Windows with native Cython. The deterministic plateau regression now exercises the Digits threshold with both caps: 10000 iterations / 1000 checks / cap, and 11991 iterations / 1200 checks / early stop for the 30000 budget. It compares omitted `return_info`, explicit `False`, and `True`: default returns remain `None`, and final positions, strengths, uint8 distances, dominance and own-degree are exactly equal. Subsequent NumPy and Numba RNG values are exactly equal. A nonconstant-dominance case also compares explicit False/True.

Cython's private RNG state is local to the call and not exposed; it cannot be directly read by tests. Its final trajectories are equal across return modes, and source inspection confirms the optional metrics are constructed only after the unchanged loop, without RNG calls. These checks provide regression evidence for the tested cases, not a proof of every possible input.

All 40 new runs pass state and stopping-accounting assertions. Every run that had already stopped early under the 10000 cap reproduces its iteration count, checks, final stalled counter, accuracy and prediction hash under the 30000 cap. Input/RNG fingerprints and parameters match across the two configurations. The algorithm and criteria remain unchanged.

```powershell
& 'C:\Users\fbrev\miniforge3\Scripts\conda.exe' run -n pypcc-dev --no-capture-output python docs/benchmark_backend_convergence.py --iterations 30000 --output-dir docs/benchmarks/convergence_30000_reproduction
& 'C:\Users\fbrev\miniforge3\Scripts\conda.exe' run -n pypcc-dev --no-capture-output python -m unittest -v test_numpy_regression test_numpy_synchronous test_sequential_safety test_propagation_metrics
```

- [30000 individual runs](convergence_30000/backend_convergence.csv)
- [30000 summary](convergence_30000/backend_convergence_summary.csv)
- [Paired comparison with 10000](convergence_30000/comparison_to_10000.csv)
- [30000 warmups](convergence_30000/backend_convergence_warmups.csv)
- [30000 protocol and environment](convergence_30000/backend_convergence.json)

Observing an actual early stop on Digits would require an additional authorized experiment, such as increasing the budget further. This follow-up does not change the requested limits or criteria and does not claim that any unexecuted budget will suffice.

## Digits stopping diagnostic

The optional continuous Numba diagnostic confirms that the score continues setting records while sampled classifications are nearly stable: at most three changed labels per late checkpoint, but 88-241 late record resets per seed. No production kernels changed. States and the next RNG value match monolithic calls for all five seeds. See the [diagnostic report, trajectories and CSVs](digits_stopping_diagnostic/report.md) for magnitudes, stall counters, reset frequencies, overhead and sampling limitations.

## Exploratory alternative stopping candidates

A separate [candidate report](stopping_candidates/report.md) evaluates classification, scalar-dominance variation and hybrid rules retrospectively on existing Digits traces, and on newly collected continuous full-budget Wine references. Original-OR candidates are the most promising exploratory options: they save about 39-41% of Digits iterations with a mean accuracy cost of 0.32-0.34 pp while reproducing the original stopped Wine outcomes in these five seeds. No candidate is implemented in production; endpoint sampling, per-seed losses, unmet evidence requirements and selection bias are documented.
