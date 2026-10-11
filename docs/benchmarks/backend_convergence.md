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
