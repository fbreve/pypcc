# Windows backend benchmark: Wine and Digits

This optional benchmark compares the three sequential backends and includes synchronous NumPy as an **experimental, algorithmically different reference**. It does not change any backend, algorithm, default, or stopping rule.

## Reproduce

Run from the repository root with the native Cython extension already built:

```powershell
conda run -n pypcc-dev python docs/benchmark_backends.py
```

The script requires Windows and a native `.pyd`; it rejects backend fallback. Results are written to `docs/benchmarks/`. To select another budget or seeds, use `--iterations` and `--seeds`. Benchmarking is optional and is excluded from CI.

## Protocol

- Five measured executions per dataset/configuration, using seeds **0, 1, 2, 3, 4**: **40 executions** in total. These are five different labeled splits, not five timings of each seed.
- Full Wine (178 nodes) and Digits (1,797 nodes) feature matrices, standardized with `StandardScaler`; one symmetric Euclidean 10-NN graph is built per dataset and reused by all configurations. This is transductive evaluation: all feature vectors participate in scaling and graph construction.
- Exactly three initially labeled nodes per class: 9 Wine labels and 30 Digits labels. Accuracy is measured only on the remaining 169 and 1,767 nodes. Each seed's known indices and input hashes are retained in JSON; paired CSV rows have identical data, graph and label hashes.
- `p_grd=0.5`, `delta_v=0.1`, `deltap=1`, `dexp=2`, **`max_iter=10000`, `early_stop=False`, `es_chk=2000`**, for every configuration. Every measured run executes 10,000 iterations and stops at `max_iter`. Counts follow the fixed budget; no kernel instrumentation is needed. This does **not** measure time to convergence or early-stop behavior. `es_chk` is inactive with early stopping disabled.
- Backend order is cyclically rotated between repetitions and datasets. Two two-iteration warmups are performed per configuration/dataset before timing full runs.
- NumPy and Numba RNGs are independently seeded before each execution. Cython retains its existing internal xorshift seed **123456789**, reset on each propagation call. The split seed still changes its initial labels. Identical random trajectories are not assumed; the effective backend seed is recorded in CSV. No RNG implementation was changed to force equal predictions.
- Wall-clock time uses `perf_counter`. Shared loading/scaling/graph construction is recorded once per dataset. Per-fit preparation includes model construction, graph copies, particles, dominance, lookup table and buffer initialization up to the propagation boundary. Propagation includes the production entry-point validation. Postprocessing covers label decoding and own-degree normalization. Input invariants and accuracy checks occur after the timer.
- The timer wraps only the public call boundary. The existing production propagation loop executes directly. There is no per-particle observer. The script rejects new Numba propagation signatures during measured runs.
- Means and sample standard deviations (`ddof=1`) use five runs. Time variability combines differing labeled splits/trajectories and wall-clock noise; it does not estimate timing noise for one fixed split. Speedup is the ratio of mean propagation times; paired ratios and paired accuracy differences are also saved.

## Environment

Windows 11 (10.0.26300), Intel Core i7-14700K, 28 logical CPUs; independent Conda environment `pypcc-dev`. Python 3.12.15, NumPy 2.4.6, Numba 0.68.0, Cython 3.2.4, setuptools 84.0.0, SciPy 1.18.1, scikit-learn 1.9.1. Cython was compiled natively with Visual Studio 2022/MSVC 14.44. Source baseline: `e3b0ab6` (master after PR #3). The JSON records the interpreter, platform and binary SHA-256.

## Results

All values are mean +/- sample standard deviation. Accuracy deviations are percentage points. Speedup refers to **propagation**, relative to sequential NumPy.

| Dataset | Configuration | Preparation (ms) | Propagation (s) | Unlabeled accuracy (%) | Speedup |
|---|---|---:|---:|---:|---:|
| Wine | NumPy sequential | 0.113 +/- 0.059 | 0.942498 +/- 0.006567 | 95.50 +/- 0.32 | 1.00x |
| Wine | Numba sequential | 0.104 +/- 0.005 | 0.005313 +/- 0.000079 | 95.15 +/- 1.14 | 177.38x |
| Wine | Cython sequential | 0.081 +/- 0.011 | 0.004045 +/- 0.000306 | 95.27 +/- 1.11 | 233.01x |
| Wine | NumPy synchronous (experimental) | 0.089 +/- 0.010 | 0.903630 +/- 0.015654 | 95.74 +/- 1.28 | 1.04x |
| Digits | NumPy sequential | 0.198 +/- 0.009 | 3.143061 +/- 0.011445 | 88.12 +/- 3.01 | 1.00x |
| Digits | Numba sequential | 0.201 +/- 0.023 | 0.017934 +/- 0.001458 | 87.99 +/- 2.02 | 175.26x |
| Digits | Cython sequential | 0.219 +/- 0.038 | 0.017334 +/- 0.001596 | 88.73 +/- 1.83 | 181.33x |
| Digits | NumPy synchronous (experimental) | 0.220 +/- 0.033 | 1.016040 +/- 0.015515 | 88.31 +/- 2.48 | 3.09x |

All 40 runs used 10,000 iterations (standard deviation 0), with `max_iter` as the stopping criterion. No claim of convergence follows from these results.

| Dataset | Configuration | Paired speedup | Accuracy difference vs NumPy sequential (pp) | Different predictions |
|---|---|---:|---:|---:|
| Wine | Numba sequential | 177.42 +/- 3.31 | -0.36 +/- 1.07 | 2.2 +/- 1.3 |
| Wine | Cython sequential | 234.07 +/- 17.50 | -0.24 +/- 0.90 | 2.0 +/- 0.7 |
| Wine | NumPy synchronous (experimental) | 1.04 +/- 0.02 | +0.24 +/- 1.15 | 1.6 +/- 0.9 |
| Digits | Numba sequential | 176.10 +/- 12.89 | -0.12 +/- 3.21 | 102.6 +/- 47.0 |
| Digits | Cython sequential | 182.47 +/- 15.55 | +0.61 +/- 3.84 | 109.6 +/- 62.1 |
| Digits | NumPy synchronous (experimental) | 3.09 +/- 0.04 | +0.19 +/- 4.01 | 103.6 +/- 60.1 |

Sequential prediction differences can arise from RNG and floating-point trajectories; they are not evidence of a backend correctness defect. Synchronous differences additionally reflect its distinct state-update dynamics. Five labeled splits at one fixed budget do not establish accuracy superiority.

## Initialization and first Numba execution

| Dataset | Load (s) | Standardize (s) | Build graph (s) | Shared total (s) | Maximum degree |
|---|---:|---:|---:|---:|---:|
| Wine | 0.017302 | 0.002510 | 0.002354 | 0.022167 | 31 |
| Digits | 0.019481 | 0.001073 | 0.049476 | 0.070030 | 35 |

The first Numba propagation took **0.740008 s**, including JIT compilation, for just two iterations; the immediately warmed two-iteration call took **0.00002630 s**. These are excluded from the result averages. The independent RNG-seeding helper's first call took **0.176751 s**, also excluded. Cold propagation includes compilation and execution; it is not a standalone compiler-only measurement. The Digits warmups reused the same compiled argument signature and did not repeat the initial compilation cost. Imports and environment startup are outside the fit timer.

## Potential bottlenecks and limits

- Sequential NumPy spends most of its measured fit time in propagation. Source inspection suggests Python work per particle, temporary array allocation and many small NumPy reductions as likely contributors. This benchmark did not profile those components individually, so their separate costs are not established.
- Compiled propagation is only about 4-5 ms on Wine and 17-18 ms on Digits. Initialization/call overhead and timing noise therefore matter proportionally more; the slight Numba/Cython gap is not a universal ranking.
- Shared preparation is larger than compiled propagation in both datasets. When preparing a new graph for each classification, propagation-only speedups overstate the gain for the complete workflow. Charging this run's once-measured shared preparation to each fit projects a Cython speedup of about **36.6x** on either dataset, rather than 233x/181x. This is a projection, not a repeated end-to-end measurement. Reusing a graph amortizes that cost.
- Numba's cold-start cost is substantial relative to its warmed propagation. Workloads with one call must account for it separately.
- Synchronous NumPy offers about 1.04x on Wine and 3.09x on Digits here. It remains an experimental variant and can classify nodes differently; its speed cannot be treated as an equivalent sequential optimization.
- Measurements are local wall-clock results under normal Windows scheduling. No CPU affinity, threadpool cap or power-policy control was imposed. Five seeds are an initial comparison; a convergence benchmark or repeated timing within each seed would answer different questions.

## Artifacts and verification

- [Reproducible script](../benchmark_backends.py)
- [Individual measured executions](backend_comparison.csv)
- [Summary and paired statistics](backend_comparison_summary.csv)
- [First calls and warmups](backend_comparison_warmups.csv)
- [Environment, shared timings, input hashes and known indices](backend_comparison.json)

The 40 records were checked for identical paired input hashes and hyperparameters. The boundary wrapper was compared with direct production calls on a deterministic small graph: predictions and subsequent RNG output matched for all four configurations, and preparation + propagation + postprocessing equaled total fit time. All measured executions passed labeled-node, finite-dominance, dominance conservation and position invariants. No production source or existing test was modified, and no optimization was implemented.
