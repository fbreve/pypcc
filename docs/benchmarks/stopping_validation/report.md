# PR #6: fixed-threshold experimental stopping validation

## Result

The original-OR-range prototype improves measured propagation time on the ten new Digits seeds, with a small accuracy loss. It does not activate on Wine, Iris or Breast Cancer, and its observation overhead makes those runs slower. These results support further research, **not a permanent API/default change**. The original PCC implementations and default stopping behavior remain unchanged.

## Protocol fixed before evaluation

Commit `65fb06c` records the prototype, tests and [protocol](protocol.json) before any new dataset/seed measurements. The candidate was not retuned after inspecting results:

- Original stop **OR** mean-maximum-dominance range `<=5e-4` in **three consecutive 500-iteration windows**.
- Original checks occur at iterations 1, 11, 21, ...; windows are 1..501, 501..1001, etc., with 51 observed scores including both endpoints. The first alternative trigger allowed is **5001**, after iteration 5000. Prior windows may contribute to the streak. An excursion at a shared endpoint affects both windows. Original stopping takes priority if both fire simultaneously.
- All 51 scalar observations contribute to each range; this is not merely an endpoint delta. It does not establish that the scalar is bounded between its sampled checks, or that the complete dominance matrix/classifications are stable.
- New seeds **10-19**, Wine/Digits plus **Iris and Breast Cancer**, three timing repetitions per seed with alternating original/prototype order: **240 individual timed runs**, 120 pairs, 40 independent seed cases.
- Same standardized transductive feature matrix, one shared 10-nearest-neighbor graph per dataset and three initial labels per class; identical initial arrays and seed per paired execution.
- `p_grd=0.5`, `delta_v=0.1`, `deltap=1`, `dexp=2`, `early_stop=True`, `es_chk=2000`, `max_iter=30000`. No graph, labeling or algorithm changes. Saved node indices and data/graph/label fingerprints permit reproduction.

The reference uses the **original criterion with a 30000 cap**, not forced uninterrupted execution: Wine and Iris can stop sooner. Earlier PR #5 seed results were not mixed into these statistics. Accuracy uses only unlabeled nodes. Candidate decisions never consume true labels or accuracy.

## Implemented scope and timing

The prototype is isolated in [experiments/stopping_range.py](../../../experiments/stopping_range.py), outside the public API. It advances unchanged Numba sequential kernels in continuous blocks and evaluates the original reduction/counter plus the range observer outside the compiled particle loop. It never reseeds RNG, changes particle state, or replaces a production function. NumPy/Cython and the synchronous variant are untouched. The observer supports only this explicit research protocol, not a public `PCC` option.

Timed propagation includes one-time input validation, all Python dispatch/block bookkeeping, original metric checks, range observation and final metric return. It excludes JIT, RNG seeding, initial-state copies, correctness verification, graph setup and output writing. Setup/initialization are recorded separately; initialization timing includes capturing a reusable state snapshot and is not a normal public-API fit benchmark. This is an actual prototype wall-time comparison, not a conversion from saved iterations.

Both paths receive warmup; measured runs add no Numba signatures. First/warm calls and environment/binary metadata are recorded. Each seed's mean time over three repeats enters the aggregate, with sample SD across ten paired seeds; the 30 repetitions are not treated as 30 independent classification samples. Speedup is original mean time divided by prototype mean time; a value below one means the prototype is slower.

## Iterations and accuracy

Mean +/- sample SD across ten seeds. Accuracy differences are paired percentage points against the original run on the same seed. Original and prototype individual accuracies are also saved in CSV.

| Dataset | Original iterations | Prototype iterations | Alternative triggers / 10 | Iterations saved (%) | Accuracy change (pp) | Worst change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Wine | 9844.0 +/- 2494.9 | 9844.0 +/- 2494.9 | 0 | 0.00 +/- 0.00 | +0.000 +/- 0.000 | +0.000 |
| Digits | 30000.0 +/- 0.0 | 18901.0 +/- 2875.2 | 10 | 37.00 +/- 9.58 | -0.300 +/- 0.131 | -0.509 |
| Iris | 7841.0 +/- 3196.2 | 7841.0 +/- 3196.2 | 0 | 0.00 +/- 0.00 | +0.000 +/- 0.000 | +0.000 |
| BreastCancer | 30000.0 +/- 0.0 | 30000.0 +/- 0.0 | 0 | 0.00 +/- 0.00 | +0.000 +/- 0.000 | +0.000 |

## Real propagation time including observation

| Dataset | Original (ms) | Prototype (ms) | Speedup, ratio of means | Original accuracy (%) | Prototype accuracy (%) |
|---|---:|---:|---:|---:|---:|
| Wine | 5.511 +/- 1.370 | 8.085 +/- 2.152 | 0.682x | 95.148 +/- 1.471 | 95.148 +/- 1.471 |
| Digits | 62.784 +/- 2.279 | 42.889 +/- 6.649 | 1.464x | 89.593 +/- 2.828 | 89.293 +/- 2.783 |
| Iris | 4.344 +/- 1.920 | 6.255 +/- 2.665 | 0.694x | 91.277 +/- 4.216 | 91.277 +/- 4.216 |
| BreastCancer | 13.237 +/- 0.667 | 21.016 +/- 1.156 | 0.630x | 91.226 +/- 2.189 | 91.226 +/- 2.189 |

Digits cuts iterations by approximately 37% and wall time by approximately 32%, yielding 1.46x speedup despite observation cost. Mean accuracy loss is 0.300 pp, worst 0.509 pp. This held-out-seed result is consistent with the earlier exploratory tradeoff, without selecting a new threshold.

Wine and Iris keep original stopping iterations and accuracy exactly, but take approximately 47% and 44% more propagation time, respectively. Breast Cancer keeps all 30000 steps and accuracy, with approximately 59% added time. Absolute thresholds therefore show no new-dataset stopping benefit in this evaluation; retaining the original rule prevents lost original stops, but does not remove monitoring overhead. The prototype is not a universally beneficial optimization.

## Stopping reasons and invariance checks

Counts below are per seed; every seed replays identically across its three timing repetitions.

| Dataset | Original early / cap | Prototype original / alternative / cap |
|---|---:|---:|
| Wine | 10 / 0 | 10 / 0 / 0 |
| Digits | 0 / 10 | 0 / 10 / 0 |
| Iris | 10 / 0 | 10 / 0 / 0 |
| BreastCancer | 0 / 10 | 0 / 0 / 10 |

For all **90 non-alternative timing pairs** (30 seeds), final positions, strengths, uint8 distances, dominance and own-degree are bitwise equal, and the next Numba random number agrees exactly with original propagation. Thus observational block boundaries do not change the tested default trajectories when the alternative is unused. Stopping reason and effective iterations agree too.

For all **30 alternative pairs**, the actual prototype state/RNG is validated against an unchanged monolithic original-dynamics prefix of exactly the prototype's stopping length, with stopping disabled solely for verification. Those prefix verification calls are outside timings. They demonstrate continuity; independent capped calls are not used to build a hypothetical trajectory. State invariants additionally check finite/bounded/sum-one dominance, preserved labeled nodes, valid particle positions and uint8 distances.

Deterministic prototype regressions cover original stop, cap/zero budget, earliest alternative activation, simultaneous-trigger original priority, an interior/shared-endpoint excursion resetting the window streak, and exact state/RNG compatibility. Production regressions still execute all three backends. **47 tests passed in Windows**, using regenerated, natively MSVC-compiled Cython. The generated C and production sources are unchanged. The CI matrix runs the tests on Python 3.10 and 3.12; long dataset experiments remain optional and outside CI.

## Limitations and interpretation before any permanent proposal

- This is a **Numba sequential prototype** only. It does not establish benefits, overhead, RNG equivalence or threshold suitability for alternative implementations of the observer or other backends.
- Ten seeds per dataset and two new datasets expand the evidence but do not establish generality. The original dataset graphs are held fixed; only labeled splits and propagation RNG vary. All scaling is transductive and follows the existing benchmark, not a supervised train/test protocol.
- Millisecond-scale timings remain noisy; three repeats and order alternation reduce bias but do not provide a broad machine/performance study. Timing overhead belongs to this Python prototype, not an intrinsic lower bound for the mathematical criterion.
- The threshold operates on an aggregate mean, whose variations depend on graph/dataset behavior. Stable aggregate values do not prove stable per-node dominance, stable classifications, or correct predictions.
- Accuracy savings tradeoffs use the original criterion as reference. They do not compare every dataset to uninterrupted 30000-step accuracy or to a known optimum.
- No threshold was modified, no accuracy-driven selection was made after new results, and no further successful settings were searched. Breast Cancer's negative result is retained.
- Preserving state/RNG when inactive does not mean preserving classifications when the alternative fires. Digits' measured loss is the expected consequence of stopping an otherwise identical trajectory earlier.

The results justify continuing an optional research prototype if the maintainer accepts the measured Digits accuracy cost. They **do not yet justify a permanent API option or changing defaults**: observation overhead harms three of four datasets and neither new dataset benefits. A separate decision on acceptable worst-case accuracy loss and further independent validation would precede any permanent proposal. No permanent implementation is proposed in this PR.

## Reproduction and artifacts

The fixed protocol is [protocol.json](protocol.json); environment versions and native Cython binary hash are in [metadata.json](metadata.json). From the repository root in Windows `pypcc-dev`, with the native extension rebuilt from current source:

```powershell
& 'C:\Users\fbrev\miniforge3\Scripts\conda.exe' run -n pypcc-dev --no-capture-output python docs/validate_stopping_range.py --output-dir docs/benchmarks/stopping_validation_reproduction
& 'C:\Users\fbrev\miniforge3\Scripts\conda.exe' run -n pypcc-dev --no-capture-output python -m unittest -v test_numpy_regression test_numpy_synchronous test_sequential_safety test_propagation_metrics test_stopping_range_prototype
```

Use a separate output directory to preserve committed results; do not use Python `-O`, which disables experiment assertions. The script checks that the prototype constants match the frozen protocol and rejects non-Windows execution or absence of native Cython.

- [Reproducible benchmark](../../validate_stopping_range.py)
- [All 240 individual timed runs](runs.csv)
- [120 paired comparisons and validation outcomes](paired_runs.csv)
- [40 seed-level averages](seed_summary.csv)
- [Aggregate statistics](summary.csv)

Previous benchmark results remain unchanged. The work is isolated on `experiment/stopping-range-validation`; retain the PR in draft, with no merge or direct master changes.
