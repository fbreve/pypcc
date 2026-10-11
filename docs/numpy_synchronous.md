# Experimental synchronous NumPy PCC

`ParticleCompetitionAndCooperation(impl="numpy", update_mode="synchronous")`
opts into a variant of PCC. The default `update_mode="sequential"` preserves
the existing sequential implementation, including its random-number stream,
parameter defaults, graph construction, label mapping and stopping criteria.
The additional argument is appended to the constructor and both NumPy entry
points, preserving positional calls. Other backends, including `impl="auto"`,
reject synchronous mode explicitly even when they would fall back to NumPy.
Numba and Cython are unchanged.

This variant is experimental. It may change classification, convergence and
runtime. It does not promise equal predictions, accuracy or random streams
between modes. It uses NumPy vectorization, not multithreaded particle walks.

## Iteration contract

1. Read initial positions, strengths, dominance and distances. Select all
   destinations using the existing random/greedy probabilities and distance
   weighting. Greedy zero-weight walks fall back to random walks.
2. Aggregate all influences on unlabeled targets and commit their dominance
   together. There is no per-particle dominance or movement update in this phase.
3. Every particle uses the **same final aggregated dominance** for its strength
   and shock check. Ties permit movement. A rejected visit still updates strength,
   distance and, for a random walk or greedy fallback, own-degree.
4. Shorten each particle's distance from its initial current node, with the
   existing 255 saturation rule. Commit positions only after the shock checks.

Labeled dominance remains fixed. Isolated particles, invalid current positions
or classes, and visits to invalid neighbors leave their particle state unchanged.
Unused neighbor padding is masked before lookup. Inputs are assumed to satisfy
the usual PCC domain: at least two classes, initial dominance on the simplex,
strength in [0,1], delta_v in [0,1], and deltap in [0,1].

## Donor-specific aggregation

For an unlabeled node, let $D_k$ be its **initial** dominance of class $k$ and
let $s_p$, $y_p$ and $t_p$ denote initial strength, class and selected target of
particle $p$. With $c$ classes and the configured $\Delta_v$, define:

$$H_k = \sum_{p:t_p=v,\ y_p=k} s_p\frac{\Delta_v}{c-1},\qquad
B_j = \sum_{k\ne j}H_k,\qquad L_j=\min(D_j,B_j).$$

Class $j$ is a donor. Its loss $L_j$ is split **only among other visiting
classes**, proportional to their initial influence:

$$T_{j\to k} = \begin{cases} L_j H_k/B_j & j\ne k\text{ and }B_j>0,\\
0 & \text{otherwise}.\end{cases}$$

$$D'_k = D_k-L_k+\sum_{j\ne k}T_{j\to k}.$$

Each loss is bounded by the donor's available initial mass. For $B_j>0$,
$\sum_{k\ne j}T_{j\to k}=L_j$; for $B_j=0$, $L_j=0$. Therefore $D'_k\ge0$
and $\sum_kD'_k=\sum_kD_k=1$, implying $D'_k\le1$. No particle can spend mass
received earlier in the iteration. A single visitor matches the sequential
PCC dominance formula; competing visitors generally do not match an ordered
sequence of visits.

For example, $D=(0.05,0.35,0.60)$ and $H=(0.40,0.20,0)$ give
$L=(0.05,0.35,0.60)$ and $D'=(0.75,0.25,0)$. With two equally strong opposing
visitors to $D=(0.5,0.5)$, the result remains $(0.5,0.5)$ rather than awarding
the node to the last visitor.

The archived experiment used global redistribution of the total loss among
all visiting classes, including donors. The donor-specific rule was selected
explicitly for this PR; the archived code was consulted, not restored.

Implementation aggregates influence with `np.add.at` and computes donor
pressure and gains with matrix operations over affected nodes. These are
reductions, not sequential PCC state transitions. Permuting forced visits
preserves the mathematical result; floating-point summation order may cause
tiny numerical differences. A final clip and row normalization removes
roundoff drift; it is not the conservation mechanism. The tests check the
unclipped rule's expected values using analytical examples.

Random selection is batched and consumes a different stream from the
sequential algorithm, even for the same seed. Greedy roulette intervals are
half-open so a zero threshold cannot choose a zero-weight neighbor. Memory
scales with the selected particles' neighbor matrix and affected nodes by
classes. Donor calculations use a classes-by-classes competitor matrix.

## Verification

```console
python -m unittest -v test_numpy_regression test_numpy_synchronous
```

The existing nine sequential tests are preserved. New deterministic tests
cover constructor compatibility, explicit backend rejection, default/explicit
sequential state and RNG equality, saturated same-class collisions, opposing
classes, analytical donor transfers, conservation over repeated visits,
permutation invariance, snapshot greedy selection, fixed labeled nodes,
final-state shock checks and strength, rejected-visit distances, uint8
saturation, greedy fallback, invalid neighbors and inactive particles.

An additional local comparison loaded the unchanged NumPy implementation
from the base commit and compared complete seeded trajectories and subsequent
RNG output against the new default path. Benchmarks remain optional and are
excluded from regression CI.

## Wine and Digits benchmark protocol

Run from the repository root:

```console
python docs/benchmark_numpy_synchronous.py --seeds 10 --iterations 1000 --repeats 3 --output results.json
```

Both modes share the same standardized dataset, k-NN graph (`k_nn=10`) and
three labels per class selected with `default_rng(seed)`. Seeds are 0–9.
Each mode resets its own NumPy RNG to that seed before each run; the streams
are not assumed equivalent between modes. Use `p_grd=0.5`, `delta_v=0.1`,
`deltap=1`, `dexp=2`, exactly 1,000 iterations and `early_stop=False`.
The original fixed-iteration table below timed `fit_predict`, including state
initialization, excluding graph construction and warm-up. The current script
times propagation only, as required for the early-stopping validation below;
its timings therefore exclude initialization and prediction postprocessing.
Execution order alternates; each seed uses the
median of three timings. Accuracy is measured only on initially unlabeled
nodes. The output includes all timings, seed results and environment versions.

These fixed-iteration measurements do not establish equal convergence or
performance under default early stopping. Changes in accuracy reflect both
the update semantics and the different random consumption. They are a small
paired experiment, not evidence of superior general accuracy.

## Measured results

Environment: Windows 11 (10.0.26300), Python 3.13.13 (conda-forge, AMD64),
NumPy 2.5.4 and scikit-learn 1.9.1.

Times are means across seed medians; accuracy is mean +/- sample standard deviation across seeds.

| Dataset | Sequential time | Synchronous time | Sequential accuracy | Synchronous accuracy | Speedup (seq/sync) |
|---|---:|---:|---:|---:|---:|
| Wine | 0.0867 s | 0.0973 s | 93.55% +/- 1.80 pp | 93.67% +/- 1.93 pp | 0.89x |
| Digits | 0.2900 s | 0.1106 s | 79.78% +/- 2.18 pp | 78.86% +/- 2.88 pp | 2.62x |

Wine, with nine particles, is slower in synchronous mode in this measurement.
Digits, with 30 particles, is faster. Both datasets differ in predictions for
all ten seeds.

Wine: paired accuracy difference (synchronous minus sequential) +0.12 pp, sample standard deviation 2.21 pp; wins/ties/losses 5/2/3.

Digits: paired accuracy difference (synchronous minus sequential) -0.92 pp, sample standard deviation 2.50 pp; wins/ties/losses 3/0/7.

Per-seed results:

| Dataset | Seed | Sequential seconds | Synchronous seconds | Sequential accuracy | Synchronous accuracy | Different predictions |
|---|---:|---:|---:|---:|---:|---:|
| Wine | 0 | 0.0864 | 0.0958 | 94.67% | 90.53% | 13 |
| Wine | 1 | 0.0887 | 0.0966 | 92.31% | 92.31% | 8 |
| Wine | 2 | 0.0832 | 0.0991 | 91.72% | 94.67% | 7 |
| Wine | 3 | 0.0896 | 0.0965 | 94.67% | 95.86% | 4 |
| Wine | 4 | 0.0827 | 0.1007 | 93.49% | 94.08% | 5 |
| Wine | 5 | 0.0911 | 0.0974 | 92.31% | 90.53% | 5 |
| Wine | 6 | 0.0828 | 0.1001 | 96.45% | 94.67% | 7 |
| Wine | 7 | 0.0884 | 0.0943 | 91.12% | 94.08% | 10 |
| Wine | 8 | 0.0847 | 0.0975 | 92.90% | 94.08% | 7 |
| Wine | 9 | 0.0896 | 0.0946 | 95.86% | 95.86% | 6 |
| Digits | 0 | 0.2897 | 0.1066 | 82.00% | 79.12% | 320 |
| Digits | 1 | 0.2863 | 0.1170 | 76.12% | 78.27% | 407 |
| Digits | 2 | 0.2869 | 0.1139 | 79.34% | 74.14% | 345 |
| Digits | 3 | 0.2886 | 0.1070 | 81.04% | 79.40% | 357 |
| Digits | 4 | 0.2942 | 0.1146 | 82.68% | 84.04% | 275 |
| Digits | 5 | 0.2929 | 0.1124 | 77.93% | 80.48% | 341 |
| Digits | 6 | 0.2916 | 0.1057 | 77.59% | 76.68% | 326 |
| Digits | 7 | 0.2918 | 0.1091 | 80.08% | 79.91% | 375 |
| Digits | 8 | 0.2902 | 0.1102 | 78.95% | 75.50% | 378 |
| Digits | 9 | 0.2880 | 0.1090 | 82.06% | 81.10% | 295 |

## Final validation with default early stopping

Reproduce from the repository root:

```console
python docs/benchmark_numpy_synchronous.py --early-stop --es-chk 2000 --iterations 500000 --seeds 10 --repeats 3 --output early-stop-results.json --csv early-stop-results.csv
```

This validation retains the same dataset standardization, graphs, three
labeled nodes per class, seeds 0–9, `p_grd=0.5`, `delta_v=0.1`, `deltap=1`
and `dexp=2`. Only stopping parameters differ from the fixed-iteration study:
`early_stop=True`, `es_chk=2000`, `max_iter=500000`. Each mode is warmed up
with two untimed iterations before the three repetitions for each seed;
execution order alternates by seed and repetition. The RNG is reset before
every repetition, preserving identical starting conditions within a mode.

The benchmark's existing `set_graph` calls do not pass `k_nn`. Consequently,
although graph construction uses `k_nn=10`, `set_graph` retains the maximum
degree for the stopping calculation: 31 in Wine and 35 in Digits. This
unchanged behavior yields `stop_max=128` and `stop_max=342`, respectively.
It is reported here rather than changed during validation.

The production API does not expose iteration counts. A benchmark-only probe
wraps the NumPy step and observes the actual `np.mean` results already used
by the stopping criterion; it neither recomputes the statistic nor changes
its return value, random calls, comparisons or control flow. Both wrappers
are restored after each run. Production code and existing tests are unchanged.

Timing covers only the propagation call, excluding patches/setup, graph
construction, particle initialization, final label mapping and own-degree
normalization. The Python observation overhead is included, without correction,
for both modes. These times are not directly comparable to the earlier table,
which measured the entire `fit_predict` call with fixed iteration counts.

The stopping audit verifies checkpoints at effective iterations 1, 11, 21,
and so on. Strict improvement resets the counter; equality or decrease
increments it. Early termination must occur on the first checkpoint with
`stop_cnt > stop_max`: 129 consecutive non-improving checks for Wine and 343
for Digits. Otherwise the executed count must equal `max_iter`. The CSV
records every measured execution, its actual iteration count, stopping reason,
audit values, accuracy on initially unlabeled nodes, order and cross-mode
prediction mismatch count.

Across-seed summaries use each seed's median of three propagation timings.
All standard deviations are sample standard deviations (`ddof=1`). Accuracy,
iterations and paired accuracy differences use ten independent seed results,
not 30 pseudo-independent repetitions. Execution-level time summaries over
all 30 timings per dataset/mode are also reported separately. Speedup denotes
sequential time divided by synchronous time; both ratio of mean seed timings
and mean of paired seed ratios are reported.

### Early-stopping results

[All 120 measured executions](numpy_synchronous_early_stop.csv) are recorded
separately. Environment: Windows 11 (10.0.26300), Python 3.13.13 (conda-forge,
AMD64), NumPy 2.5.4 and scikit-learn 1.9.1.

| Dataset | Mode | Propagation time, seed medians (s) | Effective iterations | Unlabeled accuracy | max_iter runs |
|---|---|---:|---:|---:|---:|
| Wine | sequential | 0.4920 +/- 0.1470 | 5216.0 +/- 1244.9 | 95.15% +/- 1.27 pp | 0/30 |
| Wine | synchronous | 0.4904 +/- 0.1255 | 5049.0 +/- 1288.5 | 95.15% +/- 1.18 pp | 0/30 |
| Digits | sequential | 19.4941 +/- 5.2255 | 52670.0 +/- 10310.7 | 89.65% +/- 2.16 pp | 0/30 |
| Digits | synchronous | 7.6829 +/- 2.2514 | 55519.0 +/- 10332.2 | 90.08% +/- 1.91 pp | 0/30 |

Paired statistics across ten seeds:

| Dataset | Accuracy difference, sync-seq (pp) | Time difference, sync-seq (s) | Ratio of mean times | Paired speedup | Different predictions | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|---:|
| Wine | +0.00 +/- 0.88 | -0.0015 +/- 0.1446 | 1.00x | 1.04x +/- 0.36 | 2.6 +/- 1.8 | 6/1/3 |
| Digits | +0.43 +/- 2.87 | -11.8112 +/- 4.1892 | 2.54x | 2.62x +/- 0.73 | 73.1 +/- 39.6 | 4/1/5 |

Time distribution across all 30 executions per dataset/mode (not independent accuracy samples):

- Wine/sequential: 0.4983 +/- 0.1861 seconds.
- Wine/synchronous: 0.4901 +/- 0.1210 seconds.
- Digits/sequential: 19.3692 +/- 5.2115 seconds.
- Digits/synchronous: 7.6407 +/- 2.1966 seconds.

Per-seed results (iteration counts and predictions repeated exactly in all three repetitions):

| Dataset | Seed | Seq iterations | Sync iterations | Seq median (s) | Sync median (s) | Seq accuracy | Sync accuracy | Difference (pp) | Speedup | Different predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wine | 0 | 5301 | 7331 | 0.7681 | 0.7173 | 95.27% | 95.86% | +0.59 | 1.07x | 1 |
| Wine | 1 | 4241 | 4621 | 0.3762 | 0.4460 | 95.27% | 93.49% | -1.78 | 0.84x | 3 |
| Wine | 2 | 6451 | 4871 | 0.5737 | 0.4831 | 95.86% | 96.45% | +0.59 | 1.19x | 3 |
| Wine | 3 | 3681 | 3811 | 0.3291 | 0.3772 | 95.86% | 96.45% | +0.59 | 0.87x | 1 |
| Wine | 4 | 7721 | 3641 | 0.6834 | 0.3545 | 95.27% | 95.27% | +0.00 | 1.93x | 0 |
| Wine | 5 | 4551 | 5841 | 0.3964 | 0.5587 | 92.31% | 92.90% | +0.59 | 0.71x | 1 |
| Wine | 6 | 5451 | 4111 | 0.4795 | 0.3957 | 95.86% | 95.27% | -0.59 | 1.21x | 5 |
| Wine | 7 | 5281 | 5601 | 0.4740 | 0.5393 | 94.67% | 95.27% | +0.59 | 0.88x | 5 |
| Wine | 8 | 5711 | 6751 | 0.5064 | 0.6573 | 94.08% | 94.67% | +0.59 | 0.77x | 3 |
| Wine | 9 | 3771 | 3911 | 0.3329 | 0.3753 | 97.04% | 95.86% | -1.18 | 0.89x | 4 |
| Digits | 0 | 48281 | 57881 | 14.4325 | 6.4270 | 89.59% | 87.78% | -1.81 | 2.25x | 48 |
| Digits | 1 | 42371 | 59381 | 12.7049 | 6.5170 | 83.93% | 91.00% | +7.07 | 1.95x | 161 |
| Digits | 2 | 43171 | 49581 | 12.8941 | 5.3765 | 89.08% | 86.02% | -3.06 | 2.40x | 103 |
| Digits | 3 | 67241 | 49041 | 20.0867 | 5.3803 | 90.89% | 90.89% | +0.00 | 3.73x | 57 |
| Digits | 4 | 57171 | 41331 | 23.6576 | 5.7957 | 91.06% | 91.23% | +0.17 | 4.08x | 27 |
| Digits | 5 | 59931 | 54061 | 24.4999 | 8.4737 | 90.61% | 91.40% | +0.79 | 2.89x | 35 |
| Digits | 6 | 43881 | 54931 | 18.8412 | 9.8460 | 91.28% | 91.06% | -0.23 | 1.91x | 54 |
| Digits | 7 | 39651 | 45061 | 16.4446 | 6.9224 | 89.19% | 92.30% | +3.11 | 2.38x | 97 |
| Digits | 8 | 59681 | 74111 | 24.9041 | 11.1007 | 90.89% | 89.25% | -1.64 | 2.24x | 81 |
| Digits | 9 | 65321 | 69811 | 26.4756 | 10.9896 | 89.98% | 89.87% | -0.11 | 2.41x | 68 |

All 120 stopping audits passed. None reached `max_iter`. Prediction differences
occurred in nine Wine seeds and all ten Digits seeds. Time variability was
observed despite deterministic iteration counts and predictions; individual
timings are retained rather than discarded. These are local wall-clock
measurements with observation overhead, not a guarantee of acceleration on
other machines. Early-stopping time differences include both per-iteration
cost and differing numbers of iterations; they must not be interpreted as pure
kernel throughput. The paired accuracy standard deviations do not support a
claim of general accuracy superiority for either mode.

The probe was also checked against uninstrumented execution on a deterministic
small graph in both modes: early stopping at iteration 21 and a capped run of
12 iterations produced exactly equal full states and subsequent RNG output.
All 29 existing regression tests passed unchanged. No production code,
mathematical rule, backend or stopping criterion was modified for this validation.
