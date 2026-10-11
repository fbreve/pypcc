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
Time covers `fit_predict`, including state initialization, excluding graph
construction and warm-up. Execution order alternates; each seed uses the
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
