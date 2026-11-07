# Canonical undirected k=3 vs k=10 Digits mobility: completed results

**Status:** Experimental only, dependent **draft PR #184** on top of #183/#182; all unmerged. No production-code changes. The [protocol](numpy_undirected_sparsity_mobility_protocol.md) was committed **before outcomes**. This experiment is intentionally distinct from older **directed** 10-NN class lockout results and the prior official union-10NN 50k prevalence study. Previous prospective P10/default potential stopping study [#174](https://github.com/fbreve/pypcc/pull/174) already addressed the stopping comparison on 48 held-out cases; this run does not retune stopping.

## Frozen runs and validation

- [GitHub Actions 42377610276](https://github.com/fbreve/pypcc/actions/runs/42377610276): **4/4 seed jobs plus mandatory 8-case summary PASS**. [Aggregate summary artifact 13576152825](https://github.com/fbreve/pypcc/actions/runs/42377610276/artifacts/13576152825). Raw complete per-window JSON, per-class statistics, graph component sizes, observed/unobserved run timings and dependency versions: seeds 20107 **13576835497**, 20201 **13576651293**, 20309 **13576680575**, 20411 **13576352041**.
- [PCC regression 42377610215](https://github.com/fbreve/pypcc/actions/runs/42377610215) **green** at the same experimental source commit.
- Preselected Digits seeds **20107, 20201, 20309, 20411**, exactly 2 labeled nodes/class, same initially labeled nodes between k choices for a seed. `StandardScaler` and original production `pcc_graph.build_knn_graph`; two **reciprocal, self-loop-free, undirected union** graphs: k=3 versus k=10. Both graphs had **one connected component** across all four seeds; k=3 mean degree **4.40**, k=10 **14.04**. This is a sparse-connectivity comparison, **not** a disconnected graph test.
- Native NumPy sequential, `p_grd=.5, delta_v=.1, deltap=1., dexp=2.`, 100,000 iterations with `early_stop=False`. **16 real native fits:** one ordinary and one uninterrupted per-step observed 100k fit for each of 8 seed×k conditions. Every observed fit has 100001 unique physical iteration callbacks (0–100000), 100 disjoint 1000-step windows per class, and no duplicated terminal physical callback. Instrumented and uninstrumented fits match **all six entire final arrays, complete native stop info, entire global MT19937 state and next 16 draws exactly** (8/8), with the true final low-level ownership state and RNG independently checked. No tolerance or dropped cases.
- The predeclared operational **whole-class** lockout requires five successive 1k windows of zero accepted moves by *every* class particle, maximum class strength ≤0.01 and no class-winning outgoing neighbor entry. Since graphs have no self-loops and each native particle has exactly one destination attempt per full iteration, changed position across consecutive full iterations is an accepted move, including out-and-back paths that would escape sampled endpoint detection. This is an observed operational state, **not** proof of permanent trapping.

## Results

| Seed | k=3 accuracy | k=10 accuracy | k=3 minus k=10 (pp) | k=3 max stationary steps | k=10 max stationary steps | Class lockouts (k3/k10) |
|---|---:|---:|---:|---:|---:|---|
| 20107 | 81.99% | 83.91% | −1.913 | 10 | 15 | 0 / 0 |
| 20201 | 88.97% | 89.53% | −0.563 | 10 | 12 | 0 / 0 |
| 20309 | 88.97% | 89.76% | −0.788 | 8 | 11 | 0 / 0 |
| 20411 | 87.17% | 90.04% | −2.870 | 7 | 12 | 0 / 0 |

**Zero lockouts across all 8 runs**, zero 1000-step class-windows with zero accepted movements across **8,000 class-windows**, zero full-system iterations without *any* particle moving across **800,000 actual steps**, and all **160 actual particle trajectories** changed position at least once. The maximum individual stationary duration was **15** consecutive iterations, in k=10 seed 20107. For this small controlled cohort, k=3 did not induce class lockout even at 100k, despite having about one-third the average degree of k=10.

k=3 yielded lower held-out accuracy on the *same initially unlabeled nodes* in all four paired seeds, with mean signed difference approximately **−1.534 percentage points** (range −2.870 to −0.563 pp). This does not establish a general topology effect with n=4 and should not be conflated with particle-lockout mechanism: there were no qualifying lockouts in either graph. The two graph constructions have different stochastic movement paths, so identical numeric seed does not mean synchronized cross-graph RNG trajectories.

## What remains unsupported

- **Disconnection was not exercised:** both production graphs had exactly one component. Deliberately disconnected, bottlenecked or much lower-degree reciprocal graphs could produce different behavior. This study does not imply that undirected graphs can never exhibit whole-class lockout.
- **No mathematical convergence guarantee:** 100k is a fixed horizon; longer runs or other seeds can behave differently.
- **No conclusion about the default stop rule:** the experiment intentionally disabled early stopping and did not retest previously completed stopping-cohort work.
- **No intervention evidence:** the absence of lockout offers no support for force floors, particle relocation or resets. Old cases involving a **directed** graph cannot be presented as ordinary PyPCC default topology.

## Decision

Keep #184 **draft and unmerged**. Preserve #182/#183. There is no current evidence supporting a production particle reset, stopping-rule change, or smaller default k. A separately preregistered stress test could investigate genuine disconnected or narrow-bridge **undirected** graphs with independent new seeds, accompanied by a controlled comparison to the default connected topology; do not choose new settings based on the four observed seeds.
