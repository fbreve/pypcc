# Completed undirected nested 0/1/8/32-bridge PCC study

**Status: frozen 16/16 native experimental comparisons PASS; draft PR #186, unmerged.** The [protocol](numpy_undirected_multibridge_protocol.md) was committed before outcomes. This is a stacked *research-only* branch above draft [#185](https://github.com/fbreve/pypcc/pull/185), #184, #183 and #182. No PyPCC production source, graph builder, propagation kernel, stochastic-draw rule, default k, particle movement rule or early stopping policy was modified.

## Provenance and strict reproducibility

- [GitHub Actions 42384358421](https://github.com/fbreve/pypcc/actions/runs/42384358421): four seed jobs plus a mandatory **16/16 full-cohort aggregate PASS**. Complete [automated scientific report, artifact 13579281394](https://github.com/fbreve/pypcc/actions/runs/42384358421/artifacts/13579281394).
- Full raw per-seed JSON containing **all 1000 native class-windows, 100 true crossing windows, actual per-bridge crossing tallies, class/particle crossing counts, particle stationary streaks, accurate stop metadata, held-out accuracy and experimental timing**, plus exact `pip freeze`: seed **22103** artifact `13578629029`, seed **22229** `13578614838`, seed **22307** `13578549590`, seed **22441** `13578809068`.
- [PCC regression 42384358432](https://github.com/fbreve/pypcc/actions/runs/42384358432): **all 4/4 Linux/Windows Python 3.10/3.12 jobs PASS** at the tested experimental-source commit `28a671aeb00fc3f790a6084159910ce6acc1dd6f`.
- Standardized sklearn Digits (1797 vertices, 10 classes); **two original-index-parity components, 899 + 898 vertices**, each independently built using *the original production* reciprocal, loop-free **union-10NN** kNN builder. Graph internal edge set is **identical in all four bridge arms**. Exactly 1 home particle per class per side (20 particles), held fixed within seed. Fresh seeds `22103,22229,22307,22441`, never used in earlier bridge protocols.
- Doses **0,1,8,32** are nested subsets of the same 32 deterministic shortest cross-partition Euclidean pairs, with **no repeated endpoint vertex** (greedy disjoint endpoints, stable ties). The 1-edge case exactly matches prior #185's bridge; every new edge is reciprocal; no self loops or other edge changes; graph has two components only at 0 and exactly one at all positive doses.
- Native sequential NumPy PCC, `p_grd=.5,delta_v=.1,deltap=1,dexp=2,early_stop=False,max_iter=100000`. Each of 16 conditions used **two fresh independent 100k fits**, one ordinary and one true **uninterrupted every-iteration observed** fit (100001 unique real state observations), totaling **32 full native fits**.
- All 16 observed/unobserved same-graph pairs agree **bitwise** on all **six complete final model arrays**, native original stopping metadata, *entire* global NumPy MT19937 state and next 16 generated uniform draws. The genuine final low-level state, including **raw ownership before normalization**, and full RNG state also agree exactly. Synthetic tests validate crossing out-and-back, disallowed bridge crossings, 1000-step window counters, skipped/duplicate callbacks and nested graph construction; no original algorithm code was changed.

## All 16 conditions

| Seed | Bridges | Used bridges | Accepted cross-partition moves | Particles ever crossing (of 20) | Operational class lockouts | Zero-movement class windows | Maximum individual consecutive unmoved iterations | Unlabeled accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 22103 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 75.86% |
| 22103 | 1 | 1 | 2 | 1 | 0 | 0 | 22 | 75.80% |
| 22103 | 8 | 3 | 195 | 4 | 0 | 0 | 18 | 75.41% |
| 22103 | 32 | 18 | 3,788 | 15 | 0 | 0 | 9 | 76.76% |
| 22229 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 88.35% |
| 22229 | 1 | 1 | 206 | 2 | 0 | 0 | 18 | 87.90% |
| 22229 | 8 | 8 | 2,515 | 4 | 0 | 0 | 11 | 88.07% |
| 22229 | 32 | 32 | 7,013 | 16 | 0 | 0 | 13 | 87.56% |
| 22307 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 81.09% |
| 22307 | 1 | 1 | 90 | 2 | 0 | 0 | 12 | 80.87% |
| 22307 | 8 | 8 | 850 | 6 | 0 | 0 | 14 | 80.98% |
| 22307 | 32 | 32 | 5,645 | 16 | 0 | 0 | 11 | 81.20% |
| 22441 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 79.07% |
| 22441 | 1 | 1 | 190 | 2 | 0 | 0 | 14 | 77.60% |
| 22441 | 8 | 8 | 1,890 | 4 | 0 | 0 | 12 | 79.12% |
| 22441 | 32 | 31 | 6,101 | 13 | 0 | 0 | 10 | 79.35% |

**Zero operational class lockouts in all 16 runs; zero 1000-iteration class-windows with zero accepted movement out of 16,000 class-windows.** All **320 particle-run trajectories** moved at least once (the maximum stationary streak of any particle was **22** steps in 100k, not anywhere near 100k). The operational rule stays fixed from earlier experiments: a whole class must have five consecutive 1000-step zero-accepted-move windows, maximum particle strength ≤0.01 and zero class-winning outgoing neighbor entries. This absence **does not** prove a rare lockout cannot ever happen.

## Paired dose-response patterns

Mean over four matched seeds:

| Bridge count | Mean accepted cross-partition moves | Mean unlabeled accuracy | Mean accuracy change vs no bridges |
|---:|---:|---:|---:|
| 0 | 0 | 81.09% | — |
| 1 | 122.0 | 80.54% | −0.549 percentage points |
| 8 | 1,362.5 | 80.89% | −0.197 percentage points |
| 32 | 5,636.75 | 81.22% | +0.127 percentage points |

At 32 bridges the mean crossing frequency was **about 0.282% of the 2 million particle-iteration destination attempts per run**; only 13–16 of 20 particles ever crossed within an individual run (vs 1–2 with one bridge). Not all graph edges are used: at the 32-edge dose the number of bridges actually crossed was **18/32, 32/32, 32/32, 31/32** in the four seeds. Each traversal is counted even if the particle returns within the same sampling window.

Paired true-label accuracy differences in percentage points are visibly nonmonotonic:

| Seed | B1−B0 | B8−B0 | B32−B0 | B32−B1 |
|---:|---:|---:|---:|---:|
| 22103 | −0.056 | −0.450 | +0.900 | +0.957 |
| 22229 | −0.450 | −0.281 | −0.788 | −0.338 |
| 22307 | −0.225 | −0.113 | +0.113 | +0.338 |
| 22441 | −1.463 | +0.056 | +0.281 | +1.745 |

At B=1 the accuracy was lower than B=0 in **all four** selected seeds, but the small n=4 and this single highly artificial partition are insufficient to assert that a single connection generally worsens classification. The high-bridge condition helped **3/4 seeds compared with disconnection**, but did not improve the fourth. Thus a more connected graph supports more **physical traffic**, not necessarily better predictions.

## Interpretive guardrails and decision

1. **Exact graph contrast, not an exact RNG trajectory match across arms.** Each dose changes the graph topology and degree-biased stochastic transitions; matching numeric seeds is *not* common-random-number control. The rigorous full-RNG/state parity applies **within each same-graph ordinary-versus-observed pair**, not between doses.
2. **Traffic is not a substitute for correct propagation.** Two million particle-step attempts per run do not represent independent trials, nor do 1000 windows per class. Physical circulation and prediction accuracy are correlated only descriptively; the experiment cannot demonstrate causal mediation through crossing count.
3. **Convergence and generalizability remain unknown.** 100k steps is a capped reference horizon, four seeds and one partitioned dataset are exploratory, and the graph's two components were created by original index parity rather than natural clusters. A larger network, alternative cut/bridge selection, different labeled-node placement and true naturally disconnected real datasets would require separate preregistered validation.
4. **Previous #185 full-union10 results are not a dose-arm:** that graph differs in many within-partition edges as well as cross edges. Do not use its higher accuracies to infer how many bridges are optimal for the split graph.
5. **No mechanism for a reset is established:** no class ever satisfied operational lockout in this cohort. Resetting strength or position, changing potential stopping, changing default k, or claiming directed graph lockout prevalence for original undirected PCC would be unsupported.

**Recommendation: keep PR #186 and upstream #185/#184/#183/#182 DRAFT and UNMERGED.** The present study completes the predeclared edge-dose question without changes to PCC. A next scientific analysis, if warranted, should focus on a genuinely distinct source of data (different partitions/datasets, difficult low-supervision classes) rather than continue to fit bridge counts to the same four seeds.
