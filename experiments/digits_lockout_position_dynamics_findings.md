# Digits positional lockout dynamics: classes migrate into hostile true-class regions

## Sources and integrity

GitHub Actions run **41949641070**, artifact `digits-position-dynamics-label2` (ID **13380556707**) and `digits-position-dynamics-label5` (ID **13381225434**); experimental commit `9cf9aee`. Both matrix jobs passed. Each seeded instrumented 50k, no-stop prediction equals the uninstrumented same-seed control. Six selected seed/supervision trajectories were previously characterized; the focused seeds 701 and 809 are **diagnostic, not independent held-out validation**.

Method: Digits with StandardScaler, 10-NN, same graph and seed for the nested two-vs-five labeled-per-class conditions, per-class checkpoint every 1,000 iterations; recorded particle positions, accepted moves, strengths, access to class-winning outgoing neighbors, current-position *true* class and outgoing true-class fraction. **Ground-truth classes are only used offline** for diagnosing the particle locations, not during propagation.

## All qualifying lockouts terminate at wrong-class nodes

| Seed | Labels/class | Locked class | Final particle positions at 50k | Ground-truth classes of positions | First 1k window with zero accepted moves | Class-competitive outgoing neighbors at 50k |
| ---: | ---: | ---: | --- | --- | ---: | ---: |
| 701 | 2 | 2 | 1546;1546 | 1;1 | 8k | 0/20 |
| 701 | 5 | 2 | 1546;1546;99;1546;1546 | 1;1;1;1;1 | 8k | 0/50 |
| 809 | 2 | 4 | 1546;1546 | 1;1 | 35k | 0/20 |
| 809 | 5 | 4 | 314;314;1085;314;314 | 6;6;6;6;6 | 10k | 0/50 |
| 809 | 2 | 9 | 1546;1546 | 1;1 | 19k | 0/20 |

At the final checkpoint, *every affected particle* was positioned on an example with a **different true class** from the particle's class. Each trapped particle's current outgoing 10 neighbors were also all of the *local node's* true class, yielding **zero true-own-class neighboring entries**. The outgoing neighbors' **current PCC dominance** likewise provided zero class-winning entries. This does not show the particles chose ground-truth labels; the true classes are diagnostic metadata.

The nodes involved are *not just node 1546*:
- **1546, 99:** actual Digits class 1, outgoing neighborhoods with 10/10 class-1 examples.
- **314, 1085:** actual Digits class 6, outgoing neighborhoods with 10/10 class-6 examples.

A static reconstruction of the same Digits 10-NN graph (scikit-learn 1.8, standard scaling; validate exact neighbor identity if library tie behavior changes) has directed in-degrees of **4** for node 1546, **5** for node 99 and **1** each for nodes 314 and 1085. This suggests narrow entry routes into strongly homogeneous local neighborhoods; directed reachability alone does not prove these states are absorbing under PCC updates.

## Temporal examples: asymmetry and early partial confinement

- **Seed 701, 2 labels, class 2:** at 5k, both particles occupied true-class-2 positions, still accepting ~1,995/2,000 moves in the latest 1k interval. At 6k, one was already elsewhere (true-class-1 local neighborhood). By **7k**, both were at node 1546, no winning outgoing neighbor and max strength ~5.12e-12, although the 6k→7k interval contained 1,435 accepted movements. **From 7k→8k onwards there were zero accepted movements**, and both remained at node 1546 to 50k.
- **Seed 701, 5 labels, class 2:** by **5k**, three of five particles had reached node 1546 while some were still moving; by **10k**, four were at 1546 and one at 99, all wrong-class positions, max strength zero. No class-2 accepted movements from 7k→8k through 50k. More class-2 particles alone did not rescue the class.
- **Seed 809, 2 labels, class 4:** one particle reached node 1546 by **27k** while the other still moved; at **34k**, both were there, and 34k→35k was the first fully stationary window. Every following window remained stationary through 50k.
- **Seed 809, 2 labels, class 9:** one particle reached node 1546 by **8k** but the other remained mobile for another ~10k iterations; only by **18k** were both at node 1546 and the next full window had zero accepted moves.
- **Seed 809, 5 labels, class 4:** none of the class-4 particles ended at node 1546. Instead four converged to **node 314** (true digit 6) and one to **node 1085** (also true digit 6). At 10k, all five were already in those two positions, max strength ~4.64e-31 and zero class-winning outgoing neighbors. The class was stationary from the 9k→10k window onward.
- **Seed 809, 5 labels, class 9 (not locked):** 3,678/5,000 accepted movements in 49k→50k, max strength 1, with 31/50 winning outgoing adjacency entries at 50k. Its positions were dispersed and it retained access to winning neighbors even though some positions were not true-class-9.

An important distinction: the operational lockout protocol requires **five consecutive** zero-movement windows, low strength and no class-winning neighboring entries. For **seed 701, five labels, class 4**, all five particles happened to finish at node 1546, max strength ~6.89e-193 and zero winning neighbors, but its *first* zero-movement window was **49k→50k**. Therefore it is **not** counted as an operational lockout by the predeclared definition, despite appearing to be approaching that condition at the chosen horizon.

## Interpretation / limitations

Results strongly support a **localized cross-class positional trap** mechanism, not a generic failure of initial labeled-home purity. Initially pure home neighborhoods can be left during motion; particles may enter homogeneous neighborhoods of another ground-truth class and later be unable to win any outgoing target. This can happen at multiple graph nodes and in both supervision levels. In some trajectories only one particle is trapped initially; the whole class stalls once its remaining particles lose mobility.

The graph is a **directed 10-NN** adjacency matrix. The observed failure mechanism may involve directed neighbor geometry, stochastic particle selection, and the collision/strength feedback; this study **cannot isolate** which factor is necessary. Trapping is not proved mathematically irreversible, nor is changing the graph guaranteed to improve predictive accuracy. Fixed 50k caps and selected seeds do not establish general convergence or efficacy of any remedy. Do not merge experimental stopping or rescue rules to `master`.

## Recommended next controlled question

Investigate, **on a new predeclared seed sample**, whether requiring reciprocal neighbor relations (or building a symmetric graph while maintaining comparable degree) changes the incidence of class-particle lockout, *holding supervision and other PCC parameters fixed*. First specify how graph modifications change graph connectivity and effective degree. Treat it as a separate algorithm/graph research experiment, not a production fix. Any proposed improvement must also be evaluated for accuracy and runtime and compared to the unmodified PCC baseline.
