# Wine seed 83: particle-level transition into permanent lockout

Source: GitHub Actions run **41915450126**, artifact `particle-lockout-transition/particle-lockout-transition.csv`, commit `a78a6c2`. The targeted workflow passed. Observed final predictions exactly matched the uninstrumented 50k control.

## Protocol

Sequential experimental NumPy PCC, Wine seed 83, 10-NN and two particles/class, no early stop, maximum 50k. Defensive full-state snapshots every **100 iterations** from 38k–50k, plus opt-in particle-level event logging (origin, destination, accepted/rejected, strength before/after, destination's own-class dominance before/after update). The event callback is disabled outside this range. A separate no-callback run confirms matching final predictions; exact identity of full internal RNG trajectories is not proven by label comparison alone.

## Last accepted movement and location

| Class-1 particle ID | Last accepted movement iteration | Position at 48k | Strength at 48k | Number of winning neighboring adjacency entries at 48k |
| ---: | ---: | ---: | ---: | ---: |
| 2 | **47,841** | **8** | **0.0022666** | **0/10** |
| 3 | **47,571** | **8** | **0.0002407** | **0/10** |

The 100-step window **47,900–48,000** and every subsequent window through 50k had zero accepted moves for either class-1 particle. The earlier 1k-resolution audit confirmed no further moves through 100k. Both remained at node 8. This local stagnation precedes the class-1 dominance-mass collapse already documented.

## Recovery versus final lockout

At **39,000**, both particles were also at node 8, with no presently winning neighbors. Their strengths, however, were **0.21494** and **0.22542**. During 39,000–39,100, each particle accepted **39 movements** and ended the window with strength **1.0** and access to 10 winning neighbors. An initial target not previously controlled by class 1 was successfully conquered during that recovery window. Therefore **having zero winning neighbors at a checkpoint is not sufficient by itself for permanent immobility**.

At 47,000, both class-1 particles still had strength 1 and between 4 and 10 winning outgoing neighbors. Particle 3 became trapped near 47,571; particle 2 made its last accepted move at 47,841. By 48,000 both were confined to node 8 and too weak to recover in observed continuation. Between 47,500 and 48,000 the aggregate accepted moves across the two particles fell sharply:

| End of 100-step window | Accepted movements / 200 | Movements accepted only after target conquest |
| ---: | ---: | ---: |
| 47,500 | 177 | 0 |
| 47,600 | 129 | 1 |
| 47,700 | 98 | 0 |
| 47,800 | 96 | 0 |
| 47,900 | 18 | 1 |
| 48,000 | 0 | 0 |

A few accepted movements into previously non-winning destinations occurred shortly before permanent failure; their presence alone did not guarantee sustained recovery.

## Mechanism and limits

The **production** `pcc_numpy.py` and `pcc_numba.py` kernels, as well as the experimental candidate, update particle strength from the destination class dominance **even on rejected moves**, while position only advances on accepted moves. This is consistent with a feedback loop: local confinement to hostile neighbors lowers strength, thereby limiting the capacity for future dominance takeover. It does **not** establish a code bug or a safe production modification.

**Next diagnostic:** evaluate the *exact one-visit update rule* on frozen 39k and 48k checkpoints, using measured strengths and fixed hypothetical strengths (e.g. 0.1, 0.2, 0.5, 1.0), without changing propagation. The resulting count of immediately conquerable neighbors distinguishes whether a stronger particle has an available local escape path, but is **not** a counterfactual full trajectory or proof that changing the algorithm improves accuracy. Keep stop criteria experimental; do not merge to `master`.
