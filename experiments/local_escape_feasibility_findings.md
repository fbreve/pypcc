# Frozen-neighborhood one-visit escape feasibility (Wine seed 83)

Source: GitHub Actions run **41916015295**, artifact `local-escape-feasibility/wine-local-escape-feasibility.csv`, commit `5908c10`. The targeted workflow and PCC regression passed.

## Protocol
Experimental NumPy PCC, same graph and seed 83 as preceding lockout audits, no stopping. Frozen snapshots at 39k, 40k, 47k, 47.5k, 47.6k, 47.8k, 47.9k, 48k, 49k, 50k. For each of the two class-1 particles, apply **only one potential visit** to each outgoing neighbor independently, on a copy of that neighbor's dominance row. Compare measured particle strength with fixed strengths 0.01, 0.05, 0.1, 0.2, 0.25, 0.5, and 1.0. The perturbations are not applied to the actual propagation trajectory. Reference predictions match the uninstrumented run.

**Finding:** At 39k, 47.9k, 48k, 49k and 50k, neither particle could immediately reach any winning neighbor or conquer one with any tested strength (including 1.0). Both were at node 8 at those checkpoints. For example:

| Checkpoint | Actual strengths (particles 2/3) | Winning neighbors | Neighbors winning after one visit with strength 1.0 | Best own-class-minus-competitor gap after strength 1.0 |
| --- | --- | --- | --- | --- |
| 39k | 0.21494 / 0.22542 | 0/10 | 0/10 | -0.4492 |
| 47.9k | 0.00652 / 0.000406 | 0/10 | 0/10 | ~-0.933 |
| 48k | 0.002267 / 0.000241 | 0/10 | 0/10 | ~-0.944 |
| 49k | ~4.4e-10 / ~4.8e-10 | 0/10 | 0/10 | -0.900 |
| 50k | ~1.15e-14 / ~4.74e-15 | 0/10 | 0/10 | -0.900 |

At 40k, both class-1 particles had escaped to node 106 with strength 1 and 10 winning neighbors each. At 47k, particles 2 and 3 occupied nodes 38 and 116, with 4 and 10 winning neighbors respectively.

The single-visit calculation cannot explain the **39k→40k recovery**: a neighbor initially losing to another class could become winning through accumulated attacks over multiple visits, or due to other particles' interactions. This does **not** imply that force is irrelevant. Conversely, rescuing only strength after 48k might or might not restore mobility, because repeated attacks plus responses from other classes matter.

**Next discriminating test:** perform controlled, **experimental-only** interventions *at the 48k checkpoint* on a continuous trajectory: (1) restore only class-1 particle strength to 1.0, (2) relocate only the class-1 particles to their original labeled home nodes, (3) both, and (4) unmodified control. Measure accepted movements, class-1 dominance mass, accuracy and agreement through a common later cap. This tests whether strength or position is causally limiting in the observed instance; it is not proposed as a new default algorithm.
