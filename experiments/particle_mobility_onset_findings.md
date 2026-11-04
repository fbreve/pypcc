# Onset of class-1 particle immobility in Wine seed 83

Source: GitHub Actions run **41904990332**, artifact `particle-mobility-onset/wine-particle-mobility.csv`, head `c02fc21`. Targeted experiment and PCC regression passed. The latest historical real-data pilot was canceled by CI concurrency; no inference is made from that cancellation.

## Protocol

One uninterrupted, no-stop sequential NumPy PCC trajectory, Wine seed 83, 10-NN graph, 2 labeled points/class. Read-only state checkpoints and particle-event counters every 1,000 iterations from 0 to 100,000; final initially unlabeled predictions exactly matched an uninstrumented 100k control run. Every class has two particles, thus up to 2,000 valid visits per 1k-step window. A “winning neighbor” is a neighboring node at which the particle’s class is tied for the maximum dominance **at the checkpoint**; this is a local accessibility proxy, not a proof that the particle could or could not change a neighbor’s dominance during a future visit.

## Key transition

| End of 1k window | Accepted class-1 moves out of 2,000 | Max particle strength | Unique current positions | Winning neighbors across two positions | Class-1 dominance mass, initially unlabeled | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 46k | 1,932 | 1.0 | 2 | 17/20 | 57.789 | 83.72% |
| 47k | 1,954 | 1.0 | 2 | 14/20 | 57.624 | 83.72% |
| **48k** | **1,286** | **0.002267** | **1** | **0/20** | 57.339 | 83.72% |
| **49k** | **0** | **4.813e-10** | **1** | **0/20** | 55.132 | 84.88% |
| 50k | 0 | 1.150e-14 | 1 | 0/20 | 53.913 | 84.30% |
| 60k | 0 | 1.130e-63 | 1 | 0/20 | 43.564 | 80.23% |
| 70k | 0 | 1.042e-117 | 1 | 0/20 | 17.366 | 62.79% |
| 100k | 0 | 5.410e-261 | 1 | 0/20 | 10.174 | 58.72% |

**Last interval containing any accepted class-1 moves:** (47k,48k]. **First entire 1k window with zero accepted moves:** (48k,49k]; every remaining one of 52 consecutive 1k windows through 100k also had zero accepted moves. At 48k, both particles were co-located, their strengths were low and none of the 20 outgoing adjacency entries (10 for each particle) favored class 1. All **2,000 destination visits** for class 1 were still valid in the 49k window; a zero accepted-move count does not mean no attempted moves.

## Earlier recovery episodes complicate a simple absorbing-state interpretation

- Around 17k–22k one class-1 particle became very weak (e.g., minimum strength ~3.5e-17 at 18k), but the other remained at strength 1. Accepted moves stayed around 970–980 per 1k, and both particle strengths recovered to 1 by 23k.
- At the 39k checkpoint, both particles were co-located, **no winning neighbors at that instant**, and both strengths were around 0.215–0.225. Yet accepted moves in the 39k–40k window numbered 1,842, and both strengths recovered to 1. The absence of winning neighbors at a snapshot is **not sufficient to establish irreversible confinement**.
- At 47k both particles were strong; at 48k they were co-located and near zero strength. This suggests an abrupt transition into self-reinforcing rejection and strength decline, but causal details require inspection around the last accepted movements.

## Mechanistic hypothesis and limits

The kernel updates each particle's strength from the visited target's **own-class dominance even if its movement was rejected**; its position changes only if its class ties for maximum dominance after visiting the target. Thus repeated attempts at hostile neighbors can decrease strength, reduce future ability to overturn neighbors, and exacerbate immobility. The class-1 dominance mass can continue falling because other classes still visit and attack its nodes. This is a plausible feedback loop, **not yet proof of an algorithmic defect**, and the earlier 39k recovery means the exact final transition deserves inspection at finer resolution.

Next: targeted per-particle 100-step windows between 46k–50k (and control at 38k–40k), including exact last accepted iterations, pre/post target dominances and strengths; verify that added observation does not change predictions. Do not introduce a stopping heuristic or modify `master` based on this selected seed.
