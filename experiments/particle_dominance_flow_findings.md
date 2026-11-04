# Particle-level dominance transfer audit: Wine seeds 37 and 83

Source: GitHub Actions run **41898112890**, artifact `particle-dominance-flow` (`particle-class-flow.csv`, `particle-pairwise-flow.csv`), commit `bc983c5`. All **13** workflows at that commit passed, including PCC regression.

## Method

Instrument the **experimental sequential NumPy** kernel only, with per-visit counters of valid particle destinations, greedy choices, accepted moves, and actual donor-to-visiting-class dominance transfers (self-reduction cancels, so is excluded). Collect class mass and classification checkpoints on initially unlabeled nodes every 5,000 iterations from 50,000 to 100,000 in two uninterrupted runs. The instrumented and control runs agree on final predictions. For each class and window, the recorded net transfer is compared to observed class-mass change and must agree within 1e-7.

**Integrity:** Maximum absolute mass-balance residual among all class/window records was ~1.01e-12 for seed 37 and ~6.75e-14 for seed 83. These confirm that the observed class-mass redistribution is explained by measured particle transfer operations within rounding error; they do not prove every aspect of the algorithm is correct.

## Aggregates over iterations 50,001–100,000

Each class has two particles, each making 50,000 iterations: 100,000 valid visits per class.

| Seed | Class | Accepted moves | Transfer received | Transfer donated | Net class mass change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 37 | 0 | 94,382 | 497.792 | 518.325 | -20.533 |
| 37 | 1 | 90,564 | 541.086 | 522.609 | +18.477 |
| 37 | 2 | 58,843 | 32.603 | 30.546 | +2.056 |
| **83** | 0 | 99,568 | 28.153 | 26.247 | +1.906 |
| **83** | **1** | **0** | **3.676×10⁻¹³** | **43.739** | **-43.739** |
| **83** | 2 | 99,069 | 62.658 | 20.825 | +41.834 |

In seed 83, **both class-1 particles made no accepted moves in any of the ten 5k intervals**, despite 100,000 valid visits, all to unlabeled nodes; 50,007 were flagged greedy. Their received dominance transfer in successive 5k windows decayed from ~3.676e-13 at 50k→55k to ~2.063e-226 at 95k→100k. This strongly suggests a locked-out or near-extinct competing particle class, but does not establish exactly when or why its mobility vanished.

Seed 83 pairwise flows 50k→100k:
- class 1 donated **36.411** to class 2, **7.328** to class 0, and received effectively zero from either;
- class 0 donated **26.247** to class 2;
- class 2 donated **20.825** to class 0.

Seed 37 instead exhibits large **bidirectional exchanges**: class 0→1 ~514.158, 1→0 ~494.174, with 90,564 accepted class-1 moves. Large late flips here occur with continued particle mobility, so seed 37 and seed 83 are **not** the same failure mode.

## Mechanistic interpretation and next check

The sequential kernel updates an unlabeled destination by removing up to `step = part_strength * delta_v / (c - 1)` from each donor class and transferring it to the visitor class. Acceptance requires the particle's own class to be at the destination's maximum dominance. A class that loses access to winning neighbors can stop moving; weak strengths can limit its ability to reclaim dominance, allowing competitors to drain its former territory. This feedback loop is a **hypothesis** about seed 83, not yet a proven implementation bug or an expected algorithm property.

Next, inspect **the complete 0–100k trajectory** for seed 83 at 1k resolution, including accepted moves, particle strengths, positions, and count of accessible winning neighbors by class. A read-only state callback and matched uninstrumented control must establish the point at which class-1 mobility vanishes. Do not alter production PCC or the experimental stopping policies until this is understood.
