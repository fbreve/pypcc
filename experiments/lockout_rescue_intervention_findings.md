# Controlled intervention study of Wine seed 83 particle lockout

Sources: GitHub Actions run **41920906992** (successful), artifact `particle-lockout-rescue/wine-lockout-rescue-summary.csv` and `wine-lockout-rescue-trace.csv`, commit `3258310`. The preceding frozen-neighborhood experiment passed (run **41916015295**, artifact `local-escape-feasibility/wine-local-escape-feasibility.csv`). The first rescue workflow failed to import due to an indentation mistake in the newly added validation code; corrected in `3258310`.

## Setup

Experimental sequential NumPy backend, Wine seed 83, 10-NN graph, two initially labeled points per class, no early stopping. All four variants start from identical state and RNG seed. Immediately after completing iteration **48,000**, apply one of:

- **Control:** do not mutate particle state.
- **Strength only:** set strengths of the two class-1 particles to 1.0 without moving them.
- **Home relocation only:** move each class-1 particle to its original labeled home node without changing strength; *historical distance tables are deliberately left unchanged*, which is a limitation of this surgical intervention.
- **Both:** apply both mutations.

Observe through **60,000** iterations. The callbacks record accepted moves, particle strengths, class-1 dominance mass and predicted labels every 1,000 iterations. The instrumented control's final predictions match a separate uninstrumented 60k reference; *all four variants have exactly matching observed prefixes up to 47k*. The model and code are not changed in `master`.

| Intervention | Class-1 accepted moves, 48,001–60,000 (of 24,000 attempted) | Accepted, first 1k | Class-1 mass at 60k | Class-1 predicted nodes at 60k | Accuracy on initially unlabeled at 60k | Agreement with control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 0 | 0 | 43.564 | 43 | 80.23% | 100% |
| Reset strength to 1 | 21,907 | 1,018 | 57.949 | 55 | 83.72% | 93.02% |
| Relocate home | 22,856 | 1,959 | 58.204 | 55 | 83.72% | 93.02% |
| Both | 22,856 | 1,959 | 58.204 | 55 | 83.72% | 93.02% |

Relocating home alone and relocating **plus** resetting strength produce matching summary trajectories from 49k through 60k (identical accepted counts and accuracy; tiny deviations in one window's accepted count are absent between those two variants). Their intervened 48k states do differ in particle strength, but both rapidly recover to mean strength 1.0 by 49k. Strength reset alone also reestablishes ongoing mobility and class-1 dominance, though its trajectory differs from relocation.

In the control, both class-1 particles remain at node 8, with zero accepted moves through 60k; mean strength decays from ~0.00125 at 48k to ~6.21e-64 at 60k and class-1 mass falls from 57.339 to 43.564. Under strength reset only, accepted moves resume in the first 1k and by 49k both strengths are 1.0; under position relocation alone, accepted moves resume nearly immediately and by 49k both strengths are 1.0. The resulting difference of ~3.49 percentage points in accuracy vs control is **case-specific**, not evidence of universal benefit.

## Combined mechanistic interpretation

The frozen one-visit audit found zero immediately conquerable neighbors from node 8 even at strength 1.0 both at 39k (temporary trap) and at 48k (persistent trap). The strength-only rescue establishes that **multiple visits and evolving neighborhood states** can nonetheless produce escape when strength is restored. The home-relocation rescue establishes that starting from favorable territory can independently restore mobility and ultimately strength. These observations support a self-reinforcing interaction between position, movement acceptance and particle strength rather than a simple static one-step escape threshold.

**Importantly:** These counterfactuals do not prove a bug or justify replacing the PCC movement/strength rule. They also change the trajectory and its future interaction with competing particles; the force reset and relocation perturb the model in different ways, and the historical distance table remains unchanged after relocation. Some variants achieve the same final labels without having identical hidden internal state.

## Implications for stopping criteria

This selected case is clear evidence that apparent label stability can conceal weakening and imminent trapping of a class's particles, followed by large class-mass redistribution and classification drift. A robust stopping policy cannot be justified solely by consecutive unchanged labels or P95 dominance statistics. Do not merge early-stopping experiments into `master`.

**Recommended next validation:** To determine whether the lockout is present as an expected characteristic or an implementation defect, compare this sequential update/acceptance rule to the original MATLAB PCC reference implementation (including strength updates after rejected attempts, tie handling and particle move semantics). Separately, use multiple seeds/datasets and preregistered evaluation conditions before interpreting intervention gains as generalizable. Avoid expanding stop heuristics based on this one selected seed.
