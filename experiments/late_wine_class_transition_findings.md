# Wine late class transitions at 1k resolution

Source: GitHub Actions run **41877986218**, artifact `late-class-transitions` (`late-wine-class-counts.csv` and `late-wine-class-transitions.csv`), commit `5310d4a`. The targeted workflow and all 11 other workflows, including PCC regression, passed.

## Protocol

Experimental NumPy PCC, Wine (3 classes), seeds **37** and **83**, 10-NN graph and two labeled samples per class, no early stopping, single uninterrupted trajectories to 100,000 iterations. Record initially unlabeled node class counts, accuracy and top-two dominance margins every **1,000 iterations from 50k through 100k**. The callback receives a defensive copy; final predictions match the uninterrupted model output. There are **172 initially unlabeled nodes** per seed. Results are selected challenging seeds, not a representative performance estimate.

| Indicator | Seed 37 | Seed 83 |
| --- | ---: | ---: |
| Adjacent-checkpoint label flips, total | 117 | 55 |
| 1k intervals with any flip | 29 / 50 | 23 / 50 |
| Maximum flips in one interval | 16 | 9 |
| Flips with **prior** top-two margin >= 0.5 | 57 / 117 | 6 / 55 |
| Flipping intervals with P95 absolute dominance change < 0.001 | 2 / 29 | 9 / 23 |
| Accuracy at 50k | 87.21% | 84.30% |
| Accuracy at 100k | 74.42% | 58.72% |
| Class counts at 50k (0,1,2) | (61, 56, 55) | (69, 52, 51) |
| Class counts at 100k (0,1,2) | (39, 79, 54) | (71, 6, 95) |

**Transition counts**, summed over adjacent sampled checkpoints (a node can flip more than once):
- Seed 37: class 0→1: 59; class 1→0: 37; class 1→2: 10; class 2→1: 11. 0↔1 back-and-forth is substantial, so the trajectory does not exhibit a simple one-way class collapse.
- Seed 83: 1→2: 39; 1→0: 7; 0→2: 7; 2→0: 2. Class 1 effectively loses most of its assigned nodes, and prediction accuracy declines sharply.

Seed 37 has many high-margin class reversals at the 1k observation scale; for example, at 79k the five changed nodes had a median **prior** class margin of 1.0. This is not just near-tie oscillation. Seed 83 shows a more directional migration from predicted class 1 to class 2, generally at smaller prior margins.

## Why P95 alone cannot be a reliable stopping signal

On 1k intervals with a changed label, **2** (seed 37) and **9** (seed 83) nevertheless had P95 per-node absolute dominance change < 0.001 on the initially unlabeled nodes; for several intervals the P95 was numerically near zero. A percentile deliberately ignores small affected minorities, so it can miss abrupt, important changes in a handful of nodes. This does **not** establish that the global hybrid necessarily stopped there, because its label-unchanged condition would reject the flip itself. However, a preceding stable window can still trigger an irreversible premature stop *before* later changes.

These diagnostics do not establish whether numerically large swings follow particle dynamics, changes to one class's dominance mass, sparse event propagation or a numerical pathology. Avoid proposing yet another patience/tolerance threshold based only on these two seeds.

## Follow-up: class mass, sparse flips and numeric invariants

Source: GitHub Actions run **41896181046**, artifact `late-class-transitions` (extended fields), commit `653dd2c`. Targeted workflow passed.

| Numeric diagnostic (50k→100k) | Seed 37 | Seed 83 |
| --- | ---: | ---: |
| Dominance mass class 0, 50k→100k | 56.74→36.21 | 61.78→63.68 |
| Dominance mass class 1, 50k→100k | 61.03→79.50 | 53.91→10.17 |
| Dominance mass class 2, 50k→100k | 54.24→56.29 | 56.31→98.14 |
| Largest deviation from per-node dominance sum 1 | 1.38 × 10⁻¹⁴ | 1.12 × 10⁻¹⁴ |
| Non-finite dominance entries at sampled checkpoints | 0 | 0 |

A class's dominance mass is the **sum of the class dominance over the 172 initially unlabeled nodes**. Therefore the three masses sum to approximately 172, regardless of the predicted class counts. Seed 83 loses **43.74** units of class-1 dominance mass while class 2 gains **41.83**. This is a persistent redistribution of dominance, not just an argmax artifact. The snapshot simplex invariant holds to floating-point precision and no non-finite dominance was observed, so there is **no evidence of NaN or normalization breakdown in the sampled states**; this does not by itself establish correctness of the dynamics.

More decisively, a high-percentile statistic can hide large changes in a few nodes:

- Seed 83, **78k→79k**: 1 label flip, P95 per-node absolute dominance change ~1.36×10⁻¹⁵⁸, but the **maximum is 0.521**.
- Seed 83, **83k→84k**: 2 label flips, P95 ~1.44×10⁻¹⁶⁶, but **maximum is 0.550** (median change among flipping nodes 0.325).
- Seed 83, **99k→100k**: 1 flip, P95 ~1.70×10⁻²⁵⁶, maximum **0.323**.
- Seed 37, **66k→67k**: 1 flip, P95 approximately machine epsilon, maximum **0.100**.

Across changed intervals, **11/52** exhibited P95 < 0.001 even though labels flipped. The max-change diagnostic distinguishes sparse substantial shifts from global small drift. However, using the maximum as a stop rule does **not** solve the problem of future changes after a quiet window, and may eliminate savings. It needs prospective evaluation before use.

**Current decision:** neither label stability nor the current hybrid criterion is safe to integrate. No production changes are justified. Before further parameter tuning, investigate how class-wise dominance redistribution arises in the PCC dynamics, preferably across additional seeds and with a non-stop baseline.
