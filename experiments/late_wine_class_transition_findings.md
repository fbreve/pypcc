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

## Follow-up

Add read-only class-wise summed dominance mass, max absolute per-node dominance change, changed-node magnitude of dominance change and simplex/nonfinite diagnostics to the **same** targeted workflow. This can distinguish a rare-node large movement (masked by P95), gradual redistribution across classes and potential numerical-state corruption without adding a new stop condition.
