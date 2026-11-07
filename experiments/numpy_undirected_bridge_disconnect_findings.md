# Undirected graph disconnection vs a single bridge: frozen 12-case PCC pilot

**Result:** all registered cases and checks PASS. [Precommitted protocol](numpy_undirected_bridge_disconnect_protocol.md). This study is experimental-only, on stacked draft [#185](https://github.com/fbreve/pypcc/pull/185), depending on draft #184/#183/#182; **nothing is merged into `master`** and no PCC model movement/stopping/graph-building production source was modified.

## Reproducibility and strict integrity

- [GitHub Actions workflow 42380326406](https://github.com/fbreve/pypcc/actions/runs/42380326406): **4/4 independent seed jobs and mandatory aggregate 12/12 report green**. [Summary artifact 13576883869](https://github.com/fbreve/pypcc/actions/runs/42380326406/artifacts/13576883869).
- Raw full per-class/per-window state, all 100 class-checkpoint windows per 10 classes, per-particle motion and crossing counts, accurate stop metadata, source environment inventories: seed **21107** artifact `13577861129`, **21221** `13577737019`, **21317** `13577766910`, **21401** `13577985184`.
- [PCC regression workflow 42380326252](https://github.com/fbreve/pypcc/actions/runs/42380326252) **green** for this tested commit.
- Standardized sklearn Digits: 1797 vertices and 10 classes; fixed even/odd original-ID partition of **899/898 vertices**. One labeled vertex of **each class in each partition**, exactly two labeled per class and 20 native particles total. The same preselected labels are used in all three graph arms within a seed.
- Three reciprocal undirected no-self-edge graph variants:
  - `full_union10`: unmodified production union-10NN on all 1797 points, connected; mean degree **14.043**.
  - `split_disconnected`: union-10NN built independently by the original builder within each partition; exactly **two connected components** of 899 and 898 nodes, mean degree **13.978**, no cross edges.
  - `single_bridge`: exactly the same split graph **plus one undirected cross-partition edge**, endpoints selected by minimum standardized squared Euclidean cross distance, one connected component, mean degree **13.979**. Its edge set and reciprocal-arc counts differ from split by **exactly 1 undirected edge / 2 directed arcs**.
- Frozen **four novel seeds** 21107, 21221, 21317, 21401. NumPy sequential, `p_grd=.5,delta_v=.1,deltap=1,dexp=2,early_stop=False,max_iter=100000`. 12 graph×seed conditions × one uninstrumented + one uninterrupted, every-iteration instrumented full native fit: **24 actual 100k fits**.
- Instrumented vs ordinary full fits matched **exactly, 12/12**, on all six complete final state arrays, original native stop metadata, all NumPy global MT19937 fields, and next 16 random draws. The last raw low-level state before ownership normalization and complete RNG were independently verified. **100001 actual physical observations per instrumented fit**, plus one final notification without a duplicate step. All four synthetic graph/crossing and 100-window definitions passed the 6-test suite per job.

## The complete cohort

| Seed | Condition | Components | Mean degree | Cross-partition moves | Operationally locked classes | Zero-acceptance 1000-step class windows | Longest individual stationary streak | Held-out accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 21107 | full | 1 | 14.043 | 925,877 | 0 | 0 | 11 | 83.23% |
| 21107 | disconnected | 2 | 13.978 | 0 | 0 | 0 | 15 | 68.88% |
| 21107 | single bridge | 1 | 13.979 | 2 | 0 | 0 | 21 | 68.26% |
| 21221 | full | 1 | 14.043 | 950,740 | 0 | 0 | 11 | 86.83% |
| 21221 | disconnected | 2 | 13.978 | 0 | 0 | 0 | 10 | 79.91% |
| 21221 | single bridge | 1 | 13.979 | 269 | 0 | 0 | 12 | 80.81% |
| 21317 | full | 1 | 14.043 | 932,200 | 0 | 0 | 9 | 91.11% |
| 21317 | disconnected | 2 | 13.978 | 0 | 0 | 0 | 15 | 84.75% |
| 21317 | single bridge | 1 | 13.979 | 163 | 0 | 0 | 12 | 84.75% |
| 21401 | full | 1 | 14.043 | 949,144 | 0 | 0 | 11 | 83.91% |
| 21401 | disconnected | 2 | 13.978 | 0 | 0 | 0 | 13 | 83.51% |
| 21401 | single bridge | 1 | 13.979 | 131 | 0 | 0 | 12 | 83.57% |

**All 12 runs:** 0 operational class lockouts, **0 / 12000** full 1000-step class windows with zero accepted movements, maximum per-particle consecutive stationary streak **21** steps. Every one of the 12×20=240 particle-run trajectories changed position at least once. Absence of this **specific operational criterion** does not prove that every rejected attempted move is harmless, that graph disconnection can never impair class propagation, or that a positive lockout is impossible on other graphs.

**Full-vs-split** crossings between even and odd vertex groups: **3,757,961** observed accepted cross-partition moves over all 4 full-union10 fits, compared with **565** traversals along the unique bridge edge over all 4 one-bridge fits. These are full trajectory event counts, *not* independent class- or node-level statistical observations; the full graph has many cross edges. Split has precisely zero, as demanded by its component constraint.

## Paired accuracy effects and limits

All accuracies are on the **same 1777 initially unlabeled ground-truth vertices** within each seed; the graphs differ, not the labels. Signed accuracy differences (percentage points):

| Seed | Bridge − disconnected (one-edge contrast) | Full − disconnected (many-edge contrast) |
|---|---:|---:|
| 21107 | −0.619 | +14.350 |
| 21221 | +0.900 | +6.922 |
| 21317 | +0.000 | +6.359 |
| 21401 | +0.056 | +0.394 |
| **Mean, n=4** | **+0.084** | **+7.006** |

The single reciprocal bridge produced **small and direction-variable** accuracy changes in these four cases and did **not** restore the much higher original full union-10NN connectivity/mixing. However, **the full graph differs in many more edges, not just the one bridge**: even the within-partition kNN neighborhoods were recomputed separately in the split design. Therefore the full-minus-split accuracy difference is not a controlled estimate of a single-edge, mean-degree or connectivity-only effect. The bridge-minus-split condition is the precise one-edge topology change but still changes future stochastic movement trajectories; same numeric seed across graph arms does not imply synchronized random draws.

The large variability of the full-minus-split accuracy difference (0.394 to 14.350 pp) and the slightly negative one-edge effect for seed 21107 rule out simplistic claims that graph connectedness alone guarantees improved accuracy. More crossings do not **by themselves** prove correct propagation, causation or convergence. The partition is an artificial parity construction and is **not** an independent-draw sample of realistic natural cluster cuts. The 100k cap is not a proof of convergence, nor are 4 seeds a representative prevalence sample. One-shot hosted runner times are descriptive only.

## Decision and next scientific question

- Keep #185 and its dependencies #184/#183/#182 **draft/unmerged**.
- Do **not** reset particle strength/position, add forced moves, alter k, or change the native potential stopping rule: no operational class lockout occurred on this controlled cohort, and the large full-vs-split accuracy differences cannot be attributed to lockout.
- Distinguish **node/class accessibility**, **physical accepted movement**, **inter-component traffic**, and **classification accuracy**. The study shows one-bridge physical connectedness and actual sparse mixing are not interchangeable.
- An independent next-stage experiment, *if scientifically justified*, could pre-register several bridge counts on a fixed within-partition edge set to quantify whether inter-group traffic and classification respond monotonically. Another axis is genuine single-class isolation or zero-homed components, but those present inherently unidentifiable classes and require careful framing, not automatic claims about PCC bugs.
