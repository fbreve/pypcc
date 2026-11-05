# Canonical undirected PCC: prespecified 100k label-stability stopping results

## Provenance and integrity

Protocol frozen before running in [`canonical_undirected_long_stopping_protocol.md`](canonical_undirected_long_stopping_protocol.md). Code: [`audit_canonical_undirected_long_stopping.py`](audit_canonical_undirected_long_stopping.py). GitHub Actions run **42012373215**: all four dataset jobs **passed**. Artifact IDs: Iris **13408505728**, Wine **13408276141**, Breast Cancer **13407418915**, Digits **13408696147**. Source commit `b0c29ee`.

Four *new* seeds per dataset: **2609, 2707, 2801, 2903**. Production `pcc_graph.build_knn_graph`, k=10 **undirected union**, StandardScaler, two initially labeled nodes/class. Experimental sequential NumPy PCC. Compare 100k cap to predeclared label-change stability stopping with 1k interval and patience **3, 6, 10**. The 100k reference is **capped**, not a mathematical convergence guarantee. Each observed 100k reference predicted exactly the same final labels as an independent no-observer no-stop run (**16/16**), and all graph builders verified 100% reciprocity.

## Overall comparison

| Stop policy | Early stops /16 | Mean iterations saved | Mean *wall time* saved | Average agreement with 100k | Worst agreement | Mean accuracy difference vs 100k | Worst accuracy loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Patience 3 | **16/16** | **69.44%** | **69.39%** | **98.99%** | **97.09%** | **-0.494 pp** | **-2.907 pp** |
| Patience 6 | **11/16** | **51.50%** | **51.35%** | **99.40%** | **97.09%** | **-0.356 pp** | **-2.907 pp** |
| Patience 10 | **10/16** | **38.44%** | **38.33%** | **99.72%** | **99.31%** | **-0.101 pp** | **-0.694 pp** |

Mean elapsed-time savings reflect *one* runner measurement per case; no statistical claim on runtime variance. For patience 10 the highest positive accuracy change was **+0.694 pp**; patience 6 reached **+1.744 pp** in one Wine case, illustrating why accuracy is not necessarily monotonic with later stopping.

### Dataset-specific outcomes

| Dataset | Patience 3: stops/4, saved time, mean accuracy delta | Patience 6: stops/4, saved time, mean accuracy delta | Patience 10: stops/4, saved time, mean accuracy delta |
| --- | --- | --- | --- |
| Iris | 4/4, **90.74%**, **-0.694 pp** | 4/4, 82.15%, -0.694 pp | 4/4, **64.60%**, **-0.174 pp** |
| Wine | 4/4, **90.73%**, **-0.727 pp** | 4/4, 85.50%, -0.436 pp | 4/4, **70.41%**, **-0.145 pp** |
| Breast Cancer | 4/4, **44.88%**, **-0.133 pp** | 0/4, 0.00% iterations, 0.00 pp | 0/4, **0.00% iterations**, 0.00 pp |
| Digits | 4/4, **51.20%**, **-0.422 pp** | 3/4, 38.48%, -0.295 pp | 2/4, **18.72%**, **-0.084 pp** |

**Breast Cancer illustrates an important practical tradeoff:** At patience 10, the rule did not stop in any of the four instances and merely added minor overhead (mean measured runtime **0.4% longer** than an uninstrumented 100k reference). Therefore patience 10 is **not** a universal guaranteed speedup. Conversely, patience 3 stopped in all Breast Cancer cases and saved ~44.9% wall time, at small average accuracy cost, but its worst overall accuracy loss occurred on Wine.

### Selected critical cases

- **Wine seed 2707:** patience 3 (7k) and 6 (10k) both lost **2.907 pp**, while patience 10 stopped at 38k and lost only **0.581 pp**.
- **Iris seed 2609:** patience 3 stopped 17k and lost 0.694 pp, patience 6 stopped 30k and lost **2.083 pp**, while patience 10 stopped 53k and lost 0.694 pp. Increasing patience does **not** imply monotonically improved accuracy.
- **Wine seed 2903:** patience 6 at 27k was **1.744 pp more accurate** than the 100k reference, while patience 3 (15k) and 10 (51k) were each +0.581 pp.
- **Digits seed 2707:** patience 3 stopped 29k (-0.732 pp), patience 6 at 45k (-0.675 pp), patience 10 ran all 100k and incurred no classification difference but no savings.

## Late drift of 100k references: 50k vs 100k

| Dataset | Mean number of changed predictions, same initially unlabeled nodes | Maximum number changed | Mean change in accuracy at 100k vs 50k |
| --- | ---: | ---: | ---: |
| Iris | 0.50 | 1 | 0.000 pp |
| Wine | 0.50 | 1 | 0.000 pp |
| Breast Cancer | 9.00 | 13 | +0.354 pp |
| Digits | 12.25 | 25 | +0.563 pp |

Even the 100k trajectories are not necessarily converged; some predictions still change after 50k and accuracy sometimes decreases. Do **not** confuse class-particle trapping, label-change stability over selected checkpoints, or capped reference agreement with guaranteed convergence.

## Recommendation and reproducibility scope

**Patience 10** provides the most conservative accuracy/efficiency balance **among the three fixed settings on this small cohort**: average agreement 99.72%, worst agreement 99.31%, mean accuracy change -0.10 pp and mean wall-time saving 38.33%. It still stops early in only 10/16 runs and fails to speed Breast Cancer in 4/4 cases; 100k is an arbitrary cap. **Patience 3** is faster but incurred a selected worst loss of -2.907 pp, too large to recommend as a default without stronger evidence.

Next independent generalization axis: change labeled fraction to **five labeled points/class** on *new seeds* and canonical undirected production graphs; separately measure other graph degrees and accelerated backends if promising. Keep the same observational endpoints and stop criteria, without tuning after results. Do **not** merge stopping rules into `master`, nor alter particle dynamics (force floors/reset), based on these data.
