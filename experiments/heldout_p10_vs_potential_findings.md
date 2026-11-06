# Complete held-out comparison: production potential stop vs experimental label P10

**Frozen protocol:** [heldout_p10_vs_potential_protocol.md](heldout_p10_vs_potential_protocol.md). This study was not tuned after observing previous Iris hard cases; none of those seeds is included.

## Reproducibility and CI provenance

- Source / protocol / CI: [PR #174](https://github.com/fbreve/pypcc/pull/174), initial head `f36ffb515bc1d3ccdb62f72a7a22c85c26264e8d`.
- Results: [GitHub Actions 42158950255](https://github.com/fbreve/pypcc/actions/runs/42158950255): **8/8 dataset×supervision jobs + dependent summary job all successful**. Production regression [42158950174](https://github.com/fbreve/pypcc/actions/runs/42158950174) passed.
- Automatically generated, full cohort report artifact **13476618538**, name `heldout-p10-vs-potential-summary`. The summary enforces **exactly 8/8 CSV artifacts, 48/48 condition triplets and 192/192 policy rows**, with no missing/duplicate seed/policy rows.
- Raw per-stratum data (artifact name / ID):
  - Iris, 2/class: `heldout-pcc-iris-2` / **13476046775**
  - Iris, 5/class: `heldout-pcc-iris-5` / **13476111986**
  - Wine, 2/class: `heldout-pcc-wine-2` / **13475184804**
  - Wine, 5/class: `heldout-pcc-wine-5` / **13475783318**
  - Breast Cancer, 2/class: `heldout-pcc-breast_cancer-2` / **13475658935**
  - Breast Cancer, 5/class: `heldout-pcc-breast_cancer-5` / **13475874379**
  - Digits, 2/class: `heldout-pcc-digits-2` / **13476284030**
  - Digits, 5/class: `heldout-pcc-digits-5` / **13477540783**
- All **48** NumPy candidate/production no-stop 10k parity gates passed; all production kNN graphs were the required **union-symmetrized undirected** graphs. New seeds: **5407, 5501, 5609, 5701, 5801, 5903**; exactly **2 or 5 labels/class**, fixed per seed across policies.

## Final results (prospective 48-case cohort)

| Metric | Production potential stopping | Experimental label-stability P10 |
|---|---:|---:|
| Stopped before 200k | **48/48** | **36/48** |
| Mean percentage of iterations saved | **78.12%** | **58.60%** |
| Mean wall time saved vs 200k no-stop | **77.80%** | **57.95%** |
| Mean final-label agreement with 200k no-stop | 98.62% | **99.57%** |
| Worst signed accuracy difference vs 200k no-stop | **-2.778 pp** | **-2.778 pp** |
| Cases with >2 pp loss vs 200k no-stop | 5 | **1** |

**Direct paired actual accuracy P10 minus potential:** P10 higher in **11/48** cases, potential higher in **16/48**, tied in **21/48**. Mean signed difference **+0.068 pp** (median 0.000 pp; extremes **-1.481 pp** to **+2.504 pp**). Signed gains and losses substantially cancel; the tiny positive mean is **not** evidence that P10 improves typical accuracy. There are three cases with >2 pp gain for P10 vs potential, none with >2 pp loss in the reverse direction, but this is a descriptive tail count not a validated worst-case bound.

### Important per-stratum tradeoffs

| Dataset | Labels/class | Mean P10−potential accuracy (pp) | P10 better / potential better / ties | Potential stops /6 | P10 stops /6 | Mean potential steps | Mean P10 steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Iris | 2 | +0.694 | 2/0/4 | 6 | 6 | 14,018 | 22,500 |
| Iris | 5 | -0.494 | 1/3/2 | 6 | 6 | 6,678 | 25,167 |
| Wine | 2 | +0.291 | 2/0/4 | 6 | 6 | 16,266 | 36,500 |
| Wine | 5 | +0.409 | 2/0/4 | 6 | 6 | 6,368 | 31,833 |
| Breast Cancer | 2 | -0.295 | 1/4/1 | 6 | **0** | 78,426 | **200,000** |
| Breast Cancer | 5 | +0.268 | 3/3/0 | 6 | **0** | 29,306 | **200,000** |
| Digits | 2 | -0.310 | 0/4/2 | 6 | 6 | 144,968 | **97,333** |
| Digits | 5 | -0.019 | 0/2/4 | 6 | 6 | 54,073 | **49,000** |

For **Breast Cancer**, label P10 never stopped in **12/12** executions (both supervision levels), while the existing potential criterion stopped in **12/12**; it costs drastically more iterations without any consistent accuracy gain. For **Digits**, P10 required fewer iterations, but its mean accuracy was slightly lower in both supervision regimes. This conditional behavior is why a blanket default replacement is not justified.

### Prespecified adverse-case audit

**More than 2 pp loss vs the same 200k no-stop reference:**
- Iris, 2 labels/class, seed **5407**: both potential and P10 **-2.778 pp** vs reference (potential 18,541 steps; P10 16k).
- Iris, 2 labels/class, seed **5801**: potential **-2.083 pp**; P10 recovered the reference accuracy (+2.083 pp direct advantage).
- Iris, 2 labels/class, seed **5903**: potential **-2.083 pp**; P10 recovered the reference accuracy (+2.083 pp direct advantage).
- Wine, 2 labels/class, seed **5903**: potential **-2.326 pp**; P10 better but not fully equal to reference.
- Breast Cancer, 5 labels/class, seed **5501**: potential **-2.504 pp**; P10 ran all 200k and matched the reference by construction.

**No-stop reference is not static:** across 48 cases, **38** had at least one differing prediction at 100k versus 200k, with **222** changed initially unlabeled predictions in total. The largest case had **27** changed predictions (Breast Cancer, 2/class, seed 5609). Some paths can change and later return between endpoints, as separately demonstrated by prior selected-Iris 500k instrumentation; these 100k/200k comparisons do not measure such transient changes.

## Scientific conclusion and decision

1. The existing production potential criterion is **usually more computationally economical** in this cohort (about 19.5 percentage points more mean iteration saving). Label P10 better matches fixed long-run labels and has fewer **large losses relative to that cap**, but not consistently better ground-truth accuracy.
2. The observed **+0.068 pp** average true-label accuracy difference is practically small relative to stratum-to-stratum and seed-level variation; it neither demonstrates equivalence nor a reproducible accuracy advantage. Six seeds per dataset/supervision stratum and four datasets are insufficient for a universal statement.
3. **200k is a reference horizon, not a convergence guarantee.** Agreement with long-run predictions cannot substitute for ground-truth accuracy, and true accuracy is not monotone in iterations. The 2- and 5-label conditions also use different sampled labeled nodes, so they are not a strict paired causal manipulation.
4. One-off GitHub Actions wall times are descriptive; control-run CPU benchmarking would be needed before claiming actual speed ratios in deployment.
5. **Recommendation: do not promote P10 to the production default or merge the experimental branch.** Retain the current potential stopping criterion, preserve P10 only as an experimental comparator, and focus next on **reducing the cost of current potential monitoring without changing its numerical/stop semantics**. Accelerated production Numba/Cython no-stop Digits previously exhibited appreciable monitoring overhead, but this should be remeasured with a controlled, fixed-iteration design before implementing changes.

This final study closes the specific question of a direct P10 default replacement **for the tested conditions**. More algorithms or post-hoc hybrid thresholds should not be added to production without an independent prespecified quality-and-cost study.
