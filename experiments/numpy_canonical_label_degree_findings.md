# Canonical undirected union10: degree-based labeled-home selection — complete audit

**Outcome: all frozen 36/36 same-graph observed/unobserved NumPy native comparisons PASS**, on experimental unmerged draft [#188](https://github.com/fbreve/pypcc/pull/188) stacked above unmerged #187/#186/#185/#184/#183/#182. [Protocol preregistered before new results](numpy_canonical_label_degree_protocol.md). This study modifies **no production PCC source, backend, movement rule, RNG, default graph/k, stopping mechanism or API**.

## Provenance and strict reproducibility

- [Scientific GitHub Actions run 42399270746](https://github.com/fbreve/pypcc/actions/runs/42399270746): **12/12 Linux Python 3.12 dataset×seed jobs, plus a required complete-cohort 36/36 aggregate PASS**. [Full machine-gated report, artifact 13585241293](https://github.com/fbreve/pypcc/actions/runs/42399270746/artifacts/13585241293).
- [Production PCC regression 42399270741](https://github.com/fbreve/pypcc/actions/runs/42399270741): all Linux/Windows × Python 3.10/3.12 jobs PASS on the same tested experimental-source commit **`c8acb28b90acdbdd1d8924630b4f3b0ca541b40d`**.
- All **12 raw per-seed artifacts** include: exact original graph SHA256, class-eligible quartile pools, every truly selected labeled vertex ID, selected degree and oracle true-label-neighbor share, common evaluation subset node IDs, original own-unlabeled accuracy, all 100×class native mobility windows, individual maximum stationary streaks, exact native terminal-state/RNG parity gates, times and complete dependency `pip freeze`:
  - Digits 24103 `13584638094`, 24223 `13584024240`, 24317 `13584428192`, 24439 `13585310879`.
  - Wine 24103 `13585026099`, 24223 `13584383226`, 24317 `13584368291`, 24439 `13584529025`.
  - Breast Cancer 24103 `13584288400`, 24223 `13585245534`, 24317 `13584568633`, 24439 `13583854165`.
- Original sklearn Digits (**1797 nodes, 10 classes**), Wine (**178 nodes, 3 classes**) and Breast Cancer (**569 nodes, 2 classes**), globally standardized. **The exact same original connected, fully reciprocal, union-symmetrized production kNN=10 graph for all 3 label strategies and all 4 seeds within each dataset**; no artificial index-parity split, bridge or other edge modification. Graph SHA256 and all structural metrics exactly match across label/seed conditions.
- Four fresh preregistered seeds **24103,24223,24317,24439**, two initially labeled homes per true class, **20/6/4 particles** respectively. Three fixed **oracle class-balanced** strategies: random from full class; low degree randomly selected from the lowest-ranked `max(2,ceil(class_size/4))` class members; high degree from the highest-ranked same count, with stable node-ID tie-breaking. The low/high degree sampling ranges are disjoint by node; the degree is the **actual symmetrized graph degree**, not directed kNN outdegree. The ground-truth class is used only for eligibility and retrospective diagnosis, not as a learned label beyond the 2/class supplied to PCC.
- **Evaluation set control:** every within-seed primary comparison uses the **identical common unlabeled intersection of ALL three alternative initial training masks**; graph fixed, and no item selected as any strategy's labeled home is evaluated. The ordinary per-arm remaining-unlabeled accuracy is archived only as secondary. The common intersection can vary by seed, as expected, but it is constant **across all three strategies within a seed**.
- **36 conditions × (ordinary original native fit + single uninterrupted true per-step observed original native fit) = 72 real no-stop 100k fits**, with actual **100001 unique physical snapshots per observed run**, no prefixes/replays. Each original same-graph/same-labeled-mask pair agrees **bitwise** on all **six complete final state arrays**, native stop metadata, all original global NumPy MT19937 fields, next 16 uniform draws, and the true low-level native final snapshot including **raw ownership before final normalization**. Graph construction and observer were reused without modifying native PCC or prior audited observer code.
- Original native sequential `p_grd=.5,delta_v=.1,deltap=1,dexp=2,max_iter=100000,early_stop=False`. Fixed iteration horizon, not a convergence claim.

## Primary matched-common-set classification outcomes

| Dataset | Random mean | Low-degree mean | High-degree mean | Low − random | High − random |
|---|---:|---:|---:|---:|---:|
| **Digits** | **87.38%** | 84.93% | **90.94%** | **−2.460 pp** | **+3.553 pp** |
| **Wine** | **94.54%** | 94.53% | **94.85%** | −0.006 pp | +0.313 pp |
| **Breast Cancer** | **92.87%** | 88.69% | 92.37% | **−4.171 pp** | −0.492 pp |

These are four-seed descriptive means on matched within-seed common evaluation nodes, **not** population confidence intervals or a recommendation to change the production labeling policy. A mean over seeds is not a pooled dataset count or an equivalence test.

### Individually observed signed paired accuracy differences (pp)

| Dataset | Seed | Low − random | High − random | High − low |
|---|---:|---:|---:|---:|
| Digits | 24103 | +0.115 | −2.129 | −2.244 |
| Digits | 24223 | **−7.250** | +3.395 | **+10.644** |
| Digits | 24317 | −0.403 | **+6.214** | +6.617 |
| Digits | 24439 | −2.301 | **+6.732** | +9.033 |
| Wine | 24103 | **+3.727** | −0.621 | −4.348 |
| Wine | 24223 | −0.625 | +1.250 | +1.875 |
| Wine | 24317 | −2.500 | −1.250 | +1.250 |
| Wine | 24439 | −0.625 | +1.875 | +2.500 |
| Breast Cancer | 24103 | −1.795 | +1.616 | +3.411 |
| Breast Cancer | 24223 | **−6.631** | −3.943 | +2.688 |
| Breast Cancer | 24317 | **−9.874** | −3.770 | +6.104 |
| Breast Cancer | 24439 | +1.616 | +4.129 | +2.513 |

High degree outperformed random for **3/4 Digits** and **2/4 Wine and Breast Cancer** seeds, so a universal high-degree superiority claim is untenable. Low-degree class-balanced homes underperformed random in **3/4 seeds each** across all three datasets despite a sometimes large positive isolated seed. The 7.250 pp Digits worst low-vs-random and 9.874 pp Breast Cancer low-vs-random differences are *not* caused by changes in topology or evaluation subset: both are exactly matched within seed. These results support sensitivity to initial label placement but not prove that degree itself is the mechanistic cause, since vertex identity, true decision boundary and density change with labeled choices.

### Actual selected-node degree and retrospective class-neighbor agreement

| Dataset | Home strategy | Average selected node degree | True-label same-class neighbor fraction (retrospective, NOT model input) |
|---|---|---:|---:|
| Digits | random | 13.71 | 91.4% |
| Digits | low-degree | 10.35 | 90.3% |
| Digits | high-degree | 20.14 | 95.6% |
| Wine | random | 13.12 | 90.1% |
| Wine | low-degree | 10.38 | 84.7% |
| Wine | high-degree | 19.38 | 93.8% |
| Breast Cancer | random | 13.94 | 94.2% |
| Breast Cancer | low-degree | 10.94 | 90.8% |
| Breast Cancer | high-degree | 20.38 | 96.9% |

Higher degree selection also yielded greater **true-class neighborhood agreement** in all three datasets on these samples, a potential confound and interesting diagnostic, **not** an input to PCC or proof that high degree causes cleaner labels. Ranks use the undirected union-symmetric degree and true-label eligibility; this is a controlled ***oracle class-balanced stress study***, not a ready-made active-learning policy for unlabeled data.

## Particle mobility and negative lockout finding

- **0 operationally locked classes across 36/36 actual no-stop native runs** and **zero 1000-step class windows with all-class-particle zero accepted moves among 18,000**: Digits 4 seeds×3 strategies×10 classes×100 windows = 12,000, Wine 4×3×3×100 = 3,600, Breast Cancer 4×3×2×100 = 2,400.
- Existing preregistered criterion unchanged: at least **five consecutive** class 1000-step windows with zero accepted movements, class max particle strength ≤0.01 and zero class-winning outgoing neighbor entries. The absence of this specific metric does not prove the PCC cannot ever experience rare stationary particles, rejected moves or poor coverage.
- Large accuracy changes were possible **with the same connected undirected graph** and no operational class immobility. Thus neither class traps nor loss of graph connectedness is necessary for an observed accuracy difference, and physical mobility alone is not a sufficient predictor of accuracy.

## Interpretation limits and decision

1. **Not a usable unsupervised sampling strategy:** low/high-degree eligibility is restricted *within ground-truth classes*, which are not known for unlabeled points in a real active learning workflow. A deployable scheme would need to infer or query class membership and independently demonstrate benefit. This experiment is useful for diagnosing initial-home sensitivity, not a new PCC feature.
2. **Degree ≠ global center/periphery.** Local undirected neighbor count does not measure graph eccentricity, betweenness, influence in PCC or distance from class boundaries. The retrospective true-label neighborhood agreement is **not** a clean intervention: high-degree choices also tended to be purer in this sample.
3. **Four seeds per dataset and a 100k fixed cap do not establish significance, invariance, convergence or universal benefit.** Class-aware label selection, training mask overlap, population representativeness, finite-run RNG trajectories and feature scaling all bound applicability. Different training labeled masks change the model's trajectory; the exact RNG state parity is between *same-label observed and ordinary* fits, not between random/high/low arms.
4. **No original PCC change justified.** Do not introduce resets, forced moves/strength floors, alter original graph builder, switch default stopping, or silently adopt degree sampling in production. The original undirected graph exhibited no class-level immobility on this cohort.
5. **Future independent test:** a practical follow-up could separate label purity vs degree using matched degree/purity bins *retrospectively* with new datasets/seeds, but true class-neighbor purity is unobservable without labels. Alternatively an explicitly feasible unlabeled-node selection strategy would require a completely new prespecified active labeling protocol and a fair oracle-label-query budget. **Do not tune selection on these same fixed four seeds.**

**Decision: keep PR #188 and dependencies #187/#186/#185/#184/#183/#182 DRAFT and UNMERGED.**
