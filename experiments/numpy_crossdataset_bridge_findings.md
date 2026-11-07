# Independent Wine and Breast Cancer bridge-dose study — complete results

**Status:** preregistered, research-only and unmerged [draft PR #187](https://github.com/fbreve/pypcc/pull/187), stacked above unmerged drafts #186/#185/#184/#183/#182. [Frozen protocol](numpy_crossdataset_bridge_protocol.md) committed **before Wine/Breast Cancer outcomes**. No production source, stopping rule, particle movement/strength, RNG, graph construction, native backend or API change.

## Independent CI and reproducibility

- [Scientific GitHub Actions run 42395549817](https://github.com/fbreve/pypcc/actions/runs/42395549817): **8/8 dataset×seed jobs + mandatory complete 40/40 aggregate job PASS**.
- [Machine-validated full-cohort report artifact 13583004319](https://github.com/fbreve/pypcc/actions/runs/42395549817/artifacts/13583004319). Raw JSON includes all 100 true crossing windows, all per-class 1000-step windows, per-edge/class/particle acceptance counts, class strength/neighbor/mass, full graph SHA fingerprints, exact native parity gates, timings, and dependency `pip freeze`:
  - Wine 23111 `13582819514`, 23209 `13583331912`, 23321 `13583059045`, 23417 `13583113758`.
  - Breast Cancer 23111 `13583387655`, 23209 `13582849406`, 23321 `13583531178`, 23417 `13583158684`.
- [PCC regression run 42395549762](https://github.com/fbreve/pypcc/actions/runs/42395549762): **all 4 Linux/Windows × Python 3.10/3.12 regression jobs PASS** at tested source commit `2b57789d99403c1f41050224b579abad5640710e`.
- Preselected **four new seeds** `23111,23209,23321,23417`, two **different** real datasets: `sklearn.datasets.load_wine` (178 nodes, **3 classes**) and `load_breast_cancer` (569 nodes, **2 classes**), globally `StandardScaler` normalized. Split by *original row-index parity*: Wine **89/89**, Breast Cancer **285/284** vertices. Both inner graphs are **connected production union-10NN reciprocal** graphs, no isolates/self-loops; every class exists on both sides. Exactly **one initially labeled node of each class on each side**, fixed across graph arms per seed: 6 Wine particles and 4 Breast Cancer particles.
- Controlled graph doses **B=0,1,8,32** reciprocal vertex-disjoint shortest-distance bridges, all nested on the **identical** per-dataset within-partition graph, with **two components only at B=0**. The 32 bridges have globally stable order/digests across the four label-sampling seeds, no repeated endpoint. Fifth **original full production union-10NN** arm on the complete dataset is a separate *descriptive reference*, **not** one of the controlled bridge doses, as both intra- and cross-partition edges change.
- Sequential native `ParticleCompetitionAndCooperation(impl='numpy')`, `p_grd=.5,delta_v=.1,deltap=1,dexp=2,max_iter=100000,early_stop=False`. Fixed horizon, not convergence. **2 datasets × 4 seeds × 5 arms × (one ordinary native 100k fit + one true uninterrupted per-step observed native 100k fit) = 80 real fits, 40 exact native pairs**, 100001 genuine physical iteration observations per observed fit, without prefix replay or extra physical terminal state. Each observed member matches **bitwise** on all **six entire final model arrays**, original native stop metadata, all global MT19937 RNG fields, next 16 uniform random draws, and true raw ownership low-level final state. Original NumPy/Numba/Cython/PCC kernels and model API untouched.

## Per-dataset and per-bridge empirical outcomes

| Dataset | Graph variant | Actual cross-partition edges | Mean accepted cross-partition moves / 100k | Cross-move range (four seeds) | Mean true-label held-out accuracy | Mean Δ accuracy vs B0 |
|---|---|---:|---:|---:|---:|---:|
| Wine | B0 | 0 | 0 | 0–0 | **94.33%** | — |
| Wine | B1 | 1 | 516.0 | 388–660 | 94.48% | **+0.145 pp** |
| Wine | B8 | 8 | 3,583.8 | 3,016–3,881 | 94.48% | **+0.145 pp** |
| Wine | B32 | 32 | 17,535.2 | 16,765–18,867 | 94.48% | **+0.145 pp** |
| Wine | original full union10 | 662 | 312,824.0 | 310,086–317,191 | 95.78% | **+1.453 pp (not controlled)** |
| Breast Cancer | B0 | 0 | 0 | 0–0 | **84.87%** | — |
| Breast Cancer | B1 | 1 | 104.2 | 25–147 | 78.14% | **−6.726 pp** |
| Breast Cancer | B8 | 8 | 651.8 | 351–799 | 86.50% | **+1.637 pp** |
| Breast Cancer | B32 | 32 | 2,492.0 | 2,071–2,861 | 87.57% | **+2.699 pp** |
| Breast Cancer | original full union10 | 2,085 | 190,490.8 | 185,571–194,259 | 86.86% | **+1.991 pp (not controlled)** |

All accuracies use known ground truth only on **the same initially unlabeled vertices within each dataset×seed**: 172 Wine and 565 Breast Cancer observations. Single-run hosted times are archived but are not calibrated performance benchmarks.

## All preregistered signed paired accuracy differences (percentage points)

| Dataset | Seed | B1−B0 | B8−B0 | B32−B0 | Full union10−B0 (descriptive only) |
|---|---:|---:|---:|---:|---:|
| Wine | 23111 | −0.581 | +0.581 | 0.000 | +1.163 |
| Wine | 23209 | +1.744 | +1.163 | +1.163 | +3.488 |
| Wine | 23321 | 0.000 | 0.000 | 0.000 | +0.581 |
| Wine | 23417 | −0.581 | −1.163 | −0.581 | +0.581 |
| Breast Cancer | 23111 | **−16.814** | −1.770 | +3.717 | +3.009 |
| Breast Cancer | 23209 | **−10.619** | +6.726 | +6.903 | +4.071 |
| Breast Cancer | 23321 | 0.000 | +0.177 | 0.000 | −0.708 |
| Breast Cancer | 23417 | +0.531 | +1.416 | +0.177 | +1.593 |

In Breast Cancer, the observed substantial **B1 loss is driven by two strongly adverse seeds** (23111, 23209), not a consistently modest loss across all four. It appears with few actual accepted crossings (25 and 117) yet **no operational class lockout**. A single bridge changes local stochastic dynamics and model competition; there is no direct evidence here that *the crossing count itself* causes the accuracy loss. At B32, Breast Cancer improved in three seeds and tied in one relative to B0, whereas Wine changes remained much smaller overall. Four seeds per dataset are not enough to assert a universal class-count mechanism, prove quality equivalence, or estimate worst-case prevalence.

## Operational class mobility and null finding

- **0 of 40** runs had whole-class operational lockout; **zero complete 1,000-iteration class-windows without accepted movement** across **10,000 class-window records**: Wine 4 seeds × 5 graph arms × 3 classes × 100 windows = **6,000**, Breast Cancer 4×5×2×100 = **4,000**.
- The preregistered class-lockout definition remains five consecutive 1,000-step zero-movement class windows, maximum class-particle strength <=0.01 and no class-winning outgoing neighbor. More particles can move within a component without crossing between components; lack of crossings is **not** evidence of class lockout. Conversely, a substantial accuracy loss without lockout is consistent with the earlier experimental distinction between physical mobility, class competition and predictive quality.
- Number of installed vs utilized edges is preserved in every raw sample. Even with full canonical union10, thousands of actual crossings do not constitute independently sampled experiments; graph degree, number of classes, dataset size, input geometry and location of the labeled home nodes all differ between Wine and Breast Cancer.

## Interpretation and decision

1. **Distinct conclusions across datasets:** Digits (#186) had near-zero mean B32 vs B0 accuracy change, Wine here had **+0.145 pp**, Breast Cancer here had **+2.699 pp**, with a notable **−6.726 pp B1** average in Breast Cancer. This is descriptive sensitivity, not proof of a stable dose-response or a generally beneficial bridge count. Do not pool the datasets into a single statistic without accounting for very different sample sizes and supervision.
2. **Structural counterfactual only within doses:** B0↔B1↔B8↔B32 share the identical inner edge set and within-seed labeled homes, so the treatment is a controlled number/placement of undirected bridges. However, different graph degrees change stochastic visit probabilities and future random-stream consumption. **Identical numeric seed does not imply an identical random tape across different graph arms**. State/RNG parity gates hold **within same-graph ordinary versus observed pairs**, not across graph arms.
3. **Original full union10 is not a B dose:** Wine full has **662** and Breast Cancer full **2,085** cross-partition edges and, importantly, recomputes many within-partition neighborhoods. Its sometimes-higher accuracy cannot be attributed solely to number of bridges or connectivity.
4. **Graph partitions are artificial and use original dataset index parity.** The data were globally standardized and graph split deliberately. This is not an independent sample of natural graph cuts, nor does it demonstrate robustness to naturally disconnected data, new class labels, sample-selection policies or all PCC backends. Fixed 100k does not prove convergence or late-horizon stability.
5. **No intervention justified:** no tested class became operationally immobile, while different accuracies occurred anyway. Do **not** add particle reset, minimum strength/forced movement, modify production graph builder/k, switch potential stopping, or claim old directed-kNN trapping prevalence in the canonical undirected PCC.

**Decision: keep #187 and all upstream experimental PRs #186/#185/#184/#183/#182 in DRAFT/UNMERGED state.** A next valuable investigation should address actual graph cuts or supervision placement as independent generalization dimensions, or return to production optimization priorities; additional bridge tuning on these fixed seeds risks post-hoc overfitting.
