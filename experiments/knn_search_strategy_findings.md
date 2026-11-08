# Frozen exact kNN search-engine audit: result and engineering decision

**Completed independent original-master scientific experiment (DRAFT/UNMERGED [PR #193](https://github.com/fbreve/pypcc/pull/193)).** [The protocol](knn_search_strategy_protocol.md) was committed before trials or outcomes, and production `pcc_graph.py`, kernel sources, graph semantics, package definitions and defaults remain **unaltered**.

## Fully validated evidence and provenance

- [GitHub Actions scientific run **42443181429**](https://github.com/fbreve/pypcc/actions/runs/42443181429): **Linux native, Windows native, mandatory two-OS strict aggregator ALL GREEN** at tested code head `debdb4ba7b554039b18d6d01576de2fd0b91858e`.
- [Machine-checked complete report artifact **13603040575**](https://github.com/fbreve/pypcc/actions/runs/42443181429/artifacts/13603040575); **original raw timing + graph/neighbors SHA + synthetic deviations + installed dependency versions**: [Linux artifact **13602701988**](https://github.com/fbreve/pypcc/actions/runs/42443181429/artifacts/13602701988) and [Windows artifact **13602288205**](https://github.com/fbreve/pypcc/actions/runs/42443181429/artifacts/13602288205).
- Exactly **1,440 original-master full graph construction pairs** (4 real datasets × 3 fixed k × 3 alternative sklearn search engines × 20 alternating control/trial repeats × 2 OS), **960 genuine separate NearestNeighbors fit+query samples** (4×3×4 methods×10×2 OS), **180 prespecified duplicate/equidistant/constant/noncontiguous tied synthetic graph cases** (5×3k×2 Minkowski p×3 alternatives×2 OS), and **48 native PCC semantic control pairs** (4 datasets×3 alternative algorithms×2 NumPy native backends Numba/Cython×2 OS).
- Original source is obtained from unchanged `git show 8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128:pcc_graph.py` and imported into an isolated Python module. All three alternative builders are constructed by **substituting ONLY `algorithm='auto'` with the explicit original sklearn `brute`, `kd_tree`, or `ball_tree` literal** in independent otherwise-identical original source modules. No handcrafted CSR clone or production patch. Every whole-graph timer includes exact source NearestNeighbors fitting, querying, CSR symmetrization, padded ordered `neib_list` materialization and degrees.
- Verified all 36 original real dataset×k×alternative comparisons matched the baseline complete **ordered padded int64 graph and degree arrays exactly**, on **both OS**. All 48 actual 1000-iteration Numba/Cython same-backend/reference-vs-candidate native controls returned identical **six final array fingerprints, full original native stopping metadata, held-out accuracy and backend identity**. Cross-backend RNG equivalence is neither assumed nor tested.
- Standard production PCC regression at this pre-findings source head was green on **Linux Python 3.10/3.12** at time the findings were prepared; the Windows jobs were still running. Adding this documentation-only findings file may trigger another CI workflow. Do not call unverified final-head CI green until it actually concludes.
- Initial [failed scientific CI run **42443062334**](https://github.com/fbreve/pypcc/actions/runs/42443062334) disclosed and preserved: the external native diagnostic called the original Numba benchmark's `steady_state=True` without first compiling the signature. Fix `debdb4ba7b554039b18d6d01576de2fd0b91858e` added the original two-iteration `measure(...steady_state=False)` warm-up before 1000-step **untimed** native state checks; no graph, labels, measured candidate engine, timing cohort or scientific decision threshold was changed. Both OS and strict aggregate then passed.

## Original sklearn `auto` resolved methods

For **all three k=1,3,10** on **both OS**:

| Data (standardized original sklearn fixtures) | Nodes × features | Actual original `auto` method |
|---|---:|---|
| Iris | 150 × 4 | `kd_tree` |
| Wine | 178 × 13 | `kd_tree` |
| Breast Cancer | 569 × 30 | `brute` |
| Digits | 1,797 × 64 | `brute` |

The original scikit-learn `auto` is an explicit *data-dependent exact algorithm choice*, not synonymous with brute or KD-tree universally. Small low-dimensional data were dispatched to KD-tree; the 30- and 64-feature cases were dispatched to brute on these installed library versions.

## Original-master whole-graph paired k=10 results

Each cell shows **original `auto` → explicit alternative median**, in milliseconds, plus the **median of the 20 paired full builder timing ratios expressed as reduction**. Medians and paired-ratio medians are separate statistics; do not derive the precise percentage from the ratio of the printed medians.

| Host | Dataset | Original auto ms | Brute time / paired speedup | KD-tree time / paired speedup | Ball-tree time / paired speedup |
|---|---|---:|---|---|---|
| Linux | Iris | 1.347 | 1.165 / **+12.80%** | 1.328 / −3.36% | 1.339 / −2.61% |
| Linux | Wine | 1.755 | 1.305 / **+25.51%** | 1.750 / −0.56% | 1.690 / +2.93% |
| Linux | Breast Cancer | 3.593 | 3.569 / −0.49% | 12.377 / **−242.49%** | 9.121 / −159.09% |
| Linux | Digits | 21.180 | 21.085 / +0.81% | 172.066 / **−713.26%** | 139.022 / −556.52% |
| Windows | Iris | 1.531 | 1.268 / **+17.54%** | 1.445 / −2.71% | 1.490 / −4.64% |
| Windows | Wine | 2.035 | 1.437 / **+29.11%** | 2.067 / +3.91% | 1.968 / +7.80% |
| Windows | Breast Cancer | 4.130 | 4.152 / −1.70% | 13.629 / **−243.63%** | 10.760 / −153.44% |
| Windows | Digits | 23.552 | 23.492 / +0.89% | 182.664 / **−677.09%** | 151.592 / −543.43% |

**All 12 real dataset×k arms matched fully exact original graph outputs for all explicit engines**, so real-data semantic parity alone is insufficient to choose a production replacement. In low-dimensional Iris/Wine, explicit `brute` was **~13–35% faster** depending on k and OS, but the improvement on the higher-dimensional datasets that already selected brute was negligible. Explicit KD/ball trees slowed high-dimensional graph construction by multiples as high as ~9.6× on the selected hosts. The separate `fit` vs `kneighbors(X)` timing records confirm the NN query itself dominates Digits and that KD/ball-tree methods substantially worsen it; all 960 phase rows are archived.

### Prespecified tied-distance and duplicate-sample safety study

These synthetic tests **intentionally include exact ties and duplicate input points** and enforce complete symmetric output identity, not just approximate equal distances.

| Explicit engine | Linux exact / 30 | Windows exact / 30 | Failing cases / OS | First incompatible example |
|---|---:|---:|---:|---|
| `brute` | 24 | 24 | 6 | Duplicate-pair rows, k=1, Manhattan p=1: **1 edge missing and 1 unexpected** |
| `kd_tree` | 28 | 28 | 2 | All-equal rows, k=10, Manhattan p=1: **1 edge missing and 1 unexpected** |
| `ball_tree` | 28 | 28 | 2 | All-equal rows, k=10, Manhattan p=1: **1 edge missing and 1 unexpected** |

Thus different sklearn exact search algorithms can produce **nonidentical ordered production PCC graphs when distances tie**, including changed undirected edge sets *after symmetrization*. A graph can be reciprocal and valid under each method but still differ from the original input graph, influencing downstream PCC outcomes. A speedup is not semantic equivalence under these conditions. These failing cases are retained; the experiment was not declared failed merely because a candidate violates the strict drop-in behavior contract.

## Exact frozen universal engine decision

The preregistered gate required: (1) exact complete graphs on *all* real **and adversarial synthetic** inputs, on Linux and Windows; (2) ≥5% paired median **full builder** gain on at least **three of four k=10 datasets per OS**; and (3) no >5% median regressions on any of the 12 real dataset/k cells per OS.

| Candidate | Exact all fixtures both OS? | k10 datasets gaining ≥5% (Linux / Windows) | Severe regression on high-dimensional inputs? | Decision |
|---|---|---|---|---|
| `brute` | **No** (24/30 synthetic per OS) | **2/4, 2/4** | No material high-dimensional gain; not enough universal benefit | **Reject universal replacement** |
| `kd_tree` | **No** (28/30 synthetic) | 0/4, 0/4 | Yes: many-fold slower on BC/Digits | **Reject** |
| `ball_tree` | **No** (28/30 synthetic) | 0/4, 1/4 | Yes: many-fold slower on BC/Digits | **Reject** |

**Final decision: KEEP current production `NearestNeighbors(algorithm="auto")` unchanged.** Do not add an empirically fit dataset-name or dimensionality-based override based on these four datasets; a prospective new input-dependent decision rule would require independent predeclared tests, protection of tied-distance and self-neighbor behavior, plus separate benefit measurements. No kernel/graph/default changes and no PR merge were made.

## Next engineering decision

- This result does **not** negate the independently demonstrated minimal original-graph **CSR-to-dense-array optimization [#192](https://github.com/fbreve/pypcc/pull/192)** (two expressions plus permanent tests), already validated in [experimental #191](https://github.com/fbreve/pypcc/pull/191). That optimization removes temporary arrays **without changing nearest-neighbor ranking** and showed **4.4–16.14%** paired whole-builder speedups across 24 real graph cases in independent Linux/Windows runs. Keeping `auto` unchanged is compatible with integrating #192 after explicit review.
- Any new proposal to accelerate the search itself (batching, alternate exact query implementations, candidate tree thresholds or parallelism) must **preserve entire ordered original graph outputs on ties/duplicates and original self-neighbor handling**, not merely improve mean neighbor distances or classification score. Different platform sklearn/CPU/thread configurations must be held constant. Preserved graphs imply identical model inputs; also keep the existing native regression gate. Avoid another unplanned large trial using these same synthetic fixtures as the optimization-tuning set.
- The current GitHub runner pilot is not a controlled workstation benchmark, and 20 technical timing repetitions do not establish universally valid wall-clock speedups or p-values. Keep this research-only PR #193 **DRAFT/UNMERGED**.
