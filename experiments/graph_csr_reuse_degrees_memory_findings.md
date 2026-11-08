# Independent V2 CSR scatter memory study — PASS under frozen gates

**Scientific verification and candidate-promotion gates: PASS** in independent research [PR #197](https://github.com/fbreve/pypcc/pull/197), based DIRECTLY on production `master` `5a8cab9978cf367b8e02341ac856f297b7f2c2c5`. The [V2 protocol](graph_csr_reuse_degrees_memory_protocol.md) was committed before V2 source edits/CI outcomes, and retained V1's exact no-regression thresholds. Failed first-attempt scientific [PR #196](https://github.com/fbreve/pypcc/pull/196) remains independent and unmerged, with its NO-GO reported rather than retroactively excused.

## Verified immutable scientific run

- [**Fully green V2 scientific CI run 42487280916**](https://github.com/fbreve/pypcc/actions/runs/42487280916) — Linux original benchmark, Windows original benchmark and mandatory strict two-OS aggregation all SUCCESS at `6c6be58c0c4c81add0c545b51f125d3892ce1375`. [Machine-verified all-cohort aggregate artifact **13619748821**](https://github.com/fbreve/pypcc/actions/runs/42487280916/artifacts/13619748821), raw all-sample [Linux **13620341088**](https://github.com/fbreve/pypcc/actions/runs/42487280916/artifacts/13620341088) and [Windows **13619703787**](https://github.com/fbreve/pypcc/actions/runs/42487280916/artifacts/13619703787).
- Exactly **1,400/1,400 alternating, complete, actual-original-vs-V2 graph builder wall-time pairs**, **350/350 independently isolated `tracemalloc` peak-allocation pairs** (700 individual samples), **192/192 original native complete six-array PCC state/stopping/accuracy comparisons** across three installed original NumPy/Numba/Cython backends on two new label seeds, both graph construction paths and both original stop modes, and **40/40 original invalid/tied/noncontiguous behavior cases** on Windows/Linux. All graph arrays **bitwise equal**, including full sorted original neighbor order, all `-1` padding and `int64` degree/neighbor shape/dtype. Both OS matched input and original result hashes. All raw timings retained, including negative gains.
- Original master reference was loaded directly from pinned original Git blob, not approximated in a reimplementation. Exact candidate source guard only allows the two CSR scatters to (1) reuse the already-allocated original `neib_qt` degree vector, and (2) allocate large temporary row/column indexes using **SciPy's original safe CSR index/indptr dtype**, assigning `adj.indices` directly into the **unchanged int64 output**. No graph topology, edge dedup, original sklearn `algorithm='auto'`, label selection, RNG, stopping, native kernel, packaging or defaults changed.
- Source-head [ordinary PCC regression run **42487280832**](https://github.com/fbreve/pypcc/actions/runs/42487280832) passed Linux Python 3.10 and 3.12; Windows 3.10 and 3.12 were still executing at the time this findings document was drafted. CI after docs will be reviewed independently. Do not assume the final documentation-head CI is already complete.

## Original complete-builder tracked peaks and paired timing

These are **5-run original/candidate paired `tracemalloc` peak medians**, measuring Python/NumPy-tracked allocations with returned graphs live. They **do not represent** system RSS, total native memory or GPU peak. The final `neib_list` and `neib_qt` themselves remain identical `int64` allocations, and are included in this tracked peak; transient savings arise from fewer/smaller intermediate arrays. All graph timings were sampled separately in 20 full-builder alternated paired repetitions, without tracing.

| Prespecified case | Original traced peak (MiB) | V2 peak (MiB) | Median tracked peak saving (both OS) | Paired time reduction Linux / Windows |
|---|---:|---:|---:|---:|
| 1,000 nodes / 30,000 directed edges | 3.522 | 2.786 | **20.90%** | +11.28% / +2.08% |
| 10,000 nodes / 100,000 directed edges | 12.765 | 10.339 | **19.01%** | +14.46% / +3.84% |
| 30,000 nodes / 100,000 directed edges | 14.753 | 12.172 | **17.49%** | +14.48% / +1.07% |
| Digits k10 / original `build_knn_graph` | 1.830 | 1.464 | **20.01%** | +0.10% / +0.07% |
| Digits k10 / original `build_graph_from_edge_index` | 2.314 | 1.948 | **15.82%** | +11.01% / −2.54% |
| Iris k1 / edge-index small-case | 0.029 | 0.029 | **−1.32%** | −2.33% / −4.17% |
| **NEW** eight isolated nodes | ~0.004 | ~0.004 | **+5.15%** | −2.23% / +1.79% |
| **NEW** 32 nodes / 16 edges | ~0.010 | ~0.010 | **+2.20%** | +3.25% / +5.40% |

Paired peak reductions for identical inputs were the same to two decimals across OS due to deterministic tracked allocations; **independent host full-builder timings differ** and were not pooled. The previously failing Iris k1 edge-index case now shows a **1.32%** increase, which lies within the unchanged 5% bound. The new tiny cases also satisfy the frozen memory and runtime limits. These experiments are limited to the selected graph sizes and GitHub runners; a worst-case padded hub graph may be dominated by the permanent dense output rather than temporary indices.

## Frozen V2 production candidate decision

The rule was defined **before** V2 measurements and intentionally NOT weakened after V1's negative outcome:

1. Exact complete graph/invalid input/native six-final-state/stopping agreement in all tested contexts — **PASS**.
2. At least **10%** reduction in paired median full-function tracked-peak memory for **all three named high-edge synthetic graphs** on **each OS** — **3/3 Linux, 3/3 Windows: PASS**.
3. No >**10%** paired median *full builder time* regression for **any** of 35 conditions/OS — **zero: PASS**.
4. No >**5%** paired median tracked-memory peak regression for **any** of 35 conditions/OS — **zero: PASS**.

**Overall: PASS.** This is a green light for a **different, minimal, current-master-based production candidate PR** consisting of the small `pcc_graph.py` edit plus permanent adversarial graph equivalence tests, followed by the full Linux/Windows Python3.10/3.12 wheel/sdist CI. **It does not authorize merging research PR #197, #196 or their benchmark tools into `master`**. Graph construction peak-tracing improvements are not necessarily total memory/RSS improvements; the public graph remains dense padded int64.

## Engineering implications

The remaining substantial memory consumer for irregular, high-degree graphs is the *required dense padded `neib_list`*, whose storage size is fixed by the public PCC representation and can dwarf temp index allocations. Any sparse public-graph API is a separate model-level redesign requiring significantly broader cross-backend equivalence and scientific review, not a continuation of this two-site cleanup. The same original native stopping criteria and backend choices remain intact.
