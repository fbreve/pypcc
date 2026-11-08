# Exact CSR-offset graph materialization: measured findings

**FROZEN SCIENTIFIC PROTOCOL COMPLETED — candidate was measured *after* preregistration** in independent experimental [draft PR #191](https://github.com/fbreve/pypcc/pull/191), branched from production `master` `8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128`. No merge. Neither PCC native propagation (NumPy/Numba/Cython) nor graph topology, neighbor ordering, label choice, RNG, early stopping, public API or dependency specification was changed; the production candidate diff is limited to **two identical CSR-to-dense-neighbor-offset replacements** in `pcc_graph.py` (`build_knn_graph` and `build_graph_from_edge_index`).

## Provenance, verification and reproducibility

- [Frozen protocol](graph_dense_layout_protocol.md) was committed first (`aaead32d468ec3c6aeda9050bea0b001a18191e1`), before patch/CI. Candidate source patch commit `7b25f674fe4c840535e4293f06b105a81cd791e1`.
- [Final fully green two-OS scientific run **42441649629**](https://github.com/fbreve/pypcc/actions/runs/42441649629) at code/test head `88888a5dc83a696654f69a3ee53852f7cfb8f6d2`: original Linux and Windows study jobs plus the mandatory exact-complete aggregation **SUCCESS**. [Machine-verified report artifact **13601514619**](https://github.com/fbreve/pypcc/actions/runs/42441649629/artifacts/13601514619), all 360 per-OS original-vs-candidate timing pairs + actual phase samples, raw graph hashes and native equivalence states [Linux artifact 13601569427](https://github.com/fbreve/pypcc/actions/runs/42441649629/artifacts/13601569427) and [Windows artifact 13601639433](https://github.com/fbreve/pypcc/actions/runs/42441649629/artifacts/13601639433).
- A separate **first fully green independent replication run [42441589093](https://github.com/fbreve/pypcc/actions/runs/42441589093)** was made before a **test-only** improvement in the handling of duplicate samples; it too validated all complete graph/phase/native cohorts. The test correction is scientifically necessary: original sklearn `kneighbors` with tied duplicate input rows can return the *query row itself* after the first position, causing the original `build_knn_graph` to return degree less than `k` when it removes the diagonal. The candidate **preserves** this original behavior, rather than silently modifying input graph semantics.
- Production regression `PCC regression` for the tested code/test head passed on Linux Python 3.10 and 3.12 when documentation was prepared; Windows status was still being checked. A docs-only findings commit should trigger separate generic regression; check the current head before merging.
- **Strictly 720 paired full-graph builds** (4 real standardized datasets × 3 fixed k × 30 original/candidate pairs × 2 OS), **240 original-phase profile samples** (4 datasets × 30 × 2 OS), and **96 independent original-vs-candidate native model state checks** (4 datasets × 2 label seeds × 3 original backends × 2 original stopping modes × 2 OS), plus 5 adversarial edge-index cases per OS. All 12 real dataset/k graph outputs and all 5 edge-index output families matched the *actual pristine original Git blob* bit-for-bit on **entire `neib_list` and `neib_qt` including order, shape, dtype and padding**. All 96 native PCC checks passed exact six final state-array hashes, native iteration counts/stop reasons and held-out accuracy when using original vs candidate graphs **within the same backend and seed**.
- **No performance speed threshold was a CI gate.** Every measurement was retained. Windows and Linux ran on separate, uncontrolled hosted hardware and timings were never pooled; native backends have distinct RNGs so their results were not falsely required to be identical across backends. The native 1000-iteration cap and actual potential check do **not** prove long-horizon convergence. This is a candidate optimization, not a proposed default/behavior change.

## Full graph construction — final validated run

Elapsed milliseconds; each entry shows original median → candidate median and **median of the 30 paired time ratios**. The percentage improvement is `100×(1−median(candidate/original))`, so it is not necessarily exactly the percentage reduction computed from the separate pooled medians.

| OS | Dataset | k | Original ms | Candidate ms | Paired median time reduction |
|---|---|---:|---:|---:|---:|
| Linux | Iris | 1 | 1.051 | 0.955 | +9.49% |
| Linux | Iris | 3 | 1.115 | 1.005 | +9.52% |
| Linux | Iris | 10 | 1.260 | 1.160 | +8.13% |
| Linux | Wine | 1 | 1.385 | 1.265 | +8.31% |
| Linux | Wine | 3 | 1.496 | 1.368 | +9.24% |
| Linux | Wine | 10 | 1.702 | 1.580 | +7.00% |
| Linux | Breast Cancer | 1 | 2.254 | 1.887 | +16.14% |
| Linux | Breast Cancer | 3 | 2.539 | 2.129 | +16.01% |
| Linux | Breast Cancer | 10 | 3.427 | 3.006 | +12.16% |
| Linux | Digits | 1 | 17.006 | 15.781 | +7.27% |
| Linux | Digits | 3 | 17.838 | 16.446 | +7.87% |
| Linux | Digits | 10 | 20.986 | 19.503 | +7.11% |
| Windows | Iris | 1 | 1.375 | 1.265 | +9.07% |
| Windows | Iris | 3 | 1.485 | 1.351 | +10.47% |
| Windows | Iris | 10 | 1.947 | 1.686 | +8.34% |
| Windows | Wine | 1 | 2.095 | 2.061 | +4.94% |
| Windows | Wine | 3 | 2.354 | 2.290 | +4.40% |
| Windows | Wine | 10 | 2.606 | 2.325 | +12.42% |
| Windows | Breast Cancer | 1 | 3.609 | 2.975 | +13.49% |
| Windows | Breast Cancer | 3 | 4.022 | 3.463 | +13.93% |
| Windows | Breast Cancer | 10 | 6.166 | 5.422 | +13.58% |
| Windows | Digits | 1 | 25.388 | 24.477 | +7.20% |
| Windows | Digits | 3 | 22.585 | 20.880 | +7.64% |
| Windows | Digits | 10 | 25.998 | 24.251 | +6.56% |

**All 24 original-full-graph benchmark cells improved (none of the four datasets at any prespecified k/OS regressed in median paired wall-clock timing)**. The improvement was **4.40%–16.14%** on these hosts, robust to the independent first completed CI run even though absolute hosted CPU times changed. This describes the CI host and tested datasets, not all processors or large high-degree graphs.

## Attribution to original k=10 phases

Milliseconds; rows are stage medians from 30 independent measured samples with a shared original CSR adjacency for each scatter pair.

| OS | Dataset | Original NN fit+query | Sparse reciprocal CSR | Original dense materialization | Candidate dense materialization | Median candidate/original scatter |
|---|---|---:|---:|---:|---:|---:|
| Linux | Iris | 0.779 | 0.348 | 0.137 | 0.035 | 0.245× |
| Linux | Wine | 1.160 | 0.376 | 0.160 | 0.039 | 0.243× |
| Linux | Breast Cancer | 2.351 | 0.586 | 0.481 | 0.078 | 0.161× |
| Linux | Digits | 18.135 | 1.250 | 1.594 | 0.197 | 0.125× |
| Windows | Iris | 1.110 | 0.525 | 0.173 | 0.064 | 0.356× |
| Windows | Wine | 1.756 | 0.616 | 0.245 | 0.081 | 0.331× |
| Windows | Breast Cancer | 4.261 | 1.208 | 0.666 | 0.177 | 0.259× |
| Windows | Digits | 22.027 | 2.038 | 2.007 | 0.503 | 0.240× |

The frozen hypothesis was confirmed: per-vertex NumPy allocation overhead exists in the ordered CSR scatter, and replacing it gives roughly **64%–88% less scatter phase time** on these hosts, but the original nearest-neighbor search dominates particularly for Digits, so the **complete builder gain** is smaller. The paired end-to-end measurement, not microphase time, is the primary outcome.

## Engineering decision

- **Recommend a tiny standalone production candidate PR** containing only the two changed `pcc_graph.py` scatter expressions and a focused permanent regression test for exact undirected graph order, ties and edge-index empties. All scientific scripts, raw benchmark archives and experimental CI should stay in **draft PR #191**, independent of the minimal production patch.
- Do **not** merge this experimental scientific PR wholesale; the benchmark harness and workflow are not needed to ship a two-line graph optimization. No change to original propagation behavior is approved by these timings alone.
- Re-run complete native regression on the minimal production candidate, including Cython/Numba original backends on Linux and Windows. Consider a clean Windows native workstation benchmark before any speed claim for the user's specific hardware; runner benchmarks are evidence only for their hosts.
- This improvement reduces graph-building latency when a graph is constructed; repeated fits using the same graph amortize this cost, so there may be little benefit to workflows dominated by already-compiled propagation.
