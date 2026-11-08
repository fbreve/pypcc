# Frozen CSR index width memory experiment — completed, NO-GO

**Scientific completeness and exact model/graph equivalence: PASS. Frozen production promotion threshold: NO-GO.** Keep independent [research PR #196](https://github.com/fbreve/pypcc/pull/196) **DRAFT/UNMERGED**. Protocol `graph_csr_index_memory_protocol.md` committed before the proposed graph-code edits or CI timings; anchored to already optimized `master` **`5a8cab9978cf367b8e02341ac856f297b7f2c2c5`**.

## Complete scientific evidence

- [Fully green scientific run **42486312580**](https://github.com/fbreve/pypcc/actions/runs/42486312580): Windows, Linux and mandatory independent two-OS strict aggregate all **SUCCESS**. [Machine-checked complete report **artifact 13619677146**](https://github.com/fbreve/pypcc/actions/runs/42486312580/artifacts/13619677146); full raw timing, traced peaks, original graph fingerprints, exact native states and environment: [Linux **13619468328**](https://github.com/fbreve/pypcc/actions/runs/42486312580/artifacts/13619468328), [Windows **13620210093**](https://github.com/fbreve/pypcc/actions/runs/42486312580/artifacts/13620210093).
- Separate [ordinary PCC regression **42486312507**](https://github.com/fbreve/pypcc/actions/runs/42486312507) **4/4 Linux/Windows × Python 3.10/3.12 all SUCCESS** on scientific code head `c3cf294e04da28b9f01defd6e140e0eb06ae70eb`.
- Original reference compiled directly from frozen Git master blob, with exact source-diff guard limiting candidate to two scatter sites. **1,320/1,320 paired original-vs-candidate complete graph builder timed comparisons**, 330/330 independently isolated paired whole-builder `tracemalloc` peaks (660 individual original/candidate samples), **192/192 NumPy/Numba/Cython original same-backend six-array final state/stopping/accuracy parity checks** (original sequential fixed and early stop, 2 seeds per dataset and both graph sources), and 40/40 tied/noncontiguous/invalid graph behavior controls all passed on Linux and Windows. Complete **ordered padded int64** neighbor list and original int64 degree outputs unchanged across all 33 conditions/OS. No time results or outliers were filtered.

## Representative verified host results

Each figure is **median of 5 paired complete-builder tracked peak ratios**, reported as a percentage reduction. These are **tracemalloc-tracked Python/NumPy allocation peaks**, *not* OS RSS, GPU allocation or exact native-heap memory peaks. The public int64 padded output is retained in the traced function and is **not** reduced by this candidate.

| Condition | Original traced peak (MiB) | Candidate traced peak (MiB) | Paired reduction Linux / Windows |
|---|---:|---:|---:|
| 1,000 nodes / 30,000 directed edges | 3.565 | 2.835 | **20.49% / 20.49%** |
| 10,000 nodes / 100,000 directed edges | 12.536 | 10.186 | **18.75% / 18.75%** |
| 30,000 nodes / 100,000 directed edges | 14.753 | 12.401 | **15.94% / 15.94%** |
| Digits k10, original kNN builder | 1.830 | 1.478 | **19.25% / 19.25%** |
| Digits k10, original edge-index builder | 2.314 | 1.962 | **15.22% / 15.22%** |
| Iris k1, original edge-index builder | 0.029 | 0.031 | **−5.64% / −5.64%** |

The two frozen *memory-regression* violations were both **Iris k=1 edge-index**: one on Linux and one on Windows, each representing about **1.7–2 KiB more tracked peak**. The other small cases also showed minor negative or positive memory changes, but no other case exceeded 5% regression. All three high-edge cases improved at least 10% on both OS (3/3 each). Every complete-builder paired median timing regression stayed within the frozen 10% bound (0 violations across 66 OS×condition cells). Larger cases benefited in memory; small graph fixed allocations and dtype metadata overhead appear disproportionate. That is a possible explanation, **not a measured causal attribution**.

The exact frozen gate was conjunctive: no graph/native divergence; ≥10% tracked-peak improvement for *all three* dense high-edge cases on both OS; no >10% paired wall-clock regression in any condition; **no >5% paired tracked memory peak regression in any condition**. The final condition failed for Iris k1 edge-index on both OS. **NO-GO.** Do not reinterpret the 5% bound as an absolute kilobyte threshold after seeing results or integrate only the successful subsets into `master`.

## Hypothesis and follow-up, not an integration authorization

The first candidate used dynamic `adj.indices.dtype` and `adj.indptr.dtype` for temporary scatter row/column positions and avoided the full int64 RHS neighbor copy, while preserving the original int64 return arrays. It **did not** eliminate an independently duplicated int64 `degrees` vector: both constructors create `neib_qt = np.diff(adj.indptr).astype(np.int64)` and then redundantly recompute the same degree array as `degrees = (adj.indptr[1:]-adj.indptr[:-1]).astype(np.int64)`. A *separately preregistered second candidate* could remove the duplicated vector by directly reusing `neib_qt` as the scatter's repeat counts, in addition to native-width scatter indices. That is a distinct, simple memory optimization that could address fixed overhead in small cases, but the first results cannot be reused to claim it works.

Any follow-up needs an **independent branch based directly on the present production master**, a frozen protocol with new synthetic seeds/graph sizes and a more diverse validation cohort before candidate outcomes, exact same native full-array/RNG/stopping parity, complete independent both-OS benchmarks, and no post-hoc change to the declared acceptance criteria. Do not stack the first failed experimental PR into production or merge the benchmark harness.

**Scope caveat:** The dominant persistent memory cost for a hub-heavy graph is the required dense padded int64 `neib_list`; temporary-index improvements don't shrink that output. A radically sparse output API would be a distinct model redesign not justified by this experiment. 
