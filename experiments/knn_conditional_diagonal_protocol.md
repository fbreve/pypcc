# Frozen V1: conditional diagonal cleanup in ORIGINAL kNN graph constructor

**SCIENTIFIC DRAFT; not a production PR.** Exact frozen `master` base `c1073ae53ea785031de9e474b678e94f44a2124d`, after the completed [#209](https://github.com/fbreve/pypcc/pull/209) stage attribution. This protocol is committed **BEFORE any candidate source modification or CI measurements**.

## Rationale, exact single source change
The current `build_knn_graph` uses `NearestNeighbors(algorithm="auto").fit(data).kneighbors(data)[:,1:]`, builds `rows` and `cols`, creates sparse directed CSR, symmetrizes via `adj=adj+adj.T`, then unconditionally invokes `adj.setdiag(0);adj.eliminate_zeros()`. #209 found this CSR cleanup stage non-negligible, while most graphs do not contain a self-edge. The previously merged [#195](https://github.com/fbreve/pypcc/pull/195) skips unnecessary cleanup in **edge_index** graphs; kNN must be independently audited because ties/duplicates can put the query itself at a **nonfirst position** (a documented failure mode of simplifying the self-neighbor exclusion). Retain the exact current `algorithm="auto"`, `kneighbors(data)`, first-result discard, old graph dtype, sorted CSR order, int8 edges, dense padded int64 outputs, zero/remove cleanup on actual self-loops, and the explicit CPU-oriented dense `neib_list` representation.

Change ONLY in `pcc_graph.build_knn_graph`:

```python
    adj.setdiag(0)                          # remove self-loops (safety)
    adj.eliminate_zeros()
```

to exactly:

```python
    if np.any(rows == cols):
        adj.setdiag(0)                      # remove actual self-loops
        adj.eliminate_zeros()
```

No change to `build_graph_from_edge_index`, graph construction engine/metric/parameters, graph order, PCC set_graph/build_graph, all NumPy/Numba/Cython kernels, seeds, convergence, stopping, public API or installation metadata. Predicate uses the **existing** directed `rows` and `cols` and must check all entries (not just the first neighbor). When any self-edge is present, execute the unchanged original cleanup; when none, skip two operations. Since skipping can change CSR ordering, **entire output `neib_list` and `neib_qt` including padding must match bitwise**; if not, NO-GO, even if connectivity is equivalent. Extra bounded temporary allocation for the n×k boolean predicate is allowed only for demonstrated CPU improvements.

## Frozen candidate-vs-original Git source experiment
1. Use `git show c1073ae53ea785031de9e474b678e94f44a2124d:pcc_graph.py` to compile actual frozen **reference** source and verify that production candidate `pcc_graph.py` differs from it by exactly this one hunk. Guard all other production files, `benchmarks/` and source/kernel/config against modification. Do not treat a hand-reimplemented graph builder as reference.
2. Two GitHub-hosted independent OS jobs: Linux and Windows, Python 3.12, all BLAS/Numba thread variables set to 1, native `pcc_step`/Numba/NumPy installed. Record host/compiler/package identities and original source SHA. Synthetic local fixtures and standardized sklearn datasets only; no external downloading or OpenML.
3. Exactly 21 graph cases/OS: four original standardized real datasets Iris/Wine/Breast Cancer/Digits × **k=1,3,10** (=12), plus three original-style duplicate/tie stress groups (`duplicate512` 512×8, `duplicate2048` 2048×16 and `all_identical64` 64×4) × **k=1,3,10** (=9). Fixed independent RNG 44001+n for the first two groups, identical zero rows for third. Explicitly determine how many of original directed `rows==cols` survive the `[:,1:]` kNN slice, and require **at least one case with real self-loops**, and at least one without. Ties may resolve across host differently; compare exact original/candidate graph arrays **within each host**, cross-host graph fingerprints for the four original real datasets, and cross-host input array fingerprints in every case.
4. Original and candidate warmups unmeasured. **25 alternating source-reference/candidate complete `build_knn_graph` CPU timing pairs** per case/OS (21×25×2 = **1,050 original-vs-candidate paired full graph builds**); independently **five original/candidate matched `tracemalloc` peak pairs** per case/OS (21×5×2=**210 tracked allocation peak pairs**). Baseline and candidate functions allocate new outputs on every call and their output arrays must be bitwise identical, writable, C-contiguous, same int64 dtype/shape/padded -1 values, original CSR neighbor order, original input not mutated. Measure elapsed with no tracing; measure memory separately with own tracemalloc; no dropped samples. Do not confuse the sparse/dense stage improvement with whole graph wall CPU or the fitted PCC time.
5. Original `build_graph_from_edge_index` must remain byte-for-byte unchanged, including its documented n=0, loop/duplicate, and isolate behaviors. Test original vs candidate **exception types/messages** for original invalid graph inputs and verify known explicit edge source unchanged. For tie inputs, validate graph identity regardless of query's own index rank; do not force or alter sklearn semantics.
6. **96 actual native original-vs-candidate full PCC pairs** total (4 real sklearn k=10 datasets × 2 independent fixed class-label mask seeds **42027 and 42053** × native NumPy/Numba/Cython × both fixed/early stopping modes × 2 OS). Propagation seed **42059**, 1,000 iteration cap and `es_chk=10`. Compare all six native final state hashes exactly, detailed stop counters/reason/effective count, true backend module/name/no fallback, NumPy RNG state where available and exact Numba C/F native dominance layout. No cross-backend RNG parity assertion.
7. Third **mandatory exact-completeness aggregator** fails for any missing reference/candidate graph pair, changed graph array/neighbor order, incomplete 25/5 pairs, missing observed self-loop and loop-free cases, invalid/changed exception type, missing native parity, missing/changed corpus, unverifiable sources or incorrect OS. Scientific CI must **not** fail solely due to a scientifically valid CPU NO-GO. Standard four-platform package regression on experimental head must be green before any production promotion.

## Prospective GO vs NO-GO CPU/memory gates (immutable)
- GO only if all exact function/native and completeness gates pass AND **at least two of the four real k=10 datasets on EACH OS save >=5% paired-median FULL graph-builder CPU time**; no case of any 21 cases/OS may regress >**10%** full-builder CPU time; and no case/OS may exceed `16 * n * k + 1048576` bytes additional median paired tracked Python/NumPy allocation peak (a permissive explicit CPU-for-memory tradeoff; `tracemalloc` is not process RSS).
- Otherwise NO-GO; leave production unchanged; retain raw negative artifacts and do NOT retune cutoffs or delete adverse tied cases.
- Even if scientific GO, **never merge this research PR**. Extract a separate **two-file** minimal master-based production candidate (only `pcc_graph.py` hunk + permanent focused tests), require its own four-platform regression, compare exact source Git blob, then consider squash merge. Avoid stacked research branches, rejected ideas and deferred background work.

This experiment tests only a guarded skip of redundant original CSR diagonal cleanup. It does not test changing sklearn search engines, removing reciprocal edges, converting dense graphs to CSR, or modifying the particle algorithm.
