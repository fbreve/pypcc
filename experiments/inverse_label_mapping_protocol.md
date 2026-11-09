# Precommitted V1: inverse mapped-label conversion CPU versus temporary memory

**EXPERIMENTAL DRAFT ONLY.** This research branch is based directly on production `master` commit `15695279884a82d0c71b4cb395187a04a1c8a698`, which already contains #201, #203 and #205. The protocol is written/committed **before the candidate source edit, GitHub CI or native experiment**. Do not combine with earlier research PRs, especially one-hot node initialization NO-GO #202. The deliberately padded dense `int64 neib_list` is an intentional CPU optimization and must not be converted to sparse/CSR. Bounded transient allocations to gain CPU are acceptable.

## Hypothesis; exact sole candidate source change

The original `ParticleCompetitionAndCooperation.fit_predict` inverse mapping at the end is:

```python
            # Unmap labels before returning
            final_labels = np.full(self.node.label.shape, -1, dtype=np.int64)
            for idx, lbl in self.idx_to_label.items():
                final_labels[self.node.label == idx] = lbl
            return final_labels
```

Replace **only** this block with:

```python
            # Unmap labels before returning
            final_labels = np.full(self.node.label.shape, -1, dtype=np.int64)
            if self.c < 8 or self.node.label.size < 1024:
                for idx, lbl in self.idx_to_label.items():
                    final_labels[self.node.label == idx] = lbl
            else:
                valid = (self.node.label >= 0) & (self.node.label < self.c)
                final_labels[valid] = self.unique_labels[self.node.label[valid]]
            return final_labels
```

This prospective dispatch guard is an informed but **not scientifically established** choice: pre-experiment local abbreviated NumPy pilot suggests the mask/fancy indexing regresses for binary labels but improves for many classes. It may not be retuned after formal results. The existing original code remains for `c<8` and `n<1024`. Both branches produce a new independent writable C-contiguous `np.int64` output. For valid labels 0..c-1, vectorized sorted `unique_labels` indexing is exactly the inverse `idx_to_label`. For invalid internal `node.label` (negative/ >=c), preserve original output **-1** rather than allowing NumPy negative or out-of-range indexing; these cases must be explicitly tested even if native propagation normally restores unknown labels to a valid class. No change to mapping in #205, original external input validation/range policy, graph building, any native propagation, RNG, early stop, `node.label`, full-state rollback, dictionaries, or any other code.

## Frozen measurements and gates

1. Audit source by checking only this one exact hunk differs between the **actual original Git `pcc.py` blob** and candidate; compile original blob in a separate Python module. Extract both real original/candidate inverse mapping blocks byte-for-byte into small callables for micro (including return, allocated int64 output and all predicates); never replace the baseline with a reimplementation. Match `node.label`, sorted true class labels and dictionaries, and verify actual equality/ownership and invalid-label -1 behavior.
2. Focused tests: 0/1/1023/1024/1797/30000 nodes, 2/3/7/8/10/50 classes, all-valid/invalid indices including -1,-2,c,c+1, int64 extremes; original class label values nonconsecutive, negative including int64 extremes (but -1 excluded as external sentinel), unique label class mappings, output shape/dtype/C order/writability/no alias, input `node.label` untouched, noncontiguous original internal label views, and real fitted NumPy/Numba/Cython random-seeded paths. The full existing regression must pass for Linux/Windows, Python 3.10/3.12.
3. **True original-vs-candidate inverse-block micro** per OS: `(150,3),(569,2),(1023,10),(1024,10),(1797,10),(30000,2),(30000,7),(30000,8),(30000,10),(30000,50),(100000,2),(100000,10),(100000,50),(100000,50-invalid)` = **14 fixed fixtures**, with deterministic `np.random.default_rng(38011 + n + c + invalid)` and valid class indexes uniformly distributed (including explicit 5% invalid `node.label` positions in invalid stress), plus class label values `-1000000 + 271 * index` (none equals -1). **25 alternating original/candidate paired CPU repeats** per fixture/OS; separately **5 balanced `tracemalloc` peak pairs** per fixture/OS; warmups unmeasured, GC not inside CPU timing. 700 CPU pairs and 140 traced peak pairs across two OS; store complete raw paired rows, dtype, layout, SHA and provenance.
4. **Complete real native original-vs-candidate PCC**: StandardScaler + original union-undirected kNN graph for sklearn Iris/Wine/Breast Cancer/Digits; three labels per class, independent label RNG seeds **38027 and 38053**, explicit propagation seed **38059**, `max_iter=1000`, `es_chk=10`, `early_stop=False/True`, same NumPy/Numba/Cython backends; **48 whole fit pairs/OS = 96 native comparisons**. Compare six output arrays SHA256 (predictions, dominance, positions, strengths, uint8 distances, owndeg), complete stopping counters/reason/iteration count, NumPy RNG state (where measurable), actual native no-fallback identity and correct C/F dominance layout in both models, plus seeded known-label assignments and input/graph fingerprints; record whole fit/init/propagation/postprocessing timestamps. Do **not** infer cross-backend RNG equality; compare each backend to itself.
5. Third mandatory aggregate must reject missing/incomplete two-host results, missing 14 fixture groups and exactly 25/5 pairs, invalid shapes/nonfinite timing or peak, source mismatch, different platform input hashes, invalid outputs/states and missing native comparisons. **Performance NO-GO is a valid completed experiment and must not fail scientific CI.**
6. Prospective **GO** requires (i) all complete exact scientific and four-platform regression; (ii) >=**25% paired median whole-inverse-mapping CPU savings** for each of `(30000,10),(30000,50),(100000,50)` on BOTH Linux and Windows; (iii) no more than **10% CPU regression** for small/binary/control cases `(150,3),(569,2),(1023,10),(1024,10),(1797,10),(30000,2),(30000,7),(30000,8),(100000,2),(100000,10),(100000,50-invalid)` on either OS; (iv) <=**10% median whole-fit regression** in each real dataset/backend/stop cohort (over 2 seeds) per OS; (v) extra `tracemalloc` peak on any micro <= `24*n + 524288` bytes (CPU-for-memory bound; it is NOT process RSS); (vi) unchanged owned output int64 and deliberately dense graph.
7. If GO, retain experiment draft and create **separate minimal production PR directly from then-current master** containing only this one hunk and permanent tests; require independent four-platform CI and review before any merge. If NO-GO, leave production unchanged with reproducible artifacts. Never tune the decision gate or threshold after reviewing measurements.

The earlier one-off abbreviated local pilot is exploratory only and does not constitute a native whole-function or PCC end-to-end benchmark.
