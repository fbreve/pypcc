# Precommitted V1: sparse-index `fit_predict` label mapping CPU-vs-allocation audit

**Scientific DRAFT, never merge experiment directly.** Source base: post-#203 production master `a2fd27adf53155f378f12eeeb7cf67ec0625e195`. Protocol committed **before source edits/CI measurement**. Distinct from rejected #202 dominance one-hot and rejected #199 in-place owndeg: neither one is stacked. This is a CPU optimization allowed to use a bounded temporary index array to save repeated full vector scans. Do not replace or compress the **intentional dense int64 padded `neib_list`**, which is retained to optimize propagation CPU.

## Code hypothesis and strict source scope

After unchanged input validation, int64 prepared-label coercion, `np.unique` class ordering, and dictionaries `label_to_idx`/`idx_to_label`, replace ONLY the following original mapping hunk inside `ParticleCompetitionAndCooperation.fit_predict`:

```python
            self.mapped_labels = np.full(self.labels.shape, -1, dtype=np.int64)
            for lbl, idx in self.label_to_idx.items():
                self.mapped_labels[self.labels == lbl] = idx
```

with the following strictly equivalent hybrid:

```python
            self.mapped_labels = np.full(self.labels.shape, -1, dtype=np.int64)
            if self.labels.size < 1024:
                for lbl, idx in self.label_to_idx.items():
                    self.mapped_labels[self.labels == lbl] = idx
            else:
                known_rows = np.flatnonzero(self.labels != -1)
                self.mapped_labels[known_rows] = np.searchsorted(
                    self.unique_labels, self.labels[known_rows]
                )
```

The <1024 fallback is a **prospectively fixed guard** to avoid unnecessary short-array temporary allocations; a preliminary uninstrumented local NumPy fragment suggested sparse-positions `searchsorted` saves CPU on long, sparsely labeled vectors but is not full PCC evidence, and the threshold is not tuned after the CI results. For valid input, `prepared_classes` is sorted, every known label is in it, and -1 is the only unknown sentinel. For n>=1024, temporary `known_rows` is an intp vector of labeled home row indices; the allocated output remains a **new, writable, independently owned, C-contiguous int64 array** with unknown entries -1. Class indices are unchanged. No change to labels, backend, RNG, seeding, state reset/rollback, graph, propagation, return-label inverse map, early stop or dtype/range validation.

## Fixed controls and outcome gates

1. Guard that **only this single hunk** differs between original Git `pcc.py` and candidate and compile the **actual original frozen Git blob** in a separate Python module. Micro must execute the *actual extracted mapping block* from each source, not a rewritten hand-coded imitation. The candidate must not modify `benchmarks.runner.measure` (it was fixed separately in #203).
2. Focused unit/regression: exact mapping for n=0/1/150/569/1023/1024/1797/30000, classes 2/3/7/8/10/50 as applicable; several nonconsecutive/negative/exreme signed int64 labels including -1 sentinel; all-labeled, all-unknown, 3 labels/class, 50% and 90% labeled; writable dtype/layout/copy isolation; class dictionaries; fit state rollback and invalid parameter gates unchanged. Full Linux/Windows Python 3.10/3.12 package CI.
3. Native **matched micro**: `(150,3,3/class),(569,2,3/class),(1023,10,3/class),(1024,10,3/class),(1797,10,3/class),(30000,10,3/class),(30000,50,3/class),(100000,50,3/class),(100000,2,3/class),(30000,50,50%-labeled),(30000,10,50%-labeled),(100000,50,90%-labeled)`. Use `np.random.default_rng(37011)` class-balanced deterministic label assignment and int64 nonconsecutive actual labels; independently check `np.unique` and `label_to_idx` are original; preserve same object states on both functions. **25 alternating original/candidate time pairs** per case/host, separately **5 balanced traced allocation peak pairs** per case/host, GC excluded from timed call; warm-ups excluded. That's **300 CPU pairs and 60 `tracemalloc` pairs per OS**. Report all pairs, class maps, hashes, file provenance, Python/NumPy/OS/CPU. This measurement covers mapping hunk, not full fit and not whole algorithm.
4. Real full PCC seeded parity: `sklearn` Iris/Wine/BreastCancer/Digits after original StandardScaler and undirected union kNN=10. Use 3 labeled/class with two independent label seeds `37027,37053`; same explicit seed 37059, 1000 max iterations, `es_chk=10`, both early_stop flags, original unforked NumPy/Numba/Cython backends, two OS. **48 original vs candidate native fit state pairs per OS**, 96 total. Check all 6 original-vs-candidate exact final SHA256 arrays, complete original stopping info/reason/effective iterations, the true backend callable identity and Numba C/F layout (now reliable after #203), and NumPy RNG state; independent no-fallback checks. Record full-fit, initialization/dispatch, propagation and postprocessing timings; no cross-backend equality claims. Extra large synthetic CPU runs are not substitutes for these complete comparisons.
5. Independent third aggregator rejects incomplete/missing source provenance, missing 12 micro cases, non-matching cross-OS fixture fingerprints, incomplete 25/5 pairs, invalid/missing native results, native parity mismatches and timing-metric NaNs. CI **must not fail simply because the CPU optimization is NO-GO**.
6. Frozen **GO** criterion: (a) all exact-functional and scientific-completeness gates pass; (b) >=**25% paired-median CPU saving in mapping** at each OS for all three sparse target sizes `(30000,10),(30000,50),(100000,50)`; (c) <=**10% CPU regression** on any sparse smaller case (150,569,1023,1024,1797) or 100000×2 on either OS; (d) **no >10% full-fit median regression** on *any* real dataset/backend/stop mode over the two label seeds at each OS; (e) mapped-label temporary allocation tracked-peak increase must be <= **24 × n + 524288 bytes** over original for every case/OS (a bounded CPU-for-memory tradeoff, not an unrealistic requirement to reduce bytes; `tracemalloc` is **not** RSS or native allocation); (f) correct dtype/layout/ownership and original dense graph. Other dense 50/90% micro cases must maintain >= -10% CPU saving, and their tracked-peak under (e). **No threshold changes after outcome**. If GO, extract *separate minimal* master-based production candidate (not scientific experiment) then require own four-platform regression before any integration.

No permanent source changes until the experiment is GO and separately independently validated.
