# Frozen V2 review: remove legacy-path overhead via specialized Cython step

**Prerecorded before making the V2 executable change.** The initial clean-master cross-revision evaluation (workflow [42282512047](https://github.com/fbreve/pypcc/actions/runs/42282512047), commit `8e97a99ea1f48e371541a3375141b7eff59ec828`) yielded **240/240 native fits and 80/80 exact three-way state and stop-info matches**, but *failed its predetermined performance gate*: Linux Digits had **30.22% more time** with the candidate's default-disabled cache, although opt-in cached Digits was **30.31% faster than untouched master on Linux** and **34.01% faster on Windows**. Complete v1 artifact: `clean-master-cython-cache-summary`, ID **13535693263**.

## Frozen V2 hypothesis and implementation constraints

The v1 change put a **runtime `cache_enabled` conditional inside the Cython particle-update hot loop**, which can inhibit compiler optimization on some platforms. V2 must eliminate that inner-loop conditional from the **legacy/default-disabled** kernel:

- Preserve the original `pcc_step_sequential` body **verbatim from the pinned master source**. This function must retain its original Cython argument signature and cannot reference `row_max_cache` or `cache_enabled`. This safeguards the default-disabled path (and public one-step `pcc_step`).
- Add a second **private, specialized `pcc_step_sequential_cached`** function containing the same particle-update code but with a direct `row_max_cache[next_node] = max_dom` after the existing local max scan for unlabeled nodes. No runtime cache-flag conditional inside either particle loop. The duplication is experimental; do not approve a production merge without reviewing maintainability and alternative specializations.
- The outer `pcc_propagate` loop chooses `pcc_step_sequential_cached(..., row_max_cache)` when opt-in cache is active, otherwise `pcc_step_sequential(...)`. This is **one conditional per iteration, not per particle visit**. The existing potential checks, node-order summation, stopping threshold and RNG remain unchanged.
- The new trailing opt-in arguments `experimental_rowmax_cache=False` and verification flag remain default-disabled; no changes to `pcc.py`, NumPy/Numba, graph construction, or `master`.

## Fresh V2 independent-binary evaluation

Use **exactly the same frozen cross-revision three-mode benchmark, full-state and stopping metadata gates and OS×dataset matrix** as the original `clean_master_cython_cache_protocol.md`, with **new seeds 8101, 8209, 8303, 8407** instead of previously observed 7703/7807/7901/8009. The benchmark CLI and summary program remain identical except for those fixed seeds. Two operating systems Windows/Linux, canonical Digits/Breast Cancer graphs, two labels/class, 3×20k fixed monitoring +1×100k default potential +1×5k no-stop per seed, three separately compiled/process-isolated binaries/variants. **240 timed fits; 80 three-way exact state+stop parity comparisons**, 4 raw artifact matrix jobs, one mandatory summary. Do not change runtime parameters after seeing results. Retain all outliers. Any array/stop difference is an immediate rejection.

Performance gates remain unchanged from v1:
1. Cached-on Digits median native 20k propagation saving vs untouched master >=**15%** on **both** Windows/Linux.
2. Cached-on Breast Cancer median regression not below **-5%** on either OS.
3. Default-disabled candidate median regression not below **-8%** on **any** OS×dataset.

Both additional hypotheses are exploratory: (a) v2 removes the Linux default-disabled regression; (b) opt-in acceleration remains worthwhile. A pass would support a focused code review only. The v1 results must be preserved as a failed experiment, not silently overwritten or generalized to V2. **No merge to master without user approval and production quality review.**
