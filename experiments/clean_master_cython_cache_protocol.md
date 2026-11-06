# Frozen clean-master Cython cache evaluation — separately compiled binaries

**This protocol is frozen before collecting the cross-revision run.** Candidate is a clean branch directly from current `master` (base commit at creation: `2902009c80b8edd3fc5f4961769444563fcf5b8a`). The ONLY change to executable production source is `pcc_step.pyx`, copied without modification from the already validated opt-in prototype in draft #177. There is **no modification to `pcc.py`**, Python/Numba backends, public model API, graph construction, the master branch or production defaults.

## Difference from earlier experiments

Draft #177 compared legacy and cached Cython calculations compiled **inside a single experimental extension**, achieving ~30% Digits whole-fit savings on both Windows and Linux, with 6,480 numeric edge tests and 96 independent full-state/stop equivalence pairs. Such a same-binary comparison cannot isolate any impact of adding an opt-in branch on the **default-disabled** code path versus the **untouched master** build.

Here GitHub Actions must **check out `master` and the candidate in distinct directories**, build Cython native extensions **separately** using the same installed Python/Cython/NumPy versions on each OS, and execute each binary in **separate Python subprocesses** from its own checkout directory. Each invocation records the actual loaded binary path and SHA-256; the external aggregator verifies that each `master` subprocess imported only the master extension and that candidate variants loaded only the candidate extension.

## Fixed comparison conditions

- Matrix: Linux and Windows, standardized **Digits (10 classes)** and **Breast Cancer (2 classes)** with canonical **undirected union-kNN=10** production graphs via `pcc_graph.build_knn_graph`. Exact **2 initially labeled nodes/class** selected by independent fixed `np.random.default_rng(seed)` per class, identical for all variants.
- **New seeds 7703, 7807, 7901, 8009**, all included.
- Three independently compiled/process-isolated modes:
  1. **master:** untouched Cython from pinned GitHub base commit, original `pcc_propagate`.
  2. **candidate_off:** independently built candidate binary, original `pcc_propagate` call with the new opt-in arguments **omitted**, proving that default-off behavior is backward compatible.
  3. **candidate_on:** same candidate binary with `experimental_rowmax_cache=True`.
- Fixed workloads:
  - **20k forced monitoring:** `early_stop=True`, `es_chk=2000000`, `max_iter=20000`, `stop_max` explicitly calculated from the same canonical formula and verified too high for early exit; **3 paired blocks per seed**.
  - **Actual standard potential:** `early_stop=True`, `es_chk=2000`, `max_iter=100000`; **1 block/seed**.
  - **No stopping:** `early_stop=False`, `es_chk=2000`, `max_iter=5000`; **1 block/seed** to check default-off and candidate-on argument behavior when monitoring is inactive.
- Other scalar parameters exactly production defaults: `p_grd=0.5, delta_v=0.1, deltap=1, dexp=2`; fixed seed per case, without any shared process-global RNG across runs.
- **2 OS × 2 datasets × 4 seeds × (3+1+1) blocks × 3 variants = 240 measured separate-process runs and 80 three-way exact comparisons**. All pairs (including all three modes) must have exactly identical Python-level propagation metadata, iteration counts, predictions, particle positions, strengths, distances, dominance and raw ownership matrices. Do not weaken exact `np.array_equal` to tolerances. Record actual stop modes, and require full 20k/5k execution in the forced/no-stop modes.
- The worker measures ONLY the native propagation call; initialization, fresh sklearn graph, serialization, cross-process launch, compilation and comparison are excluded from the timer. Within a block, vary the order of the three variants cyclically based on seed and repeat number. Each native output is serialized to NPZ with the metadata next to it. The aggregation compares actual arrays from independently built extensions, not just hashes.
- **Raw traceability:** each OS×dataset CI job archives all per-run JSON/NPZ, paired CSV, installed versions, native extension hashes and both Git commit SHAs. Summary job validates the complete frozen 80-triplet cohort with no exclusions and reports medians by OS and dataset, both candidate modes relative to actual master.
- Use Python 3.12 on each OS, same dependencies for both checkouts within an OS runner. Do not combine the candidate and master site-packages, and do not import both extensions in one process.

## Precommitted decision rules

1. Any mismatch of final state, `propagation_info`, configured stopping, graph or binary path/identity **rejects the candidate**, regardless of performance.
2. Record the **median of four per-seed median whole-propagation time savings** (positive = faster) for each OS/dataset/variant; include all negative outliers.
3. Exploratory speed criteria: **candidate_on vs master** Digits forced workload median saving >= **+15% on both OS**, and Breast Cancer forced median >= **-5% on both OS**. **candidate_off vs master** must not show a forced-workload median slowdown worse than **8%** on either dataset/OS; if it does, investigate default-off dispatch and compiler effects before adoption.
4. Even if all conditions pass, that authorizes **review of a minimal production patch**, NOT an automatic merge. Require review of memoryview lifetime/no allocations when disabled, ABI/backward-compatible default signature, sanitizer/boundary tests and the repository's existing regression CI. CI-host timing and compiled Cython versions may not generalize to other machines.

**The production `master` remains untouched until explicit integration decision.**
