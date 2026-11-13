# P2-07 — Retrospective metadata-only array and snapshot budget, not execution approval (2026-11-13)

**Source:** The original complete, cross-platform passing P2-04/05 [run #43475652889](https://github.com/fbreve/pypcc/actions/runs/43475652889) and its **existing aggregate metadata only**. This PR is stacked on #267 and the immutable P1 scientific protocol SHA-256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`.

Unlike P2-04/05, this P2-07 check does **not** download the official UCI data, parse records, create graphs or label masks, compile or JIT any backend, import PyPCC, or execute **any** particle steps. Only archived JSON source/mask/graph **hashes and aggregate graph dimensions** are consumed. No original data rows, original class vector, selected label indices, mask vectors, graph edges, predictor matrices, particle state, or predictions are exported.

## Frozen input and provenance

- Linux P2-04/05 metadata archive `14063840028`, Windows archive `14063660306`, and cross-platform hash concordance archive `14063161582`, from run `43475652889` at PR #267 head `1a86cf0b1f9350f6a98cd1287e9b77edd390c87a`.
- Require the original P1 and three UCI IDs 212/850/292 in exact order, source class-support counts, graph SHA-256 frozen in #267, exact `neib_list` int64 dense dimensions/padding, eight masks per source, and all 72 native backend-specific seeds from the original P1 SHA-256 derivation.
- Direct Git checks of unchanged `pcc.py` blob `01606aaee860c3b9bc84710e7362a2ef81046d45`, `pcc_graph.py` blob `3e8943462555b4feafbd162125b51e41f29d7ab7`, and the original isolated milestone observers:
  - NumPy #253, observer blob `d642bd0f329eceb4f3dc3111b5f8c91f0e085cac`.
  - Numba #255, observer blob `e10942e96be623d47d5537d45bd365f366bba548`.
  - Cython #256, observer blob `281daf0e66a567e587d0bd3e21dc34329a234b95`.

No source code is modified or recompiled.

## Memory geometry formula, not a process-RSS claim

The unchanged `pcc.py` creates the main per-fit structures: original dense int64 `neib_list[n,w]` with padding -1 and `neib_qt[n]`, (planned co-resident) original+transformed `float64` feature matrices `X[n,d]`, four int64 node/label vectors, three particle index vectors, one particle-strength vector, `uint8 dist_table[n,p]` (Fortran layout), `float64 dominance[n,c]`, `owndeg[n,c]`, two class-sized and four degree-sized scratch buffers, and a 257-element distance-weight table. We estimate **array payload byte counts only**, distinguishing transient final-label or feature buffers from retained model state. Simultaneously treating all arrays as co-resident is a *planning scenario*, not a verified call-stack lifetime.

For each of the eight approved milestones `(0,1,10,100,1000,5000,25000,50000)`, the original NumPy/Numba observers copy:

`positions[p]int64 + strengths[p]float64 + dist_table[n,p]uint8 + dominance[n,c]float64 + owndeg[n,c]float64`.

The Cython observer additionally copies `rowmax[n]float64` and one `uint32` native RNG state per milestone; Cython schedule/RNG final-state arrays and Numba schedule are accounted for separately. Here `p=3c` (three original labels per class), and `w` is the **measured graph width** from the frozen P2-04 report, not an invented kNN degree. No snapshot metadata/header or Python allocation overhead is treated as a native-array upper bound.

The three original observers enforce **64 MiB per fit for planned checkpoint arrays**; this new no-fit auditor checks all nine source/backend combinations independently. The theoretical **72 × 64 MiB = 4.5 GiB** is a deliberately loose all-fit scenario **if** all fits were allowed to use their entire individual cap; this is **not** a requirement to persist raw checkpoints on disk, and is not a statement of actual memory residency. The tighter total is separately calculated from real archived node/class/degree geometry and the eight approved milestones. Snapshot payload bytes are not equal to total process RAM, which also includes Python, NumPy/scikit-learn native code, compiled Cython/Numba/JIT, allocator fragmentation and temporary workspaces.

## CPU, time, disk and export policy — explicit blockers, no false GO

The P1 design requests exactly **72 fits × 50,000 iterations = 3,600,000 planned PCC iterations**; with **3 labels/class** and class counts (3, 2, 2), it implies **25,200,000 potential particle-update opportunities** across all 72 fits. That is simple workload arithmetic, **not measured CPU time, instructions, wall-time or cache effects**.

A metadata-only audit cannot determine the actual operating-system process peak RSS in each backend, compiled toolchain working set, true CPU/wall time on each of the three real graphs, overhead from graph construction, or postprocessing/export. It also cannot prove physical execution host capacity, disk/log retention, number of concurrent fits, and privacy/retention handling. Therefore:

- `P2_07_snapshot_payload_within_original_64MiB_per_fit_cap` may be **true** after checking each backend/dataset.
- `P2_07_total_CPU_wall_time_budget_verified=false`.
- `P2_07_native_backend_peak_RSS_budget_verified=false`.
- `P2_07_real_fit_artifact_disk_and_log_budget_verified=false`.
- **`P2_07_full_resources_gate_pass=false`** and no execution authorization.
- **`P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `PCC_real_data_fits_executed=0`.**

No CPU or memory performance numbers will be fabricated by extrapolating the static state footprint or by pretending the GitHub Actions runner is the final sealed experimental machine.

## CI expectations and preservation

Linux and Windows Python 3.12 each verify the same **archived** metadata and original Git source objects without installing packages or importing PCC. Five adversarial tests use synthetic JSON only and reject corrupted graph width, nonexistent masks, altered backend seeds, authorization drift and missing source identity. A separate comparison job requires identical static budgets across platforms and ensures that the full P2 resource gate is **still blocked**.

Only the new resource-accounting JSON (including source Git SHA, array geometry, scalar workload and observed runner free disk/CPU metadata) is archived for 30 days, never source records. Prior P2-04/05 artifacts expire around 2026-12-13; a long-term scientific artifact archive would require a separate preservation decision.

**After this stage:** establish a reproducible final execution host/toolchain and a **separately authorized resource measurement plan** for CPU time, RSS and disk/logs, then finalize P2-07 and P2-08 with an exact technical SHA for explicit human review. No automatic transition into P3. The current PR remains **DRAFT**, stacked, no merge into `master`.
