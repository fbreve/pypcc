# P2-07B — Isolated original three-backend synthetic CPU and memory pilot (2026-11-13)

**Research only; no P3 real-data fits authorized.** This supplement is stacked on metadata-only P2-07 #268 and inherits frozen P1 scientific SHA `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. No production code, benchmark hotloop, model default or observer code is changed; no merge.

## Why this step

P2-07A proved memory array *payload geometry* and the 64 MiB snapshot cap for all 72 **proposed** fit units. That does not bound actual process RSS, CPU time, memory allocator/JIT overhead or disk. This P2-07B provides **real measured but strictly SYNTHETIC** CPU and process peak-working-set pilot observations, to inform the upcoming final-host budget. **It does not measure or extrapolate to 50,000 iterations on any approved UCI dataset**.

## Original benchmark source and binary identity

Use original unmodified independent paired full-`fit_predict` benchmarks at pinned research commits, with each benchmark and observer source Git blob independently verified before import:

| Backend | Original full-fit benchmark Git blob | Original observer Git blob |
| --- | --- | --- |
| NumPy #253 | `cc35e3e3107b2f24b419a6d04efe35ebd8b2d4d5` | `d642bd0f329eceb4f3dc3111b5f8c91f0e085cac` |
| Numba #255 | `241a0500ac4ed33d8c348ac29bb74d62cb98bf4b` | `e10942e96be623d47d5537d45bd365f366bba548` |
| Cython #256 | `72a17badcdb78d3353591fa56dfa3ecc835aa819` | `281daf0e66a567e587d0bd3e21dc34329a234b95` |

Cython *original production* and *isolated observer* native binary bytes are **reused** from original archived P2-01C #262 workflow #43434870932 after validating original sealed binary lengths, SHA-256, CPU architecture, exact CPython patch/ABI and the 13 previously pinned dependency versions. The original native .so/.pyd files are **not** recompiled and Cython is not patched. Numba source code is also unchanged; original Numba kernel JIT compilation occurs as needed **in the synthetic child**, never on real data. P2-01B/01C original full toolchain byte identities and original wheel bytes are still not proven: do not treat passing this pilot as final P2-01 lock.

## Exact SYNTHETIC workload and measurement scope

Each platform (Linux and Windows CPython 3.12) starts exactly three separate child interpreter processes, one per backend. All three use their **original** benchmark implementation with:

- 500 deterministic synthetic graph nodes, 10 active neighbors per node and width 16 dense int64 padded neighbors (constructed by the original benchmark's fixed synthetic ring fixture; **not** the three real kNN graphs).
- Two fake classes with three labeled synthetic nodes/class = six particles; seed 71, sequential update, 1,000 iterations, `early_stop=False`.
- The **original synthetic benchmark's** eight-milestone schedule: 0, 1, 10, 50, 100, 250, 500, 1000 — not the P1 50,000-iteration schedule. No P1 masks, real-source labels or UCI archive is read.
- Original warmups excluded from original paired full-fit CPU samples. Two AB/BA pairs; NumPy one fit per pair arm, and Numba/Cython 10 fits per pair arm (necessary for Windows CPU clock resolution and Cython original benchmark's minimum).
- The original benchmark assertions must check observed vs unobserved **within each backend** for final labels, full mutable state and (where supported) native/Python RNG. No cross-backend state or CPU identity is claimed.

Measurements:
- The original benchmark measures `time.process_time_ns` over **complete synthetic `fit_predict`** calls, excluding setup/graph construction. Report both paired CPU samples and medians/ratios **within the backend**.
- The separate child records elapsed process CPU and wall-clock time **including imports, warmup and paired benchmarking** from the beginning of its benchmark code (not including initial interpreter process startup). The parent additionally records entire spawned-process wall time.
- Memory is measured for the **whole process lifetime** after those synthetic calls: Linux `getrusage(RUSAGE_SELF).ru_maxrss` (KiB → bytes), Windows `GetProcessMemoryInfo.PeakWorkingSetSize` (bytes). **Their semantics are different**, and neither is incremental cost of a single PCC fit: they include libraries, retained benchmark data, Numba JIT, Python runtime, original native Cython imports and allocator effects. No equality/efficiency ranking between Linux and Windows is implied.
- A 180-second *per synthetic child* safety timeout ensures that test cases cannot silently run indefinitely. It is **not** an approved cap for real P3 cases or a statement of feasibility for 72 cases.

The CI compares only **same original source** and completeness/parity scope between OS. It explicitly refuses cross-platform equality requirements for CPU or resident memory. Each job separately saves safe numeric metadata, the original paired CPU report and full-state **SHA** fingerprints (no underlying mutable arrays). It never exports graph edges, label masks, raw source records, images or copied checkpoint values.

## Scientific/approval boundaries

This P2-07B **does execute synthetic PCC `fit_predict` calls** on a constructed artificial graph. It is therefore incorrect to call it zero-PCC. What remains strictly **zero** is all **real-source** PCC propagation and all P3 scientific fits. It is not permission to run, time or warm any real UCI fit.

Passing P2-07B can set `all_synthetic_observer_pair_parity_pass=true` but must preserve:
- `P2_07_real_fit_CPU_wall_budget_verified=false`;
- `P2_07_real_fit_peak_RSS_budget_verified=false`;
- `P2_07_full_resources_gate_pass=false`;
- `P2_01_final_execution_toolchain_lock_pass=false`;
- `P2_execution_preflight_sha256=null`;
- `P2_execution_authorized=false`; `real_PCC_fits_executed=0`.

**Next true blockers:** choose intended final execution host and full native/wheel/runtime/ABI lock; set a maintainer-reviewed 72-case CPU/time/RSS/disk/log budget with an acceptable non-real-data validation strategy; finalize a distinct immutable P2 technical manifest with a **separate explicit maintainer approval of its exact SHA**. No automatic P3 transition or publication/production claim, even after all synthetic CI is green.
