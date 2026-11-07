# Current production master: native fixed-iteration performance baseline (completed)

**Status: FROZEN DATA WORKLOAD COMPLETE; all 216/216 measurements + 72/72 warm-ups validated.** [Independent draft PR #190](https://github.com/fbreve/pypcc/pull/190), based directly on untouched production `master` **`8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128`**. Frozen [protocol](current_master_performance_protocol.md) was committed *before any benchmark measurements*. This PR changes no PCC production source or algorithm. The document audit [#189](https://github.com/fbreve/pypcc/pull/189) and experiment stack #182–#188 remain separate/unmerged.

## Verified final run and environment

- **Final CI [42413317973](https://github.com/fbreve/pypcc/actions/runs/42413317973)**: **both real Linux/Windows Python 3.12 jobs + mandatory strict 216/216 aggregate SUCCESS** at code head `f0ed9a3ddee924fbcf5b3aadfc561b8b6be41145`. [Final complete machine-checked report, artifact 13590607109](https://github.com/fbreve/pypcc/actions/runs/42413317973/artifacts/13590607109). Full raw six-array state SHA digests, every recorded timing, native backend identities and ordering, exact fixture arrays/hashes, environment, source fingerprints, compiled Cython extension SHA256, all warm-up records and external memory samples: [Linux artifact 13590272689](https://github.com/fbreve/pypcc/actions/runs/42413317973/artifacts/13590272689), [Windows artifact 13590776516](https://github.com/fbreve/pypcc/actions/runs/42413317973/artifacts/13590776516).
- **Standard production regression [42413317972](https://github.com/fbreve/pypcc/actions/runs/42413317972)**: four of four Linux/Windows × Python 3.10/3.12 jobs **SUCCESS** at the same verified experimental baseline head.
- Linux hosted runner: **AMD EPYC 9V45 96-Core Processor**, Python 3.12.16, compiled Cython extension SHA256 `c8b42bcaf505d46d1820dd5e0dc11c5e659836ea4b3683a41ed3b1ca352110d5`. Whole subprocess sampled max RSS **305.1 MiB** and wall **53.36 s**, including source load/build warmups, graph/data preparation, timing and artifact writing.
- Windows hosted runner: **AMD EPYC 7763 64-Core Processor**, Python 3.12.10, compiled Cython extension SHA256 `1a0dd2d5c37fbd2945bee84da002f75315eb50d12f3f5ea5fe7ac0bc0d646567`. Whole subprocess sampled max RSS **199.3 MiB**, wall **121.95 s**, same whole-process scope. RSS sampled externally at 50 ms intervals, **not** precise native per-fit peak nor memory optimization.
- All `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS` set to 1. Actual installed native package versions and benchmark source hashes recorded in environment manifests and `pip freeze` raw artifacts; compiler-specific flags are **not established** merely from a Python compiler banner. Cython/setuptools can be absent from the active runtime environment because isolated source build already produced the compiled binary.
- **Source identity:** both jobs used the same Git commit and each separately verified `git diff` empty for every frozen production PCC kernel, graph/validation source, `benchmarks/` harness and `setup.py` versus pinned master. Windows checkout CRLF versus Linux LF can change *raw checkout file* SHA-256 while preserving exact **Git blob/commit** identity. The source hash inventory is stored per OS rather than falsely requiring matching raw-byte fingerprints. All six case **fixture array SHA-256** identities were checked cross-OS and did match.

## The actual validated numerical workload

Official original production reciprocal **union k=10NN**, standard `StandardScaler` preprocessing, full sklearn **Wine (178 nodes/3 classes), Breast Cancer (569/2), Digits (1797/10)**. Exactly **3 labeled home vertices per class**, 2 preregistered new masks per dataset (`25013,25117`), 3 original sequential backends (`numpy,numba,cython`), native original propagation seed 0 (Cython retains its separate fixed xorshift algorithm), original `p_grd=.5,delta_v=.1,deltap=1,dexp=2`. **10,000 fixed no-stop iterations** and **6 fully counterbalanced backend-order permutations** per fixture, two initial 2-step warm-ups/backend/fixture. Thus **108 measured full `fit_predict` executions per OS = 216 total, with 36 warm-ups/OS = 72**, each without kernel probes or observer. One loading/scaling/graph construction per dataset/OS, reused for both label masks, separate from full-fit timing.

**Validation passed:** all 216 rows present, 72 expected warm-ups, original six-array native final-state SHA256 hashes and held-out accuracy exactly repeated across six executions for each fixed fixture/backend, original 10k-iteration cap and stopping metadata in every sample, all six permutations per fixture without duplication, original binary and backend identity (no silent fallbacks), preparation graph count exactly 3 per OS, exact six input graph/label fixture fingerprints across OS. NumPy/Numba/Cython RNGs/trajectories intentionally differ and should not be assumed equivalent across backends. No outliers removed.

## Actual propagation medians for 10,000 fixed iterations

All entries **milliseconds**; host medians over 2 label masks × 6 repeated timings (12 measured timings per cell). Never pool Linux/Windows.

| Host | Dataset | NumPy ms | Numba ms | Cython ms | Paired median Numba / Cython |
|---|---|---:|---:|---:|---:|
| Linux | Wine | 848.035 | 5.853 | **2.664** | **2.173×** |
| Linux | Breast Cancer | 580.765 | 4.415 | **1.902** | **2.314×** |
| Linux | Digits | 2780.220 | 18.539 | **10.923** | **1.701×** |
| Windows | Wine | 1911.988 | 7.883 | **6.818** | **1.153×** |
| Windows | Breast Cancer | 1329.017 | 5.658 | **4.877** | **1.159×** |
| Windows | Digits | 6449.692 | **28.725** | 30.564 | **0.942×** |

The paired ratio is median of **same-fixture, same-repetition** Numba/Cython propagation time ratios, not the ratio of pooled medians. A ratio >1 means Cython was faster. Numba was faster on **Windows Digits at this final CI runner**; thus there is no universal Cython lead or reason to change the current auto-backend preference. This fixed-iteration test does **not** measure performance of Cython's already-merged potential stopping cache, which is relevant when monitoring is active. Accuracy per backend and all timing quartiles are in the complete report.

## Preparation cost vs full native fit

These are **different measured timing boundaries**: original data load/standardization/initial mask, original graph build **once**, then complete `fit_predict` medians that **exclude graph build**; full fit includes init/dispatch, native propagation and postprocessing. Do not add preparation to each fit unless the application really builds a new graph for every call.

| Host | Dataset | One-time data/prep (ms) | One-time graph build (ms) | Cython complete-fit median (ms) |
|---|---|---:|---:|---:|
| Linux | Wine | 3.094 | 1.939 | 2.782 |
| Linux | Breast Cancer | 4.500 | **12.644** | **2.031** |
| Linux | Digits | 8.867 | **14.522** | **11.187** |
| Windows | Wine | 7.591 | 3.454 | 7.056 |
| Windows | Breast Cancer | 8.514 | **40.424** | **5.136** |
| Windows | Digits | 17.614 | 28.991 | 31.086 |

On the actual sampled Breast Cancer hosts, a single kNN graph build cost **6.2× (Linux) or 7.9× (Windows)** as much as a warmed Cython full fit on the same dataset. Thus if a real user **builds a new graph for every fit**, graph preparation may dominate end-to-end runtime even after optimized native kernels. If a graph is **reused many times**, this one-time cost is amortized and hot propagation remains the relevant target. Avoid claiming that the graph *always* dominates.

## Important runner/order sensitivity: earlier attempts are retained, not pooled

This baseline required multiple **engineering-only CI repair commits**, exceeding the initial protocol's optimistic *at-most-one correction cycle*: this is an explicitly reported **procedural deviation**, not concealed as perfect preregistration. The frozen **datasets, masks, graph, 10k cap, algorithms, six permutation orders, all time/memory metrics, and OS matrix were never retuned or filtered**. The original failing CI run artifacts/logs remain visible:
- First original [42412432753](https://github.com/fbreve/pypcc/actions/runs/42412432753): both native jobs finished, but the aggregator downloaded Linux/Windows artifacts into the *same merged output directory* and lost host-specific paths; [original regression 42412432725](https://github.com/fbreve/pypcc/actions/runs/42412432725) failed because a test imported optional monitoring-only `psutil` in a minimal runtime that never needed that monitor.
- Later [42412770664](https://github.com/fbreve/pypcc/actions/runs/42412770664): both native jobs finished, but the aggregate incorrectly treated the **40-character Git commit SHA** as a 64-character file SHA256. Fix did not change a numerical source.
- Later attempts identified that isolated compiled Cython does not require `Cython` in the runtime env and that platform newline normalization changes raw checkout source SHA256. The final aggregator accepts missing build-only runtime packages and verifies same Git commit plus per-job unchanged-master source guard and identical actual fixture SHA256 across OS. The **final run fully passes**.
- **Runner hardware changed across attempts**: a prior Windows runner used Intel Xeon Platinum 8370C while the final Windows runner used AMD EPYC 7763; Linux EPYC model also varied. On a prior run, Windows Digits Cython's median was fractionally faster than Numba's, whereas in the *final* validated run Numba was faster. That reversal is not due to any kernel change and is a concrete reason **never to pool/recommend tiny cross-host timing differences as reproducible speed gains**.

## Engineering decision and next independent experiment

1. **No production code changes** and **no merge from this draft solely because baseline gates passed**. The current `master` is competitive on compiled PCC propagation; pure NumPy remains far slower (hundreds to thousands of ms per 10k vs native single-digit/tens of ms), but remains important for reference/diagnostic semantics. All figures are host-specific and 10k no-stop only.
2. **Best next target is graph-build profiling and exact-semantic vectorization**, especially the construction of dense ordered neighbor lists from the symmetrized sparse CSR graph in `pcc_graph.build_knn_graph`. Current source creates thousands of small per-row `np.arange` arrays in a Python list comprehension before concatenation; this is a plausible source of avoidable overhead, **not a measured attribution**. First partition total graph time into (a) nearest-neighbor search, (b) sparse CSR symmetrization and (c) dense neighbor-array materialization, with original arrays and neighbor order unchanged. Only then compare a candidate implementation in a **separate, independent master-based experimental PR**.
3. **Exact before/after graph invariants** should include bitwise equality of `neib_list`, `neib_qt`, padding and original CSR neighbor order, representative naturally tied distances and duplicated rows, original kNN plus edge-index graph, and then all six full PCC state arrays/seeded RNG and stop metadata for each backend. Compare same OS/controller with preregistered order, warmup and representative data sizes. Do not change adjacency sorting, edge reciprocity or self-loop handling.
4. A separate native `early_stop=True` baseline on current Cython/Numba is appropriate **only if profiling finds potential monitoring important**; the fixed-10k results are not a proxy for default-stop runtime. Existing historic 3,600+3,600+432 benchmark campaigns are already archived and should not be repeated indiscriminately.

**Decision: retain PR #190 DRAFT/UNMERGED; archive the exact final success and failed CI attempts. The audit PR #189 and all scientific #182–#188 remain independent of production.**
