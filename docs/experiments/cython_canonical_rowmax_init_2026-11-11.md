# PyPCC — approved canonical Cython row-maximum cache initialization (2026-11-11)

**Accepted, source-exact performance change; current production baseline:** [`e1b54d7936f08db76473ac622d0598f0e950942e`](https://github.com/fbreve/pypcc/commit/e1b54d7936f08db76473ac622d0598f0e950942e), squash merge of the **minimal production [PR #242](https://github.com/fbreve/pypcc/pull/242)**. Separate scientific [PR #241](https://github.com/fbreve/pypcc/pull/241) is **closed WITHOUT merge**. No other experimental code was shipped.

## Technical change and semantic boundary

The previously merged Cython per-node potential cache (#178/#179) maintained each node's current maximum during particle updates but still initialized the cache by reading the complete freshly created `dominance[n_nodes, classes]` matrix. For the *exact* high-level `ParticleCompetitionAndCooperation.fit_predict` that initialized this state, the initial maximum is already known: **1.0 for a labeled home, and the original IEEE-754 `1.0/c` value for any unlabeled node**. The accepted change initializes these maxima from the already available mapped labels in **O(n_nodes)**, rather than reading all **O(n_nodes × classes)** entries. The original node-index sum order at every stopping checkpoint is unchanged.

Production changes only `pcc.py` and `pcc_step.pyx`, plus permanent `tests/test_cython_canonical_rowmax_init.py`. A trailing `experimental_canonical_rowmax_init=False` native Cython low-level argument defaults to the original, general-purpose full-matrix scan. The high-level implementation enables the fast path only for the **exact PCC base class**, the **actual Cython dispatcher**, `update_mode='sequential'` and `early_stop=True`. Low-level callers with arbitrary initial dominance, user subclasses, non-Cython backends and no-stop calls retain the original path. The default cached/uncached native particle loops, moves, dominance/strength arithmetic, RNG draws, stop checks, source graph, data layouts and intentionally dense padded int64 neighbor matrix are unchanged.

## Prospectively frozen source-original native CPU evidence

- [Original frozen scientific protocol](https://github.com/fbreve/pypcc/blob/experiment/cython-canonical-rowmax-init-v1/experiments/cython_canonical_rowmax_init_protocol.md), committed **before** any original/candidate measurements.
- [CPU clock resolution addendum](https://github.com/fbreve/pypcc/blob/experiment/cython-canonical-rowmax-init-v1/experiments/cython_canonical_rowmax_batch_resolution_addendum.md) committed before the final fresh two-host trial: on Windows process CPU is quantized (~15.625 ms), so *both OS and both arms* use predetermined full-fit batch sizes 16/64/128/256 depending on workload. The earlier incomplete 8-fit campaigns were **not pooled** into final results; acceptance thresholds were unchanged.
- [One true original-vs-candidate native build/full-fit run](https://github.com/fbreve/pypcc/actions/runs/43023683759): both Linux and Windows source-native runner jobs SUCCESS, six original/candidate counterbalanced paired CPU blocks per cohort, original true-Cython `fit_predict` (not profiler-simulated cache misses), two independent label seeds per cohort, native package version/source guards, 80 experimental subprocesses per OS. Its *automated auditor* failed due solely to GitHub artifact ZIP path, not to an unpassed semantic comparison. The raw collected data were retained unchanged.
- [Independent immutable-archive-only auditor run 43024935347](https://github.com/fbreve/pypcc/actions/runs/43024935347) **SUCCESS**, **`CPU_GO`**, **zero prospective CPU/wall/memory/correctness failures**. That auditor re-read precisely the *same* existing original source/CPU archives rather than rerunning or selecting measurements; its read-only ZIP layout parser was corrected after the original run. [Decision and full tables artifact 13857290377](https://github.com/fbreve/pypcc/actions/runs/43024935347/artifacts/13857290377); original [Linux 13855499960](https://github.com/fbreve/pypcc/actions/runs/43023683759/artifacts/13855499960), [Windows 13856048639](https://github.com/fbreve/pypcc/actions/runs/43023683759/artifacts/13856048639).

### Frozen primary: complete-fit **process CPU reduction**

**Exact reciprocal undirected 16-neighbor ring; 100,000 nodes, 50 classes; 3 labeled nodes/class; monitoring on; 2 fixed label masks.** Positive percentages are **original-versus-candidate median saving** in genuine total `fit_predict` CPU, not in a phase microbenchmark.

| Host | Iterations | Label seed 89021 | Label seed 89119 |
|---|---:|---:|---:|
| Linux | 0 | **10.28%** | **12.84%** |
| Linux | 5 | **10.48%** | **11.79%** |
| Windows | 0 | **9.35%** | **9.98%** |
| Windows | 5 | **11.33%** | **10.96%** |

All **8/8** preregistered primary host × iteration × seed cohorts exceeded the frozen >=5% complete-fit CPU threshold. Secondary reciprocal 30,000-node × 50-class cases improved 6.12–7.50% Linux and 3.31–9.27% Windows. **Real Digits, Wine and Breast Cancer**, monitored and fixed 2,001-iteration scenarios, had mostly small mixed changes; all satisfied the preregistered full-fit CPU and wall no-regression bounds. Python `tracemalloc` peak tests passed their fixed ceiling, but this is **not** a measurement of all OS RSS or native allocations. Neither an overall average over all workloads nor a speedup on all graphs is implied.

## Exactness and independent production integration

Scientific comparisons required matching original graphs and labels, **ten complete terminal array hashes** (prediction, node and particle fields, dominance, distance, owner degree and mapped labels), seeded Cython replay, original native stop/check metadata, per-checkpoint potential cache verification, prior-fit array ownership, and invalid-input rollback. The arbitrary input low-level `pcc_propagate` maintained the full original cache path. The candidate's exact original two production source files were reconstructed from pinned Git blobs and independently verified to be **byte-for-byte identical** to the scientific candidate source.

The resulting [production PR #242](https://github.com/fbreve/pypcc/pull/242) changed only three files, received **8/8 exact-head GitHub workflow successes**, including [4/4 Linux/Windows × Python 3.10/3.12 wheel/sdist and full unit regression](https://github.com/fbreve/pypcc/actions/runs/43025923618), then was **squash merged** as [`e1b54d7`](https://github.com/fbreve/pypcc/commit/e1b54d7936f08db76473ac622d0598f0e950942e). This is a tested workload-specific production CPU improvement, not evidence for tuning graph topology, moving immobile particles by force or modifying the Numba greedy movement.

**Future benchmark baseline:** use `e1b54d7` (or a later explicitly pinned master commit), not the earlier `4c1c410` Cython build. Raw GitHub Actions files have finite retention; links and the decision table are preserved here, but this text is not a byte-for-byte backup of the raw JSON.
