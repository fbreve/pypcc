# Clean-master seeded replay pilot: results and next gate

## Exact experiment and provenance

- [CI 42316409541](https://github.com/fbreve/pypcc/actions/runs/42316409541): **9/9 backend×dataset runs and 1/1 full-cohort summary PASS**, no exclusions.
- Frozen protocol: [seeded_prefix_replay_protocol.md](seeded_prefix_replay_protocol.md); raw per-stratum CSVs uploaded as GitHub Actions artifacts.
- Aggregate result artifact **13549829987** (`seeded-prefix-replay-complete-report`). Raw dataset/backend artifacts: `iris` NumPy **13551131384**, Numba **13551475176**, Cython **13550234467**; `breast_cancer` NumPy **13551525012**, Numba **13551196416**, Cython **13551101424**; `digits` NumPy **13550289392**, Numba **13550877344**, Cython **13550553190**.
- **108/108 fresh-instance fits executed**, of which **90 milestone fits** produced 90/90 raw summary rows and **18 duplicate full-cap ordinary high-level model fits** served as full-state validation controls. **18/18 strict six-array parity controls PASS**. Those duplicate fits are represented by parity fields in the raw CSV, not separate per-run CSV rows.
- Exactly **3 datasets × 3 real production backends × 2 frozen seeds 11003/11117 × 5 milestones (1, 11, 101, 501, 2001)**, two labeled vertices/class on the same canonical **undirected union-10NN** graph and no early stop. All actual backend identities, graph reciprocity, full-cap metadata, array hashes, accuracy and finite state gates validated.

## Descriptive retrospective accuracy on initially unlabeled vertices

| Dataset | NumPy | Numba | Cython |
|---|---:|---:|---:|
| Iris | 87.15% | 88.19% | 86.11% |
| Breast Cancer | 82.74% | 84.16% | 81.33% |
| Digits | 79.21% | 75.91% | 77.80% |

Each percentage averages only **two seeds** and is **not** a meaningful estimate of expected classification accuracy across varied label selections or a cross-backend seeded path equivalence result.

All **72/72 between-milestone intervals** (3 datasets × 3 backends × 2 seeds × 4 intervals) had at least one **net change in assignment**. At the final interval, 501→2001, the sum of nodes whose assignment differed between these two observations across both seeds was:
- Iris: NumPy **9**, Numba **6**, Cython **11**.
- Breast Cancer: NumPy **304**, Numba **284**, Cython **268**.
- Digits: NumPy **998**, Numba **1049**, Cython **1027**.

These are **net differences at the two observed endpoints**, NOT the total number of label flips inside an interval, which could be much higher. A zero in an interval would not establish actual convergence.

## Scientific and engineering decision

- The pilot demonstrates that seeded *replay of independent fits* can produce internally consistent bounded diagnostics using the **unmodified current master**, rather than resurrecting stale native instrumentation. The experimental wrapper only requests backend `return_info=True`; every terminal outcome matched a separate normal public `fit_predict` invocation bitwise on all six state arrays.
- It **does not** validate an uninterrupted continuous-run trace, restore RNG continuation states, identify the iteration of an intermediate flip, or establish P10/potential stop correctness. It repeats propagation work from zero at each milestone and is inappropriate for dense 100k–500k traces.
- It does not solve historical draft #18: that PR contains richer, efficient in-kernel continuous-run observations plus unexecuted historical scientific protocols, but its current three kernel diffs are incompatible with the integrated Cython cache and seeded propagation and would require a new, separately audited port.
- **No source or master changes** in this PR; leave it experimental. Do not merge #18 or claim its preflight authorizes a new real pilot. If useful, the next strictly staged work would be a separate **NumPy-only, opt-in continuous tracer** based on current master, with side-by-side full-state, stop and RNG parity; only later consider Cython/Numba after independent correctness and overhead validation.
