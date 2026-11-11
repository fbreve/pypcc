# Pre-measurement addendum: fix original Windows process-CPU clock resolution

**Scientific protocol amendment committed BEFORE the next original/candidate performance experiment.** The underlying frozen protocol at `experiments/cython_canonical_rowmax_init_protocol.md` and all source/graph/state/RNG/stop/CPU acceptance thresholds remain completely unchanged. This document changes **ONLY the deterministic number of original complete `fit_predict` calls per timed batch**, in the direction already explicitly permitted by the original protocol for Windows clock quantization.

## Observed failure and why the first runs cannot decide GO/NO-GO

- [Initial run 43022098012](https://github.com/fbreve/pypcc/actions/runs/43022098012) had a Windows canonical-source checkout-line-ending guard error *before any Windows timing*. Same original production source Git object; corrected without changing candidate.
- [Corrected run 43022568921](https://github.com/fbreve/pypcc/actions/runs/43022568921): Linux source, state and complete 80-subprocess original/candidate study finished successfully. Windows built real original/candidate sources and passed canonical/native correctness checks, but eventually failed in an **original Wine** 8-fit batch: the Windows `time.process_time_ns()` reports CPU in 15,625,000-ns quanta and one batch read the same clock value before and after its entire genuine fit loop. The original harness correctly refused to calculate ratios with zero CPU. Its window clock measurements demonstrated poor resolution; this is NOT evidence of any source candidate speedup/slowdown, and there is no complete 2-OS experiment.
- Both original and candidate whole-fit measurements from those two preliminary runs are **excluded as entire incomplete experiment attempts**. They are preserved in Actions logs/artifacts, not discarded selectively; no samples are transplanted into the new two-host comparison.

## Frozen uniform per-cohort CPU batch sizes (BOTH hosts, BOTH arms, BOTH label seeds)

| Cohort family | Original 8 fits/batch | Newly fixed real fits/batch |
|---|---:|---:|
| Reciprocal undirected ring 100,000 nodes / 50 classes, monitored 0/5 steps | 8 | **16** |
| Reciprocal undirected ring 30,000 nodes / 50 classes, monitored 0/5 steps | 8 | **64** |
| Real Digits / 10 classes, monitored and fixed 2001 steps | 8 | **128** |
| Real Wine / 3 classes, monitored and fixed 2001 steps | 8 | **256** |
| Real Breast Cancer / 2 classes, monitored and fixed 2001 steps | 8 | **256** |

These values are selected prospectively based on coarse Windows ~15.625-ms process-CPU resolution and the already specified relative workload sizes; they do **not** depend on observed original-vs-candidate speedups. Six independent original/candidate block comparisons per cohort, exactly 3 blocks in each of the `ORIGINAL-CANDIDATE-CANDIDATE-ORIGINAL` counterbalanced subruns, are unchanged. Every variant/seed/stop configuration gets the exact same count within its family on both OS. Use `time.perf_counter_ns` separately for wall. If ANY batch still has zero process CPU, **fail closed** and mark experiment unresolved; do not choose another batch size post hoc to force a desired numerical result.

**Required remeasurement:** full original frozen source native compilation and ALL 20 cohorts ×4 subruns = 80 separate original/candidate subprocesses on **EACH** Linux/Windows host, new immutable result artifacts, new strict independent frozen source/state/stop/CPU/memory gate. Old Linux-only results from 43022568921 CANNOT substitute. All ten native terminal array SHA256, input/graph exactness, low-level arbitrary dominance no-opt-in, canonical check, initial rowmax IEEE value, original Cython RNG, original stopping checks, every pair and previous-fit rollback remain mandatory.

**No change whatsoever** in original source baseline SHA `4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2`, experimental candidate source generation, sampled case/labels, original correct undirected dense padded int64 neighbor matrix, 5% BOTH 100k×50 primary CPU savings threshold, -5% secondary CPU and whole wall limits, original memory budget, raw-data completeness or NO-MERGE policy. This is a measurement resolution correction *before new complete data*, not an optimization retune or a new hypothesis.
