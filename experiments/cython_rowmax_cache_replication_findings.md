# Cython global-potential row-max cache: Windows/Linux replication completed

**Pre-registered phase-2 protocol:** [cython_rowmax_cache_replication_protocol.md](cython_rowmax_cache_replication_protocol.md). This is a separate replication of the positive phase-1 result in draft PR #177. It is still **experimental only**, with both opt-in flags defaulting to `False`; `master` is unchanged.

## Full traceability and hard correctness checks

- [Actions workflow **42276168148**](https://github.com/fbreve/pypcc/actions/runs/42276168148): **2/2 numerical stress jobs, 4/4 independent replication jobs, 1/1 aggregate summary job passed**.
- Phase-2 cohort: canonical undirected production union-10-NN on **Digits** (10 classes) and **Breast Cancer** (2 classes), exactly two labeled nodes/class. **Four fresh seeds: 7307, 7403, 7507, 7603**.
- Windows and Linux compiled their Cython extension from the **same candidate source** using native system compilers, Python 3.12.
- Each OS×dataset×seed: **5 alternating original/cached pairs of 20k fixed work**, plus **1 original/cached pair of genuine 100k-capped default potential stopping**. Total **192 timed fits, 96 exact paired tests**. The aggregate checker validated every full-state array, every stop-info field, full reference caps, condition count and timing row, with no excluded samples.
- Independent direct Cython numerical stress tests: **3,240 cases on Linux + 3,240 on Windows = 6,480**, all passed exact full-state and stopping-metadata equality, and every cached check independently matched the original full global potential calculation. Frozen factors: 2/3/5/10/32 classes, `p_grd` 0/0.5/1, `delta_v` 0/1e-12/0.1, `deltap` 0.4/1, `stop_max` 0/1/2, C/F `dominance` memory layout, seeds 311/733, caps 1/11/65, and a genuinely isolated labeled home node.
- An initial protocol comment miscounted the product as 96 fits/48 pairs and the edge grid as 1,620/OS; these **arithmetic-only corrections were committed before final results**. The configurations/seeds/stop parameters did not change.
- CI raw CSVs + pip versions:
  - Linux Digits **13532543401**; Linux Breast Cancer **13532363917**
  - Windows Digits **13532294101**; Windows Breast Cancer **13533711005**
  - Linux numerical stress **13533566006**, Windows numerical stress **13533098781**
  - Mandatory complete-cohort summary artifact **13533711142**.

## Whole-fit time savings (original vs incremental row-max cache)

Positive means the candidate uses **less total wall-clock time** than the same-binary legacy alternative. Figures are the **median of four per-seed medians** for the 20k repeated workload; default-potential rows include only one run/seed and are descriptive.

| OS | Dataset | Repeated 20k fixed-work savings | Real 100k-cap production-potential savings |
|---|---|---:|---:|
| Linux | **Digits** | **+30.07%** | +30.21% |
| Windows | **Digits** | **+30.03%** | +31.88% |
| Linux | Breast Cancer | +6.21% | +5.66% |
| Windows | Breast Cancer | +4.92% | +4.78% |

The strict final-state and complete stopping-metadata equality gates passed for every one of those trials, including the real stopping policy. The fixed-work Digits improvement was robust to the new seeds in the tested CI environments: per-seed medians on Linux ranged **+29.88% to +30.84%**, and on Windows **+29.19% to +30.91%**. Some individual Windows Breast Cancer timings were noisy (range **−11.18% to +25.14%**), despite a positive median; preserve these outliers and do not claim every execution is faster.

## Frozen phase-2 decision gate

- Digits fixed-work speed-up median **>=+15% on each OS**: **PASS**, +30.07% Linux / +30.03% Windows.
- No Breast Cancer fixed-work median regression worse than **−5% on either OS**: **PASS**, +6.21% Linux / +4.92% Windows.
- All numerical cache/check, six full-state arrays and complete stop metadata **exact**: **PASS** (6,480 synthetic cases and 96 repeated pairs).

**Decision: PASS for further production-readiness evaluation, but NOT approval to merge.**

## Safety and future work

- This is a **cached per-node maximum**, not an `O(1)` floating-point sum. Global checks still sum maxima in original node-index order, avoiding changed addition associativity. `experimental_rowmax_cache` remains **False** by default in the draft.
- Repeated paired timing on a single hosted GitHub machine per OS/dataset is not a hardware-population speedup guarantee. Runtime also includes `fit_predict` setup and postprocessing, the same in both paths.
- The reference original calculator and optimized path are compiled into the **same experimental extension**. For a production change, establish a clean branch based on current `master` containing only the Cython modification, and compare **untouched-master native binaries against candidate binaries** on both Windows and Linux with identical inputs/seeds. Verify default behavior, packaging, all known regression tests, and memory/lifetime concerns before any integration.
- Preserve existing 10-step check times, strict `>` comparisons and full particle dynamics. A true global running sum remains a **different and numerically riskier** experiment; no basis to pursue it now.

No changes to production `master`, the Numba or NumPy implementations, or the public model API in this experiment.
