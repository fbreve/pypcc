# Complete study: experimental Cython cache of per-node maxima for global potential

**Pre-registered protocol:** [cython_incremental_rowmax_cache_protocol.md](cython_incremental_rowmax_cache_protocol.md). The candidate remains opt-in on an experimental branch; both new flags default to `False`, and the production `master` is untouched.

## What is new relative to earlier PCC optimizations

The Cython production movement kernel already computes `max_dom` across all classes of a visited node after any dominance update, to decide whether a particle can move. This candidate uses that already-computed **post-update** maximum to incrementally update an optional `row_max_cache[node]`, only for unlabeled nodes whose dominance may change. At each existing 10-step potential check, it calculates `sum(row_max_cache[i] for i in increasing node index) / N` using the **same summation order** as the full legacy `_pcc_calc_mmpot`. It changes repeated potential scan cost from `O(N*C)` to `O(N)` while adding cache writes during particle updates. It is **not** a global running double accumulator, which would change floating summation associativity.

Prior PRs #44/#65/#82–#87 fused the **local** dominance maximum and did not maintain this global stopping-metric cache. PR #176's alternative column-major rescanning was rejected after slowing Digits Cython ~41.49%.

## Provenance and semantic equivalence

- [Workflow **42274430336**](https://github.com/fbreve/pypcc/actions/runs/42274430336): **all four dataset jobs and required aggregated audit PASS**.
- Fixed complete cohort: **4 canonical production undirected union-10-NN datasets × four fresh seeds 6907/7001/7103/7207 × two stopping policies × two Cython implementations** = **64 timed fits**, **32 original/cached paired checks**. Zero exclusions.
- Mandatory **32/32 exact six-array state parity checks PASS** for classifications, particle positions/strengths/distance tables, dominance matrix, normalized ownership; **32/32 complete `propagation_info` dictionary matches PASS** (effective iterations, stop reason, number of checks, stall counter and stop threshold).
- Before timed trials, **16 untimed 2,000-iteration cache verification runs** independently checked every metric against the original full matrix scan, **200 checks per run = 3,200 per-check scalar equalities**, all passing.
- Raw CSV + environment metadata artifacts: Iris **13531368732**, Wine **13532397068**, Breast Cancer **13530928824**, Digits **13532951715**. Prespecified complete-cohort report artifact **13532836913**.
- Benchmark timing is of complete `fit_predict` calls, excludes the slow verification mode and alternates legacy/cache order by seed.

## Speed change (positive means faster; percentage of original whole-fit time saved)

| Dataset | 20k forced same work, mean | Median | Production-potential default (100k cap), mean |
|---|---:|---:|---:|
| Iris | **+2.41%** | +2.53% | +0.81% |
| Wine | **+2.27%** | +2.35% | +2.91% |
| Breast Cancer | **+6.64%** | +6.79% | +6.64% |
| Digits | **+36.49%** | +36.30% | **+36.92%** |

- The **four Digits fixed-work** savings were in the tight range **+36.18% to +37.17%**. The corresponding original default early-stop runs all reached the 100k cap under both variants, identically, and showed **+36.48% to +37.51%** savings.
- The fixed-work Iris results were positive in all four seed cases, but one original-stop Iris trial measured a **−2.49%** cache slowdown (shared CI runner noise matters).
- No false convergence or altered particle RNG trajectory was detected under the exact parity gates; this does not prove equality for arbitrary inputs.

## Preregistered exploratory decision

The Digits **20k median whole-fit improvement of +36.30%** exceeds the required +10%; the worst per-dataset median is **+2.35%**, safely above the allowed −5% regression floor. Therefore the **exploratory performance decision is PASS: worth independent replication**.

## Limitations / next steps

1. **Do not merge yet.** Single observed timing per pair on shared Linux GitHub runners is not an independent cross-platform replication. Actual production Cython builds on Windows (MSVC) could differ materially.
2. Replicate in **Windows and Linux**, with multiple repeated pairs, balanced order and fresh seeds; separately check 2/3/5/10/32-class graphs and irregular/isolated nodes, `delta_v=0` and very small updates, `p_grd=0/0.5/1` and near-stop boundaries for exact state/stop behavior. No relaxing correctness tests to preserve speed.
3. Consider a direct memory and ABI/extension-compatibility review, including no allocations when the opt-in flag is disabled. The existing per-particle conditional may slightly affect legacy timing in a same-binary A/B comparison; compare to an untouched historical baseline before integration.
4. Preserve the existing **10-step** stop monitoring cadence and original strict `>` potential comparison. Never replace the canonical-order sum by an O(1) running floating accumulator without a separate numerical-consistency study.

**Decision:** preserve this candidate in draft PR #177 for further validation, original production behavior unchanged.
