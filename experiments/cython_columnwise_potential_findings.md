# Result: opt-in Cython columnwise potential calculator — rejected

**Preregistered protocol and gates:** [cython_columnwise_potential_protocol.md](cython_columnwise_potential_protocol.md). This experimental branch is not to be merged into `master`.

## Evidence and CI integrity

- Final corrected matrix: [GitHub Actions 42255286492](https://github.com/fbreve/pypcc/actions/runs/42255286492): **4/4 comparison jobs + aggregate summary PASS**.
- Artifacts with raw CSV and dependency versions: Iris **13523340269**, Wine **13521808636**, Breast Cancer **13523185509**, Digits **13522271817**. Full predeclared report: **13522576393** (`cython-columnwise-potential-summary`).
- Exactly **64 timed fits** on actual compiled Cython, **32 original/opt-in pairs** across four standardized canonical undirected union-10-NN datasets, four frozen new seeds `6503,6607,6701,6803`, and two stopping regimes.
- **32/32 strict bitwise six-array state equality checks PASS** (predictions, particle positions, particle strengths, distance tables, node dominances and normalized ownership). **32/32 exact full propagation-info dictionary parity checks PASS**, including effective iterations, checks, reason for stopping, stop maximum and stall counts. This establishes tested-scope semantic preservation, *not universal proof*.
- CI **first attempt 42255055255 failed because of a bug in the experimental test assertion**: it inspected Fortran contiguity of a snapshot generated with NumPy `.copy()`, which creates a C-ordered copy. The Cython extension had compiled successfully. Corrected in commit `9e033cd653ab4f903080b4eead31885b61c176e5` to inspect the **original** `model.node.dominance` layout. The final corrected jobs pass.
- Both the original function and alternative were compiled into the **same experimental extension**; the new trailing opt-in flag defaults to `False`, so the legacy computation remains selected unless explicitly requested in the experiment.

## Runtime result: worse in every tested stratum

Numbers below represent **measured percent reduction in total fit runtime**, `100*(1 − T_columnwise/T_legacy)`. Negative numbers mean the alternative was **slower**.

| Dataset | Forced identical 20k monitor workload | Actual production potential with 100k cap |
|---|---:|---:|
| Iris | **−6.75%** | **−5.49%** |
| Wine | **−7.60%** | **−9.15%** |
| Breast Cancer | **−22.81%** | **−20.91%** |
| Digits | **−41.49%** | **−42.51%** |

- Digits forced-workload **median improvement −41.56%**; preregistered minimum was **+10%**.
- Worst forced-workload median across datasets **−41.56%**; preregistered allowed regression floor was **−5%**.
- Therefore **the predefined performance criterion fails decisively**.
- The extra scratch vector scans, writes and final pass are plausible contributors, but no instruction-level CPU profiling isolated them, so no single causal explanation is asserted. Four seed pairs/dataset on GitHub runners are exploratory timings; here the regression is large and repeated.

## Decision

**Reject/discard the columnwise Cython calculator candidate.** It preserved behavior in the tested cases, but was slower in every dataset and regime. No merge, no production API change, no default change, and no need to continue this branch's optimization. Keep the earlier [measured original monitoring overhead](production_potential_monitoring_overhead_findings.md) as motivation for a different, independently evaluated approach; do not relax metric sampling or stopping semantics without explicit accuracy verification.

The experimental PR should be **closed without merging** after recording the findings, so the repository's open PR backlog reflects only active work.
