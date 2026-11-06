# Results: production potential-monitoring cost with identical trajectories

**Preregistered protocol:** [production_potential_monitoring_overhead_protocol.md](production_potential_monitoring_overhead_protocol.md). Original production backends only, without code modifications.

## Provenance and correctness

- Full workflow [42253321060](https://github.com/fbreve/pypcc/actions/runs/42253321060): **4/4 dataset jobs + dependent validation/summary job all passed**.
- Environment and raw CSV artifacts: Iris **13520019680**, Wine **13521038587**, Breast Cancer **13519854793**, Digits **13519939904**; generated complete-cohort report **13520319640**.
- Exactly **96** timed fits, **48** paired full-state comparisons. Every pair required the two runs to complete **20,000 iterations**, from the same canonical undirected union-symmetrized 10-NN graph, same labeled nodes/seed and the same actual production NumPy, Numba or Cython backend.
- Potential monitoring stayed on its existing **every-10-steps** schedule when enabled, but `es_chk=2000000` deliberately made its stop threshold larger than the cap's maximum number of metric checks. This **isolates monitoring cost**, not a new stopping default.
- All **48/48 bitwise equality gates passed** for six complete state arrays: predictions, particle positions, particle strength, particle distance tables, node dominance, and normalized node ownership. This strongly validates that the existing monitor does not modify the *tested* dynamics or RNG trajectories.
- Two monitoring orders were balanced across four new seeds **6101,6203,6301,6407** per dataset; Numba specialized warmups for both conditions and dataset layout occurred outside timing.

## Measured paired overhead (on/off ratio minus one)

| Dataset | NumPy | Numba | Cython |
|---|---:|---:|---:|
| Iris | +0.87% | +4.10% | +7.21% |
| Wine | +1.61% | +6.65% | +3.83% |
| Breast Cancer | +3.57% | +17.42% | +25.64% |
| Digits | +0.55% | **+19.72%** | **+50.24%** |
| **Pooled mean over 16 pairs per backend** | **+1.65%** | **+11.97%** | **+21.73%** |

For **Digits Cython**, +50.24% means `T_on ≈ 1.5024 × T_off`, so monitoring accounts for roughly `(T_on−T_off)/T_on ≈ 33.4%` of measured monitored runtime. This is **not** a 50% improvement that a perfect optimized monitor would provide; even removing the monitoring cost entirely could reduce observed monitored runtime only by ~33.4%, ignoring second-order effects.

For **Digits Numba**, +19.72% corresponds to about **16.5%** of monitored wall time. The monitoring percentage can look small in NumPy because particle processing dominates its slower interpreter-based loop, not because its metric calculation is free.

The pooling averages across different datasets and runner conditions and therefore does **not** represent a universal application speedup. Single-pair noise is visible (e.g. Wine Cython one observed -8.97%); preserve negative measurements rather than cherry-picking. The per-dataset four-seed median and range are stored in the generated summary artifact.

## Decision / follow-up

**Do not change the criterion, its timing, or the monitored values** based on a faster run alone. There is a credible target for a **separate strictly experimental Cython calculator optimization**, particularly for 10-class Digits, where the current `_pcc_calc_mmpot` scans each dominance row before moving to the next. The Cython backend's dominance matrix is generally Fortran-contiguous; a different *memory access order* may be benchmarked while preserving the per-row maximum and original row-index accumulation order exactly. Any candidate must prove bitwise equality of final state, actual stopping iteration, stop counters and predictions across representative graphs/seeds, and show reproducible end-to-end wall-time improvement. Cython only first; do not simultaneously modify Numba or NumPy.

Keep this study as a draft; **`master` unchanged**.
