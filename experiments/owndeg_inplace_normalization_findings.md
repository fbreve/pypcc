# Frozen in-place `owndeg` normalization experiment — NO-GO (CPU tradeoff)

**Decision: KEEP PRODUCTION `master` UNCHANGED.** This independent [draft PR #199](https://github.com/fbreve/pypcc/pull/199) is a scientific/benchmark archive, not a production merge. The [protocol](owndeg_inplace_normalization_protocol.md) was committed before candidate/source changes and CI measurements, based on immutable original master `ababc48986b40f083d5930e922f844aa2ea418b2`. The original dense padded `neib_list` remains untouched; it exists deliberately for CPU-efficient particle movements and is not being restructured.

## Completed evidence and provenance

- [Fully successful scientific two-OS native job and mandatory aggregator **42498300904**](https://github.com/fbreve/pypcc/actions/runs/42498300904) at experimental source-head `e61055d97cc294702023d30451a9c54a9245ca62`.
- [Complete machine-checked scientific report artifact **13624482644**](https://github.com/fbreve/pypcc/actions/runs/42498300904/artifacts/13624482644); all original/candidate raw timings, same-backend full state hashes, NumPy tracked memory peaks, complete source/package inventory: [Linux raw **13624333262**](https://github.com/fbreve/pypcc/actions/runs/42498300904/artifacts/13624333262) and [Windows raw **13624635820**](https://github.com/fbreve/pypcc/actions/runs/42498300904/artifacts/13624635820).
- Precisely **96/96 original-versus-candidate full PCC fit native comparisons** on original real Iris/Wine/Breast Cancer/Digits k10 symmetric union graphs, 2 fresh class-balanced masks (seeds 32003/32027), original three NumPy/Numba/Cython sequential propagation engines, both original fixed/early stopping modes, original 1000-iteration cap, Linux and Windows. Exact match in **all six final array hashes, predictions, backend identity, original actual iteration counts and stop metrics and unlabeled accuracy** within each original/candidate same-backend pair; no cross-backend RNG equality assumed. No graph or kernel/label/default change.
- **480/480 original/candidate paired normalization CPU timing repetitions** and **60/60 separately isolated `tracemalloc` tracked-peak memory pairs**, all original/numeric output/layout exact, including all-zero rows and 0-node matrices. Six predeclared original `float64` Fortran shapes; no time observation dropped, balanced call order. Native total fit times and postprocess intervals also retained for every fit.
- The model source reference was **the untouched pinned Git blob compiled independently**. Only production source candidate edit: `self.owndeg = self.owndeg / row_sums` → `np.divide(self.owndeg,row_sums,out=self.owndeg)` at the end of the unchanged propagation wrapper. All original returned arrays and state contents matched; the internal postprocess buffer *identity* intentionally differs (reuses the original instead of allocating a new normalized array), not the numeric outputs.
- The original [first scientific CI **42498055725**](https://github.com/fbreve/pypcc/actions/runs/42498055725) passed Linux but Windows failed **before any scientific measurements**, because `git show master:pcc.py` was decoded using Windows default non-UTF-8 locale while `Path.read_text(encoding='utf-8')` used UTF-8; non-ASCII Unicode arrows in the original file made the exact-source guard disagree. Fixed the original-read helper to explicitly `encoding='utf-8'` in commit `e61055d97cc294702023d30451a9c54a9245ca62`. No source candidate, label/graph inputs, micro workload, thresholds or measurement code changed. The entire independent Linux/Windows science rerun passed.
- Separate standard PCC regression on the same code-head [run 42498300902](https://github.com/fbreve/pypcc/actions/runs/42498300902) had both Linux Python3.10 and 3.12 jobs green when writing this report; Windows jobs were still executing. Since this is a NO-GO for production, no approval/integration is contingent on later results. A docs-only findings commit may trigger new generic/scientific workflows; cite this known successful scientific code-head run.

## Frozen isolated postprocessing outcomes

Each CPU result is the reduction `100×(1−median(candidate_time/original_time))` over 40 paired runs per OS. Each memory result is the corresponding median paired reduction over 5 distinct `tracemalloc` whole-normalizer peak pairs. Input `owndeg` is copied equally into a fresh buffer on both sides before the original sum/zero handling; tracked peak includes the returned normalized matrix and is **not** whole-process RSS, GPU memory, exact native heap or propagation runtime.

| Input nodes × classes | Linux original→candidate CPU µs | Linux paired CPU saving | Windows original→candidate CPU µs | Windows paired CPU saving | Traced peak saving (both OS) |
|---|---:|---:|---:|---:|---:|
| 0 × 2 | 6.26 → 6.35 | **−1.29%** | 5.10 → 5.00 | +1.92% | +4.36% |
| 178 × 3 | 9.80 → 10.17 | **−13.54%** | 7.90 → 8.15 | −3.16% | +26.53% |
| 569 × 2 | 11.15 → 11.44 | **−9.32%** | 9.30 → 9.60 | −3.81% | +26.99% |
| 1797 × 10 | 36.65 → 37.06 | +15.14% | 25.90 → 26.10 | −6.58% | +39.78% |
| 30,000 × 10 | 698.59 → 586.95 | **+20.69%** | 986.05 → 627.35 | **+36.75%** | **+47.02%** |
| 30,000 × 50 | 6832.15 → 4508.63 | **+20.94%** | 4691.95 → 2932.50 | **+37.42%** | **+49.38%** |

**Do not calculate the exact paired percentage from the ratio of printed separate medians:** numerator/denominator medians were evaluated independently; row percentages are the median **paired ratios**, and can differ substantially from the visible ratio of medians (e.g. Linux 1797×10). A displayed negative saving is a candidate slowdown. On the larger matrices memory peaks nearly halved, while observed CPU improvements on Linux did not meet the frozen large-size threshold and two smaller shapes slowed beyond the allowed limit. Thus the original memory-for-CPU design concern remains relevant even for postprocessing.

## Exact preregistered adoption gate

The protocol requires all complete original native states equal (**PASS**); in BOTH 30k-row shapes, ≥25% median paired **CPU** saving on **each OS** (**Linux 0/2, Windows 2/2**); ≥20% `tracemalloc` tracked peak saving on both large matrices on both OS (**2/2 each**); no >8% CPU slowdown on any of four small shapes (**FAIL: two cases on Linux, 178×3 and 569×2**); and no >10% paired full PCC `fit_predict` median wall-time regression within any dataset×backend×mode across the two fixed seeds (**PASS: zero such groups**).

**Final candidate decision: NO-GO under the unchanged precommitted CPU/memory rules.** The scientific verification succeeded, but the candidate did not. Do not relax thresholds after outcomes, do not make an input-size conditional fallback from this dataset, and do not create a production PR or merge this scientific PR. Preserve the study to inform future optimization priorities.

## Practical interpretation

- The dense neighbor matrix is an intentional faster-CPU representation and was **never modified** here; saved memory in an unrelated postprocessing stage is not evidence that a CSR neighbor-list API would improve CPU performance.
- Saving one float64 `n×c` output copy matters for large n/classes, but the postprocessing accounts for only a subset of `fit_predict` runtime, and all 48 original-fit pairs per OS showed no material overall slowdown under the frozen 10% rule.
- To explore more optimization opportunities, prioritize **measurable end-to-end CPU bottlenecks** in the original compiled backends or safely remove needless graph setup overhead without adding CPU cost; do not rerun the existing 3,600-fit campaigns or revisit the intentional dense representation without a separate algorithmic justification.
