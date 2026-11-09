# Frozen trial: source-exact Numba dominance reduction/max fusion, 2026-11-09

**Research only. Never merge this branch.** Immutable production `master` baseline: `4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2` (including merged #228). This protocol is committed **before** implementation, benchmarks, and numerical outcomes. Production sources are byte-guarded throughout.

## Motivation / independent hypothesis

The current sequential Numba step visits the entire `dominance[next_node, :]` row TWICE on a successful visit to an **unlabeled** node: once to subtract the per-class reduction and then again to find the destination maximum used in the original movement decision. Both loops are O(classes); C-contiguous dominance is already installed for true Numba when classes >=8. We propose computing the maximum of *reduced* per-class values in that very same original-order arithmetic loop and then comparing the **post-addition** owning-class value once. For a labeled destination, keep the original separate row-max scan verbatim. All numerical dominance updates remain in EXACT original order (including `sum_reduc += reduc_val` ordering), without fastmath or reordering. No change to RNG consumption, probability/neighbor choice, distance table, strength, stopping checks, or graph.

Historical #65/#82 experimented with related fusion in **Cython** (not this Numba C-layout source); #213 cached global potential at stopping checks and failed its preregistered gate. This is a **different Numba-only native hot-loop hypothesis**, not a rerun or rescue of those negative outcomes.

## Source and semantic controls

- Read and byte-check exact original `git show 4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2:pcc_numba.py` and other protected production files; compile actual complete original Numba source in an isolated module and a candidate derived from that exact Git source by ONE uniquely guarded source-region replacement inside `_pcc_step_numba_sequential`. Do not substitute hand-simulated toy kernels or edit the real `pcc_numba.py`.
- Candidate keeps original dominance subtraction, addition, and all other statements. Record the maximum from the reduced row inside original `for i in range(c)`, then consider the final updated `dominance[next_node, label]`. For labeled rows preserve the original O(c) max scan. No external state/cache/buffer, no new allocations or output API. Original Numba C-order for c>=8; F otherwise. Must preserve all six native array hashes plus predictions, mapped labels, original next native MT19937 draw, true native stop reason/count/checkpoints/iterations; no per-step RNG shift.
- Original and candidate use the original public `PCC.fit_predict` path, graph installed once per independent arm, and source-exact genuine Numba propagators with fair JIT warmups excluded. The baseline contains #228's Numba-specific argmax, so a temporary, scoped identity mapping must ensure **both** original and candidate take the same actual Numba postpropagation path; check that it is executed. Original source dispatch and postprocessing are not silently changed.
- Verify exact replay for seeded `p_grd=0` and `p_grd=1` at cap 2 and cap 1001; validate invalid-seed rollback and previous-fit array ownership, no memory-layout drift. All hashes are computed outside timers. The monitored and fixed full-fit modes remain distinct and both are tested. Reject incomplete evidence as invalid experiment, not performance NO-GO.

## Prospective corpus and timing

- Real standardized undirected original ordered union-10 kNN Iris, Wine, Breast Cancer and Digits, plus 8,192-node reciprocal undirected cycle with 32 classes and exactly 3 labeled homes per class; NO digraphs, no replacement of deliberately padded dense int64 neighbor layout.
- Two independently sampled genuine balanced masks for each dataset, new `label_seed` **87523** and **87697**, original data seed=0, `p_grd=.5, delta_v=.1, deltap=1, dexp=2, es_chk=2000`. Cap 10,000 native iterations (both fixed and original early stopping).
- Per OS exactly 5 datasets ×2 masks ×2 modes = 20 cohorts. Each cohort uses 5 counterbalanced original/candidate **batches of 8 consecutive actual original-fit calls** (1600 total timed complete fits per OS), plus separately compiled 2-step warmups, 3 independent traced Python/NumPy peak pairs per dataset/mode, two 2/1001-step forced-choice policies per mask, and a monitored 2001-step prefix. Batched `process_time_ns` and wall clocks avoid Windows short-process CPU quantization without changing sampling after observing results. No discarded outliers, no pooled Windows/Linux timing.
- All original/candidate paired state fingerprints across eight independent fit results and prior-fit ownership checked AFTER timers. Also independently compare stop_info and native RNG endpoints after identical seeded fits. Timestamp-only phase observations are NOT evidence of CPU savings.
- Full CI on GitHub-hosted Linux and Windows Python 3.12: identical source, input masks/ordered graph/output hashes, complete count, strict independent both-host GO/NO-GO aggregator. Separate regular Linux/Windows Python3.10/3.12 package regression matrix.

## Precommitted performance gate

Performance GO **only if ALL**:
1. Exact source/semantics/graph/stop/RNG, all 20 cohorts/host and independent cross-host checks PASS.
2. >=**6%** median **complete-fit process CPU saving** in genuine Digits (10 classes) BOTH fixed and monitored modes on EACH OS; >=**8%** complete-fit process CPU saving in synthetic reciprocal 8,192×32 both modes on EACH OS.
3. No other real dataset/mode has >5% complete-fit paired CPU regression; no synthetic wall regression >5%; no individual primary cohort wall regression >5%.
4. Candidate full-fit `tracemalloc` peak <= original+8*n+1MiB for every selected traced condition. This is Python/NumPy memory only, NOT native RSS.

If any gate fails, outcome is **SCIENCE COMPLETE, PERFORMANCE NO-GO** and close experiment **without merge**. Never relax thresholds after seeing results. If GO, extract ONLY a separately reviewed minimal master-based source change plus permanent invariants into a distinct production PR with its own four-platform CI. No merge here.