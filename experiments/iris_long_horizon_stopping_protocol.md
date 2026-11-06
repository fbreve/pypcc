# Preregistered diagnostic: Iris stopping under 500k reference horizons

**Experimental branch only. This protocol is frozen before observing the new CI outcomes.** No changes to production kernels, default settings, graph construction, particle dynamics or `master`.

## Why this study

The prior prospective 100k canonical undirected dataset cohort (draft PR #171, workflow [42143714591](https://github.com/fbreve/pypcc/actions/runs/42143714591)) found a notable hard case, **Iris seed 5003**, with default *production potential* stopping accuracy drops compared with the **100k no-stop** reference: NumPy -9.03 percentage points, Numba -11.11 pp and Cython -6.25 pp. This result conditions case selection; **the hard case must not be treated as fresh held-out evidence**. A 100k reference is not proof of convergence, so re-evaluate it at longer fixed horizons before recommending any criterion replacement.

## Frozen cohort, settings, and outcomes

- One diagnostic seed **5003** (previously identified as difficult) and two *new* seeds **5209, 5303**, all on Iris; these two provide limited independent sensitivity checks, **not** statistical generalization.
- Same standardized sklearn Iris data, production `pcc_graph.build_knn_graph` with union-symmetrized **undirected** kNN=10, validated by `canonical_prepare`; two labelled samples per class, chosen deterministically for each seed. Install via `set_graph(..., k_nn=10)` to preserve potential-stopping calibration.
- Actual production backends `numpy`, `numba`, `cython`, with no fallback. Common model settings: `p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0`, fixed seed numeric value, sequential movement, `return_info=True`.
- Each backend's **independent no-stop references** at **100k and 500k**; additionally NumPy at **250k**. Each reference is freshly initialized and run with exactly the stated cap.
- Each backend's original production potential stopping with `early_stop=True, es_chk=2000, stop_criterion='potential'`, cap **500k**.
- Experimental **NumPy-only label-stability patience 10** (`label_change`, check every 1000, tolerance 0) at cap **500k**, using existing `experiments.benchmark_inloop_global_stopping.run`; do not adapt the patience or interval after observing results.
- **Mandatory candidate/production no-stop 100k prediction-parity gate** on every seed; abort if it fails. This is a final-output parity control at 100k only, not full state or 500k proof.
- For each policy/cap: report effective iterations, accuracy on originally unlabeled nodes, agreement and changes vs the **same backend's 500k no-stop** and its 100k reference, saved steps, wall time vs own 500k no-stop, and the non-stopping 100k→500k prediction/accuracy drift. For NumPy also 250k→500k drift. Compare candidate P10 to production NumPy potential, not to a different backend's unaligned RNG trajectory.
- A backend-specific 500k cap is a **fixed comparison horizon**, never a mathematical convergence claim. Report signed accuracy changes separately from absolute per-case differences; more iterations need not improve accuracy.
- CI matrix: three separate jobs, one per frozen seed; independent raw CSV artifacts retained; Numba prewarmed on a separate 25-step Iris fit with seed 29 *before timing*. All comparisons based on true executed iterations. One observed runtime per policy is descriptive only.

## Interpretation gates

1. Report whether the 100k no-stop **accuracy and predictions change by 500k**, especially for seed 5003, before interpreting a large stop loss as simply premature convergence.
2. Decide whether P10 has **better label agreement and accuracy** than production potential for the same seed on the same reference; account for extra iterations, and separate seed 5003 from previously unseen cases.
3. If P10 does not materially improve both quality and cost across the fresh cases, do **not** recommend a default replacement; if it improves quality but costs more, report the tradeoff rather than declaring superiority.
4. Preserve both policies as-is; no graph/particle rescue or RNG changes. Follow-up on more datasets/seeds only under a separately frozen protocol.
