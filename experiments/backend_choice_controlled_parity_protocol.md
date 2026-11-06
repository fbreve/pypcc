# Predeclared backend parity diagnosis with controlled destination choice

**Frozen protocol before executing outcomes.** Experimental audit of actual production backend propagation functions, not a proposed PCC algorithm change. No code changes to production, no merges into `master`.

## Question

Do NumPy, Numba, and Cython differ in particle/state updates even when their randomized **destination choice cannot affect the result**? Separately, do NumPy and Numba agree on canonical undirected kNN graphs when only weighted greedy sampling is enabled? An identical numeric seed alone does **not** control common random numbers across backends.

## Two independent test families

1. **Single-choice diagnostic (positive structural control).** Synthetic, *undirected* graph of 8 disconnected vertex pairs, each vertex connected only to the other in its pair. Degrees are all one. Exactly one endpoint per pair has a label, alternating between classes 0 and 1. Both values of `p_grd` (0.0, random branch; 1.0, greedy branch), seeds **4409, 4513**, and independent no-stop caps **1, 2, 10, 100, 500**. NumPy, Numba and Cython must choose the same actual target at every particle update *regardless of RNG internals*, even though the Bernoulli branch may consume different random values. Compare final predictions, particle current nodes, particle strength, node dominance, ownership weights and particle distance tables across all 3 pairs of backends. Discrete state must match exactly; floating arrays must match within absolute **1e-12**, zero relative tolerance. A failure is evidence of non-RNG update differences **in this synthetic setup**, not yet a production-graph classification defect.

2. **Canonical-graph comparison (observational, not a pass/fail exact-parity gate).** Production `pcc_graph.build_knn_graph` with verified undirected union-symmetrized 10-NN; standardized Iris, Wine, Breast Cancer, Digits; two labels/class; fixed new seeds **4409, 4513**; `early_stop=False`, two regimes `p_grd=1.0` (always attempt weighted greedy sampling) and `p_grd=0.0` (random walk; deliberately different random index-selection APIs). Independent caps **1, 2, 5, 10, 50, 200, 1000**. Compare actual *production* NumPy and Numba final full state at each checkpoint. No assertion of identical outcomes on these real graphs; log exact discrete matches and maximum float state deviation, plus prediction agreements. The `p_grd=1` regime is **not** guaranteed to share a random tape: zero-weight fallback or numerical threshold boundaries can make streams diverge, and Numba may use different floating-point evaluation order. Its early-checkpoint parity is suggestive, not a guarantee.

## Execution and controls

- All fits use `ParticleCompetitionAndCooperation(impl=...)`, `set_graph(graph,degrees,k_nn=k)`, `fit_predict(... seed=seed, max_iter=cap, early_stop=False, return_info=True)`. Verify the actual backend callable's exact name and that the executed iteration count equals `cap`.
- Reinitialize model, particles and RNG on each independent cap, preserving identical graph, labels and numeric seed.
- Compare six state outputs: predictions, particle current nodes, particle strength, distance table, node dominance, and normalized ownership weights. Do not compare working buffers or memory layout.
- In the forced single-choice test, verify actual graph symmetry, absence of loops, and degree one before running. Its disconnected/low-degree construction is **only a kernel diagnostic** and is not a valid stand-in for canonical 10-NN classification performance.
- The canonical dataset graph is built from `experiments.audit_canonical_undirected_lockout.canonical_prepare`, which validates reciprocity, uniqueness and row shape.
- CI matrix one job per dataset, each emits a CSV. Synthetic single-choice positive controls run in the Iris job only, to avoid repeated identical work across all four jobs.
- Keep this audit in an isolated experimental branch/PR based on PR #169's experimental head; no merge into production or inference about stopping-rule superiority.

## Interpretation criteria

If forced-choice full-state parity passes, the *tested deterministic* transitions are consistent; remaining prediction differences in ordinary PCC can still arise from RNG consumption, dynamic route-specific numeric differences, or untested paths. If forced-choice parity fails, identify first offending state field/checkpoint before attempting a kernel change. `p_grd=1` canonical results are diagnostic; any divergence should be investigated at the earliest checked cap, not used to proclaim a correctness regression. Larger statistical validation needs a separate protocol.
