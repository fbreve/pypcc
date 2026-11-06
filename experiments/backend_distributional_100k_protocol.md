# Predeclared held-out production-backend distributional check

**Phase B frozen before running:** this is a distinct follow-up to the controlled-choice parity result in [backend_choice_controlled_parity_findings.md](backend_choice_controlled_parity_findings.md). The earlier phase found NumPy/Numba exact full-state agreement under `p_grd=1`, but differences under `p_grd=0`. This experiment assesses *practical final-label and accuracy differences under the ordinary `p_grd=0.5` production default*, without enforcing common RNG trajectories.

## Design and immutable parameters

- Actual production `ParticleCompetitionAndCooperation` with explicit `impl=numpy|numba|cython` and backend callable verification, **default `p_grd=0.5`**, `delta_v=0.1`, `deltap=1`, `dexp=2`. No algorithm, PRNG or default changes.
- Datasets: standardized Iris, Wine, Breast Cancer and Digits; graph through production `pcc_graph.build_knn_graph`, **10-NN undirected union-symmetrized**; `k_nn=10` on `set_graph`; two labeled observations/class drawn once per seed. Graph symmetric/unique/loop-free validation inherited from `canonical_prepare`.
- **New held-out seeds, frozen:** `4603, 4703, 4801, 4903, 5003, 5101`, six per dataset, yielding **24 dataset-seed cases** and **144** backend/policy runs (3 backends × 2 policies).
- Two policies per backend for each input: `early_stop=False` with exactly **100,000** steps; `early_stop=True, stop_criterion='potential', es_chk=2000`, capped at **100,000**. All use `return_info=True` and the *same seed integer* with fresh model fits; their RNG streams are **not** common-random-number aligned across backends.
- Dataset-specific Numba warmup with separate seed 29, 25 steps before timing, matching the correction in [canonical_production_backend_stop_timing_amendment.md](canonical_production_backend_stop_timing_amendment.md).
- Reuse the already validated `experiments.audit_canonical_production_backend_stop.compare` without changing its implementation or production kernels. This writes per-case backend/policy accuracy on unlabeled nodes; stop iterations; stopped-vs-own-no-stop agreement and accuracy change; full-run cross-backend agreement; actual execution time.
- One CI matrix job per dataset, raw CSV artifact per job.

## Questions (descriptive, not a formal equivalence trial)

1. Under default stochastic movement, how much do **100k no-stop** final-label agreement and accuracy differ across NumPy, Numba, and Cython when label splits are held constant but RNG algorithms differ?
2. Does **production potential stopping** reduce steps/time, and how much does it change prediction/accuracy *relative to the same backend's own 100k no-stop run*?
3. Are any striking backend/dataset outliers visible that merit new paired-RNG experiments?

Compute dataset-stratified, per-backend mean/min/max no-stop accuracy, 6-seed mean, cross-backend full-run agreement and per-backend stop-rate/label agreement/accuracy deltas. A 100k cap is **not convergence**; six seeds per dataset and unaligned RNG mean these are *descriptive observations only*, not claims of statistical equivalence, statistical significance, universal backend superiority, or a guaranteed speedup. Wall times are individual CI runner observations, not a calibrated speed benchmark.

## Decision rule

No production kernel or default stopping rule may be changed from this study. If an accuracy or stopping gap appears material, preserve raw data and isolate a distinct follow-up with a stronger statistical design and/or controlled common-random-number replay. **Do not merge this experimental work to `master`.**
