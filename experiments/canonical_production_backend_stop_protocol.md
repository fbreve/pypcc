# Predeclared production-backend baseline check: NumPy, Numba, Cython

**Study frozen before execution**; do not adapt seeds or criteria after results.

## Goal

The two- and five-labeled-per-class stopping experiments used an isolated
candidate NumPy propagator to evaluate label-stability P10 against the
**existing production NumPy potential stopping rule**. Before considering a
new stopping criterion in accelerated implementations, verify that the
existing **production** Numba and Cython backends exhibit comparable
stopping outcomes on the **canonical undirected PCC graph**.

## Cohort, controls and measured outcomes

- Datasets: sklearn Iris, Wine, Breast Cancer, Digits, standardized with
  StandardScaler.
- **New seeds: 4001, 4093**, two per dataset (8 dataset-seed cases).
- The production `pcc_graph.build_knn_graph` supplies an undirected
  union-symmetrized graph with `k_nn=10`, with all active edges valid,
  symmetric, unique and loop-free. Two labeled observations per class
  from the established deterministic `canonical_prepare`.
- Three explicit installed backends: `numpy`, `numba`, `cython`.
  Fail if requested implementation is missing or silently falls back.
  The same graph, labels and seed are used in every backend.
- For each backend: (a) `ParticleCompetitionAndCooperation(impl=...)`
  with `set_graph(graph,degrees,k_nn=10)`, `max_iter=100000`,
  `seed=...`, `early_stop=False`, `return_info=True`;
  and (b) same inputs with `early_stop=True, es_chk=2000,
  stop_criterion="potential"`. This is **the existing PCC rule**,
  not the experimental label rule.
- Require 100k full-cap execution without stopping; positive executed
  iterations <=100k for all stopping runs; finite and valid output
  labels. Record per-backend, per-seed accuracy on initially unlabeled
  nodes; early-stop prevalence, iterations and measured wall time
  relative to its own backend's no-stop reference; prediction agreement
  between stopped and no-stop output and across different backends'
  no-stop outputs.
- Warm up the Numba JIT **on a separate small Iris invocation** outside
  the timed sections so compilation time does not dominate. Neither
  Cython nor NumPy requires JIT warm-up. Wall-time differences across
  backends remain exploratory single-run measurements on CI runners,
  not a robust throughput benchmark.
- For each seed, report whether no-stop NumPy, Numba and Cython final
  predictions match **exactly** and their pairwise agreement otherwise.
  **Cross-backend exact parity is measured, not assumed**; even with
  identical seeds, distinct RNG internals/update order may legitimately
  create different trajectories. Compare stopped outputs against each
  backend's **own** no-stop reference, not only a NumPy reference.

## Interpretation

The fixed cap of 100k is *not* a mathematical convergence guarantee.
Only two seeds per dataset are included in this smoke-check cohort,
so neither backend robustness nor accuracy ranking is established.
If no-stop outputs disagree materially, investigate backend parity
before interpreting an apparent stop-criterion improvement. If they
agree, production backend-specific default stopping can be treated
as a more solid baseline for any subsequent optional stopping change.
There is **no** candidate label-stability implementation in Cython
or Numba in this study. No `master` change is authorized.
