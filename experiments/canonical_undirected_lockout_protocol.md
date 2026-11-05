# Canonical undirected-PCC lockout prevalence and practical impact

**Prespecified before running this cohort, 2026-11-05.** The former `compare_dominance_stopping.prepare` function built **directed** kNN graphs, unlike the published PCC and production PyPCC, which use the **undirected union** of kNN neighborhoods. See [`GRAPH_BASELINE_CORRECTION.md`](GRAPH_BASELINE_CORRECTION.md). All results from those directed studies are **not** estimates for canonical PCC.

## Fixed design

- **Graph construction:** call the unchanged **production** `pcc_graph.build_knn_graph` on StandardScaler-transformed sklearn datasets. `k_nn=10`, default Euclidean metric; graph includes reverse edges and deduplicates. Validate reciprocity, no self-edges, valid indices and variable degree. No independent or experimental graph builder.
- **Data:** Iris, Wine, Breast Cancer, Digits. Each run selects **2 labeled observations/class** with `np.random.default_rng(seed).choice(..., replace=False)`, same as prior experiments.
- **New fixed seeds:** **1709, 1801, 1901, 2003, 2111** (5/dataset, 20 dataset-seed cases total; 90 class-case records). These seeds were not used in the prior directed-graph cohort or selected Digits investigations.
- **Dynamics:** unchanged experimental sequential NumPy PCC rules with **no stopping and no interventions**, `50_000` iterations; **1k nonoverlapping windows**. This tracks the canonical graph without changing the production graph or PCC propagation.
- **Operational lockout (same predeclared threshold as directed study):** at least five successive 1k windows with **zero accepted moves** by **all** particles in a class, all valid visits, maximum class-particle strength **<= 0.01**, and **zero** class-winning outgoing neighboring entries at the checkpoint.
- **Control:** separate same-graph, same-label, same-seed, full-50k run with event/state observation callbacks **absent**; final predictions must match exactly. No stopping or tuning.

## Outcomes

**Primary:** per-dataset frequency of runs with >=1 lockout out of 5 and classes with lockout out of total classes. Secondary: time of first event, subsequent movement recovery, number and duration of zero-movement windows, particle strength, outgoing winning-neighbor count, class dominance mass; number of predictions changed between 25k and 50k, overall accuracy at 25k and 50k on the **same initially unlabeled nodes**, and per-class recall at 25k/50k. Accuracy is **descriptive**, not a causal estimate of harm from lockout: if no event occurs, accuracy may still evolve, and an event might occur with minimal predictive effect.

**Integrity:** full 50k, exact expected valid visits per class/window, positive degree, symmetric graph, finite dominance/strength, per-node dominance sums within 1e-8, particle strength bound excess <=1e-8 (no clipping), final predictions match both 50k checkpoint labels and same-seed uninstrumented control. Report max numerical excursion.

## Decision rule

If **zero** lockouts occur across all 20 cases, the observed proportion is 0/20 (descriptive; does not prove impossibility). Under an independent-binomial illustration, a rough one-sided 95% upper bound on a common case-level lockout rate is `1-0.05**(1/20)` ≈ 13.9%; no claim of cross-dataset homogeneity is warranted. If any occur, report incidence **without tuning** the criterion, then explore a **new, separately declared intervention comparison** (force floor, conditional reset, original control) only on an independent cohort to assess accuracy, premature rescue and compute cost. In either case, do **not** change the production PCC/defaults or merge to `master`.
