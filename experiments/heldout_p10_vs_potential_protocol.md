# Preregistered held-out: potential stopping versus label P10 on canonical PCC graphs

**Frozen before running the new cohort. Experimental only; no production modifications.** This protocol follows earlier selected hard-seed Iris diagnostics (#172/#173), but **none of those seeds are included**. The experiment is based on draft PR #173; PR nesting is solely to isolate development, not to merge experimental code to `master`.

## Primary question

Under the existing default *sequential* production NumPy PCC dynamics (including `p_grd=0.5`), does experimental label-stability stopping at **1,000-iteration checks / patience 10 / zero label-change tolerance** provide a favorable **paired trade-off of accuracy, label fidelity, and execution cost** compared with the existing production potential rule (`es_chk=2000`)? In particular, does the result hold across datasets and supervision densities, rather than selected Iris hard cases?

## Frozen inputs and four policies

- **Datasets:** Iris, Wine, Breast Cancer, Digits via `sklearn.datasets`, standardized with `StandardScaler` using established `canonical_prepare`.
- **Graph:** actual production `pcc_graph.build_knn_graph` with **undirected union-symmetrized 10-NN** (`k_nn=10`), checked for reciprocal edges, unique neighbors and no self-loops. Explicit `set_graph(graph, degrees, k_nn=10)` for production model calibration.
- **Supervision:** exactly **2 and 5 labeled nodes per class**; draw a single deterministic subset for each dataset/seed/supervision combination and reuse it for all policies.
- **Fresh seeds** (never used in earlier protocols): **5407, 5501, 5609, 5701, 5801, 5903**. There are **4 datasets × 2 supervision levels × 6 seeds = 48 independent dataset/supervision/seed comparisons**. All seeds/datasets are reported, including failures, without post-hoc exclusion.
- **Horizon:** production NumPy no-stop **100,000** iterations (short reference) and **200,000** (longer reference); actual original **potential** stopping capped at 200k; preexisting experimental NumPy **label-change P10**, checked every 1,000 steps, capped at 200k. These are exactly four policy outputs per case, **192 policy runs**.
- Production uses `ParticleCompetitionAndCooperation(impl="numpy")`, `return_info=True`, `p_grd=0.5`, `delta_v=0.1`, `deltap=1`, `dexp=2`, and explicit `early_stop` configuration. Candidate uses **existing** `experiments.benchmark_inloop_global_stopping.run`, without edits to particle or stopping implementation.
- **Safety gate for every case:** an additional **10,000-step no-stop candidate replay** must have exactly the same final labels as production 10k no-stop (both started from the same graph/labels/seed); failure aborts the job before interpreting a stopping comparison. This is a short-run final-label parity check, not a proof of equal 200k dynamics. Candidate and production 100k parity was already established in earlier independent cohorts.
- Separate model/RNG initialization for every policy. No cross-backend comparison is attempted: it is restricted to production NumPy versus an experimental NumPy candidate with established seeded no-stop prediction parity.
- **Raw output:** per dataset×supervision matrix job, one CSV containing **6 seeds × 4 policies = 24 rows**. Include cap, executed iterations, accuracy on initially unlabeled nodes, stop indicator, number of unlabeled nodes, final-label agreement, absolute label disagreement and accuracy delta relative to 100k and **200k no-stop** references, between-policy paired accuracy difference (P10 minus potential), actual wall time and steps saved against the 200k production reference, graph stats, and parity gate. Write completed cases incrementally (flush after every seed) so failures retain evidence.
- CI records dependency versions and enforces full 100k/200k reference iteration counts. Matrix splits **8 jobs**, one per dataset/supervision; original PyPCC regression workflow runs separately.

## Analyses specified before results

1. **Primary paired endpoint:** per-case `accuracy(P10) - accuracy(potential)` in percentage points, reported overall and separately in each of **eight dataset/supervision strata** (mean, median, minimum, maximum; number favoring each policy and ties). Signed differences must not be mistaken for magnitude of losses.
2. **Secondary cost:** fractions of 200k steps and measured wall-clock time saved under each stopping rule; report per-stratum stop rate, and number of instances where the criterion **fails to stop** before cap.
3. **Reference horizon sensitivity:** accuracy change and number of label changes **100k → 200k** within *the same* no-stop trajectory initialization. Matching two endpoints does not establish convergence, given known intermediate reversals in Iris.
4. **Fidelity and adverse cases:** agreement with fixed 200k labels, number of different labels, individual accuracy **losses exceeding 2 pp** relative to each backend's 200k no-stop reference and direct P10-minus-potential accuracy losses exceeding 2 pp. Retain all outliers.
5. **Reporting restrictions:** six seeds/stratum are **not sufficient** for robust confidence intervals or broad scientific claims. Report descriptive 48-case results and stratification; no unqualified statistical-significance or equivalence claims. The same seed can induce different label subsets between supervision densities: supervision strata are **not paired causal interventions**. Fixed 200k labels are **not convergence, optimal labels, or ground truth**; actual known labels are used for accuracy only on initially unlabeled nodes. Individual CI timing measurements are exploratory, not calibrated speed benchmarks.

## Decision

**No automatic production adoption irrespective of outcome.** Do not adjust P10/potential thresholds, seeds, datasets or exclusion rules after results. If a pattern is promising, next independent study should broaden to additional datasets and quantitatively characterize late changes at 200k+; accelerated backend integration requires its own tests. If the result is mixed, retain the existing production default. No modifications to `master`, existing RNGs, particle resets, graph algorithms or backend kernels.
