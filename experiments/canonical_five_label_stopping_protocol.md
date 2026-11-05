# Prespecified five-labeled-per-class canonical PCC stopping validation

Recorded **before new run results**. This is a sensitivity study of
label-stability stopping to an increased initial supervision budget.

- Data: sklearn **Iris, Wine, Breast Cancer, Digits**; standardization by
  StandardScaler; graph from the unchanged production
  `pcc_graph.build_knn_graph` with **k=10 and undirected union
  symmetrization** (reciprocity required).
- Previously unused seeds **3109, 3203, 3301**, three per dataset:
  **12** dataset-seed cases.
- **Five labeled observations per class**, selected from each class
  by the same `np.random.default_rng(seed).choice` selection rule
  used by `canonical_prepare`. Graph and labeled set are identical
  across all policies within each case.
- Experimental sequential NumPy PCC, original particle dynamics,
  **100,000-iteration no-stop reference**, plus two fixed
  `label_change` stopping policies, check interval **1,000**:
  **patience=3** and **patience=10**. These were selected before
  this new cohort based on the prior two-label study; do not add
  more settings in response to results.
- Diagnostic observation at 50k and 100k in reference only;
  verify callback-observed 100k predictions equal final predictions,
  and the entire 100k observed reference matches an independent
  same-graph same-seed **no-callback** 100k run.
- Metrics: full cap versus actual stopping iteration and wall time,
  fraction saved, predicted-label agreement on initially unlabeled
  nodes and accuracy difference in **percentage points** vs 100k;
  worst-case losses, positive changes, and reference 50k→100k
  drift. Track mean graph degree, reciprocity and label budget.
- All runs must reach a valid positive stopping iteration <=100k;
  the reference must reach 100k. Compute per-dataset and total
  descriptive summaries.

**Interpretation:** Accuracy on initially unlabeled examples here has
a different denominator from prior two-labeled-per-class cohorts.
The comparison is a *generalization/sensitivity check*, not a direct
paired causal test of changing label density. A fixed 100k reference
is not convergence. Do not change default stop criteria or particle
behavior in production `master` without larger independent
validation, including accelerated backends and other datasets.
