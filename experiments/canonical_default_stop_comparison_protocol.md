# Prospective current-default stopping baseline on canonical PCC graphs

**Frozen before new cohort results.** Earlier experiments compared experimental
label-stability stopping with a fixed no-stop horizon, but not with the
**existing PyPCC production default**, based on potential stabilization.

## Cohort and policies

- Four sklearn datasets: Iris, Wine, Breast Cancer, Digits, standardized
  using `StandardScaler`.
- New seeds **3407, 3511, 3607** (3 per dataset, 12 paired cases).
- Graphs created via **unchanged production**
  `pcc_graph.build_knn_graph`, `k_nn=10`,
  **undirected union**, validated for reciprocity and valid active edges.
  Two randomly labeled samples/class, fixed labels and graph for every
  policy within a dataset-seed case.
- Full 100,000-iteration **no-stop production NumPy PCC** reference,
  `ParticleCompetitionAndCooperation(impl="numpy")`,
  `set_graph(graph, degrees, k_nn=10)`,
  `fit_predict(..., early_stop=False, return_info=True)`.
- **Production default potential-based stopping**: exact same class and
  `set_graph(..., k_nn=10)`; `fit_predict(..., early_stop=True,
  es_chk=2000, stop_criterion="potential", return_info=True)`.
  The explicit `k_nn=10` matters: when graph is externally installed,
  `set_graph` otherwise infers a variable maximal symmetrized degree,
  changing the production `stop_max` calibration relative to
  `build_graph(k_nn=10)`.
- **Candidate label-change stopping**, using the isolated experimental
  sequential NumPy implementation: check every **1,000 iterations**,
  fixed **patience 10**, full cap 100k.
- **Mandatory parity gate:** a separate experimental no-stop run with
  the *same graph, seed, labels, cap* must produce the exact **same final
  predicted labels as production NumPy no-stop**. If it does not, mark
  the case a scientific validation failure and do not interpret
  differences in stopping as an apples-to-apples comparison. This
  gate does not establish full internal-state numerical parity.

## Primary outcomes

For each policy: number of early stops /3 per dataset, iterations saved,
measured wall-time savings *relative to the production no-stop reference*,
accuracy on the identical initially unlabeled test set, accuracy delta in
percentage points from production no-stop, and final-label agreement.
Also directly compare potential versus candidate P10 predictions/accuracy
and total wall times, with no claims of runtime distribution from a
single timing measurement.

**Do not retune** seeds, `es_chk=2000`, `patience=10`, or thresholds
after seeing the cohort. Results are exploratory, 100k cap is not
mathematical convergence; differences due to stochastic trajectory
can remain. Production and experimental backends both use sequential
NumPy to compare stopping semantics first; evaluate any promising
candidate across Numba/Cython before considering adoption.

No production change, no new particle rescue, and **no merge to
`master`** on the basis of this pilot.
