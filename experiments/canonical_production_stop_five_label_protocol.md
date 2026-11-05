# Preregistered five-label/class production potential vs label-stability stopping

**Fixed before evaluating the new cohort.** The previous canonical-PCC
study compared the PyPCC production potential-based early stopping default
with the experimental label-stability stopping on 12 held-out two-label/class
cases. Production was faster on average; label stability P10 retained more
100k-reference predictions, with variation by dataset. This study tests whether
that tradeoff persists with more labeled examples, **not** a new threshold sweep.

## Fixed design

- Datasets: sklearn Iris, Wine, Breast Cancer, Digits; StandardScaler.
- New unused seeds: **3701, 3803, 3911** (3/dataset; 12 paired cases).
- **5 labeled points per class**, selected by the deterministic `canonical_prepare`
  procedure. The **undirected union k=10** graph is built exclusively by
  production `pcc_graph.build_knn_graph`, checked for reciprocal arcs,
  and held fixed with labels and seed across runs.
- Same sequential NumPy PCC parameters and 100,000 maximum iterations.
- Compare three policies for each dataset-seed:
  - **Production NumPy no-stop** (`early_stop=False`, `k_nn=10`), 100k.
  - **Production default potential stopping** (`early_stop=True`,
    `es_chk=2000`, `stop_criterion="potential"`, `k_nn=10`).
  - **Experimental label stability** (`label_change`, 1k interval,
    **patience=10**), no other thresholds or adaptive policy.
- Before accepting each case, **same-seed, same-graph, same-label** candidate
  NumPy no-stop must produce exactly the same final prediction vector as the
  production NumPy no-stop at 100k. Prediction parity is a gate, not proof
  of full trajectory or accelerated-backend equality.

## Primary outcomes

- Cases stopping early per policy and realized iteration savings.
- Mean wall-clock saving relative to 100k **production** reference
  (one run per case, exploratory timing).
- Agreement with production 100k labels on identical initially unlabeled
  nodes, mean accuracy delta in **percentage points**, **worst individual
  loss**, and pairwise agreement between stopped policies.
- Per-dataset results and outliers, not just a pooled mean. Note that
  100k is a finite reference cap and not a convergence theorem.

**Do not tune** `es_chk`, patience, seeds, supervision or graph construction
in response to these results. No production modifications or merge into
`master` are authorized by this exploratory study.
