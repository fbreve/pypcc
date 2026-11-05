# Prespecified canonical undirected PCC early-stopping pilot

Recorded **before** running the pilot. Unlike older directed-neighbor benchmarks, **every run here** constructs its graph through production `pcc_graph.build_knn_graph` via `audit_canonical_undirected_lockout.canonical_prepare`, verifying reciprocity and preserving the two-per-class initial labels.

## Fixed cohort and comparisons

- Four sklearn datasets: Iris, Wine, Breast Cancer, Digits. Three new unused seeds per dataset: **2203, 2309, 2411**. 10-NN **undirected union** graph, StandardScaler, two labeled examples per class.
- Sequential experimental NumPy PCC, identical graph, labels, and random seed across policies. Maximum **50,000 iterations**.
- Three runs per dataset/seed: (1) no-stop 50k **capped reference**; (2) class-label prediction stability criterion `label_change` checked every **1,000** iterations, requiring **3** successive stable checks; (3) `dominance_label_hybrid` with per-node dominance-change P95 threshold **0.001** and same 1k interval and patience 3. No tuning after seeing results.
- Data collection: actual iterations, early-stop fraction, initially unlabeled accuracy, agreement with the 50k capped reference on the *same nodes*, change in accuracy (in percentage points), elapsed seconds and reference seconds, stop reason if available, graph degree statistics.
- Assert that the no-stop reference reaches 50k, produces labels for all initially unlabeled samples, has a valid undirected graph, and all stop runs execute a positive number of steps not exceeding 50k.

**Important:** 50k is a fixed reference horizon, **not a convergence guarantee**. A near-perfect match to 50k does not establish that stopping is intrinsically accurate or stable beyond this cap. Compare *actual wall time* as well as saved iterations, because checking P95 dominance can be expensive. A class-particle rescue or force-floor rule is **not** part of this stopping study.

## Interpretation

This is a **12-case pilot** following a 20-case canonical class-lockout prevalence experiment with zero observed events. It supports only descriptive comparisons, not universal stopping defaults. The previously studied directed-graph stopping thresholds are not canonical PCC evidence. Do not merge into `master` until validated across larger samples, longer reference horizons and preferably production accelerated backends.
