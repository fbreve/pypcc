# P2-02B + P2-03 — Original ordered UCI data and frozen float64 transforms (2026-11-13)

**Research-only, no PCC fit/propagation.** This joint preflight is stacked on P2-02A #265, the native research chain and P1 #258. Scientific P1 SHA-256 stays **`f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`**. Production code, algorithms and neighbor memory representation are untouched. No model, graph, label mask, row sampling, optimizer or RNG calls occur.

## Freeze the source interpretation, not merely the outer archive

The exact three already approved official UCI ZIPs (212, 850, 292) are independently downloaded once more, checked against P1 outer SHA-256 and selected member SHA-256, then parsed entirely in transient runner memory. No raw ZIP, CSV, XLSX, ARFF, feature vector or class label sequence is written to GitHub artifacts.

The parsing routines are the *original* previously audited PR #260 sources, fetched verbatim at commit `096a6c127eeeaa6b37171c99824a7a80a040aa72`, with both source Git blobs verified before importing them:

- `experiments/p1_offline_candidate_integrity.py` blob `1a38a36b8737effb71b49f90049dfdf43ae6d9ee`: strict ARFF and Raisin XLSX parsing, including formula prohibition and verified citation-only secondary worksheet.
- `experiments/p1_alternative_domain_source_quality.py` blob `1cea28ad67105923166bb1f8acc993520b370e83`: exact original Wholesale CSV headers in `Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen` order.

The P2-specific wrapper rejects any drift in 310/900/440 original rows, 6/7/6 numeric features, class distributions, original class encoding, row widths, nonfinite/missing features, duplicated feature names or source-column mix-ups. It deliberately **never sorts, excludes or deduplicates rows**.

- UCI 212: original `column_3C_weka.arff` three-class task, six ordered numeric features and final `Class` column, counts Hernia 60 / Normal 100 / Spondylolisthesis 150.
- UCI 850: `Raisin_Dataset.xlsx` from nested ZIP, exactly the original `Raisin_Grains_Dataset` data worksheet (901 rows including header, eight columns including `Class`), alongside the non-data `Citation_Request` worksheet; Besni 450 / Kecimen 450. `openpyxl==3.1.5` fixed for this experiment.
- UCI 292: `Wholesale customers data.csv` original 440 customer rows; **derived** target `Channel` codes 1:298 / 2:142; **both Channel and Region excluded from X**. Only Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen are predictors in that exact order.

The original class labels are encoded in deterministic lexicographic order **solely to record the target sequence's SHA-256**, not to select/impute/normalize the predictor matrix. No labels are sampled or hidden for PCC.

## Frozen numeric transformations

Exact `numpy==2.5.4`, Python3.12 Linux/Windows candidate runner:

- Vertebral and Raisin: parse all original numeric predictors as **float64**, then columnwise population z-score `(X - mean(X,axis=0,dtype=float64)) / std(X,axis=0,ddof=0,dtype=float64)`.
- Wholesale: require six annual spending values are all finite and nonnegative; apply **float64 `np.log1p(X)` first**, then the identical population z-score. `Region` and `Channel` never enter the numeric matrix.
- Any zero-variance, nonfinite or invalid column **FAILS CLOSED**. No mean imputation, outlier clipping, subject filtering, rows removed, or outcome-informed preprocessing.

Evidence reports include original shape, original ordered column names, class supports, a target-order SHA and **two hashes for each raw and transformed matrix**: SHA-256 of canonical little-endian float64 C row-major bytes; and SHA-256 of metadata (shape + dtype + ordered predictors) concatenated with those exact bytes. Only the hashes and aggregate validity information are published, never the matrices themselves.

Each runner reports its results independently; a third job compares Linux and Windows **all three** original column orders, target sequences, raw X hashes and transformed X hashes exactly. A numerical divergence **blocks** this candidate environment rather than silently changing a transform. Passing this is source/feature identity evidence, **not** proof that later kNN graphs or Numba/Cython propagation will be identical across platforms.

## What still remains unapproved

- P2-04: actual kNN graphs from unchanged production `build_knn_graph`, symmetrized undirected and **dense `neib_list` with `-1` padding**; graph identity, ordered neighbors and memory budget.
- P2-05: original label mask PCG64 recipe and backend-specific seed table, without PCC fits.
- P2-01/06 final native lock: historic compiler/wheel identities were not fully sealed; candidate synthetic parity has passed but does not authorize a real run.
- P2-07/P2-08: full CPU/RAM/disk/export/resource proof, exact technical SHA and **separate explicit maintainer approval**.

Required flags: `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `PCC_propagations_executed=0`, `PCC_real_source_fits_executed=0`. The study remains descriptive/exploratory only, with no claims of subject independence or cross-backend trajectory equivalence. Keep the PR **DRAFT**; do not merge into `master`.

**Status of this document:** proposed checks; only actual passing two-platform CI and digest equality qualify as P2-02B/03 candidate success.
