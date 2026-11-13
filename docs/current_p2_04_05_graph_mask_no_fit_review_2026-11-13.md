# P2-04 / P2-05 — Frozen real-source kNN graph and PCG64 masks, zero PCC fit

**Status:** candidate engineering preflight, stacked on [P2-02B/03 #266](https://github.com/fbreve/pypcc/pull/266). Scientific P1 SHA `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166` is unchanged; P2 SHA remains null and no execution is approved.

This gate is different from P2-02B/03: **it DOES build graphs on the three real official source feature matrices and DOES sample labels to create 24 real-source masks, but DOES NOT execute a single PCC propagation or fit**. Those preparatory masks and graphs exist only in transient CI memory. It does not publish original source rows, exact selected label indices, serialized masks, original label vectors, graph neighbors, raw/transformed features, or predictions. It publishes only digests/aggregate metadata, the 72 deterministic propagation seed integers (not used), and environmental evidence.

## Strict inputs

- Only UCI 212 Vertebral Column, 850 Raisin and 292 Wholesale Customers in the approved original P1 order.
- Verify P1 scientific JSON SHA and official ZIP/member SHA independently on every run; fail closed on any source drift.
- Use the exact original P1 source-parsing modules from PR #260 pinned by Git blobs `1a38a36b8737effb71b49f90049dfdf43ae6d9ee` and `1cea28ad67105923166bb1f8acc993520b370e83`.
- Re-run exact candidate P2-03 transformations with NumPy **2.5.4**, SciPy **1.18.1**, scikit-learn **1.9.1**, openpyxl **3.1.5**; verify raw/transformed X shape/dtype/ordered-feature SHA against the already passed cross-platform P2-03 digests **before** graph construction.
- Frozen P1 seed formula is checked against `experiments/p1_exploratory_reference.py` Git blob `480777758a445ba42425df69bcc5a90b97d6d0dc`. This is exact original source from P1, not a new reimplementation of the seed recipe.

## P2-04 graph construction and representation

Call the unchanged production `ParticleCompetitionAndCooperation(impl="numpy",n_jobs=1).build_graph(Z,k_nn=10,n_jobs=1)` exactly once per dataset (no `set_graph` call, no `fit_predict`). Verify graph builder/source and unchanged `build_knn_graph` defaults: Euclidean Minkowski p=2, `NearestNeighbors(algorithm="auto")`, symmetric undirected union.

The graph delivered to the PCC **remains exactly** a `neib_list` dense matrix with `int64` active column indices and **literal `-1` padding**, plus one `int64` degree vector `neib_qt`. The internal scipy sparse adjacency used *during construction* is unchanged. No CSR propagation replacement or additional graph storage representation is proposed.

Fail on self edges, invalid endpoint, duplicate, neighbor-order drift, zero degree, missing reciprocal edge, or a padding value other than `-1`. Record exact full graph SHA-256 of a documented metadata+byte serialization, as well as separate raw int64 `neib_list`/`neib_qt` SHA-256, shape, min/max degrees, undirected edge count and allocated bytes. Graphs with tied nearest-neighbor distances may differ by operating system; this is **not silently fixed**. The mandatory cross-OS hash comparison must pass or this gate fails. This graph-construction check does not itself prove the exact Numba/Cython kernel backend receives a future graph unchanged.

## P2-05 deterministic mask/seed preflight

For each UCI dataset in approved registry order, and replicates 0..7:

- From its verified original row-ordered class labels, take each class's ascending source row IDs.
- Create a new `np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261112, registry_index, replicate_index])))`.
- In lexicographically sorted original class order call `Generator.choice(original_class_indices,size=3,replace=False)` **once per class**; then sort the combined selected indices.
- Encode `-1` for unlabeled rows and `0..C-1` for selected original class values. Reject incomplete or overlapping selections. Produce a hash of the entire little-endian int64 mask vector and a hash of selected indices, but **do not publish the actual indices or vectors**.
- Derive 72 propagation `uint32` seeds from the untouched original P1 code, using the exact ASCII domain-separation formula with source UCI ID, replicate, and backend name. The seeds are **NOT given to PCC** in this preflight.

All 24 original-source masks and 72 seeds must be computed with the same class label/order and used as a single shared basis for **future** distinct backend fits, not silently redrawn independently for each backend. The native backend RNG streams are explicitly **not** assumed identical across NumPy, Numba and Cython.

## CI and status

Runs on Linux and Windows/Python 3.12, pinned candidate package versions. Unit tests use only synthetic 18-row data and aggressively mutate graph padding, neighbor IDs, degree, class support, mask schedule and source order. Actual data workflow emits safe hashes/aggregate metadata only. After two platform jobs pass, a third job verifies exact equality for:

- 3 frozen raw/transformed X identities inherited from P2-03;
- 3 complete dense graph hashes, node degree hashes and aggregated shape/edge metadata;
- 24 full-mask hashes and selected-index hashes;
- 72 propagation seeds and associated backend/replicate IDs.

The third job produces a dedicated concordance report with `P2_04_05_cross_platform_hash_gate_pass=true` **only on real successful equality**. Individual platform reports deliberately set that flag false, since one platform cannot independently claim global agreement.

**Still blocked:** P2-01/06 final native toolchain lock, P2-07 RAM/time/disk/resource cap, P2-08 exact technical SHA and explicit distinct maintainer approval. No source/production changes and no PR merges. Specifically, P2-04/05 is a **no-PCC-fit engineering preflight**, not authorization for P3 real scientific results or the 72 fits.
