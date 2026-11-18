# P1 — seed identity and exploratory analysis contract (UNAPPROVED)

**Status (2026-11-12): FOR INDEPENDENT SCIENTIFIC REVIEW.** This is a *precise proposed decision* for [P1 design PR #258](https://github.com/fbreve/pypcc/pull/258), **not** a preregistration seal, a dataset selection, a graph/mask generation authorization or a P2 execution approval. It uses the candidate shortlist and unchanged 72-task design described in [the P1 scientific decision packet](current_p1_scientific_decisions_2026-11-12.md). The reference implementation `experiments/p1_exploratory_reference.py` and `experiments/test_p1_exploratory_reference.py` handle **synthetic-only numbers**; they do not import NumPy or PCC, load real data, draw real label masks, construct a graph or invoke propagation.

## A. Freezeable seed identities — specific proposal, NOT a generated scientific seed table

Original P1 draft #258 proposes exactly **8** masks per dataset, **3 labeled nodes per class**, the same mask and graph across NumPy/Numba/Cython, a separate mask RNG and backend-native propagation RNGs.

**Mask identity**: if the independently reviewed candidate order stays `[212,850,292]`, define registry indices `0,1,2`. For replicate `r=0..7`, the *proposed* mask RNG is precisely `numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence([20261112, registry_index, r])))`. This must be instantiated **once per mask**, never reused to seed a propagation backend. Draw three distinct original row indices per original class using `choice(..., size=3, replace=False)`, with classes sorted by their final source-code spelling and eligible original indices ascending; concatenate/sort sampled indices deterministically. The actual mask-drawing and serialization implementation, NumPy version, source row/class mapping and complete per-replicate labeled-index SHA-256 must be independently checked and approved at P1/P2 **before any model fit**. These words are only a recipe: **no source-specific mask has been sampled here**. A candidate-order change invalidates the index-based mask schedule and must create a new reviewed proposal; it cannot silently reuse a former mask identifier.

**Propagation seeds**: propose a domain-separated deterministic uint32 mapping that does not consume the PCG64 mask stream. Compute SHA-256 of the ASCII bytes

```text
pypcc-P1-propagation-v1|uci:<dataset_id>|mask:<replicate_index>|<backend_name>
```

and then `seed=1+(int.from_bytes(digest[:4],"little") % (2**32-1))`. This yields a nonzero uint32, avoiding the special all-zero xorshift32 state. Proposed backend-name strings, verbatim from the existing unapproved P1 draft, are `numpy_sequential`, `numba_sequential_nopython`, and `cython_sequential_cached_or_legacy_as_defined_by_early_stop`. **A common integer seed does not make their native random streams or numerical paths identical.** The exact 72-case seed table, backend-native seed validation and Cython build/source identity must still be frozen and authorized separately. The reference tests verify only deterministic mapping and no collisions for the 72 proposed ID/replicate/backend combinations, **not actual random trajectories**.

## B. Primary outcome — fixed descriptive definition

Within each individual *(dataset, frozen labeled mask, backend)* unit, use the **same** original ordered node set, class order and undirected graph throughout 50,000 iterations, with `early_stop=False`. The primary comparison remains **iteration 5,000 versus 50,000**. On the original **unlabeled** rows only, use first-index `argmax` on each checkpoint's dominance row and calculate:

```text
flip_fraction = number of originally unlabeled nodes changing argmax(5000 → 50000)
                / number of originally unlabeled nodes
```

Do not exclude tied maxima from the denominator: record the **exact equality** top-class ties at both checkpoints separately, and do not reinterpret ties or change tolerance retrospectively. Originally labeled nodes are excluded even if their class or dominance would otherwise differ. The primary study is **not** designed to optimize stopping time, choose a best backend, infer calibrated uncertainty or prove that 50k improves accuracy. Retrospective accuracy/balanced accuracy, tie and margin secondary outcomes remain those already listed in #258, without using true unlabeled labels to choose masks, graphs, parameters or seeds.

## C. Reporting and uncertainty — avoid false replication

The proposed scientific claim is explicitly **exploratory descriptive**, not confirmatory. For each backend, report all eight masks within each dataset and their **mean, median, min and max**; then report the **unweighted average of the three dataset-level means**. These nested outcomes represent **three domains**, not thousands of independent nodes; eight masks of the same dataset are not eight independent domains, and the different backend streams are not interchangeable independent studies. Present the complete primary and secondary per-case attempts and all discrepancies.

**No 95% confidence interval, p-value, equivalence claim, hypothesis threshold, or cross-dataset generalization assertion in this initial three-domain descriptive design.** With only three deliberately chosen domains, a node-level bootstrap or simple 72-row bootstrap would be misleading. A later confirmatory study requires independently approved domains, practical effect-size threshold, larger-domain sampling rationale and a separate uncertainty method before seeing its outcomes.

The proposed synthetic-only reference summary **fails closed unless it receives exactly 72 distinct cases**, all three backend conditions for each of eight masks in each of three proposed datasets, valid `0..1` flip fractions and identical graph SHA-256 per dataset / identical mask SHA-256 per dataset×replicate across backends. If a real study is eventually authorized and a case fails, record it in the append-only attempt ledger with reason and **do not silently impute, drop or replace it**. A complete 72-case aggregate should then be withheld or qualified as incomplete pending a separately approved failure-handling amendment; no manufactured mean of successful runs.

## D. Final independent-review actions

1. Confirm **exploratory-only** reporting and this deterministic but still *proposed* mask/propagation seed mapping. If it changes, revise the protocol before science freeze.
2. Approve or reject original UCI shortlist `[212,850,292]`, Wholesale **derived Channel** target, clinical and customer/grain provenance, prior outcome exposure and sample independence claims. Changing source order implies remapping all mask identities.
3. Specify and approve exact finite numeric transformations: population z-score for Vertebral/Raisin, and `log1p` followed by z-score for Wholesale, with no source rows excluded; compare **no PCC outcomes** while selecting.
4. Approve exact `k=10`, Minkowski `p=2`, symmetric non-directed graph and existing dense padded `neib_list`, retaining a fixed graph construction/installation path; verify source and graph hashes.
5. Independently approve this P1 document set and issue a **new scientific P1 SHA** only after evidence/history/provenance/resource blockers clear. It must be different from the reviewer-draft checksum.
6. Execute **separate no-fit P2 environment/compiled-binary, graph, mask, CPU/RAM/disk and export preflight** with a **new execution-specific P2 SHA** and distinct explicit maintainer approval before even one **real-data PCC fit**.

**Review outcome at this stage:** Source-only prospective method choices have become concrete and testable, but remain **NOT APPROVED**. P1 selected registry remains empty, no P1 or P2 approval SHA exists, no real dataset mask or PCC graph was created and no real-data PCC fit was performed. Keep #258 DRAFT, unmerged.

