# P1: proposal for a prospective PCC trajectory-stability study (not approved)

**Status: scientific design draft, not a frozen preregistration.** This proposal is not an instruction to run an experiment. It has **no selected real datasets**, no approved label masks, no executable P2 manifest, no approved scientific SHA and no real-data fit authorization.

Machine-readable **proposal**: [`current_p1_prospective_design_draft_2026-11-12.json`](current_p1_prospective_design_draft_2026-11-12.json).
Inert validator: `python -m experiments.review_p1_proposal`. Its report's `draft_content_sha256_not_an_approval` is only a checksum of this *unapproved draft*; it must **never** be substituted for a reviewed scientific or execution-preflight SHA.

## Why a new design rather than replaying C2/G1

The previous studies have already examined **Banknote, Sonar, PageBlocks and Spambase**, including C2 hypotheses about exact ties, high-margin errors and label geometry, and G1 candidate-based label selection. Some historical outcomes, thresholds and seed choices were selected after observing data. They are valuable context, but **not fresh confirmatory evidence**. Frozen PR #18 and modern observer drafts #253 (NumPy), #255 (Numba), #256 (Cython) remain separate, and P0 source inventory #257 is a provenance audit rather than an approved science record.

The narrower *proposed* scientific question is **descriptive**: with a fixed graph, three labeled examples per class and unchanged sequential PCC, what fraction of originally unlabeled nodes change their predicted class between iteration 5,000 and iteration 50,000? This is **not** an attempt to establish that one backend has correct/incorrect trajectories relative to another, that confidence is calibrated, or that continued propagation causally improves accuracy.

A follow-up **confirmatory** claim would require independently selected truly unexamined datasets, prespecified practical thresholds and uncertainty assumptions, an outcome-blind registry and separate approval before inspecting results. This draft is **exploratory unless and until** those gates pass, and post hoc reclassification is not allowed.

## Proposed design — all numerical choices subject to scientific review

| Element | Draft proposal | Why it is provisional |
| --- | --- | --- |
| Dataset registry | Target **3 previously unexamined public datasets**; currently **0 selected** | Need independent provenance, licensing, raw file versions, outcome-exposure audit, feature processing and graph feasibility |
| Classes and labels | 2–5 classes; at least 10 nodes per class; **3 labeled nodes per class** | Must ensure each originally labeled class has truly unlabeled evaluation support; never filter based on PCC outcomes |
| Mask repeats | **8 independently drawn masks** per dataset, shared across backends | Need resource and independent-label assessment; do not treat multiple nodes as independent replicates |
| PCC backend conditions | Sequential NumPy, compiled Numba and compiled Cython | Keep their distinct RNGs/arithmetic; compare each observer to its original backend; never require cross-backend exact trajectory identity |
| Graph | Same sealed **undirected, symmetric, ordered dense padded `neib_list`** per dataset | Existing production graph-building, padding and feature processing must be specified and hashed at P2; do not change to sparse/CSR |
| Horizon | `max_iter=50,000`, **early_stop=False** | Fixed horizon is suitable for observing drift; this is not a stopping-rule optimization, and its compute cost needs P2 validation |
| Milestones | `0, 1, 10, 100, 1,000, 5,000, 25,000, 50,000` | Eight points fit current observer's 32-slot cap; do not reuse older C2-R1 ten-point schedule by accident |
| Proposed volume | **3 × 8 × 3 = 72 fits** | An estimate of planned task count, **not** authorization or evidence of feasible runtime/storage |

**Important:** the observed 500-node synthetic benchmark costs in #253/#255/#256 do not predict this 50,000-iteration real-data workload. The native Cython observer copies an additional row-max cache and its `xorshift32` uint32 state; native Numba currently provides continuation through the *next compiled draw* rather than a serialized internal RNG state. P2 must confirm real memory, storage, fit CPU and export CPU, without a single real fit, before anything is approved.

## Primary outcome and analysis

The proposed **primary descriptive estimand** is, for every *(dataset, frozen label mask, backend)*:

```
flip_fraction = number of originally unlabeled nodes
                whose argmax(dominance[5000]) differs
                from argmax(dominance[50000])
                ------------------------------------------------
                number of originally unlabeled nodes
```

Predeclared deterministic `argmax` tie breaking remains part of the observation definition. Report exact leading ties *separately* rather than silently excluding them from the denominator or retrospectively redefining a flip. Report original labels and original home-node indices in the immutable mask manifest; these nodes must not be scored as unlabeled outcomes.

For each backend, report all eight mask outcomes per dataset, then average across masks within a dataset, then report the **unweighted average across the selected datasets**. Publish the backend-stratified results and dataset-level values, not just a pooled per-node fraction. Nodes in the same graph are not independent replicates, and the proposed three deliberately selected domains do not support reliable population-level uncertainty. For this **exploratory-only proposal**, report the eight mask values, per-dataset **mean/median/min/max**, and the unweighted mean across datasets; **do not report a 95% confidence interval, p-value, equivalence test or population-level generalization claim**. Any later confirmatory study would need an independently approved practical threshold, independently selected domains and defensible uncertainty procedure before inspecting outcomes.

**Secondary, all disclosed:** change in accuracy and balanced accuracy on originally unlabeled nodes (ground truth accessed **only after propagation**), exact-tie fraction per milestone, change in top-two dominance margin, and the full time series of label changes. Missing true-class support makes that dataset/mask's balanced accuracy **non-evaluable**, not zero. The margin is an internal dominance statistic, **not calibrated probability**. No thresholds, confidence intervals or p-values can be added after observing outcomes. This expressly **exploratory descriptive** analysis still requires reviewer approval before P1 can be frozen; a future confirmatory label would require a separately proposed and approved protocol.

## RNG, masks and source discipline

Draft mask recipe: use a **separate** `NumPy Generator(PCG64(SeedSequence([20261112, dataset_index, replicate_index])))` to sample three row IDs per class uniformly without replacement after stable sorting by original class ID and row index. The **same** class-stratified labeled mask and one ordered graph are used in each of the three backends for that unit. This generator must **never** consume the live propagation RNG or touch graph construction randomness. Actual class IDs, mask bytes, ordering, NumPy version and per-case seed identities must be sealed and independently checked in P2.

The propagation seeds have a **proposed, not approved** domain-separated mapping specified in [the seed and exploratory-analysis contract](current_p1_rng_and_exploratory_analysis_2026-11-12.md): SHA-256 of dataset ID, replicate index and backend string, reduced to a nonzero uint32, without consuming the PCG64 label-mask RNG. The mapping and final per-fit table require independent review and freezing; a shared numerical seed does **not** produce the same RNG stream in NumPy, Numba and Cython. For a Cython binary, the native xorshift has special handling for zero seed. Seed derivation and availability remain **review blockers**; no historical seed list (10–19) or old C2 scientific preflight hash is silently reused.

The modern observers are not merged into a single executable. If the study advances, P2 must build three isolated source-exact conditions against the P0-pinned production base `ea69f69`, verify exact Python/native library versions and compiled-binary hashes, seal the same graph/labels and perform only **non-fit** storage/shape/graph checks. The baseline production kernel, cache specialization, stopping rules, padding and numerical semantics stay unchanged.

## Failure, contamination and publication gates

- Maintain a single prospectively sealed candidate registry with explicit rejection reasons *before* PCC results. If a dataset/fixture is found to have previously influenced hypothesis construction, mark it **historically exposed**; do not call it a fresh external validation.
- Declare all planned cases and independent seeds in an immutable task manifest. An unavailable dataset or failed/non-evaluable run remains in the record with cause and denominator. No silent repick, extra seeds, algorithm changes, outcome-driven retries, or threshold tuning.
- All three backends must report observation and original-production parity **within** their own source/binary, including final state, stop metadata, full prediction and appropriate native RNG checks. A cross-backend difference is not automatically an error.
- Preserve an append-only ledger and exactly one attempt per task; an interruption blocks continuation until explicit reasoned recovery approval. Never overwrite failed results to improve the sample.
- P1 reviewer must approve scientific question, registry, mask and seed policy, sample size, milestones, analysis/uncertainty and non-evaluability rules and then separately issue an **immutable scientific SHA**.
- P2 must produce its *own* execution preflight SHA binding source+compilation+data+environment+resources. Only an **explicit human approval of that exact P2 SHA** permits a first real fit. Inspect and approve that first fit before the remaining planned cases.

### What is still undecided

**P1 is not ready to freeze:** no eligible unexamined dataset has been identified and version-sealed, no independent historical exposure review is complete, no exact graph/feature procedure is committed, and the seed-mapping and confirmatory uncertainty protocol are not approved. The proposed 72 tasks and 50k horizon are therefore **not** executable commitments.

This work is deliberately narrower than the historical causal G1 study; it does not seek to test candidate label-selection interventions. It likewise does not re-run Banknote/Sonar as fresh C2-R1 replication.

The current action is only to review this P1 proposal and its inert structural checks. **There is no current permission to perform a real-data PCC fit.**
