# Frozen independent production hypothesis: Numba-only complete-fit argmax selection

**Date 2026-11-09. SCIENCE ONLY; no direct merge.** Freeze this protocol BEFORE any scientific candidate source or sampling. Exact Git base \`bcf0fed83e64a753bf8ab01dba5124749931ebc2\`, production code unchanged. This is a NEW deliberately backend-specific hypothesis motivated by rejected universal scientific #226: the source-exact \`argmax\` full-matrix candidate delivered large Numba short-fit CPU improvements but failed the preregistered **universal** Cython GO. **Do NOT reinterpret #226 as GO or alter its criteria.** Test the different source candidate below against **new independent holdout masks (84217, 84313), different synthetic ring graph sizes and a new independently frozen acceptance rule**. Preserve original dense padded int64 graph representation, backend kernels, RNG/stop settings, layouts and exception rollback.

## Only candidate source change

The exact original single postpropagation expression remains:
\`\`\`python
unlabeled = node.label == -1
node.label[unlabeled] = np.argmax(node.dominance[unlabeled, :], axis=1)
\`\`\`
Candidate expression is restricted to the actual selected native dispatcher function object already available in \`__labelPropagation\`, WITHOUT another backend selection or public setting:
\`\`\`python
unlabeled = node.label == -1
if propagate_fn is pcc_step_numba:
    node.label[unlabeled] = np.argmax(node.dominance, axis=1)[unlabeled]
else:
    node.label[unlabeled] = np.argmax(node.dominance[unlabeled, :], axis=1)
\`\`\`
When the true dispatcher is Numba (explicit or fallback) use full-row argmax; otherwise **the original expression itself** is executed for Cython, NumPy and synchronous NumPy. Do not change function arguments, native backend objects, stop checks, RNG, C/F layout, source public API, graph indices, or \`owndeg\`. Derive candidate from untouched exact Git blob \`git show BASE:pcc.py\` allowing ONLY this single source span. The predicate must use dispatcher identity, not \`impl\` string (which could fall back).

## Independent prospective workloads and complete-fit sampling

- Real sklearn-standardized Wine (178, 3 classes), Breast Cancer (569, 2) and Digits (1797, 10), original ordered reciprocal union kNN10, two NEW original class-balanced labeled mask seeds \`84217\` and \`84313\` (3 labels/class).
- Controlled reciprocal 2-neighbor rings of \`40,000×20\`, \`80,000×50\`, and \`120,000×50\` nodes/classes, genuinely **undirected**. Note changed sizes from #226's 30k×10/50 and 100k×50; these test generalization beyond the original pilot's specific shapes. No claim that synthetic rings model ordinary data geometry.
- True original NumPy, Numba and Cython complete \`fit_predict\`; original \`max_iter=0,5\`, \`early_stop=False\` in every case, additionally \`max_iter=1000\` \`early_stop=False/True\` for real Numba and Cython. Original \`seed=29,p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=200\`. **48 cohorts/OS**.
- Six fixed-order-counterbalanced original/candidate batch pairs per cohort. Use **256 fits per timed arm** for real cases, 32 for ring40k×20, 16 for ring80k×50, and 8 for ring120k×50. Every arm uses ONE originally constructed and installed graph and ONE persistent PCC model across repeated fits, with alternating independent label masks. Warm Numba separately before all timing. All whole-batch process CPU and wall-clock timers stop BEFORE original/candidate six native + mapped/selected labels SHA256 and stop/RNG/rollback verification. Windows process CPU granularity is reported without posthoc corrections; real Digits has a distinct preregistered *wall-clock* GO endpoint for this reason.
- Every original/candidate true native output, all eight state arrays including final mapped node labels, true sampled next Numba and NumPy RNG, independent prior-fit arrays and invalid-seed rollback must match bitwise; original true stop reason/check count verified using independently replayed actual native \`return_info=True\` diagnostics outside CPU timers. Verify backend function identity and appropriate original dominance C/F layout. Cross-OS strict aggregation verifies source hashes, graph/input/mask hashes, all timed states, true stop info; no sample omitted.
- Standalone NumPy semantic controls, including C/F layouts, ties, NaNs, infinities and zero-to-all unlabeled masks as in #226; candidate's NumPy-only RHS retains the exact standard \`np.argmax\` implementation. Python/NumPy \`tracemalloc\` peak for Numba \`ring_120000_50\` 0/5 and Digits 0/5/1000 with **three independent original/candidate full fits per case/OS**. This is not an OS RSS/native malloc measurement. No changes to dense neighbor storage.

## Preregistered GO/NO-GO without retuning

A full scientific run succeeds only with exactly **48 cohorts/OS ×6 real original-vs-candidate paired batches** and all native state/RNG/stop/graph/memory checks, separately in Linux and Windows.

**Candidate CPU GO** only if original-vs-candidate *complete \`fit_predict\` process CPU* median paired saving is **>=8%** at both 0 and 5 iterations for held-out **ring80k×50 and ring120k×50 with actual Numba**, in each OS (8 independent primary cells). Independently, original-vs-candidate *complete \`fit_predict\` WALL* median paired saving must be **>=4%** on **real Digits with Numba**, at 0 and 5 iterations on both OS (4 real-data gates; Windows process-clock quantization is why wall is prespecified). All other real dataset/backend/stop cohorts must have **no >8% median whole-fit WALL regression** and all other synthetic cases **no >8% process CPU regression** per OS. All matched 3-sample Python/NumPy full-fit peaks must satisfy candidate <=1.05× original + 1 MiB for every sample. Exact source, complete eight-array parity, graph, native backend, RNG, stopping, previous-state independence and invalid-input rollback are mandatory.

If **any** condition fails, NO-GO; retain original production. If all pass, open a NEW minimal production-source + permanent-test PR from then-current master, independent full Linux/Windows Python 3.10/3.12 CI, review, and only then merge. **Never merge this scientific branch or its benchmark programs.** Do not introduce further backend/shape gating cutoffs based on observed outcomes; any later distinct hypothesis needs its own new preregistration and new holdouts.
