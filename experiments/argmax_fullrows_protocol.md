# Frozen scientific CPU trial: skip indexed dominance copy in postpropagation argmax

**Date:** 2026-11-09. **SCIENCE ONLY, DO NOT MERGE.** Freeze this protocol in Git **before** benchmark candidate, CI or results. Source baseline is production \`master\` at \`bcf0fed83e64a753bf8ab01dba5124749931ebc2\` (documentation-only #224 just merged). Original files \`pcc.py\`, \`pcc_graph.py\`, \`pcc_numpy.py\`, \`pcc_numba.py\`, \`pcc_step.pyx\`, \`pcc_validation.py\`, \`setup.py\` and original wheel/build scripts must remain unchanged in the experiment branch. The deliberately CPU-efficient **dense padded int64 neighbor matrix** is NOT a candidate for replacement.

## Scientific question and exactly one candidate

Original untouched production code in \`__labelPropagation\`:
\`\`\`python
unlabeled = node.label == -1
node.label[unlabeled] = np.argmax(node.dominance[unlabeled, :], axis=1)
\`\`\`
Candidate only changes RHS:
\`\`\`python
node.label[unlabeled] = np.argmax(node.dominance, axis=1)[unlabeled]
\`\`\`
This avoids an explicit indexed (\`unlabeled_count × classes\`) copy before argmax, but must scan rows corresponding to labeled homes and may still allocate internally depending on NumPy/Fortran layout. The candidate is derived at runtime from the **exact immutable original \`git show BASE:pcc.py\` Git bytes**, permitting exactly this one-line replacement. Original and candidate must call the **same untouched** real Numba/NumPy/Cython backend, preserve C/F dominance layout, native RNG and all original call semantics, with NO runtime change in production source. This optimization is NOT the prior rejected #199 in-place owndeg normalization, #202 one-hot initialization, or #222 class discovery.

## Frozen workloads, timed batches and backends

- Actual sklearn standardized Wine (3 classes), Breast Cancer (2), Digits (10), undirected original ordered union-kNN10, class-balanced three labels/class. Undirected, symmetrically connected two-neighbor ring graphs of \`30,000×10\`, \`30,000×50\`, \`100,000×50\` nodes/classes. These rings stress many-class postpropagation without claiming that they represent typical datasets.
- Two independent labeled masks selected with \`np.random.default_rng\` seeds \`84013\` and \`84117\`, alternating between consecutive fits on the *same* single model and installed graph for each original/candidate arm; all original source parameters \`p_grd=.5, delta_v=.1, deltap=1, dexp=2, seed=29, es_chk=200\`. Original \`max_iter\` 0 and 5 \`early_stop=False\` for each of three native backends on all six graphs. On real datasets Numba/Cython also do fixed \`max_iter=1000, early_stop=False\` and monitored \`max_iter=1000, early_stop=True\`. Exactly **48 complete-fit cohorts per OS** (30 real + 18 synthetic) without posthoc exclusions.
- Each cohort: **6 complete alternated original/candidate timed-batch pairs**, with ORIGINAL then CANDIDATE in even pair indices, reversed in odd. Original and candidate each reuse one instance and the same preinstalled original ordered graph across all repetitions. For each arm, batch length is 256 fits for each real dataset, 32 for 30k×10, 16 for 30k×50, and 8 for 100k×50. Numba JIT, Cython loading, graph construction, graph installation and threadpool setup precede all timed batches; no compilation or state hashing inside timers. All genuine batches timed using \`time.process_time_ns()\` and \`time.perf_counter_ns()\`; report Windows CPU-clock quantization unchanged and per-host medians of **paired** savings, not ratio of pooled cross-host timings. Do not drop unfavorable batches.
- After timing each batch, compare **all six original native state array SHA256**, final label outputs, original \`propagation_info\` stop metadata, ordered padded graph and mask hashes for EVERY fit; keep old results as independent arrays across repeated fits. Compare true seeded NumPy/Numba RNG endpoint by separate original vs candidate replay outside timed batches; compare exact previous fitted arrays' hashes/identities after invalid-seed failure. For Cython compare original native six arrays, stop metadata and output; do NOT assert identity across different backend RNG algorithms.
- Independent isolated correctness checks cover C and Fortran \`float64\` dominance layouts, tied winners, signed zero, NaN/+Inf where NumPy argmax itself defines behavior, and unlabeled-mask fractions 0, 1%, 50%, 99% and 100% for sizes 0/1/3/1024/30000. For an \`n×0\` class matrix, compare exception types rather than inventing outputs. A small standalone local exploratory microbenchmark may motivate the hypothesis, but no local microphase improvement can promote code.
- Separately take **3 matched \`tracemalloc\` full-fit Python/NumPy peak memory pairs** for original/candidate on the 100k×50 ring Numba (0 and 5 steps) and real Digits Numba (0, 5 and monitored 1000); use fresh models, no timers or hashing inside the memory tracker. \`tracemalloc\` does NOT measure all native malloc or OS RSS; report no unsupported native-heap conclusions. Never replace \`neib_list\` with CSR to achieve a memory win.

## Frozen GO / NO-GO

Strict independent two-host aggregation must verify **all** cohorts, paired counts, exact source/git and true backend, baseline/candidate native six-array/stop/RNG parity, previous-result ownership and invalid-seed rollback. Completeness success alone is not a performance GO.

**CPU GO requires** at least **5% median paired complete-fit process CPU saving** for \`100000×50\`, **both 0 and 5 iterations**, **on Numba AND Cython**, **on each Linux and Windows host**. In addition, \`30000×50\` Numba complete-fit CPU must improve **>=3%** at 0 and 5 iterations on both hosts. No real dataset/backend/stopping cohort may regress by **>8% median complete-fit wall time** on either host, and no synthetic remaining cohort by **>8% CPU** on either host. All measured full-fit Python/NumPy traced peaks must remain <=1.05× original + 1 MiB; exact outputs/states/stop/RNG/graph checks mandatory. Treat CPU batch zeros as invalid data, never invent an adjusted quantized CPU saving.

If any strict requirement fails, **NO-GO**: preserve the experiment and evidence, close scientific draft without merge, keep original production RHS. If all gates pass, propose a *new* minimal master-based code+permanent-tests PR, with independent four-platform package CI; never merge experimental scripts, tests or benchmarks into production merely because science passed.

No host clock manipulation, posthoc threshold adjustments, benchmark trimming, cherry-picking winning datasets, synthetic-only production decision or other implicit changes after first samples.
