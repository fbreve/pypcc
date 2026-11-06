# Preregistered second phase: Windows/Linux replication and numerical edges

**Frozen before collecting phase-2 results.** Phase 1 of draft PR #177 showed **32/32 full state + stop metadata parity** and **3,200 scalar potential checks** unchanged, with **+36.30% median whole-fit saving** on 20k canonical Digits (four seeds) on a Linux GitHub runner. That cohort is now *development evidence*, not a new holdout. Do **not** alter the candidate Cython implementation or select seeds after outcomes.

## A. Independent OS replication

- Fresh canonical standardized, production **undirected union-kNN=10** datasets: **Digits (10 classes)** and **Breast Cancer (2 classes)**, each with exactly two labeled nodes/class. New fixed seeds **7307, 7403, 7507, 7603**.
- **Linux and Windows**, Python **3.12**, default Cython extension built from the same commit. Real backend must be the requested Cython `pcc_propagate`; experimental subclass delegates to that compiled function with `experimental_rowmax_cache=True`, never changes particle motion.
- For each dataset×OS×seed: **five paired blocks**, with Cython **20,000-step forced-monitoring** policy (`early_stop=True, es_chk=2000000`; stopping threshold checked to exceed possible 10-step check count). Alternate (legacy→cache) / (cache→legacy) by block and seed to reduce order bias. Require full state exact equality and all stopping-info fields identical for **every** paired block, not just final classifications. Measure `fit_predict` only.
- Plus one paired actual-production-stop run per dataset×OS×seed, **100,000 cap, es_chk=2000**, again requiring exact stopping reason/counts/full state. These stop-policy pairs are for parity, and their single-run timings are secondary exploratory values.
- **Total 2 datasets × 2 OS × 4 seeds × (5 forced + 1 real-stop) × 2 variants = 96 timed fits, 48 strict exact-state + exact-stop-metadata pairs**, with 40 of the pairs repeated forced-workload performance trials. Each OS × dataset is one CI matrix job, recording raw CSV and `pip freeze`. All observations, even slowdowns, must remain.
- Report per-OS/dataset **median of the four per-seed median percentage time savings**, and per-case min/max; avoid treating repeated within-run samples as independent experimental seeds. One shared runner per OS/dataset still cannot establish population variance or native-machine throughput.

## B. Separate numerical correctness stress

- Run a direct kernel A/B (original compiled Cython versus cache-enabled **with check-every-metric** verification) under **both Linux and Windows**.
- Graph: synthetic **undirected ring with one genuinely isolated labeled vertex**, i.e. valid degree-zero node, plus multiple classes and initially unlabeled nodes. Run both C-order and F-order `dominance` matrix views, including numerical case with **zero dominance update**.
- Fixed Cartesian grid: class counts **2, 3, 5, 10, 32**; `p_grd` **0.0, 0.5, 1.0**; `delta_v` **0.0, 1e-12, 0.1**; `deltap` **0.4, 1.0**; `stop_max` **0, 1, 2**; `dominance` layouts **C, F**; two seeds **311, 733**; caps **1, 11, 65**. This yields **1,620 scenarios per OS**, all preserving existing original particle behavior and RNG calls. Compare all mutable kernel buffers, predictions, and complete propagation metrics exactly, and ensure the debug cache scalar equals the independent full scan on **every actual check**.
- Save one raw CSV row per scenario, and record all scenarios, not only passing cases. A failure must abort CI with the exact condition.
- This is **not** a randomized numerical proof for all graphs (finite matrices, no NaNs, no arbitrary invalid neighbor lists); the adversarial zero/tiny steps, alternate memory layouts, isolated home and early-stop boundaries cover distinct risks missing from phase 1.

## Decision gate

- Any cross-platform full-state/stop-metadata or per-check potential mismatch **rejects the candidate** until explained; do not relax equality to a tolerance.
- If **both** OS stress grids pass and all **48 replication pairs** pass, consider implementation worth a **separate full review** only when median 20k whole-fit saving on Digits is **>=15% on each OS** and Breast Cancer median regression is **not worse than -5% on either OS**.
- Even a pass does not authorize merging: compare compiled candidate **default-disabled path** with untouched master on CI/hardware and consider unsupported/reused object state, memory allocation, and native sanitizer testing; quantify original default end-to-end speed independent of runner noise.
- Default flags continue `False` and **`master` remains unchanged**.
