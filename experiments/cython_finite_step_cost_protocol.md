# Additional preregistered gate: repeated public Cython single-step overhead

This follow-up is frozen **before running the single-step benchmark**. The earlier clean native propagation study passed 96 exact compiled calls / 48 matching full-state pairs with median per-stratum slowdown no worse than 5.58%. That study **did not measure the public `pcc_step.pcc_step`** API, where finite-input scanning occurs on every separate call rather than once for an entire `pcc_propagate` simulation.

## Fixed performance and correctness design

- Independent, separately compiled original `master` (`c449444d99eac16d89f4a2492d44795e490d874b`) and clean candidate binaries; Python 3.12, same NumPy/Cython versions per OS, different checkout directories and subprocess interpreters, actual native module path and hash required.
- Windows/Linux × production canonical undirected union-kNN=10 **Breast Cancer** (2 classes) and **Digits** (10 classes), two labeled nodes/class.
- Fresh seeds **9803, 9901, 10007** (unseen in the earlier propagation study); for each seed and variant, call the genuine public `pcc_step.pcc_step` **5,000 times** on the same live arrays (sequential mutations), with the standard production `p_grd=0.5, delta_v=0.1, deltap=1, dexp=2`, explicit finite 257-entry distance weights and reusable scratch vectors.
- Time the 5,000 Python→Cython calls as a **single wall-clock interval** after building graph/buffers; graph prep, process startup, compilation, imports and serialization are not timed. Each array starts from an identical deterministic copy on a separate process.
- Strictly compare **final predictions, positions, particle classes, strengths, distances, dominance, ownership**, and **array shapes/dtypes** with `np.array_equal` between original and candidate. If any element differs, fail before interpreting time; no tolerances.
- **2 OS ×2 datasets ×3 seeds ×2 independently built variants = 24 timed processes, 12 exact state pairs**. Keep every raw observation, including negative effects.
- Report per-OS/dataset per-seed percentage change `100*(T_candidate/T_master - 1)`, median and range, with baseline absolute duration in milliseconds and exact loaded native SHA-256.
- Frozen exploratory usability criterion: **median slowdown no more than +20%** in any OS×dataset stratum. If the criterion fails, keep the fix in draft and **do not merge** until the direct-step overhead has been reduced or a well-justified API control/alternative is established; do not silently exempt the single-step entry point. This constraint is about a public kernel API, not overall `fit_predict` performance.

This experiment does **not** change the input validation algorithm, previously frozen seeds, earlier propagation results or the merged Cython row-max stopping cache. `master` remains untouched until successful validation and review.
