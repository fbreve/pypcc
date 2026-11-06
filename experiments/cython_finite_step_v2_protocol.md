# Frozen V2: native finite-input scan for public Cython single-step calls

**Registered before implementing V2 or collecting its timing results.** The original V1 fix in draft PR #180 reused Python/NumPy `validate_finite` on every `pcc_step` invocation. It passed 96 independently built master/candidate 200k propagation fits and 48 exact state/stop parity gates, and all four general regression CI jobs. However, the separately precommitted **5,000-call public single-step** benchmark showed **+30.77%, +40.36%, +43.46%** slowdown on three fresh Digits/Linux seeds, clearly failing its **20% median slowdown limit**. Therefore V1 must not merge as implemented, regardless of its correct NaN/Inf detection.

## Exact, narrowly scoped V2 design

- Preserve the three already validated **active-only** preflight checks in `pcc_propagate` (once per full simulation, where measured overhead is small).
- In **`pcc_step` only**, replace the three separate `validate_finite(...np.asarray(...))` traversals with **one native Cython C-loop helper** returning an error code: first active `part_strength[:n_particles]`, then active `dominance[:n_nodes,:c]` (Fortran-friendly order when appropriate), then `dist_weights[:256]`.
- Check IEEE NaN/+Inf/-Inf with native `libc.math.isfinite`; the C loop must **only read state** and return a code before any particle mutation. Python raises the same named `ValueError` outside the helper. Bounds and invalid-shape checks continue to occur before entering the helper; no other kernel dynamics, source loops, RNG, stopping metrics or cached global potential code may be changed.
- Preserve legal NaN/Inf in **unused buffer padding** and overwritten scratch arrays. Guard both C-contiguous and F-contiguous dominance memory views, large class counts, and the weights 255/256 boundary.
- Keep the existing comprehensive direct step/propagation invalid-input regression and standard PyPCC CI. Exact full-state/stop parity against separately compiled original master remains mandatory.

## Independent performance holdout

- Same Windows and Linux, canonical undirected Breast Cancer (2 classes)/Digits (10 classes), 2 labeled/class.
- **Fresh seeds 10103, 10211, 10301**, not those used to select V2. **5,000 public `pcc_step` calls per independently compiled original-master/candidate process**; same scalar parameters, native source identity and final seven-state-array parity.
- **24 timed native processes and 12 exact original-vs-V2 comparisons.** All results included; no warmup/outlier exclusions. Counterbalance master/candidate order by seed.
- **Keep the original frozen ceiling: <=+20% median per OS×dataset**. If even native scanning fails it, reject/redesign the direct single-step fix; no performance threshold relaxation or default opt-out workaround without a separate protocol.
- Original 200k-step propagation preflight results remain historical V1 evidence; a new #180 CI run after the V2 source change must pass regression and full compiled master/candidate parity too.

**Decision:** preserve the failed V1 observations permanently, retain the candidate in draft until both correctness and independent V2 direct-step performance are satisfactory. Do not merge to `master` or close PR #36 until the clean replacement is complete.
