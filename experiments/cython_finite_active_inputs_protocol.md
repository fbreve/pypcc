# Preregistered validation: finite active probability inputs in production Cython

**Frozen before changes and CI results.** Existing draft [PR #36](https://github.com/fbreve/pypcc/pull/36) proposed checking nonfinite values in three probability inputs, but its branch predates the now-merged Cython stopping cache [PR #178](https://github.com/fbreve/pypcc/pull/178) and cannot be merged directly. This is a **clean branch from current master**, without inherited experimental stopping code. No production merge without independent validation.

## Exact change

- At the Python/Cython boundary of **both** native `pcc_step.pcc_propagate` and single-step `pcc_step.pcc_step`, after existing shape/scalar validation but **before any particle mutation, cache initialization or `nogil` step**, reject NaN/Inf in:
  - `part_strength[:len(part_curnode)]`;
  - `dominance[:neib_list.shape[0], :c]`;
  - `dist_weights[:256]`.
- Reuse the existing `pcc_validation.validate_finite` helper; no edits to core particle motion, the two specialized cached/noncached loops, RNG, potential metric, stop checks, or public model API.
- Preserve compatibility with legal overallocated numeric buffers: **do not reject NaNs/Infs exclusively in unused extra rows/columns/strengths or weight index 256+**, or contaminated scratch arrays overwritten before use. The Cython kernel only reads the active elements above.
- Calling `pcc_propagate` with `max_iter` valid, `early_stop=False` must also reject nonfinite active inputs, not only the new stopping cache path.

## Frozen correctness checks

1. Direct **both entry points**, starting from a completely legal, reproducible undirected graph with labeled and unlabeled nodes, Cython-typed buffers, and seed 101; inject NaN/+Inf/−Inf separately into active particle strength, node dominance and one of the first 256 weights. Require **ValueError identifying the affected buffer** before any mutation of ALL caller-supplied arrays. Preserve sentinel-filled unused padding with NaN/Inf as a *successful* case and leave scratch buffer sentinels unconstrained if the kernel overwrites them. Confirm `pcc_step` optional `dist_weights=None` remains backward compatible.
2. Across **canonical actual production undirected union-10-NN Iris, Wine, Breast Cancer, Digits**, 2 labeled/class, fresh seeds **9503, 9601, 9709**, compare `master` and candidate on ordinary `impl='cython'` `fit_predict` with default early stopping and without stopping; require exact predictions, particle state, distances, node dominance, ownership and (where available) all stop metadata. Cache remains **enabled by default** as merged in #178.
3. **Independently compiled native master/candidate binaries** on Windows/Linux for Breast Cancer (2 classes) and Digits (10 classes), same fixed seeds. Both ordinary stopping monitor forced to complete the same cap and `early_stop=False` (two regimes), identical native inputs. Compare all final state arrays **bitwise** and metadata. Record actual loaded native paths/hashes and SHA-1 commit revisions, and installed package versions; no use of both binaries inside a single interpreter.
4. **Timed preflight overhead** measured with at least **200,000 native propagation steps** per matched case, so short 3–5 ms noise doesn't dominate, and raw timings retained by seed and OS. Report seed-level paired elapsed-time ratios, median and min/max; do not infer universal speedups. For `pcc_step` direct-call overhead, report timings separately if measured; it is not equivalent to full propagation. A limited exploration gate is candidate slowdown no worse than **8% median per OS×dataset×regime**; any hard correctness failure forbids integration regardless of timing. If CI-runner noise produces mixed signs, gather repeated longer windows rather than silently exclude.
5. Existing full **PCC regression** on Windows/Linux Python 3.10 and 3.12. No new always-on runtime debugging, external dependencies or staged master changes.

## Decision

If exact state/stop parity and invalid-input atomic rejection pass, and a reasonable overhead bound is demonstrated, the clean PR may be considered for merging *only after review*. The original #36 should be closed unmerged **only if** the clean replacement is validated and linked, otherwise preserve it as active input-validation work. No direct merge of 2016–2026 experimental branch history.
