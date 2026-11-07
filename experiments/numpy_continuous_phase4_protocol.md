# Pre-registered phase 4: observer safety boundaries, memory, and high-frequency sampling

Recorded on experimental draft PR #182 **before phase-4 tests/results**. Reference baseline: completed phases 1–3, 128/128 strict paired comparisons, master `8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128`. No merge to master, no new model public API, no changes to particle movement, randomness, stopping rules, Numba or Cython.

## Questions and falsifiable expectations

1. **Callback failure after mutation:** a synthetic custom observer throws on a *visited* iteration-1 checkpoint (not iteration 0). It is explicitly **not transactional**: the exception must propagate; at least one live particle buffer and/or RNG position may already have changed, and no rollback of the global RNG is promised. At high-level `fit_predict`, previous Python model attributes should be restored by existing exception handling, but out-of-band buffer references cannot be treated as committed valid results. Distinguish this from the already tested safe initial iteration-0 rejection.
2. **Collector memory preflight:** the `max_bytes` budget represents **retained NumPy payload**, not transient allocations, Python object overhead, or process RSS. If the next retained row cannot fit, a `ValueError` must occur **before creating the row's copied state**, without changing `used_bytes` or `samples`. The finalization of an already visited milestone is metadata-only and must not count as another retained row. A failed capture must never increment the retained-byte tally.
3. **Dense sampling:** on unmodified canonical undirected union-10NN Iris and Digits, use exactly two labels per class, fresh fixed seeds `16007,16111`, NumPy sequential `max_iter=111`, `early_stop=False`, and **every** iteration 0…111, with explicit `max_observations=112` and `max_bytes=48 MiB`. Compare each observed fit with a separate unobserved fit from the same explicit seed: six full final arrays, full native stop metadata, global MT19937 and next 16 draws must match exactly. Require 112 physical snapshots at precisely iterations 0…111; no exclusions. Four OS×dataset jobs => **8 strict pairs / 16 fits**. Overhead is exploratory and must be reported, not filtered and not judged against the 6001-step threshold.
4. **Packaged contract:** `observer` is a private low-level opt-in in the shipped `pcc_numpy.pcc_propagate_numpy`, not a public `ParticleCompetitionAndCooperation.fit_predict` parameter. The helper under `experiments/` is source-tree diagnostics, not an installed library API. Do not promise arbitrary custom observers are read-only or safe. Evaluate wheel packaging separately from source-tree editable installs.

## Safety / decision gates

- Keep phase-4 test fixtures, documentation and workflows on the draft branch.
- If preflight guarding requires changes, modify **experimental collector only**, never PCC mathematics, RNG or stopping code.
- All dense sampling comparisons must have complete state, stopping and RNG parity and stay inside retained-payload budget; unexpected failures are recorded and investigated, not silently excluded.
- Distinguish retained bytes from peak process memory and opt-in overhead from default-off impact.
- Even if green, PR #182 remains **draft/unmerged**, pending an explicit API and risk decision.