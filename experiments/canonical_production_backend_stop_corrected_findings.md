# Corrected production-backend stopping comparison (2026-11-05)

**Experimental only; do not merge as a production stopping change.** This documents a fixed predeclared cohort and a timing-control correction. The `master` branch is unaffected.

## Inputs and provenance

- Protocol: [canonical_production_backend_stop_protocol.md](canonical_production_backend_stop_protocol.md).
- Timing correction and rationale: [canonical_production_backend_stop_timing_amendment.md](canonical_production_backend_stop_timing_amendment.md).
- Initial workflow: [42115280196](https://github.com/fbreve/pypcc/actions/runs/42115280196).
- Corrected workflow: [42116736566](https://github.com/fbreve/pypcc/actions/runs/42116736566); all 4 matrix jobs passed. Corrected artifact IDs: Iris `13457597385`, Wine `13456727589`, Breast Cancer `13456852499`, Digits `13456168599`.
- Four datasets (Iris, Wine, Breast Cancer, Digits); two predeclared seeds (4001, 4093); production union-symmetrized 10-NN, two labels/class; 100,000 iterations maximum; three explicitly selected production backends. Each stop run is compared with its *own backend's* no-stop reference.

## Invariance check

The original and corrected CSVs contain the same 48 dataset/seed/backend/policy records. **Every non-timing column matches exactly, including predictions-derived accuracies, stop iterations, stop flags, agreements and graph statistics.** Only seconds and derived wall-time fields differ. This supports that the corrected warmup changed measurement conditions, not algorithms or stopping outcomes.

The first Digits/Numba no-stop timed run dropped from `1.269796 s` (initial run, affected by warmup) to `0.157076 s` (corrected); the second from `0.196985 s` to `0.157217 s`. Single CI observations are not a stable performance benchmark.

## Corrected descriptive results

| Backend | Stops / 8 | Mean wall time saved vs own no-stop | Mean iterations saved | Mean label agreement vs own 100k no-stop | Mean accuracy change (pp) | Worst loss (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy | 6 | 52.59% | 53.66% | 99.17% | -0.291 | -0.694 |
| Numba | 6 | 39.66% | 48.80% | 99.19% | +0.133 | -0.581 |
| Cython | 6 | 32.63% | 50.95% | 99.25% | +0.153 | -0.581 |

All three fail to stop early on both Digits cases. Because checking potential still takes work, its mean elapsed-time overhead on these two Digits cases is approximately 0.85% for NumPy, 21.51% for Numba, and 51.06% for Cython. This is *measured relative to the same backend's unmonitored 100k cap* and is especially noisy for very fast Cython runs.

No-stop outputs **are not exactly identical across all three backends in any of the eight cases**. Minimum pairwise agreement between no-stop predictions ranges from 93.75% (Digits seed 4001) to 99.42% (Wine seeds). Do not treat the shared numeric seed as evidence of matched trajectories.

## RNG source audit and follow-up

Code inspection identifies distinct random-number consumption paths in the sequential production backends:

- `pcc_numpy.py`: legacy `np.random.seed(seed)`, then scalar `np.random.random()`; random-walk index via `int(np.random.random() * degree)`.
- `pcc_numba.py`: Numba-local `np.random.seed(seed)`; random-walk index via `np.random.randint(degree)` after the greedy decision draw.
- `pcc_step.pyx`: private `xorshift32` stream, random-walk index via `xorshift32(state) % degree`.

Even if initial NumPy and Numba uniform draws coincide, their different *index-selection APIs* can desynchronize subsequent random draws. Cython uses another generator altogether. Identical seed integers therefore do not provide common-random-number (CRN) control. This is an explanation for possible trajectory differences, **not proof that all observed prediction differences are solely RNG-related**. A separate isolated sampling audit is appropriate before attempting invasive random-tape injection.

## Limits and decision

Two seeds/dataset, capped references, and individual CI timings do not establish convergence, cross-backend statistical equivalence, or a superior default stopping rule. Keep production and `master` unchanged. Audit RNG stream/index selection first, then predeclare a fair larger-seed comparison only if needed.
