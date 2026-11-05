# Timing-control amendment to predeclared cross-backend audit

Original protocol: [`canonical_production_backend_stop_protocol.md`](canonical_production_backend_stop_protocol.md); first completed run **42115280196** at `ad0ace3` (four jobs **passed**, with no-stop and production potential stopping). The original protocol warmed the Numba compiler by running a short **Iris** example before collecting *every dataset's* timings.

## Issue discovered only after seeing results

The first 100k Numba **Digits** no-stop timed call (seed 4001) took **1.2698 s**, whereas the second Numba Digits no-stop (seed 4093) took **0.1970 s**. These execution-time differences were inconsistent with the same fixed dataset, shape and cap. A likely explanation is *additional JIT compilation* on the first Digits run, since the Digits 10-class dominance state can use a different array layout/signature from the 3-class Iris warmup. A separate measurement artifact cannot be ruled out. Thus **the original across-backend timing figures must not be used as controlled speed ratios**; short Cython timings are also susceptible to noisy runner scheduling.

## Strict correction

Code commit `5846215` changes **only warm-up inputs**: compile Numba once before the measurements using the **same dataset as the CI matrix job**, a separate seed **29**, and just 25 no-stop steps. The warm-up remains outside the measured sections. The predeclared **tested** seeds **4001/4093**, all graphs, labels, stopping thresholds, per-case 100k horizon and backend selection stay **unchanged**. This is a measurement-control amendment, not post-hoc selection of favorable cases or stopping parameters. All original run artifacts remain available.

A new CI run from this code should be used for timing interpretation and should be compared to the first run to verify that no-stop accuracy, stop iterations and output agreement remained unchanged. Even with warmup, each walltime measurement is a single CI observation and cannot provide a reliable backend speedup distribution. **No algorithm or production default changes are justified** by this adjustment.
