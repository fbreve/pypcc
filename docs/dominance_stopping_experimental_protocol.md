# Experimental dominance-change stopping: integration safeguards

The existing mean-maximum-dominance stopping rule remains the default. The
`DominanceChangeStopTracker` introduced in PR #163 is currently an independent
observer, not an enabled stopping mode. Do not silently change existing
`fit_predict` or backend behavior.

## Existing empirical evidence

Previous PyPCC stopping experiments reported mean iteration savings of 68.58%
with the original rule, accompanied by a mean accuracy loss of 0.167 percentage
points and a worst observed loss of 12.38 points. In the 20-dataset/200-trajectory
study, all 172 observed early stops had subsequent label changes, and 168 of
those trajectories later returned to earlier labels. These observations show
that neither mean maximum dominance nor full-matrix stability alone establishes
optimal predictive accuracy. Increasing patience reduced average losses in
that experiment but did not justify changing the default.

A separate 3,600-fit campaign had 86.67% early stops and 13.33% censored
runs. A fixed 10,000-iteration cap censored every run in one Digits experiment;
therefore cap-limited results must not be presented as uncensored stopping times.

## Requirements before enabling the optional criterion

1. Keep the existing public and backend call signatures backward compatible.
   Search callers in `experiments/` before changing any backend signature.
2. Preserve the original check schedule: after iterations 1, 11, 21, etc.
   A first dominance-change observation establishes the baseline; it must not
   count as a stalled check.
3. Validate tolerance and mode before mutating model state or consuming RNG.
   Never reset RNG between checks. Do not change the default RNG draw count.
4. Apply the same semantics to NumPy, Numba and Cython. Account for the Numba
   compiled loop and Cython `nogil` implementation; avoid a Python callback in
   either hot loop.
5. Define convergence as the maximum absolute entrywise difference between
   successive dominance snapshots. Stop after more than `stop_max` consecutive
   changes at or below tolerance; reset patience on a larger change.
6. Benchmark snapshot memory O(nodes * classes) and comparison cost every ten
   iterations. The legacy mode must not allocate extra snapshots.
7. Report effective iterations, stop reason, cap-hit rate, classification
   accuracy, and any later label changes against a sufficiently long reference.
   A smaller number of iterations alone is not evidence of better stopping.
8. Retain the original frozen stopping-candidate, convergence-dynamics,
   patience and adaptive-stopping protocols. Add a separate comparison rather
   than modifying historical experiment definitions.

## Suggested implementation sequence

- Add tests for tolerance validation, patience, stable and changing matrices,
  and state parity when the experimental option is not requested.
- Integrate the optional mode in the NumPy path and verify against the
  deterministic premature-stop example from PR #161.
- Integrate equivalent logic in the Numba and Cython compiled loops without
  changing random-number generation, then run cross-backend CI.
- Run new, separately named experiments comparing original and experimental
  modes at identical seeds and reference horizons. Do not switch the default
  based on a single synthetic example.
