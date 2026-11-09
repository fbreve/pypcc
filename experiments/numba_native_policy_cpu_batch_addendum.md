# Prospective Windows CPU clock quantization addendum (before first measurement)

Frozen base protocol `experiments/numba_native_policy_cost_protocol.md` defined 8 actual `fit_predict` calls as a minimum technical unit and warned that Windows `process_time_ns` can quantize to ~15.625ms. With real Iris/Wine and only 1,000 original iterations, one eight-fit batch can be **shorter than one process CPU clock tick**, yielding a meaningless measured zero even with valid source.

**Before any runner, calibration, candidate source or native measurement**, specify a fixed *CPU timing aggregation depth*, independent of any observed CPU result:

- Every timed CPU/wall sample remains an integral number of 8-complete-fit units.
- 1,000 original native iterations: **8 such units = 64 consecutive complete original fits per arm per clock sample**.
- 10,000 original native iterations: **2 such units = 16 consecutive complete original fits per arm per clock sample**.
- Each of four counterbalanced original-versus-wrapped measurement pairs uses these fixed unit counts. All original/wrapped fit references, SHA and object identities are checked AFTER stopping clocks. Both host corpora must meet the same exact samples.
- The analysis of each graph/mask/policy/iteration cohort still retains exactly 4 paired samples; sample counts have **not** been selected post hoc from outcomes. Total genuine timed original native fits PER OS now **54 cohorts per cap ×4 pairs ×2 arms ×(64+16)=34,560**; 69,120 across both OS.
- Windows whole CPU batches are *more reliable*, but resolution remains finite; report original process CPU and wall together. Native-only source wall timings are obtained from the true unchanged backend on the wrapped arm. Do not pool hosts, treat technical repeats as new datasets, or claim causal branch cost from changed trajectories.
- This addition changes ONLY predeclared measurement duration/depth, not any PCC inputs, graphs, native code, RNG, mutation order, stop algorithm, hypothesis, or acceptance logic (the experiment has no optimization GO).

This pre-measurement addendum supersedes only the words “8 full fits per arm per batch” as a *clock accumulation depth* in the base protocol; the eight-fit logical units and all other frozen scientific controls remain unchanged.
