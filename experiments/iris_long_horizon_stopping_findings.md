# Iris 500k diagnostic: production potential vs experimental label P10

**Protocol frozen before running:** [iris_long_horizon_stopping_protocol.md](iris_long_horizon_stopping_protocol.md). All code is experimental on draft PR #172, layered on draft PR #171. The production `master` and backend kernels were **not changed**.

## Provenance / test status

- GitHub Actions workflow [42146350603](https://github.com/fbreve/pypcc/actions/runs/42146350603): **3/3 Iris seed jobs green**. Per-seed artifacts, with raw CSV and installed Python environment:
  - **5003**: `iris-long-horizon-5003`, ID `13470781306`
  - **5209**: `iris-long-horizon-5209`, ID `13470841309`
  - **5303**: `iris-long-horizon-5303`, ID `13470342096`
- `PCC regression` [42146350585](https://github.com/fbreve/pypcc/actions/runs/42146350585): **4/4 green**, on Windows/Python 3.10 and 3.12, Linux/Python 3.10 and 3.12.
- **All 3/3** candidate NumPy vs production NumPy **no-stop 100k prediction-parity gates passed**.
- Production **undirected union-symmetrized 10-NN**, two labeled/class; original potential stopping (`es_chk=2000`) vs existing experimental label-stability P10 (1k checks), with independent 100k, 250k (NumPy) and 500k no-stop references. Every backend ran with its own seeded RNG stream. The 500k horizon is still a finite cap, **not proof of convergence**.

## Seed 5003: the pre-identified hard case (post-hoc selection)

| Backend / policy | Iterations | Accuracy | Delta vs own 500k | Disagreeing unlabeled labels vs own 500k |
| --- | ---: | ---: | ---: | ---: |
| NumPy no-stop 100k | 100,000 | 93.750% | 0.000 pp | 0 |
| NumPy no-stop 250k | 250,000 | 93.056% | -0.694 pp | 1 |
| NumPy no-stop 500k | 500,000 | 93.750% | 0.000 pp | 0 |
| **NumPy production potential** | **14,161** | **84.722%** | **-9.028 pp** | 15 |
| **NumPy experimental label P10** | **73,000** | **93.056%** | **-0.694 pp** | 1 |
| Numba no-stop 100k | 100,000 | 86.111% | +0.694 pp | 1 |
| Numba no-stop 500k | 500,000 | 85.417% | 0.000 pp | 0 |
| **Numba production potential** | **17,591** | **75.000%** | **-10.417 pp** | 15 |
| Cython no-stop 100k | 100,000 | 94.444% | +1.389 pp | 2 |
| Cython no-stop 500k | 500,000 | 93.056% | 0.000 pp | 0 |
| **Cython production potential** | **13,331** | **88.194%** | **-4.861 pp** | 9 |

**Key observation:** NumPy 100k and 500k labels were identical, but its 250k snapshot differed by one prediction. Thus two matching endpoints **do not show an unchanging trajectory**. The P10 candidate recovered +8.333 pp accuracy compared with the original NumPy potential stopping on this deliberately selected hard seed, at the cost of running 73k rather than 14,161 iterations. The P10 candidate differed from the 500k reference on just one initially unlabeled node.

## Fresh sensitivity seeds: 5209 and 5303

| Seed | NumPy no-stop 100k | NumPy no-stop 250k | NumPy no-stop 500k | Original potential (iterations; accuracy) | Label P10 (iterations; accuracy) |
| --- | ---: | ---: | ---: | --- | --- |
| **5209** | 80.556% | 80.556% | 80.556% | 23,261; **83.333%** | 19,000; **82.639%** |
| **5303** | 88.889% | 88.194% | 88.194% | 8,621; **95.833%** | 15,000; **95.833%** |

For 5209, production potential improved accuracy **+2.778 pp** vs 500k and P10 improved **+2.083 pp** (potential is 0.694 pp better, though it takes more iterations). P10 prediction agreement with the 500k reference was nevertheless **higher** (97.917%) than potential's (97.222%). For 5303, both stopping methods improved accuracy **+7.639 pp** against 500k, but neither matched the 500k predictions closely (90.972% agreement each). This warns against using final-reference prediction agreement alone as an accuracy metric.

In seed 5209, all three production no-stop backends reproduced identical unlabeled prediction accuracy at 100k and 500k (80.556%) and no label changes between those two snapshots. In seed 5303, the production 100k→500k reference drifted on one label in each backend.

## Descriptive NumPy averages across exactly these 3 seeds (NOT generalizable)

| Metric | Production potential | Experimental label P10 |
| --- | ---: | ---: |
| Mean steps saved vs 500k | 96.930% | 92.867% |
| Mean observed wall time saved vs 500k | 96.840% | 92.790% |
| Mean accuracy delta vs NumPy no-stop 500k | +0.463 pp | +3.009 pp |
| Worst accuracy loss vs NumPy no-stop 500k | -9.028 pp | -0.694 pp |
| Mean final-label agreement vs NumPy no-stop 500k | 92.593% | 96.065% |

These averages **must not be treated as an independent randomized validation**: seed 5003 was selected because the original potential criterion had failed strongly, and the two new seeds are a tiny cohort. The 500k reference is not an optimum or convergence guarantee, and single-run wall times are not calibrated benchmarks.

## Research interpretation and next experiment

1. **Potential stopping is fragile in one known hard Iris case:** the large earlier loss persists at 500k in all three production backends. This is a real fixed-horizon sensitivity, not just an artifact of choosing the 100k cap.
2. **P10 is a credible candidate** on this selected hard case, at higher step cost, but other seeds show that potential can be as good or better; no universal superiority is demonstrated.
3. **More iterations can lower accuracy or move labels away and later back**. The 5003 NumPy 100k/250k/500k snapshots show nonmonotonic endpoints; 5303's early-stop accuracy exceeds the 500k score. Stop quality must be judged against true labels and instability, not simply proximity to a later snapshot.
4. Next, instrument a long **no-stop trajectory** of the same hard case to identify the *timing* of temporary label flips and assess whether long apparent plateaus can precede changes. This must be observer-parity gated so callbacks cannot silently change the trajectory. Follow that with a separate, fully held-out multi-dataset validation before considering any change to the production stopping default.

**Decision:** do **not** merge any of these stopping or particle-dynamics experiments into `master`; retain original criterion in production.
