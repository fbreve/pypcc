# Results: read-only 500k label trajectories on canonical Iris

**Preregistered protocol:** [iris_500k_label_trace_protocol.md](iris_500k_label_trace_protocol.md). All findings are diagnostic of two previously observed Iris seeds, **not independent held-out validation**. The experimental branch/PR #173 is based on draft #172; there are no modifications to production PCC, `master`, stopping defaults or particle dynamics.

## Provenance and exact-output safety gate

- CI workflow [42147309879](https://github.com/fbreve/pypcc/actions/runs/42147309879): **2/2 jobs green**.
- Raw **500 samples per seed** (every 1,000 iterations through 500k), strict per-seed summary CSV, and installed Python version/environment:
  - Iris **5003**: artifact `iris-500k-trace-5003` ID **13470588056**
  - Iris **5303**: artifact `iris-500k-trace-5303` ID **13469744853**
- **Both 2/2 three-way parity gates PASS:** experimental candidate NumPy with read-only observer at 500k == same candidate without observer at 500k == actual production NumPy without stopping at 500k. Final checkpoint argmax also matches all three. This rules out an *observed final-prediction change* caused by the instrumentation in this cohort.
- Production graph built through production `build_knn_graph`, symmetric undirected union 10-NN, two labeled nodes per class. Same explicit seeds and `p_grd=0.5` default; **not** the earlier noncanonical directed adjacency.
- All counts of events refer to **differences between 1,000-iteration snapshots**. They are lower bounds on actual changes: within-window switches and reversals can go unnoticed. Longest no-change sequences are *observed sample-interval plateaus*, not mathematical immobility.

## Quantitative findings

| Diagnostic | Seed 5003 (previous hard case) | Seed 5303 (previous fresh case) |
| --- | ---: | ---: |
| Unlabeled nodes | 144 | 144 |
| No-stop accuracy at 100k | 93.750% | 88.889% |
| No-stop accuracy at 250k | 93.056% | 88.194% |
| No-stop accuracy at 500k | 93.750% | 88.194% |
| Snapshot label disagreements: 100k vs 500k | **0** | **1** |
| Snapshot label disagreements: 100k vs 250k | **1** | **1** |
| **Observed adjacent-snapshot label flips after 100k** | **60** | **39** |
| Distinct nodes ever differing from their 100k label after 100k | **6** | **2** |
| Nodes that differ at some post-100k snapshot but return to their 100k label by 500k | **6** | **1** |
| Longest observed no-change sequence before a later flip | **41,000** iterations | **85,000** iterations |
| P10 candidate stop iteration | 73,000 | 15,000 |
| P10 accuracy | 93.056% | 95.833% |
| P10 delta vs 500k no-stop | -0.694 pp | +7.639 pp |
| P10 disagreements with 500k final labels | 1 | 13 |
| Observed adjacent-snapshot flips after P10's stop | **61** | **61** |
| Distinct nodes ever differing from P10's stopped predictions at later snapshots | **6** | **15** |

The longest quiet sampled sequence on seed **5003** was 41 empty 1k transitions before a new observed label flip at **iteration 201,000**. For seed **5303**, 85 empty sampled transitions preceded a new flip at **429,000**. Neither proves continuous inactivity between those samples.

## What this resolves

1. **Same endpoints do not imply stable trajectories.** Iris 5003 had **identical 100k and 500k labels**, yet six different unlabeled nodes changed in at least one observed post-100k sample, and all six returned to their 100k labels by the endpoint. The earlier 250k snapshot showed just one difference and therefore understated transient dynamics.
2. **A 10-check label plateau does not imply convergence.** P10 stopped at 73k/15k, followed by at least 61 further adjacent-sample label changes in each uninterrupted reference. Both long quiet intervals (41k, 85k) also preceded further sampled changes.
3. **Predictive quality can be nonmonotonic.** For 5303, P10 produced **+7.639 pp higher accuracy** than the 500k reference, while its labels differed from the 500k endpoint on 13/144 unlabeled nodes. Whether an early stop is beneficial must be assessed from true-label performance and computational cost, not monotone proximity to an arbitrary cap.
4. **Read-only observer safety gate works on these runs.** Exactly matching 500k final predictions for candidate/observer/production makes these sampled changes credible for the tested seeds and environment, without proving every intermediate state is bitwise identical.
5. **Do not infer prevalence or tune P10 post hoc:** two selected seeds, 1k snapshot aliasing, no uncertainty bounds, and no convergence certificate. An optional more cautious stopping criterion should require a future *separate*, preregistered multi-dataset held-out comparison of classification quality, runtime, label drift and late reversals. There is **no current justification for a default change**.

**Decision:** leave `master`, production stopping, Numba/Cython dynamics and RNG behavior untouched. Keep PR #173 as a separate draft diagnostic.
