# Prespecified multi-dataset lockout cohort: completed results

## Provenance, protocol integrity and CI

Protocol frozen in [`particle_lockout_cohort_protocol.md`](particle_lockout_cohort_protocol.md) **before** results were inspected. Experimental code: [`audit_lockout_predefined_cohort.py`](audit_lockout_predefined_cohort.py).

- Failed first run **41922179323**: an exact floating-point bound check rejected particle strengths just above 1 by roundoff; no cohort-wide conclusion was drawn.
- Correction `b512934`: **only** strength-bound validation (allow excess up to 1e-8 and record raw excess), with unchanged seeds, graph, endpoint definition and propagation.
- Completed GitHub Actions run **41928357244** (all four matrix jobs successful), artifact names `lockout-cohort-iris`, `lockout-cohort-wine`, `lockout-cohort-breast_cancer`, `lockout-cohort-digits`. PCC regression at the same commit **41928357046** passed.

Four datasets (Iris, Wine, Breast Cancer, Digits), seeds **101, 211, 307, 401, 503**, 10-NN, 2 labeled observations per class, 50k no-stop iterations, 1k checkpoints. An operational class lockout requires **five consecutive windows with zero accepted moves across every particle in that class**, valid visits in those windows, maximum class-particle strength <= 0.01, and zero currently winning outgoing neighbor entries. It is a descriptive state, not proof of mathematical irreversibility.

## Primary result

| Dataset | Dataset-seed runs with >=1 lockout / 5 | Affected classes / total classes over 5 runs | Total 1k windows with zero accepted moves across classes | Lowest class window acceptance fraction | Mean 50k accuracy on initially unlabeled nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iris | **0/5** | 0/15 | 0 | 76.55% | 87.78% |
| Wine | **0/5** | 0/15 | 0 | 48.55% | 84.30% |
| Breast Cancer | **0/5** | 0/10 | 0 | 80.35% | 58.34% |
| **Digits** | **4/5** | **5/50** | **136** | **0%** | 57.72% |
| **All** | **4/20** | **5/90** | **136** | — | — |

All **20/20** final initially unlabeled predictions matched independent uninstrumented same-seed runs. Across all class checkpoint rows the largest observed floating-point strength-bound excess was **5.218e-14**, well below 1e-8. These are numerical observations, not modifications to strength.

The contrast across datasets is noteworthy but **not a controlled comparison of class count alone**: dataset scale, graph topology, class count, class balance and labeled sample distribution all differ. The Digits dataset has 10 classes and two particles per class; the other datasets have two or three classes. The held-out cohort was predeclared, but it is small and has a fixed 50k horizon.

## Digits events

| Seed | Class | First 1k window with zero accepted moves (end) | First operational lockout (end) | Consecutive zero-acceptance windows through 50k | First checkpoint when maximum class-particle strength was exactly zero | Recovery until 50k |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 211 | 4 | 27k | **31k** | 24 | 28k | No |
| 211 | 9 | 29k | **33k** | 22 | 33k | No |
| 307 | 2 | 15k | **19k** | 36 | 16k | No |
| 401 | 4 | 35k | **39k** | 16 | 38k | No |
| 503 | 1 | 13k | **17k** | 38 | 16k | No |

At each recorded operational event, the class-particle maximum strength was **exactly 0.0** and the neighboring-entry win count was zero. All five class groups remained with zero accepted movements through 50k. Their classes still retained *nonzero dominance mass on other unlabeled nodes* at 50k (respectively about 234.5, 291.3, 187.5, 205.5 and 221.9 across the 1,777 initially unlabeled Digits nodes), so **zero particle mobility is not equivalent to zero global class dominance**. With strength exactly zero, visits to unlabeled nodes contribute no dominance transfer until some other effect permits a particle to regain strength or reach a winning destination. The observed interval 0–50k contained no such recovery.

The five selected Digits 50k accuracy values range by seed and are *not* causal estimates of the damage from lockout: other classes and predictions also evolve. Observed instance-level lockout is consistent with feedback involving strength loss, collision rejection and local graph accessibility, as seen in Wine seed 83. This is an algorithmic dynamic compatible with the original MATLAB/MEX movement/strength semantics, not evidence of a Python-only bug.

## What is established and what remains uncertain

**Established:** In this frozen 20-run cohort, operational lockout occurs frequently in the Digits runs (4/5) but not in the other 15 runs; five affected classes have zero accepted moves for 16–38 consecutive 1k windows. Observers do not change final predictions. The tight first-run strength check failed only because of floating-point-level overshoot.

**Unresolved:** Whether higher class count, weak per-class supervision, graph construction or local density chiefly explains the difference; whether zero-strength particle lockout is unavoidable for some states; and whether an intervention generally improves **out-of-sample** accuracy. A 50k cap does not establish convergence.

**Next controlled study:** Compare Digits with the same graph/seed protocol and two different **labeled examples per class** counts, preferably with nested labeled sets and novel seeds; control particle count and track class-level movement. Independently investigate graph neighbor count rather than changing both simultaneously. Keep these as experimental diagnostics, not production defaults. **Do not merge any stopping heuristic or particle intervention into `master`.**
