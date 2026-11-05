# Digits nested-label lockout pilot: completed 2 vs 5 labels/class

## Provenance and integrity

Pre-registered protocol: [`digits_nested_label_lockout_protocol.md`](digits_nested_label_lockout_protocol.md), recorded before experiment. GitHub Actions run **41929685810**, artifact `digits-nested-label-2` (ID **13372081184**) and `digits-nested-label-5` (ID **13372440220**); commit `481c689`. Both matrix jobs and the PCC regression workflow **41929685748** completed successfully.

Same StandardScaler+10-NN graph for each pair; labels per class **2** versus **5**, with the former a subset of the latter; Digits seeds **601, 701, 809**; no early stop, 50k iterations, observations every 1k. An operational class lockout requires five consecutive nonoverlapping 1k windows with **zero accepted moves** for all class particles, valid destination visits, maximum class-particle strength <= 0.01, and no currently winning outgoing neighbors. A single zero-move window is **not** an operational lockout.

| Statistic | 2 labeled/class | 5 labeled/class |
| --- | ---: | ---: |
| Dataset-seed cases with >=1 lockout / 3 | **2/3** | **2/3** |
| Affected classes / 30 | **3** | **2** |
| Total zero-acceptance 1k class-windows | **91** | **85** |
| Classes with any zero-acceptance window | **3** | **3** |
| Mean 50k accuracy over initially unlabeled nodes | **57.19%** | **58.35%** |
| Same-seed final predictions matching uninstrumented control | **3/3** | **3/3** |
| Largest observed strength bound excess | **3.55e-14** | **9.41e-14** |

### Affected class trajectories

| Seed | Class | Lockout, 2 labels/class | Lockout, 5 labels/class | Notes |
| ---: | ---: | ---: | ---: | --- |
| 601 | — | None | None | No zero-movement windows at all |
| 701 | 2 | First event **12k**; 43 zero-move windows, no recovery | First event **12k**; 43 zero-move windows, no recovery | Persists unchanged in this paired condition |
| 701 | 4 | None | **No qualifying event**; one zero-move window at 50k | Current max strength at 50k ~6.89e-193; no winning neighbors — the isolated final zero window alone does not meet the five-window definition |
| 809 | 4 | First event **39k**; 16 zero-move windows, no recovery | First event **14k**; 41 zero-move windows, no recovery | Earlier lockout **with more** labeled examples |
| 809 | 9 | First event **23k**; 32 zero-move windows, no recovery | None; zero zero-move windows | This class appears protected in this paired condition |

All five qualifying class events across both arms had no subsequent recovered movements before 50k. Final mass may remain nonzero even for classes whose particles stop moving. The original 20-run dataset cohort (with **other** seeds) showed Digits lockout in 4/5 two-label cases, so observing 2/3 here is not a direct change in prevalence estimate but a separate small confirmation.

### Accuracy and inference limits

| Seed | Accuracy 2 labels/class | Accuracy 5 labels/class | Unpaired difference (5−2), percentage points |
| ---: | ---: | ---: | ---: |
| 601 | 52.28% | 55.58% | +3.30 |
| 701 | 61.34% | 63.59% | +2.26 |
| 809 | 57.96% | 55.87% | −2.10 |

Because the label sets are nested, the accuracy denominators differ (1,777 vs 1,747 initially unlabeled nodes in a dataset of 1,797 observations). Thus these deltas are **descriptive**, not matched-set accuracy differences. With only three seeds and changes to **both the number of labeled examples and the number of particles**, the study cannot establish a general effect or isolate which mechanism matters. It does refute the simplistic assumption that five labels per class **always** prevents lockout.

## Research recommendation

Do not implement a new default particle-rescue rule or claim more labels reliably reduce lockout. Before adding more seed sweeps, perform a **non-interventional graph/trajectory audit** of the strongly contrasting paired examples (Digits seed 701, class 2; seed 809, classes 4 and 9). Check labeled-home neighborhoods, class-to-class graph locality, particle positions/strength and winning neighbors at event onset. This is hypothesis-generating analysis, not further tuning on the same examples. Any proposed algorithm variant should be evaluated later on a separately held-out cohort with an explicit accuracy criterion and compute-cost measurement.

Experimental stop criteria remain unfit for `master`; production PCC remains unmodified.
