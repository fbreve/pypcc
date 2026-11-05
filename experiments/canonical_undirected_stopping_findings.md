# Canonical undirected PCC in-loop early stopping: first fixed-cohort results

## Provenance

The **prespecified** pilot protocol is [`canonical_undirected_stopping_protocol.md`](canonical_undirected_stopping_protocol.md), committed before running the new seeds. Script [`audit_canonical_undirected_stopping.py`](audit_canonical_undirected_stopping.py). GitHub Actions run **41965291725**, all four matrix jobs passed at commit `30e1932`; **PCC regression** run **41965291614** also passed. Artifacts:

| Dataset | GitHub artifact ID |
| --- | ---: |
| Iris | 13387611149 |
| Wine | 13387087734 |
| Breast Cancer | 13387586342 |
| Digits | 13387187623 |

All runs use **production `pcc_graph.build_knn_graph`** with union-symmetric 10-NN and the same two labeled nodes per class, seeds **2203, 2309, 2411**. The experimental propagation is sequential NumPy PCC. The reference is an **uninterrupted 50,000-iteration cap, not mathematical convergence**.

Rules were predeclared: compare no-stop reference to (a) `label_change` checked every 1,000 steps with patience 3; and (b) `dominance_label_hybrid` with labels unchanged *and* P95 dominance difference <=0.001, same interval and patience.

## Results

| Dataset | Label-only early stops /3 | Mean saved iterations | Mean saved elapsed time | Label-only average agreement with 50k | Label-only mean accuracy difference from 50k | Hybrid early stops /3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Iris | 3 | 85.33% | 85.21% | 98.61% | 0.00 pp | 0 |
| Wine | 3 | 78.67% | 78.45% | 97.87% | -0.58 pp | 0 |
| Breast Cancer | 2 | 35.33% | 35.29% | 98.82% | -0.35 pp | 0 |
| Digits | 3 | 20.00% | 19.73% | 99.74% | -0.17 pp | 0 |
| **All** | **11/12** | **54.83%** | **54.67%** | **98.76%** | **-0.28 pp** | **0/12** |

The **worst accuracy delta** for label-only was **-1.744 pp** (Wine seed 2411, stop at 9k). The smallest agreement with 50k was **95.93%** for that same case. The largest positive delta was **+0.694 pp** (Iris seed 2203) — stopping earlier can improve or worsen empirical accuracy; the 50k reference is not a gold standard.

Selected label-only trajectories:
- Iris stopped at **7k,7k,8k** for seeds 2203/2309/2411.
- Wine stopped at **17k,6k,9k**; seed 2411 lost 1.744 pp.
- Breast Cancer stopped at **50k(no stop),25k,22k**; two triggers lost **0.531 pp** apiece.
- Digits stopped at **39k,41k,40k**, losing **0.394 pp**, **0.169 pp**, and gaining **0.056 pp**, respectively.

The dominance+label hybrid at tolerance **0.001** **never stopped before 50k**, and took **~1.0% more time on average** than no-stop from monitoring overhead; it provided no computational savings in this cohort. This is evidence against **this exact predeclared threshold**, not every possible hybrid tolerance.

## Interpretation and next experiment

These results are a **reasonable efficiency–fidelity tradeoff**, not proof of convergence: 11/12 label-only cases save about half the iterations but make different final predictions from the 50k references (98.76% average agreement; worst 95.93%). A more conservative patience may reduce premature stopping, particularly on Wine, but could erase much of the speed gain. No threshold tuning on the observed 12 seeds. **Next test should preregister new seeds**, compare fixed patience 3, 6 and 10 with 100k no-stop reference, and measure early stops, agreement, actual elapsed time and accuracy. Continue to avoid production graph changes or particle resets. Do not merge stopping criteria into `master` based on this pilot.
