# Canonical PCC production stopping vs label-stability: five labels per class

## Predeclared provenance and integrity

Protocol: [`canonical_production_stop_five_label_protocol.md`](canonical_production_stop_five_label_protocol.md), frozen before running this cohort. Code: [`audit_canonical_production_stop_five_label.py`](audit_canonical_production_stop_five_label.py). GitHub Actions run **42019693773**: all **4/4 matrix jobs passed** at commit `18b6902`, as did **PCC regression 42019693687**.

| Dataset | Artifact name | Artifact ID |
| --- | --- | ---: |
| Iris | `five-label-production-stop-iris` | 13411243207 |
| Wine | `five-label-production-stop-wine` | 13410644623 |
| Breast Cancer | `five-label-production-stop-breast_cancer` | 13411707068 |
| Digits | `five-label-production-stop-digits` | 13411149723 |

All **12** dataset-seed cases used previously unused seeds **3701, 3803, 3911**, **five labeled observations/class**, production `pcc_graph.build_knn_graph` on standardized data with **k=10, union-symmetrized undirected graphs**, and experimental sequential NumPy for label stability. The *existing* production potential criterion used `ParticleCompetitionAndCooperation(impl="numpy")`, `set_graph(..., k_nn=10)`, default `es_chk=2000` and `stop_criterion="potential"`.

For every dataset-seed, **candidate and production no-stop 100k predictions matched exactly (12/12)**. This is final-label parity, not proof of identical intermediate states. Every initial label set was held fixed across policies within a case. The 100k production no-stop run was used as a *fixed reference horizon*, not a convergence claim. One wall-clock timing per condition: speed measurements are exploratory, not robust benchmarking.

## Summary results

| Measure | Production potential stop | Experimental label-change P10 |
| --- | ---: | ---: |
| Early stops | **12/12** | **9/12** |
| Mean actual wall-time savings vs production no-stop | **72.45%** | **52.78%** |
| Mean iteration savings | **72.71%** | **52.92%** |
| Mean final-prediction agreement with 100k reference | **99.09%** | **99.62%** |
| Worst single-case prediction agreement | **96.93%** | **98.52%** |
| Mean accuracy delta vs 100k | **+0.263 pp** | **+0.015 pp** |
| Worst accuracy loss vs 100k | **−1.073 pp** | **−0.613 pp** |
| Largest accuracy gain vs 100k | **+2.504 pp** | **+0.741 pp** |

**Mean signed accuracy differences are not fidelity measures**: improvements and degradations offset, and the 100k reference need not be intrinsically best. Report agreement, case-level loss, and runtime together.

### Dataset breakdown (three seeds per dataset)

| Dataset | Production potential early stops; mean wall saving; mean 100k agreement | Label P10 early stops; mean wall saving; mean 100k agreement |
| --- | --- | --- |
| Iris | **3/3**, **93.23%**, **99.26%** | **3/3**, **79.03%**, **99.26%** |
| Wine | **3/3**, **93.11%**, **98.98%** | **3/3**, **78.36%**, **99.39%** |
| Breast Cancer | **3/3**, **64.74%**, **98.21%** | **0/3**, **0.41%** measured (no early stops), **100%** |
| Digits | **3/3**, **38.75%**, **99.92%** | **3/3**, **53.34%**, **99.85%** |

The label P10 criterion **stopped in all three Digits cases** earlier than potential on average (mean 46k versus 60,841 iterations), but on Digits the production potential stopping achieved *better agreement* with the 100k reference. For Breast Cancer, potential stopped all 3, and P10 none: P10's perfect agreement there reflects **running to cap**, not an effective conservative early stop.

### Individual observations

- **Breast Cancer seed 3701:** production potential stopped at **25,051** iterations, with **+2.504 pp** measured accuracy against the 100k reference but only **97.14%** prediction agreement. A positive accuracy delta does not imply convergence.
- **Breast Cancer seed 3803:** production potential stopped at **35,371**, losing **1.073 pp** against 100k, whereas P10 reached full cap; illustrates potential's fidelity risk in a specific case.
- **Iris seeds 3701/3803/3911:** potential stopped at **3,431 / 10,501 / 6,051** versus P10 **16k / 30k / 16k**, yielding substantial compute savings from the current default. Accuracy did not consistently favor one rule.
- **Wine seed 3803:** potential finished with *exact 100k predictions* after **7,231** iterations; P10 stopped at 21k, lost **0.613 pp** and differed on one initially unlabeled point. More iterations are not necessarily more accurate.
- **Digits seed 3701:** potential 65,681 steps, P10 47k; corresponding accuracy deltas were **−0.057 / −0.114 pp**.
- **Digits seed 3803:** potential 35,291, P10 41k; both had **+0.172 pp** change and identical stopped predictions.
- **Digits seed 3911:** potential 81,551, P10 50k; P10's prediction agreement was 99.828% against potential 100%, although both had the same overall accuracy.

## Interpretation and decision

These new cases reinforce the **material competitiveness** of the production potential criterion already in PyPCC. Label P10 provides higher *average* agreement with a fixed no-stop cap, and a smaller worst observed negative accuracy delta, but uses much more CPU overall and never stops on Breast Cancer. A claim of universal superiority is not warranted. Thus **do not replace the default** based on these studies. Previous two-label cohort had potential stop 8/12 and P10 stop 8/12; this five-label cohort had 12/12 and 9/12 respectively. These are **different seed cohorts and labels**, not a paired causal comparison of supervision level.

**Next priority:** before implementing label-stability stopping in accelerated production backends, audit existing *production* NumPy, Numba and Cython results (no-stop and built-in potential stopping) on canonical graphs using the same seeds, labels and k calibration. This tests whether the baseline behavior and output agreement persist in the backends that matter operationally. Then assess whether an **optional** label criterion warrants implementation against that baseline and in independent larger data. An additional 100k NumPy-only seed sweep alone is unlikely to change the core conclusion. Do not modify `master`, graph construction, particle strength or rescue behavior.
