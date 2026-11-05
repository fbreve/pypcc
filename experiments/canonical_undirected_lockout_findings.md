# Canonical production undirected-kNN PCC: class lockout frequency and 25k–50k drift

## Provenance

**Prospective protocol:** [`canonical_undirected_lockout_protocol.md`](canonical_undirected_lockout_protocol.md), commit `73083eb`, recorded **before** cohort execution. Script: [`audit_canonical_undirected_lockout.py`](audit_canonical_undirected_lockout.py); workflow: [`canonical-undirected-lockout.yml`](../.github/workflows/canonical-undirected-lockout.yml).

**Completed successful run:** GitHub Actions **41964129094**, commit `90da1ad`, all four matrix dataset jobs passed. **PCC regression run 41964129148 passed.** Per-dataset data (cases, classes, 1k windows) are in:

| Dataset | Artifact ID |
| --- | ---: |
| Iris | 13386289589 |
| Wine | 13387330663 |
| Breast Cancer | 13386263951 |
| Digits | 13387216224 |

This study **does not reuse the directed** `experiments.compare_dominance_stopping.prepare` path. Instead it calls production [`pcc_graph.build_knn_graph`](../pcc_graph.py) on StandardScaler-transformed sklearn datasets. Every adjacency was verified symmetric, without self-loops or duplicate active neighbors. The seeded two-per-class initial label selection is unchanged. Every instrumented no-stop run reached 50k and had **identical final labels to a matched same-graph, same-seed no-callback control**. Max strength bound excursion across all datasets was ~1.29e-14, only floating-point roundoff. No production PCC code or defaults were changed.

## Predefined primary result

Seeds **1709, 1801, 1901, 2003, 2111**, 10-NN **undirected union** graph, two labeled samples/class, experimental sequential NumPy PCC, 50k iterations, 1k observational windows. Operational lockout defined *before running* as ≥5 consecutive 1k windows with zero accepted moves by an entire class, max class-particle strength <=0.01 and no class-winning outgoing neighbor entries, with every visit valid.

| Dataset | Mean degree | Class cases | Runs with ≥1 lockout | Classes locked | 1k class-windows with zero accepted moves | Mean accuracy 50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Iris | 13.067 | 15 | **0/5** | **0/15** | **0** | 88.61% |
| Wine | 13.831 | 15 | **0/5** | **0/15** | **0** | 94.88% |
| Breast Cancer | 15.033 | 10 | **0/5** | **0/10** | **0** | 88.32% |
| Digits | 14.043 | 50 | **0/5** | **0/50** | **0** | 89.69% |
| **Total** | — | **90** | **0/20** | **0/90** | **0** | — |

**No entire class experienced even one complete 1k window without an accepted movement** in any of these 20 canonical undirected trials. This is stronger than simply finding zero 5-window lockout events. The previous directed-Digits high lockout incidence **cannot** be generalized to the canonical PCC; the graph construction materially affected the result. This controlled, *new-seed* sample supports the interpretation that the earlier phenomenon was mainly **an artifact of using a noncanonical directed input graph in the experiments**, rather than a frequent PCC default behavior.

### Variation after 25k (not convergence proof)

| Dataset | Changed initially-unlabeled predictions per run from 25k to 50k, range | Mean changed nodes | Mean accuracy change, 50k–25k | Largest absolute accuracy change |
| --- | ---: | ---: | ---: | ---: |
| Iris | 0–3 | 1.2 | -0.28 pp | 2.08 pp |
| Wine | 0–5 | 2.0 | +0.47 pp | 2.91 pp |
| Breast Cancer | 5–12 | 8.8 | +0.07 pp | 0.71 pp |
| Digits | 9–28 | 16.8 | +0.53 pp | 0.79 pp |

These compare **the same trajectory at two fixed caps**, not true convergence. Some seeds got slightly less accurate with more iterations, while others improved. In Digits 16.8 flips on average out of 1,777 initially unlabeled nodes (about 0.95%) between 25k and 50k; no operational particle lockout was observed. There is no guarantee of stabilization beyond 50k or on other seeds/datasets.

## Scientific limits and next steps

- The **n=5 per dataset** sample is too small to prove lockout rare in the population or rule out rare exceptions. Even 0/20 event-positive cases admits an illustrative common-risk one-sided 95% bound of ~13.9% under independent-binomial assumptions; within Digits 0/5 has a much wider bound. Do not treat correlated class-windows as independent experimental repetitions.
- **Zero observed operational lockouts** means no empirical basis here for a general reset or minimum-strength intervention. These alterations can perturb the intended PCC dynamics and could worsen results. They remain a conditional research hypothesis for independently validated canonical graph failures.
- **Accuracy can vary without lockout.** Predictive stability is a separate stopping question; do not call 25k or 50k a convergence point.
- The historical directed-graph comparisons remain reproducible but are **not standard PCC baselines**. The +32.46 pp directed10→union10 Digits gain is the consequence of correcting a noncanonical graph input, not new PCC model improvement.
- **Next experiment:** on newly selected seeds and production undirected graphs only, compare a limited, prespecified selection of in-loop stop rules against full 50k no-stop references, measuring early triggers, accuracy change, prediction agreement and measured wall time. Avoid parameter sweeps chosen after observing these five seeds.

**No merging into `master`** and no change to graph, particle, or stopping defaults is supported by this prevalence study.
