# Directed 10-NN vs symmetric union: prespecified Digits cohort results

## Provenance

The protocol and seeds were frozen in
[`digits_symmetric_graph_lockout_protocol.md`](digits_symmetric_graph_lockout_protocol.md)
(commit `81fbba1`) before any new result. Experimental algorithm in
[`audit_digits_symmetric_graph_lockout.py`](audit_digits_symmetric_graph_lockout.py).
GitHub Actions workflow run **41954631665** at commit `e368391`:
all four per-seed jobs **passed**, publishing artifacts:

| Seed | Artifact ID |
| ---: | ---: |
| 907 | 13382807057 |
| 1009 | 13383040956 |
| 1103 | 13382905981 |
| 1201 | 13382393683 |

Each artifact contains `symmetric-graph-<seed>.csv`,
`symmetric-windows-<seed>.csv`, `symmetric-classes-<seed>.csv`,
and `symmetric-cases-<seed>.csv`. All **eight** instrumented variant runs
matched their paired **same-graph same-seed no-observer** control predictions
exactly. Max observed particle-strength bound excess was **6.88e-14**.
All runs reached the full 50k cap, and graph invariants (valid, unique
neighbor indices; reciprocity of the union; original 10-NN prefix retained)
passed.

## Fixed design

Digits (1,797 nodes), StandardScaler, two labeled nodes/class (20 in total),
matched seed and label selection, default experimental sequential NumPy PCC
parameters, **50,000 iterations without early stopping**, checkpoints every
1,000. New seeds **907,1009,1103,1201** were not used in prior studies.
Unlabeled accuracy is evaluated on the *same 1,777 nodes* for a given
paired seed. The two graph variants differ in adjacency construction only:

- `directed_10nn`: original directed 10-NN (10 outgoing connections/node).
- `symmetric_union`: union with reverse connections added where missing
  (preserves each original ordered 10-NN prefix, appends sorted reverse-only
  targets). Node degrees vary.

### Important structural difference

The directed 10-NN graph contains 17,970 directed edges and has only
**59.57% reciprocal arcs**. The union graph is fully reciprocal and
contains **25,236 directed arcs**, an increase of **7,266 / 40.43%**.

| Graph | Degree min | Median | Mean | P95 | Max | Reciprocal arc fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Directed 10-NN | 10 | 10 | 10.00 | 10 | 10 | 59.57% |
| Union symmetric | 10 | 13 | **14.04** | 22 | 35 | 100.00% |

The graph structure is the same for all seeds because Digits features and
graph settings are unchanged across seed-dependent label selection.

## Primary lockout and classification outcomes

The same operational lockout definition was used as in the preceding fixed
cohort: five consecutive 1k windows with zero accepted moves by an entire
class, no invalid visits, maximum strength <=0.01 and zero current
class-winning outgoing neighbor entries.

| Seed | Directed locked classes | Symmetric locked classes | Directed accuracy | Symmetric accuracy | Improvement (pp) | Symmetric agreement with directed |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| **907** | none | none | 57.68% | **91.45%** | **+33.76** | 57.79% |
| **1009** | class 2, event at 8k, node 1546 | none | 44.12% | **85.37%** | **+41.25** | 46.76% |
| **1103** | class 4, event at 24k, node 1546 | none | 52.34% | **76.03%** | **+23.69** | 55.94% |
| **1201** | class 9, event at 17k, nodes 259/1546 | none | 57.74% | **88.86%** | **+31.12** | 56.39% |
| **Mean** | 3/4 dataset-seed runs affected | **0/4** | **52.97%** | **85.42%** | **+32.46** | **54.22%** |

Across these four runs, class-level operational lockouts were **3/40**
for directed graph, **0/40** for the union graph; no lockout class recovered
before 50k in the directed runs. Note that *0 of 4* is not evidence that
union graphs eliminate lockouts in general, especially in other datasets
or outside 50k.

**Critical distinction:** Seed **907 had no qualifying class lockout in
either graph**, but its accuracy still increased by **33.76 percentage
points** under symmetrization. Therefore the observed accuracy gains
**cannot be explained solely by preventing operational class lockout**.
More fundamental changes to graph connectivity, stochastic path selection
and class interaction are implicated.

### Elapsed times

Single GitHub runner, one uninstrumented run per condition and seed:

| Seed | Directed seconds | Symmetric seconds |
| ---: | ---: | ---: |
| 907 | 21.306 | 21.639 |
| 1009 | 21.993 | 22.263 |
| 1103 | 20.975 | 21.358 |
| 1201 | 21.530 | 22.118 |
| **Mean** | **21.451** | **21.845** |

Mean increase **0.394 seconds (1.84%)**. These are one-shot wall-clock
measurements under GitHub Actions and are **not statistical performance
estimates**. The candidate backend is sequential NumPy, not all production
backends.

## Scientific interpretation and needed controls

The symmetrized union had substantially better 50k accuracy and fewer
lockout events on **all four** previously unused Digits seeds. The
direction and magnitude merit investigation, but **symmetrization
simultaneously increases node degree by ~40.4% on average**. A larger
neighborhood alone may improve classification; the trial does not isolate
reciprocity. The same input seed does not make the two stochastic trajectories
identical after changing adjacency.

A principled next pilot should compare a **directed 14-NN baseline**
(degree 14, similar to symmetric-union10 mean degree 14.04) to the
**symmetric union of directed 10-NN**, using new held-out seeds, fixed
labels, full same-seed/graph uninstrumented controls and the same primary
class-lockout definition. Because the 14-NN graph has all its edges chosen
by geometric nearest-neighbor rank rather than union reciprocity, this
matches **mean degree approximately, not topology**. To quantify graph
construction quality, run additional baselines or parameter studies later,
with separate held-out seeds and time/memory measurements across production
backends before considering a user-facing graph option.

Do **not** claim this constitutes a proven PCC algorithm improvement or
merge a new graph default, stopping rule or particle rescue into
`master` based on four seeds.
