# Digits class-home graph structure after nested-label lockout pilot

Source: GitHub Actions run **41939627653** (success), artifact `digits-lockout-graph-structure/digits-lockout-graph-structure.csv` (ID **13376465775**), commit `d6bd8ee`. The same commit passed **PCC regression** run **41939627370**. The unrelated legacy dominance real-data pilot was canceled; its cancellation is not a failure of this diagnostic.

## Method

Reconstruct the exact StandardScaler/10-NN graph and *nested* 2-versus-5 labeled-node selections for Digits seeds **601,701,809**, without performing any propagation. For each labeled home node, compare its outgoing neighbors' **ground-truth** classes with its own class; also count incoming graph edges, optionally from the same ground-truth class. Ground-truth neighborhood purity is used only as an **offline diagnosis**, not as information visible to the PCC propagation algorithm. There are 60 rows (three seeds × two supervision settings × ten classes).

## Main result: home-neighborhood purity does not explain the lockouts

Previously observed operational class lockouts occurred exactly in these five seed/class/supervision combinations:

| Seed | Class | Labeled nodes/class | Outgoing true-same-class neighbor entries | Purity | Incoming entries to labeled home nodes | From true-same-class origins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 701 | 2 | 2 | 20/20 | **100%** | 20 | 20 |
| 701 | 2 | 5 | 49/50 | **98%** | 39 | 38 |
| 809 | 4 | 2 | 20/20 | **100%** | 20 | 20 |
| 809 | 4 | 5 | 50/50 | **100%** | 67 | 67 |
| 809 | 9 | 2 | 20/20 | **100%** | 17 | 17 |

Across all 30 class configurations per arm, mean outgoing *true*-same-class neighbor purity was **95.0%** (2 labels/class) versus **94.53%** (5 labels/class). The five configurations that eventually locked had mean purity **99.6%**, compared to approximately **94.33%** for the other 55 configurations. This is a selected small sample; **do not** conclude higher purity causes lockout. It does refute a simple story in which the observed class lockout arose solely because the selected labeled homes had poor true-class-local neighborhoods.

The two class-4 configurations of seed 809 had perfectly pure outgoing home neighborhoods yet both eventually locked; increasing supervision from two to five labels advanced the first observed lockout from 39k to 14k. Conversely, seed 809 class 9 with two labels had 100%-pure outgoing home neighborhoods but locked, while five labels retained 96%-pure outgoing home neighborhoods and did **not** lock. Consequently, home purity alone is neither sufficient nor a useful discrimination of these cases.

In this reconstruction, **zero outgoing edges from any labeled home point directly to a labeled home of another class** across all 60 configurations. This is a property of the specific sampled 10-NN graphs and does not say all connected unlabeled nodes are pure, nor that particles cannot later enter hostile neighborhoods. The operational lockout measure is based on a **time-varying class-dominance neighborhood**, not a static true-class label comparison.

## Implication

Static ground-truth neighborhood impurity at labeled homes is **not** an adequate explanation for the observed lockout. A targeted next observation should trace how class-particle positions depart from home, how the *current* neighborhood dominance changes, and whether particles get trapped in local destinations with no winning neighbors. Use only the previously examined seeds/classes for this mechanism audit and reserve new seeds for eventual prospective validation. Do not modify particle rules or early stopping in `master`.
