# Prespecified Digits graph pilot with approximately matched mean degree

**Protocol frozen before collecting results from this new seed cohort.**
Motivation: the preceding four-seed Digits pilot found a large accuracy
improvement under symmetrizing a directed 10-NN graph, **but** its mean
outgoing degree rose from 10 to 14.0434. The prior graph improvement
could not be attributed solely to reciprocity.

## New cohort

- Digits, StandardScaler, original directed 10-NN prepared by
  `compare_dominance_stopping.prepare`; same two randomly chosen
  labeled examples/class for **all** graph variants within each seed.
- **New seeds**: **1301, 1409, 1511, 1601** (not previously analyzed in
  this workstream).
- Three graph alternatives:
  1. `directed_10nn`: unchanged original ordered 10 outgoing neighbors.
  2. `directed_14nn`: preserve the original ordered 10 neighbors and append
     the four nearest additional points not already included or the node
     itself (in order of nearest-neighbor rank), using the same standardized
     features. Outdegree is exactly 14 for every node; graph remains directed.
  3. `symmetric_union_10nn`: include all original 10 outgoing neighbors,
     plus all incoming edges necessary to make every edge reciprocal; order
     appended reverse-only neighbors by ascending node index; outdegree
     is variable (mean approximately **14.04**, same across seeds).
- Mean degree of 14 vs 14.04 is *approximately*, not perfectly, matched.
  Their degree distributions, edge geometries and reciprocity still
  differ. All three graphs must be verified for valid unique non-self edges.
- Experimental sequential NumPy PCC only, default PCC parameters,
  **50,000 iterations**, **no early stopping**, **1,000-iteration windows**.
  No particle-state intervention or production-code modification.
- Graph structure, node labels and seed are held fixed across all
  trajectories within a seed except for the explicit graph variant.
  Stochastic streams may diverge as graphs differ.

## Primary outcomes

1. Paired initially-unlabeled accuracy after 50k on identical **1,777**
   nodes per seed: compare directed14 to symmetric-union10, with directed10
   as original reference.
2. Dataset-seed runs with **>=1 operational class lockout** / four seeds
   under each graph; number of affected classes / 40.

The lockout endpoint is **unchanged** from earlier studies: at least five
consecutive 1k windows with zero accepted moves by all particles of the
class, all expected visits valid, max class-particle strength <=0.01,
and zero currently class-winning outgoing neighboring entries.

Secondary: graph mean/min/median/p95/max degrees, reciprocal-arc
fraction, prediction agreement, individual locked classes and
first-lockout iteration, class-particle accepted counts, wall-clock
elapsed time on the same CI runner class (one measurement/case).

## Integrity requirements

- Exact same sampled labels and original 10-NN prefix across three
  variants; directed14 is the 10-NN prefix plus four genuinely new
  nearest neighbors. Union graph is reciprocal with all original
  edges retained.
- Per-checkpoint no nonfinite node dominance or particle strengths,
  per-node dominance simplex sum within 1e-8, strength excursion outside
  [0,1] no more than 1e-8 (record exact excess without clipping);
  every expected visit valid.
- For every seed/variant, the instrumented result must match a
  same-graph same-seed uninstrumented control exactly. No early stop.
- One independent CI job per seed, output all graph, per-class,
  window-level and per-case diagnostics.

## Interpretation

Four new seeds constitute a small **exploratory degree-control pilot**,
not a definitive causal proof. A graph with constant outdegree 14 may
still differ from a symmetric union with mean outdegree 14.04 by
incoming-degree distribution, local geometry and stochastic
transition probabilities. Even successful elimination of lockout need
not imply superior accuracy across other datasets.

No tuning of the seeds, thresholds or endpoints after seeing
results. Do not change PyPCC production defaults or merge experimental
stopping/graph changes to `master` on the basis of this experiment.
