# Prespecified Digits 10-NN graph symmetrization audit

**Frozen before analysis of the results from this new cohort.**
Motivation: the preceding observational study found that particles that begin at
relatively pure labeled homes can become trapped in hostile local neighborhoods
of a **directed** 10-NN graph (see
[`digits_lockout_position_dynamics_findings.md`](digits_lockout_position_dynamics_findings.md)).
The present study is a graph-variant comparison, **not** a stopping rule or
particle-state rescue proposal.

## Cohort and pairing

- **Dataset:** sklearn Digits, StandardScaler, sklearn NearestNeighbors with
  10 neighbors excluding self exactly as in
  [`compare_dominance_stopping.prepare`](compare_dominance_stopping.py).
- **New seeds (not analyzed previously in these studies):**
  **907, 1009, 1103, 1201**. Each has two labeled examples **per class**
  sampled by the established `prepare("digits", seed)` procedure.
- Two conditions for **each** seed, with **the same initial labels**, dataset,
  underlying directed 10-NN list, PCC defaults, maximum **50,000** iterations,
  no early stopping, and 1,000-iteration observation windows:
  1. `directed_10nn`: the unmodified original ordered 10 outgoing neighbors
     per node.
  2. `symmetric_union`: an **undirected union** adjacency represented as
     symmetric directed rows. For each node, preserve the first 10 original
     neighbors in original order, then append any reverse-only incoming
     neighbors in **ascending node index** order. Active degrees vary. The
     adjacency contains neither self-links nor duplicates.
- The symmetry change **increases** some degrees, changes random-walk
  transition probabilities and can also alter RNG consumption. It is not
  a constant-degree causal isolation of reciprocity. Compute and report
  mean/min/median/p95/max degrees, directed reciprocal-edge fraction,
  and total directed adjacency entries for both variants.
- The same numerical seed initializes each separate no-stop run, but
  identical RNG trajectories after a graph change are neither expected nor
  claimed.

## Primary descriptive endpoint (fixed)

The prior operational **class lockout** definition is reused unchanged:
at least **5 successive 1k windows** with **zero accepted movements** among
all class particles, every expected particle visited a valid target on each
step, maximum class particle strength **<= 0.01**, and zero currently
class-winning outgoing neighbor entries.

Report:
1. **number of dataset-seed runs with >=1 class lockout / four seeds**,
   separately per graph;
2. **number of affected classes / 40 classes** per graph;
3. whether a previously locked class in the directed run is absent from
   the union run, or new ones emerge. Recovery after a lockout is a
   distinct indicator.

## Secondary measurements and integrity

- Per-seed accuracy on the **same initially unlabeled nodes** (the paired
  label set is identical across graph conditions), agreement of predictions
  between graph variants and runtime ratio; single wall-clock measurements
  are exploratory, not statistical performance benchmarks.
- Number of accepted movements and maximum zero-movement streak per class.
- The diagnostic must verify: (a) every expected particle visit reached a
  valid neighbor; (b) all numerical state finite, per-node dominance simplex
  within 1e-8, and strength bounds subject only to <=1e-8 roundoff;
  (c) every instrumented run's final labels exactly match a same-seed,
  same-graph uninstrumented no-stop control; and (d) no stop before cap.
- An additional graph-level check verifies symmetry, no self edges,
  uniqueness within each active adjacency, inclusion of every original
  outgoing edge, and exact preservation of the original neighbor prefix.
- Use **one CI job per seed** to bound runtime, emitting CSV for graph
  statistics, per-class lockout, per-case summary and timeline.
- Do not change the above seeds, thresholds or endpoints after seeing results.

## Interpretation constraints

Four seeds are a **small hypothesis-test pilot**, not an estimate of
population-level effects. Greater average degree and new two-way links are
simultaneous interventions; positive or negative results cannot isolate
whether reciprocity, degree, altered traversal, or randomness drives them.
The study was designed *after* observing Digits dynamics on prior seeds,
which can still influence the choice of conditions.

If symmetrization reduces lockouts, a follow-up should match or control
degree distribution and compare accuracy and computational cost on a larger
held-out set. If not, redirect mechanistic attention to the particle
acceptance/strength rule and dynamic competition. **Neither outcome
authorizes changing default graphs, early-stopping rules or particle
behavior in `master`.**
