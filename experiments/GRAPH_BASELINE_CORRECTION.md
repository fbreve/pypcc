# Critical graph-baseline correction — experimental PCC studies

**Date:** 2026-11-05. **Status:** methodological correction, not a code or production defect.

## What was checked

- The original MATLAB PCC reference, [`pcc_buildgraph.m`](https://github.com/fbreve/Particle-Competition-and-Cooperation/blob/master/pcc_buildgraph.m), constructs k-nearest-neighbor relationships and **adds reverse connections explicitly**. Thus the model receives an **undirected graph**, generally with degrees greater than k.
- Production PyPCC [`pcc_graph.py`](../pcc_graph.py), `build_knn_graph`, constructs the same type of **undirected union-kNN** topology: `A + A.T`, remove diagonal and duplicates. `ParticleCompetitionAndCooperation.build_graph` calls this function.
- Experimental [`compare_dominance_stopping.py`](compare_dominance_stopping.py), `prepare`, calls `sklearn.neighbors.NearestNeighbors` and returns the ten **outgoing directed nearest-neighbor arcs** *without symmetrizing*. Other benchmark scripts importing `prepare` inherited this directed graph, and `set_graph` installs the supplied adjacency without modifying it.

**Consequently, the experimental directed 10-NN baseline was NOT the original MATLAB or production PyPCC default graph.** The reported 50k stopping, drift, lockout, accuracy and graph comparisons are observations for that experimental directed graph unless a particular study explicitly changed the graph. They must not be represented as the behavior or validated stopping performance of the canonical undirected PCC.

## Interpretation of the large Digits accuracy difference

The previously reported +32.46 percentage point mean accuracy improvement for the union-symmetric graph over the directed 10-NN graph (four Digits seeds 907, 1009, 1103 and 1201, 50k no-stop iterations) is **not a newly discovered graph-construction improvement over the original PCC**. The study compared a noncanonical directed baseline with a union-symmetrized graph that uses the *same undirected topology type already expected by MATLAB and PyPCC*. The experiments themselves remain reproducible and meaningful as a directed-versus-undirected sensitivity test, but the former characterization as a potential production graph improvement was mistaken.

The study's union graph preserves the original 10-NN ordered neighbor prefix and appends sorted reverse-only neighbors, whereas production `build_knn_graph` emits CSR-derived neighbor ordering. Therefore the graph **edge sets** should be compared directly (including tie/self-neighbor handling), and matched numerical predictions are **not guaranteed** merely because two undirected constructions have the same topology; neighbor ordering affects stochastic sampling.

Likewise, the ongoing degree-control comparison directed10/directed14/union10 remains descriptive of graph variants, **not** a comparison of a proposed new graph with a valid production baseline. Those results do not validate canonical stopping or a default graph change.

## Corrective plan

1. **Preserve all historical experiment files, seeds and source artifacts.** Explicitly label studies that called `prepare` as *directed-graph experimental benchmarks*, not standard PCC evaluations. Do not retroactively modify protocols/results.
2. For **new** canonical evaluations, construct the graph with the unchanged production `pcc_graph.build_knn_graph(StandardScaler(...), k_nn=10)` (or `ParticleCompetitionAndCooperation.build_graph`) and verify reciprocal adjacency, degree distribution, no self-loops and valid active indices. Reuse the same graph and label set for each compared stopping policy; use no-stop controls and multiple seeds.
3. Add a **distinct canonical benchmark path or explicit graph-mode argument** rather than silently changing the historical `prepare` default and invalidating old experiment provenance. Avoid silently creating any directed benchmark when the goal is standard PCC.
4. Repeat any early-stopping, class-drift, class-lockout or accuracy claim intended to describe the canonical PCC on canonical undirected graphs. The old directed statistics **cannot** be transferred as estimates of canonical PCC behavior.
5. No changes to production `pcc_graph.py`, `pcc.py`, stopping defaults or `master` are justified by this correction alone.

## Clarifying terminology

- *Directed kNN:* A→B is present if B is in A's k-nearest list; reverse B→A is optional.
- *Undirected union/symmetric kNN:* include an undirected A—B edge when **either** A chooses B **or** B chooses A. This is what the cited original PCC code and production PyPCC do; node degrees can exceed k.
- *Mutual/intersection kNN:* include A—B only if **both** choose each other; **not** what the cited original PCC or production PyPCC does.

This note is a scientific reproducibility correction. It does not allege a regression in the production PCC graph builder.
