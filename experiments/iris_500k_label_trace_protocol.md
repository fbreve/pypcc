# Frozen diagnostic: read-only 500k label trajectory under canonical undirected Iris

**Protocol frozen before inspecting the new trace.** Diagnostic branch based on experimental PR #172; absolutely no production kernel, public API, graph, stopping default or `master` changes.

## Aim

In the previous 500k audit, NumPy Iris seed 5003 had **identical** 100k and 500k endpoint predictions but a distinct 250k endpoint (one label), demonstrating the risk of inferring convergence from matching endpoints. On fresh seed 5303 the early stop had higher accuracy than the 500k no-stop outcome. Trace actual *uninterrupted* long propagation to determine when labels temporarily change and whether long intervals of apparent stability can precede further changes.

## Frozen input

- Standardized sklearn Iris, production `pcc_graph.build_knn_graph` union-symmetrized undirected kNN=10, two labeled nodes/class, explicit original seeds **5003** and **5303** (both already observed; **purely diagnostic, not held-out evidence**).
- NumPy experimental candidate `experiments.benchmark_inloop_global_stopping.run` with existing sequential `p_grd=0.5` and original other particle parameters. Run at fixed **500000** cap with `patience=0` (never early stop), one time with an **existing read-only** `state_checkpoint_callback` every **1000** iterations and once *without* any callback.
- Independently run **actual production NumPy** no-stop with `set_graph(graph,degrees,k_nn=10)`, seed, and full 500k cap. Mandatory **three-way final prediction equality** (instrumented experimental == noninstrumented experimental == production). If this fails, report the gate failure, do not interpret the trace.
- Run experimental label-stability P10 again with 1k interval and 500k cap for an exact stop-iteration observation; compare its final prediction to the full 500k endpoint. If P10 stops, identify whether later sample checkpoints diverge from that stopped labeling, including returning to the same labels before 500k.
- Observer collects only `argmax(dominance[unknown,:])` (no mutation), sampled every 1000 iterations through 500k, plus accuracy, number of changes from prior sample, class prediction counts, mean max dominance; retain all **500** samples per seed. The true trajectory may change between checkpoints: observed flips are **lower bounds**.
- Calculate sampled changes after 100k, counts of nodes ever differing from the 100k snapshot during 100k–500k, nodes differing at 500k, count of nodes that changed and later returned to 100k labels, longest *observed* no-change sequence followed by another observed change (in 1k windows), and P10-vs-500k agreement/accuracy delta. Separately record 100k, 250k, 500k snapshot accuracies.
- CI matrix one job per seed, explicit Cython+Numba production dependencies only as installation/CI compatibility; raw checkpoint CSV and summary CSV artifacts plus installed environment metadata. The experiment uses NumPy propagation only.

## Interpretation constraints

This cannot measure continuous-time change frequency between the 1000-step observations; cannot establish convergence or monotonic accuracy, and does not estimate population prevalence because both cases were selected from prior observed results. Report findings descriptively and do not tune P10 using the observed trace. Keep default production stopping unchanged; a general stopping-policy recommendation still requires a separately predeclared held-out multi-dataset comparison.
