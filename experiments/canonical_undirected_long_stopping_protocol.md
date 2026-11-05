# Preregistered conservative label-only stopping on undirected PCC (100k reference)

**Frozen before evaluating this follow-up cohort**. The prior 12-case undirected experiment used a 50k no-stop cap: label stability (3 equal checkpoints of 1k) stopped 11/12 cases with mean ~54.8% saved iterations, mean ~-0.28 pp accuracy difference and up to -1.74 pp loss; the P95+label hybrid (threshold 0.001) did not stop at all. These previous seeds are **not reused**.

## New cohort and fixed stopping policies

- sklearn Iris, Wine, Breast Cancer, Digits. New seeds **2609, 2707, 2801, 2903** (4 per dataset = **16 paired cases**).
- Construct graph through the **unchanged production `pcc_graph.build_knn_graph`**, on standardized data, k=10, *undirected union*; exact same graph and 2 labeled nodes/class within each seed/policy. Validate reciprocity, valid edges and nonempty neighborhoods.
- Experimental sequential NumPy PCC, with same fixed default parameters. Reference is an **uninterrupted 100,000-iteration run**, not a convergence guarantee.
- Fixed label-stability variants: `label_change`, 1k check interval, `patience=3`, **6**, and **10**. Do **not** test or select more thresholds in response to results. These stability conditions count successive unchanged class predictions on initially unlabeled nodes at checkpoints. Stop can occur only after previous checkpoint exists.
- Collect a reference-only **50k checkpoint** with a read-only state observer (not as a stopping rule) to measure additional change from 50k to 100k. Confirm 100k checkpoint predictions match final labels.
- Four workflows, one per dataset, maximum 35 minutes each. CSV reports per seed/policy iterations, accuracy on initially unlabeled nodes, change vs 100k in percentage points, prediction agreement with 100k, wall time, mean graph degree, 50k-to-100k reference drift and agreement.
- Validate 100k no-stop reference against separate same-seed no-callback run for each seed; compare all numerical state checkpoints to final labels. No interventions, force floors, directed graph or production modifications.

## Outcomes and interpretation

Primary: a descriptive **accuracy–compute Pareto comparison** across the three fixed patience settings: how many cases stop early, mean and minimum agreement with 100k, most negative change in accuracy, and saved *real wall time*. Secondary: 50k→100k reference label changes/accuracy drift, dataset-dependent behavior, and systematic overlong runs.

**Do not merge** stopping rules into `master` based on small samples or a 100k reference alone. Any proposed default needs independent validation on additional datasets/seeds and accelerated production backends. Do not treat lower accuracy or label discrepancies vs a fixed cap as causal proof the cap is converged.
