# Preregistered observational audit of sustained particle lockout

**Frozen protocol recorded before accessing the new cohort results.**
Related experimental scripts: `audit_lockout_predefined_cohort.py` and workflow
`particle-lockout-cohort.yml`. This study addresses whether the Wine
seed-83 class-1 event is exceptional, not whether an intervention improves PCC.

## Sampling and computation

- Datasets: **Iris, Wine, Breast Cancer, Digits**.
- Previously unused seeds in this exploratory stopping-study sequence:
  **101, 211, 307, 401, 503** (five per dataset).
- Reproducible graph preparation: standard scaling, 10-NN,
  **two randomly sampled labeled observations per class** using each seed.
- NumPy sequential PCC experimental kernel, default PCC parameters,
  **no early stopping**, 50,000 iterations.
- Checkpoints and event counts in **50 adjacent 1,000-iteration windows**.
- One same-seed/graph/cap no-observer control for **every** dataset-seed case;
  assert all final predicted labels match. Check that all particle visits are
  counted and dominance remains finite and on the simplex at checkpoints.
- Four independent CI matrix jobs, one per dataset. No modifications to
  production PCC, no class recovery intervention, no stop-rule changes.

## Fixed, descriptive endpoint

A class-particle *operational lockout* is registered at the first observed
checkpoint where **all** these are true:

1. No moves accepted by **any particle of that class** in at least **five**
   consecutive nonoverlapping 1,000-iteration windows (5,000 iterations).
2. All expected particle visits in those windows reached valid neighbors
   (the experiment checks valid visits in each individual window).
3. The maximum current strength of that class's particles is **≤0.01**.
4. The particles' current nodes collectively have **zero** neighboring
   adjacency entries where their class ties for maximum dominance.

Thresholds are fixed descriptive diagnostics and not claimed to be an optimal
or physically absorbing-state boundary. A class might recover at a later
iteration. The first later 1,000-iteration window with a nonzero accepted
move is recorded as a *recovery*. This event is not equivalent to low
classification accuracy, algorithm nonconvergence, or an implementation bug.

## Outputs and interpretation

The script creates window-level counts and strengths, class-level first-event
and recovery summaries, and a dataset-seed summary with the number of classes
experiencing the operational event. The primary descriptive result is
**number of cases with ≥1 event / 5 runs by dataset**. Secondary results:
**number of classes with events / number of evaluated classes**, eventual
movement recovery, and class-wise dominance mass changes. Summaries also
record initially unlabeled accuracy for context, but do not claim causality.

This is a small, structured follow-up after an extensively explored case;
the new seeds reduce selection bias but don't fully eliminate it. No
convergence/stopping recommendation follows from event prevalence alone. If
events appear frequent or materially affect predictions, the next research
phase can compare MATLAB/Python trajectories under controlled graph/RNG inputs
and assess recovery interventions on a separate larger sample.

**Do not merge experimental stopping rules or particle interventions into
`master` on the basis of this cohort.**
