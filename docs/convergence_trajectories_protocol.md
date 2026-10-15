# Convergence trajectories: diagnostic protocol (PR #18)

Status: implementation planned; no historical measurements are changed by this document.

## Research questions

1. At what iteration does PageBlocks unlabeled accuracy deteriorate relative to the fixed 10,000-iteration baseline, and how do class assignments and dominance margins evolve?
2. How does Spambase improve over the same propagation trajectory?
3. How sensitive are Banknote and Sonar trajectories to the frozen historical label seeds?
4. Do dominance stability, class-change counts and existing early-stopping checks correlate with useful accuracy endpoints? Correlation is descriptive, not proof of a safe stopping rule.

## Frozen comparability

Use the historical fixture construction, preprocessing, ordered k-NN graph, three labels per class, label seeds 10–19 and original backend-specific RNG policies. Parameters: p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2, es_chk=2000; iteration cap 500,000. Do not alter the PR #16 fixed-iteration or PR #17 early-stopping archives. Never compare unequal-work elapsed times as intrinsic kernel speedups.

## Instrumentation contract

- Diagnostic mode is strictly opt-in, disabled by default and isolated from the normal fit path.
- Capture observations at a configurable set of iteration milestones, including iteration 0 when meaningful, 10,000, existing stopping-check boundaries and the final iteration.
- For each observation record dataset, label seed, backend, iteration, effective stop reason if final, unlabeled accuracy (diagnostic only), class-change count versus previous observation, dominance margin/stability summaries, state or assignment fingerprints and elapsed time separately from uninstrumented benchmarks.
- Do not feed unlabeled ground-truth accuracy into propagation, convergence decisions, hyperparameter selection or early-stopping logic. Label it explicitly as retrospective transductive evaluation.
- Preserve backend-specific numerical/RNG semantics; do not silently replace the native backend with NumPy or reset/reseed between milestones. Prefer an explicit opt-in observation hook at a safe iteration boundary over repeated fresh fits that change trajectories. If a backend cannot expose a safe hook without perturbation, document the limitation and validate an alternative before collecting data.
- Write portable machine-readable CSV/JSON with protocol, source commit, dependencies, fixture fingerprints, seed policy, checkpoint schedule and checksum manifest. Raw results and plots are generated artifacts, not committed by default.

## Validation gates before real experiments

1. Tiny synthetic fixture: observed vs unobserved run has identical final predictions, effective iteration count, stopping reason and state fingerprints for each available backend.
2. Milestone snapshots do not change RNG draws, loop order, dominance update, backend selection or early-stop semantics.
3. Repeated diagnostic runs with the same frozen fixture/backend are deterministic in classification and state hashes; timing is reported as descriptive overhead, not a performance benchmark.
4. Test edge cases: zero changes, ties, missing classes, capped vs genuinely early-stopped fits, invalid milestone schedule and partial/corrupt output.
5. CI tests must be small and portable. No expensive historical campaign in CI.

## Execution gates

Stage A: implementation and synthetic validation only.
Stage B: small PageBlocks/Spambase pilot, with bounded work and explicit cost estimate.
Stage C: Banknote/Sonar and ten-seed replication after reviewing pilot results.
Stage D: descriptive plots and comparison with PR #16/#17 endpoints; no algorithm changes or stopping-rule recommendation without separate evidence.

This file is the initial protocol, not a claim that instrumentation or experiments have already been implemented.
