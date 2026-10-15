# Convergence trajectories: diagnostic protocol (PR #18)

Status: Stage A NumPy instrumentation implemented; native instrumentation and historical experiments remain pending. No historical measurements are changed.

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

## Stage A implementation and limits

`pcc_propagate_numpy` accepts an optional `observer` at completed iteration
boundaries. Its default is `None`; the existing step, check arithmetic and break
condition are unchanged. A terminal notification includes the existing metrics,
including when the cap is zero or early stopping coincides with the cap.
Observers receive live state: this is a trusted internal diagnostic hook, and a
custom observer must not mutate arrays or draw random numbers. The provided
collector only reads arrays and RNG state and retains copies of observations.

`experiments/convergence_trajectories.py` supplies `Trajectory`, `observe_fit`,
`metadata`, `write_output` and `read_output`. `observe_fit` wraps dispatch only
on the supplied model instance, restores it even on failure, and calls one fit.
It rejects native backends, synchronous mode and fewer than two labeled classes.
No public fit signature, defaults or benchmark runner are changed.

Example (from the repository checkout, with an existing synthetic graph):

```python
import numpy as np
from pcc import ParticleCompetitionAndCooperation as PCC
from experiments.convergence_trajectories import observe_fit, metadata, write_output

neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
degrees = np.full(4, 2)
labels = np.array([10, -1, -1, 20])
truth = np.array([10, 10, 20, 20])
model = PCC(impl="numpy")
model.set_graph(neighbors, degrees)
parameters = dict(max_iter=35, early_stop=True, p_grd=.5, delta_v=.1,
                  deltap=1., dexp=2., es_chk=2000)
np.random.seed(19)  # once, before the single fit
predictions, trajectory, metrics = observe_fit(
    model, labels, truth=truth, milestones=(0, 3, 7, 10000), **parameters)
provenance = metadata(dataset="synthetic-ring", label_seed=0,
    propagation_seed=19, arrays=dict(neighbors=neighbors, degrees=degrees,
    labels=labels, truth=truth), parameters=parameters)
provenance.update(metrics=metrics, class_index_to_label=model.unique_labels.tolist())
write_output(".cache/synthetic-trajectory", trajectory, provenance)
```

Milestones must be strictly increasing nonnegative Python integers. Final
observations are unconditional; a final notification at an already observed
iteration updates that row without duplicating it. With `include_checks=True`,
the collector also samples 1, 11, 21, ... (the production check boundary schedule),
even for capped runs with early stopping disabled. Changes and stability compare
successive observations; they cannot count changes that reverse between samples.
Iteration zero has null change/stability values. Ties use NumPy argmax; supplied
labels remain fixed. Absent truth or an empty unlabeled set produces null accuracy.
The fit helper maps truth to internal class indices; truth classes absent from the
labeled set count as incorrect. Stored assignments use internal class indices.

JSON contains shared provenance and observation rows; CSV repeats dataset, label
seed and backend and encodes assignments/hashes as JSON cells. Hashes use canonical
little-endian C-order arrays, including shape and dtype, and cover assignments,
dominance, particle positions/strengths/distances, raw owndeg and NumPy RNG state.
The raw owndeg hash precedes public-fit normalization. Timing is instrumented
elapsed time, excluded from determinism comparisons. Metadata records source
commit/dirty status, protocol checksum, dependencies, platform, fixture hashes,
parameters and the caller's seed policy. The caller must supply the actual inputs,
full effective parameters and label mapping; dirty worktrees are explicitly marked.
Output goes to a new directory only. A checksum manifest is written last;
`read_output` rejects missing/partial/corrupted bundles. This is validation, not
checkpointing or resumable propagation. Each observation costs O(nodes * classes
+ nodes * particles) hashing work and retains O(nodes) assignments; this is not
suitable for unbounded historical collection without an overhead/storage budget.

### Native backend decision

Numba executes its propagation loop in nopython mode; a Python observer requires
an object-mode boundary or a separate buffered compiled interface. Cython executes
under `nogil` with a local xorshift state initialized once per propagation call.
Its public step wrapper resets that state on every call, so replaying public
steps would not preserve a trajectory. Both need separate native interface work,
including RNG preservation and overhead validation. Per the requested fallback,
Stage A stops after NumPy rather than introducing those changes in this patch.

New synthetic tests validate NumPy observed/unobserved state, predictions, stop
reason/count, RNG continuation and repeatability, plus ties, zero changes, empty
caps, schedule validation, label mapping, truth independence and output integrity.
Existing propagation-metrics and sequential-safety tests remain the native
regression evidence; they do not establish native trajectory instrumentation parity.
All-backend trajectory gates remain unmet. Next: review this hook and implement
one native observation interface at a time, then pass all-backend parity before
requesting a bounded Stage B pilot. No historical campaign is authorized here.

Local validation used Windows, Python 3.13.13, NumPy 2.5.4 and Numba 0.68.0,
after rebuilding Cython with isolated dependencies. This is local evidence only,
not an expansion of the supported CI matrix. Commands:

```text
.review-venv/Scripts/python.exe -m pip install -e ".[numba]"
.review-venv/Scripts/python.exe -m unittest tests.test_convergence_trajectories tests.test_propagation_metrics tests.test_sequential_safety -v
.review-venv/Scripts/python.exe -m unittest discover -s tests -v
```

The focused run passed 20 tests with all three backends available. The full run
passed 166 tests with two packaging archive metadata checks skipped because
`PCC_DIST_DIR` was unset; CI supplies built archives for those checks. Generated C
build output was excluded. No historical fits, archive changes or releases ran.
