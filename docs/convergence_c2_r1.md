# C2-R1 execution adapter

Scientific payload SHA-256:
`955dca598c123d059269f18fbd7fe67f4d021b3bd211d2f1da1846557989c825`.
The [approved record](c2_r1_scientific.json) is copied verbatim from the sealed
proposal. Its status text is retained unchanged; scientific approval and technical
preparation do **not** authorize real fits. Hypotheses, thresholds, exclusions and
non-evaluability gates are frozen in that record, not selected by this runner.

## Scope and contract

`python -m experiments.convergence_c2_r1` prepares a separate campaign: Banknote
and Sonar, label seeds 10/11, NumPy/Numba/Cython, 12 fits and 120 observations.
Historical milestones remain 0, 1000, 2000, 5000, 10000, 20000, 40000, 60000,
80000 and 100000; final is deduplicated. Parameters, fixtures, graph ordering,
three labels/class, native seed policies, 100000 cap and disabled early stopping
come unchanged from original C2. Backends are paired conditions, not independent
seed replicates. Historical endpoints were already examined: this is limited
prospective replication, not blinded external validation.

The adapter adds the existing opt-in full-snapshot profile to the same fits:
dominance, positions, strengths, node/particle IDs and particle home/class mapping.
It retains the CSV/JSON v2 summary and the documented NPY/JSON binary contract in
[mechanistic_snapshots_protocol.md](mechanistic_snapshots_protocol.md). No kernel,
ordinary fit default, native loop, arithmetic, update order or RNG is changed.
Native observation buffers are reused; NumPy snapshots remain preallocated.
No new per-node counters or intra-fit checkpoints are introduced.

Ten slots are allocated, with maximum 128 observations and independent 64 MiB
limits for snapshot arrays, estimated diagnostic memory and export per fit.
Conservative estimates exclude process RSS, graph, libraries/JIT and extra failed
attempts. The preflight reports per-task dimensions, edges, particles, resource
totals, historical timing projections and their limitations. Reserve 2 GiB disk
and provisionally 20 minutes; neither is a scientific stopping condition.

## Preflight, authorization and execution

Preparation is read-only toward the approved local synthesis, B/C1, the
mechanistic pilot and original C2 preflight. It verifies their stores, fixture
identities and manifests, checks PR #16/#17 compatibility, and copies only the
four frozen C2 fixtures into a new exclusive preflight directory. It never
downloads or fits real data. The execution identity pins the commit, source
hashes, backend availability, environment and Cython binary. Original C2's SHA is
checked as provenance, never replaced or accepted as the new execution SHA.

Example preparation, which performs **no fit**:

```powershell
.review-venv/Scripts/python.exe -m experiments.convergence_c2_r1 preflight --output .cache/convergence-c2-r1-preflight --original-c2 .cache/convergence-stage-c2-preflight --scientific-directory .cache/exploratory-synthesis-c2 --stage-b .cache/convergence-stage-b-pilot --stage-c1 .cache/convergence-stage-c1 --mechanistic .cache/mechanistic-pilot
```

Use the actual C1 store path from its recorded plan. Preflight refuses source
overlap, any changed reviewed input, or resource/fixture mismatch. Review the
**new** canonical payload SHA and exact-commit CI before authorizing execution.
The commands below are instructions for a future approval, **not authorization**:

```powershell
.review-venv/Scripts/python.exe -m experiments.convergence_c2_r1 init --preflight .cache/convergence-c2-r1-preflight --root .cache/convergence-c2-r1
.review-venv/Scripts/python.exe -m experiments.convergence_c2_r1 run --root .cache/convergence-c2-r1 --reviewed-preflight APPROVED_EXECUTION_SHA --max-runs 1
.review-venv/Scripts/python.exe -m experiments.convergence_c2_r1 status --root .cache/convergence-c2-r1
# Only after integration integrity passes and execution approval covers remaining fits:
.review-venv/Scripts/python.exe -m experiments.convergence_c2_r1 run --root .cache/convergence-c2-r1 --reviewed-preflight APPROVED_EXECUTION_SHA --max-runs 11
```

Each task runs continuously once. An exclusive store, lock, append-only ledger,
atomic completed manifests and strict binary/summary validation prevent partial
exports from counting as results. Valid published attempts reconcile without
refitting; completed tasks are skipped. Ambiguous attempts block continuation
until an explicit retry task/reason preserves the old attempt. No retries for
unfavorable scientific outcomes. Never run legacy C2 plus C2-R1 as 24 fits.

## Integrity and interpretation checklist

- Approved scientific record and scope unchanged; preflight SHA explicitly reviewed.
- Exact final-commit CI green for Linux/Windows and Python 3.10/3.12; synthetic
  within-backend observed/unobserved state, predictions, stop and RNG parity.
- Backend availability with no fallback; fixture/graph/mapping hashes correct;
  historical parameters/seeds compatible. PR #17 has unequal stopping/work.
- Integration result: NPY hashes, types, shapes, static identities, ten milestones,
  reconstructed summary predictions and retrospective scores all valid.
- Compare the 10k prefix against all six PR #16 technical records per backend;
  use predictions, dominance, positions, strengths and accuracy. Historical final
  `owndeg` is normalized after propagation and is not a comparable milestone hash.
  differences block continuation for investigation, never force cross-backend
  bitwise identity or silently alter floating-point behavior.
- Final audit: 12 unique complete tasks, 120 observations, no recoverable or
  ambiguous attempts; all prior dependencies preserved.
- Publish H1–H3 including negative/non-evaluable outcomes before exploratory
  extensions. No outcome-based threshold changes, default selection or stopping.
  Oracle labels only score retrospective diagnostics, never decide classifications.

Binary datasets cannot test partial multiclass ties; two seeds give limited
replication, not population inference. Positions are sampled presence, not visit
histories. Counterfactual readout does not intervene in propagation. Additional
instrumentation or causal experiments require a separate protocol.
