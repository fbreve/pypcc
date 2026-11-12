# Modern-master PCC: three-backend **source-only** preflight design

**Status: design / source inventory; not a frozen scientific study; no approval to fit real data.** This document and `experiments/preflight_three_backend_sources.py` are independent of PRs #253, #255 and #256. They do **not** merge the experiments, change PCC production, or reuse the frozen experimental preflights of archived PR #18.

## 1. What is demonstrably available

All three research observers descend from **one pinned production base** `ea69f690208620d5563c8efece6b2e04369f52c0`. Each must run in its **own** isolated checkout/environment, at the exact research head below. An observer checked out on another backend's branch is not equivalent.

| Backend | Draft PR | Pinned research head | Observer source | Synthetic evidence |
| --- | --- | --- | --- | --- |
| NumPy | #253 | `6d908e78878c4922674277e3313dbfe7f7924146` | `experiments/live_numpy_milestones.py` | Linux/Windows 3.10/3.12 4/4 PASS; synthetic paired CPU Linux/Windows PASS |
| Numba | #255 | `4c8e6df4e01d7a4dfc4efc31eb73db04d6e212e5` | `experiments/live_numba_milestones.py` | Native nopython 4/4, production regression 4/4, synthetic CPU 2/2 PASS |
| Cython | #256 | `af34f50ebabb0a7204d3f30679adaf3f00169742` | `experiments/live_cython_milestones.py`, isolated `pcc_step_live_research` | Native cached+legacy 4/4, regression 4/4, synthetic CPU 2/2 PASS |

Evidence links: [#253](https://github.com/fbreve/pypcc/pull/253) / [NumPy CPU](https://github.com/fbreve/pypcc/actions/runs/43136061873); [#255](https://github.com/fbreve/pypcc/pull/255) / [Numba CPU](https://github.com/fbreve/pypcc/actions/runs/43194770991); [#256](https://github.com/fbreve/pypcc/pull/256) / [Cython CPU](https://github.com/fbreve/pypcc/actions/runs/43233351572). The Cython research compiler must confirm original `pcc_step.pyx` Git blob `021c9a2c34f912e0d8382f05d0e21b9713ffd5fc`. No statement about general performance or real-data success follows from these synthetic passes.

The three implementations have **different random generators and arithmetic paths**, so do not require identical node dominances, positions or RNG states *across* backends. Compare observed and unobserved trajectories **within the same backend and binary**, and compare cross-backend aggregate scientific metrics only under a separately preregistered analysis.

## 2. Preflight layers and their hashes (never conflate)

1. **P0 — this source inventory, executable now, Git only:** audit the exact pinned master, three research commit SHA values, changed-file allowlists, and unchanged production Git blobs. The CLI computes a deterministic **`source_inventory_sha256`** based only on the canonical JSON source inventory. It deliberately does not import PCC, compile a binary, load a fixture, inspect labels or perform a fit. This hash is **not** a scientific-study hash and must **never** be accepted as an execution authorization.
2. **P1 — separate, future scientific preregistration:** independently define the mechanistic scientific question, hypotheses, endpoints/estimands, dataset identities and license, label budgets, prospective seeds and label allocations, exclusions, outcomes, missing/non-evaluable handling, stopping policy and sample size. Distinguish exploratory from confirmatory work and from historically examined C2-R1 results. Assign a new immutable scientific record digest after review, not the old C2-R1 payload SHA.
3. **P2 — separate, future execution preflight, no fits:** using all three individually pinned **compiled** environments, verify OS, CPU architecture, exact Python/NumPy/Numba/Cython/SciPy/scikit-learn/compiler versions, Cython extension binary SHA-256, native Numba signatures and cache provenance, controlled NumPy RNG details, dataset/fixture byte hashes, label mappings and sealed graph identity. Report actual memory/disk/CPU headroom, canonical input arrays (dtype, shape, strides, byte order), snapshot count, export design and data custody. Verify symmetric undirected adjacency, active degree, **dense padded `neib_list`** and particle home/class order. An independent reviewer must approve the **new execution-specific SHA** after P1 and P2 before any real fit.
4. **P3 — first approved real run, only after explicit approval:** within each backend, reproduce a separately frozen, independently generated prefix where such a reference exists. Compare state, stopping, native RNG continuation and snapshot provenance; stop on discrepancy. A separate recorded audit must approve continuation. Never substitute a fresh run for an unfavorable or failed result, silently regenerate labels, or reinterpret missing checkpoints after early stopping.

**P0 passing cannot imply P1, P2 or P3 passing.** The present GitHub connection cannot verify local historical PR #18 `.cache` artifacts, real fixture bytes, compiled native binaries or available disk; do not claim it did.

## 3. Common scientific observation contract (proposed, not frozen)

- **Topology:** sequential PCC on an undirected graph; explicitly verify symmetric active adjacency. Preserve the intentional **dense neighbor matrix with padding** and all ordering. No CSR conversion and no graph reconstruction between backend comparisons.
- **Iteration boundary:** 0 is initial pre-step state; i>0 means *after* complete particle iteration i and its usual stopping check. No observations beyond true early stop. Do not create partial-iteration checkpoints.
- **Minimal state fields:** `dominance`, `owndeg` **before normalization**, `part_curnode`, `part_strength`, `dist_table` (unsigned 8-bit). Include static node label and particle class/home identifiers in the input manifest; they are not evolving observed fields.
- **Additional native diagnostics:** Cython may capture per-row maximum cache and internal `xorshift32` uint32 state, including next-value derivation; Numba verifies native RNG continuation with the next compiled draw (not serialized internal bytes), and NumPy verifies the NumPy RNG state/draw. **Do not falsely promote backend-specific RNG evidence to a uniform byte-level RNG contract.**
- **Milestones:** zero and subsequent *prospectively defined* strictly increasing iteration counts, bounded by actual max_iter. The eight checkpoints `(0,1,10,50,100,250,500,5000)` are a **synthetic benchmark schedule only** and are **not** automatically the future scientific C2 schedule. Select a new schedule before data-dependent outcomes are examined; include the stop/no-stop choice and predeclared rules for unreachable milestones.
- **Storage and overhead:** each observer caps *planned* capture arrays at 64 MiB (<=32 observations). Before any future real preflight, compute its full expected bytes from actual node/class/particle dimensions, including Cython's cache/RNG fields and copies, and add metadata/serialization + RSS safety margins. If a proposed graph exceeds limits, **block and redesign the preregistration before execution**; do not silently reduce observations or swap to sparse structures. Distinguish array-copy cost, disk-export cost, compilation and graph construction from fit CPU.
- **Baseline comparison:** within each implementation's original compiled/runtime code and its own seed/state policy, assert exact evolving snapshots at independently generated prefixes; full final predictions and mutable state; stop reason/iterations; and appropriate subsequent RNG checks. Do not compare the normalized final `owndeg` directly to live pre-normalized snapshots.
- **Research integrity:** no access to oracle/test truth inside particle movement or adaptive checkpoint selection. Independent sealed labels and data, stable ordering and exact provenance; report both negative and non-evaluable outcomes. No threshold tuning based on already viewed C2 results.

## 4. Source-only audit, optional to run in a clone

A local clone must have **all four exact commits** in Git's object store and updated `origin/<branch>` remote-tracking refs for all three experimental branches. An auditor may fetch only these source refs; the script itself never connects to a service or fetches anything.

```sh
git fetch origin refs/heads/research/current-numpy-live-milestones-2026-11-11:refs/remotes/origin/research/current-numpy-live-milestones-2026-11-11
git fetch origin refs/heads/research/current-numba-live-milestones-2026-11-12:refs/remotes/origin/research/current-numba-live-milestones-2026-11-12
git fetch origin refs/heads/research/current-cython-live-milestones-2026-11-12:refs/remotes/origin/research/current-cython-live-milestones-2026-11-12
python -m experiments.preflight_three_backend_sources --output .cache/three-backend-source-inventory.json
```

Output fields are explicit: `source_validation: pass`, `source_inventory_sha256`, `execution_authorized: false`, `scientific_protocol_frozen: false`, `execution_preflight_sha256: null`, `model_fits_executed: 0` and unresolved `blockers`. If any branch moves, any pinned commit is missing, production differs, file scope expands or an exact Git blob mismatches, **fail closed**; do not update constants to make the old hash pass. Source inventory reports are metadata artifacts, never datasets.

## 5. Work sequence / authorization gates

**Now:** run P0 source-only audit via CI; keep all three observer PRs and this protocol PR **DRAFT/UNMERGED**. **Next:** design and seek explicit scientific review of a new prospective study P1 without recycling PR #18's data-dependent selection. **Later:** prepare P2 fixture and environment audit with exact execution hashes and export/CPU budgets **without fitting**. Only after human approval of P1 and the precise P2 execution SHA may a separate real-data experiment be authorized. No real C2 or G1 fit has been started by this task.

The historical [C2-R1 12-fit plan](https://github.com/fbreve/pypcc/blob/c5062fbd310c1e6b8f7a5b36c485298286691e94/docs/convergence_c2_r1.md) and [G1-R1 102-task plan](https://github.com/fbreve/pypcc/blob/c5062fbd310c1e6b8f7a5b36c485298286691e94/docs/causal_selection.md) remain separate archived designs. Their scientific SHA values and data-dependent choices cannot be silently reused as modern-master execution evidence.
