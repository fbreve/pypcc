# PyPCC P2-07C — candidate execution-resource envelope and append-only study contract (2026-11-13)

**Status: candidate operational policy for maintainer review, NOT an approved P2 execution resource gate.** This work is stacked on prospective native audit #271 and frozen P1 SHA-256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. Original production is unchanged, no merge to `master`.

The aim is to **stop repeating historical toolchain experiments** and consolidate the evidence we have in one falsifiable proposal for how to safely bound a *future* 72-case exploratory experiment. The new verification downloads **only previous GitHub Actions metadata and NEW, byte-sealed native Cython binaries**, which it SHA-checks but does not execute or import. It does **not** retrieve UCI ZIPs, load original rows, sample masks, build graphs, compile, JIT, or execute synthetic/real PCC fits.

## Pinned evidence and what it actually measures

1. **P2-07A** (#268, run [#43478839756](https://github.com/fbreve/pypcc/actions/runs/43478839756)): same results across Linux/Windows; hypothetical co-resident static array payload **650,012 bytes** per fit at its maximum, and **14,079,360 bytes** of summed eight-checkpoint array payload across **all 72 proposed** fits. Graph widths are 22, 20 and 24 (UCI 212, 850, 292). These are NOT process RSS or actual exported file sizes.
2. **P2-07B** (#269, final run [#43483012485](https://github.com/fbreve/pypcc/actions/runs/43483012485)): original same-backend, full-fit **synthetic** pilot (500 nodes, six particles, 1,000 iterations) on two platforms. The largest process peak from the Linux original pilot was **457,093,120 bytes** (approximately 436 MiB) for Numba. Linux `ru_maxrss` and Windows `PeakWorkingSetSize` have different semantics; neither is incremental per-fit RSS or a defensible 50,000-iteration upper bound.
3. **P2-01F** (#271, run [#43488066633](https://github.com/fbreve/pypcc/actions/runs/43488066633)): prospectively compiled TWO new native Cython modules on each OS, with exact original P1 sources, later archived 13 candidate dependency wheel bytes, actual compilation/link commands, hashed GCC/MSVC executable files and new module byte seals. The new SHA256s of **all four binaries are independently reverified in this P2-07C preflight**, not inferred from labels. This evidence does NOT make the mutable runner image, complete SDK, original historical P2-01C wheel set or resolved OS shared-library bytes immutable.

The archived evidence is retained for approximately 30 days and is **not itself** a durable execution-image lock.

## Candidate guardrails — deliberately conservative proposals, not measured guarantees

These are explicit **draft limits** in `docs/current_p2_07c_candidate_operational_envelope_2026-11-13.json`, not scientifically approved run parameters beyond P1:

| Item | Candidate bound | Meaning |
| --- | --- | --- |
| Number and order | **72** fixed cases: UCI 212/850/292, 8 masks per UCI, 3 backends | Original P1 order, with deterministic backend identity |
| Concurrency | **1** real fit at a time | No unreviewed parallel native process/memory contention |
| Actual iterations | **50,000** per fit, `early_stop=false` | Exactly frozen P1, no adaptive stopping or replacement |
| Host physical RAM requirement | **at least 8 GiB** | Proposed minimum for **future selected host**, not verified by CI |
| Fit process peak resident/working-set memory cap | **2 GiB** | At least 4× above largest synthetic pilot process peak, but NOT a proven real fit bound |
| Free disk before starting | **at least 12 GiB** | Proposed safety margin, not measured full P3 disk requirements |
| One fit wall-time / CPU time | **30 minutes wall / 60 CPU-minutes** | Hard stops requiring a future audited runner/monitor |
| Entire study wall-time cap | **48 hours** | Single-fit ceilings sum to 36 hours, leaving 12 hours overhead; NOT a performance forecast |
| Original checkpoint array cap | **64 MiB per fit** | Preserves original observer cap |
| Per-fit aggregate JSON / diagnostic text log | **1 MiB / 2 MiB** | Truncation/abort must be documented; never print original records |
| Per-fit scratch | **1 GiB** | Temporary working-directory allowance |
| Total persisted evidence | **1 GiB** | 72 × (1+2) MiB = **216 MiB** JSON+logs cap; remainder reserved for integrity/index records |

The literal numeric limits are *candidate engineering decisions* and must be reviewed by the maintainer before implementation or use. Passing the CI means only that these proposals are **internally consistent with prior static and synthetic records**, not that the 72 real fits will meet CPU, wall, RSS or disk limits. A future resource monitor must account for real process RSS, CPU time, disk writes and wall time and terminate/record a violating case; the current audit **does not start such a process or prove monitor behavior in a PCC run**.

## Scientific and privacy guardrails

- Case IDs follow **registry index asc → mask replicate asc → frozen backend order**, always exactly 72 unique entries. This stage computes **a hash of these case IDs and statuses**, not any real masks, graph arrays or label indices; all statuses are `NOT_RUN_NO_APPROVAL`.
- The analysis will use **only initially unlabeled nodes** to count label changes from iteration 5,000 to 50,000, including deterministic first-argmax tie handling, exactly as specified in P1; original labeled nodes never enter the primary denominator. No evaluation result may change masks, graph/neighbor order, seed or backend choice.
- Future output policy **prohibits** publication of participant/feature rows, original target-vector sequence, per-node checkpoint arrays, selected original label indices or graph neighbors in CI. Candidate reports are limited to **per-case aggregate counts**, change/tie counts, eligible denominator, accuracy/balanced-accuracy aggregates and cryptographic fingerprints. This may require review for ability to reproduce all P1 secondary metrics; do not silently remove an approved metric.
- All attempts must be **append-only**, including interrupted/failed cases. No silent replacement, rerun with another seed or failed-case dropping. An unqualified 72-case scientific summary is forbidden if even one case fails or is missing.
- The current developer-facing script is intentionally **read-only, stdlib-only** and has no executable PCC backend path. It can only validate prior source artifacts and create a no-fit proposed report.

## Mandatory STOPs remaining before P2-08

1. **The execution host is not chosen or immutably sealed.** Linux CPython 3.12.16 is a *candidate primary*; Windows CPython 3.12.10 is a crosscheck, not a user-approved machine selection. New P2-01F bin blobs and partial compiler hashes are not a complete OS/toolchain/SDK closure.
2. **The real 50,000-iteration CPU/RSS/disk resource ceiling has not been demonstrated** for the three actual graphs and backends. Synthetic measurements and static array byte-counts must not be reported as such.
3. **The candidate numeric RAM/time/disk/log/retention thresholds and export policy have not been approved.** A real resource monitor with reliable process termination and append-only failure evidence is not yet verified.
4. **The distinct immutable P2-08 technical execution manifest SHA does not exist**. Neither P1 science approval nor passing staged PRs automatically authorizes P2/P3. An explicit maintainer approval of the exact finalized SHA is still required before **any** real PCC propagation.

Regardless of success: `P2_07_full_resources_gate_pass=false`, `P2_01_final_execution_toolchain_lock_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, **zero real PCC fits**, no production edit/merge.

**CI expectations:** seven pure synthetic contract/unit tests on each OS, then a third job downloads only prior metadata + the 4 NEW compiled native files from exact run IDs, verifies all binary SHA256, provenance and proposed case/resource equations, publishes **metadata-only** report. Actual result status must be read from the Actions run, not inferred from a proposed workflow.
