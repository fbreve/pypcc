# PyPCC — P2-07I: whole-study fail-closed **administrative only** STOP policy (2026-11-13)

**Research-only stacked DRAFT on P2-07H #282.** No production source edits, no merge to master, no dataset download/row materialization, no kNN graph or label mask generation, no Python native/PCC import, no Docker invocation, no synthetic or real **PCC** propagations. Every one of the original frozen P1's 72 scientific cases remains **`NOT_RUN_NO_APPROVAL`** and the distinct final P2-08 execution SHA is **null**. All simulations here are *administrative bookkeeping*, not launched scientific workers.

## Why this stage exists

P2-07E #279 proved the Linux cgroup-v2 resource containment with standalone synthetic processes. P2-07F #280 validated the **72 exact original case identities, order and draft study-wide budget arithmetic** and three injected failures. P2-07G #281 verified Linux host **PID+process-starttime** cleanup for four artificial three-process trees. P2-07H #282 combined actual kernel OOM/tmpfs, sampled wall/CPU timeouts, normal process exit and PID1 crash with full **18/18 descendant cleanup** on the same original exact OCI image, again **only artificial stdlib processes**, not PCC or real UCI graphs. All technical stages are DRAFT and all final scientific resources/host locks remain unapproved.

P2-07I now audits the **study-wide STOP decision policy** which an eventual approved 72-case runner would need. It **does not implement or call that runner**. Instead, for every possible simulated failure, this module checks that no further case identity is marked as even administratively validated, much less actually run. Existing source-free P2-07H evidence is retrieved only as the successful [run #43547341278](https://github.com/fbreve/pypcc/actions/runs/43547341278), source-free artifact **14094039842** (expires **2026-12-14**), with exact previously verified canonical JSON SHA256 **`0f96c4c91afbd0840f50adc430f42017207b0da71e3a6126c012d50d804388db`** and original 30-event chain tail **`964a40c45f124a1aba00bfec42603bca0136a426e1dcec282fb9ff2f3fafaf48`**. P2-07I does not download the ~291 MB source OCI image. Exact P1 frozen science SHA256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`; original unapproved P2-07C resource policy SHA256 `b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04`. The existing OCI payload SHA256 remains `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8` but is **not loaded/executed** by this PR.

## Frozen original case identity

The source-of-truth is the original P1 and P2-07F `schedule`: exactly **3 UCI registry identifiers (212,850,292) × 8 label-mask replica IDs per dataset × 3 original backend labels = 72 IDs**, in source → mask → backend order. It preserves 50,000 proposed iterations per fit, eight frozen milestone iterations, `early_stop=false`, and the deliberate dense `int64 neib_list` with `-1` neighbor padding. **None of these scientific parameters, actual masks, seeds or arrays are modified, created or consumed**.

## 22 administrative-only campaigns

The 22 deterministic original-ID scenario logs consist of **one complete 72-ID schema inspection** and **21 simulated failures** (seven fault classes × first/middle/last original case index **0/35/71**).

| Fault injected artificially | Why an eventual scientific runner must STOP |
| --- | --- |
| `WALL_DEADLINE` | Single-worker wall deadline reached; no continued fit |
| `CPU_TIME_BUDGET` | Total cumulative CPU-seconds budget exceeded |
| `CGROUP_MEMORY_OOM` | Kernel observed OOM, not merely an estimate |
| `TMPFS_SCRATCH_ENOSPC` | Bounded temporary storage exhausted |
| `WORKER_ROOT_FAILED` | PID1/Python worker returns nonzero |
| `DESCENDANT_CLEANUP_UNVERIFIED` | Even if root exited, a child PID might remain |
| `STUDY_WALL_DEADLINE` | 48-hour proposed **study-wide** deadline exceeded |

**All seven reasons are synthetic metadata labels in P2-07I.** They do not mean the actual original UCI workloads suffered any of these failures.

Every one of the 22 logs has all **72 original case-ID rows** for audit completeness, with one of only three states: `ADMIN_ID_CHECKED_ONLY` (a metadata-only inspection before fault), `ADMIN_FAKE_STOP_NO_FIT` (the deliberately denied case, never launched), and `ADMIN_CASE_UNREACHED_AFTER_FAKE_STOP` (all remaining case IDs administratively not reached). This enables analysts to see how many original case *identities* were skipped after a failure, without wrongly presenting a partial scientific experiment as successful. Every record explicitly states `original_scientific_fit_state=NOT_RUN_NO_APPROVAL`, `PCC_worker_invocations=0` and `real_UCI_feature_rows_loaded=0`.

Total **1,584 metadata-only rows**, one hash-chained journal per scenario, output as a single bounded JSON-lines log. The log is created with exclusive `xb` mode, individually flushed and `fsync`ed per record, is never overwritten and has a total size limit of 1MiB. Every log row includes the previous SHA256 and the current canonical record SHA256, while the final JSON output stores each scenario's terminal SHA and the digest of the **whole persisted original journal**. This is corruption detection, not a tamper-proof signed log. The final validator forbids retries, reordering, duplicated or truncated rows, false `FIT_COMPLETED` states, wrong source hashes, falsified P2 GO, and any further `ADMIN_ID_CHECKED_ONLY` after the first simulated failure.

## Resource budget still unapproved

The same original P2-07C proposal is checked arithmetically: 72 × 1,800 wall seconds = **129,600 s (36h)** of per-case wall slots, compared with proposed **172,800 s (48h)** entire-study cap. It also proposes 72 × 3,600 CPU-seconds = **259,200 CPU-s (72 CPU-hours)** of total CPU slots; single-fit memory 2 GiB process RSS; scratch ≤1 GiB per fit; 72 MiB JSON and 144 MiB text-log caps under a candidate 1 GiB total exported-evidence cap. **These are unapproved ceiling arithmetic, not real measured resource usage, not guarantees, and not cumulative memory usage.** The absence of a real original UCI 50k-iteration resource bound still blocks P2-07.

## CI gates and precise scope

Ten adversarial unit tests run on **Linux and Windows Python 3.12**, then both runners independently retrieve and verify the pinned **P2-07H original source-free report**, generate and revalidate all 22 administrative logs and 1,584 fsync'd original case rows. A separate third runner compares the **entire** canonical JSON and audit JSONL bytes across both OSs, not merely a summary hash. The 1MiB bounded audit and report are archived; no duplicate OCI image, original feature/label data or per-node vectors are exported. Normal **PCC production regression** runs separately.

The policy is a **simulation of what a future approved runner must enforce**. It **does not implement or test** actual scientific subprocess launching, end-to-end 48h timeout, scientific crash recovery, P1 original per-case RNG verification, real graph 50k resources, full host/compiler/OS image closure, permanent OCI storage, or an execution manifest approved by its owner. It does not promise any of those gates can be closed without real evidence.

**Mandatory still-NO-GO**: `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`; every scientific case remains NOT_RUN. A distinct final P2-08 technical SHA and explicit maintainer authorization are required **before** the first real UCI PCC fit.
