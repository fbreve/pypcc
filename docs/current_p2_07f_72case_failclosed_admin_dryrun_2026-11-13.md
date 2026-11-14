# PyPCC — P2-07F: 72-case fail-closed administrative runbook (NO PCC) — 2026-11-13

**Research-only stacked DRAFT on P2-07E #279.** No production package/source modifications or master merge. The scientific P1 SHA-256 remains `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. The final P2-08 technical execution preflight SHA is still **null**, execution is **not authorized**, and **all 72 original UCI PCC fits remain NOT_RUN**. In this P2-07F stage there are **zero synthetic PCC fits, zero real PCC fits, zero Docker invocations and zero UCI/graph/mask materializations**.

## Why this step

The earlier P2-07D demonstrated 50k-iteration **synthetic** full-fit parity plus a sampled OS watchdog. P2-07E independently proved that Linux kernel cgroup-v2 memory/pid/swap limits, tmpfs scratch `ENOSPC`, and sampled CPU/wall watchdog kills can work on the **exact archived P2-01I image**, with no PCC propagated there. However, a valid isolated process is only one part of an eventual scientific run: the original scientific protocol fixes **72 exact case identities in a strict sequence**, forbids outcome-dependent reordering, silent replacement of failed fits or an early-stop change, and caps output, time and host resources.

P2-07F tests **the administrative accounting and failure propagation only**. It carries no runnable scientific launcher. It imports no `pcc`/`pcc_step` models, no UCI row-level source material, no original graph/neighbor or mask indices and no original per-case propagation seeds. The only carried dataset-related fields are frozen UCI registry IDs, original replicate IDs, backend labels and sample-count-independent budget arithmetic.

## Immutable original case order and journal

The case inventory is derived exclusively from the original frozen P1 `selected_datasets` and `fit` sections, through the previously audited P2-07C `case_order` function. Exactly **3 UCI IDs (212,850,292) × 8 mask IDs per dataset × 3 backends = 72 original case IDs**, ordered source → mask → backend. Full original backend strings are preserved: NumPy sequential, Numba sequential nopython and Cython sequential cached-or-legacy depending on original stop mode. Scientific `max_iter=50,000`, `early_stop=false`, no retries, and eight unchanged milestones `[0,1,10,100,1000,5000,25000,50000]`. Dense `int64 neib_list` with `-1` padding is not modified or rebuilt.

Four **source-free, metadata-only** trial journals are generated:

1. One 72-ID **administrative-format validation**, logging `ADMIN_CASE_VALIDATED_ONLY` for each identity. These are **NOT completed or authorized fits**.
2. Three deliberately injected **administrative aborts** at zero-based indices **0, 35 and 71**, equivalent to first, middle and last portions of the original sequence. Each failed identity has `ADMIN_CASE_BLOCKED_FAKE`; later identities are explicitly `ADMIN_CASE_CHECK_NOT_ATTEMPTED_AFTER_FAILURE`, with no replacement/retry.
3. Every entry contains only the original case ID and `scientific_case_state=NOT_RUN_NO_APPROVAL`, `PCC_execution_count=0`. A SHA-256 chain over ordered record content detects accidental reordering, truncation, duplication and later mutation. **This is an integrity fingerprint, not a signature or proof that an external adversary could not recompute the chain.**
4. Eight adversarial source-only tests per OS check wrong order, missing rows, false fit completions, silent replacement, changed resource caps, altered P2-07E source evidence and forged P2-08 authorization. The CI independently runs on Linux and Windows and requires their entire canonical JSON reports to match byte-for-byte.

The previous Linux P2-07E five-kernel-probe evidence is **read only from its already successful source-free Actions artifact** `14088600777`, [run #43532796956](https://github.com/fbreve/pypcc/actions/runs/43532796956). The manifest validates the original image gzip SHA `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`, image config SHA `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`, frozen P1 and the original **unapproved** P2-07C policy SHA `b2b7a969cf6349ec4c315cb2a91b72bf9a82868d4304a0575af246e9d85afc04`. It does not download or load the ~291 MB container.

## Candidate resource arithmetic: not measured real resource bounds

| Proposed single-fit guardrail | Candidate | Whole proposed study calculation |
| --- | ---: | ---: |
| Fit wall time | ≤ 1,800 s (30 min) | **129,600 s = 36 h** for 72 case slots |
| Fit CPU time | ≤ 3,600 CPU-s (60 min) | **259,200 CPU-s = 72 CPU-h** possible across 72 cases |
| Entire study wall time | — | ≤ 172,800 s (48 h), with 12 h scheduling overhead slack *if every fit is at wall cap* |
| Whole single-fit process RSS | ≤ 2 GiB candidate | **Not summed** as simultaneous RSS; all fits sequential |
| Snapshot array bytes per fit | ≤ 64 MiB candidate | Each separate fit only; does not claim a 72-case real export size |
| Single-fit JSON/log output | ≤ 1 MiB + 2 MiB | 72 MiB + 144 MiB, below the proposed 1 GiB total persisted evidence cap |
| Scratch disk | ≤ 1 GiB per fit | Not necessarily 72 GiB if cleanup after each case works; **not proven yet** |

These are **arithmetical ceilings from a still-unapproved proposal**, not benchmarks of any of the real UCI graphs. A 36-hour sum of proposed 30-minute maximums **does not prove** that a 72-case study can finish in 48 hours or that native backend CPU/RSS/IO remains within the bounds. Likewise, Linux cgroup-v2 evidence from P2-07E guarantees the isolated artificial hard-limit probes, **not** that the science code handles real limit violations safely or can retry a failed fit.

## Remaining gates

This PR only verifies administrative correctness, metadata provenance, failure-stop semantics and candidate limit arithmetic. It does NOT select or seal a permanent host; implement final per-backend executable scientific runner; measure any of the 72 original UCI PCC fits; bound actual real-kNN-graph 50k CPU/RSS/disk; approve a resource policy; or generate the distinct exact final P2-08 technical manifest SHA.

**Hard STOP fields**: `P2_07_full_resources_gate_pass=false`, `P2_01_final_execution_toolchain_lock_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`, and every original case `NOT_RUN_NO_APPROVAL`. Keep this PR stacked and in DRAFT. No master merge or automatic P3 launch.
