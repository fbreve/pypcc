# PyPCC — P2-07E actual Linux cgroup-v2 resource enforcement on byte-locked OCI, **NO PCC** (2026-11-13)

**Research-only stacked DRAFT on #278.** No production source edit, no master merge, no official UCI source loaded, no real kNN graph or mask built, **zero synthetic or real PCC propagations in this PR**. Frozen scientific P1 SHA256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166` remains unchanged. Technical P2 final SHA remains null, execution not authorized.

## Narrow hypothesis

The original P2-07D #273 proved synthetic 50k within-backend observer state parity and used sampled **OS process-level watchdogs** on Linux and Windows. P2-01H #276 and P2-01I #277 established an import-only OCI runtime with an **exact exported image**, independently replayed on multiple Linux runners. P2-01J #278 extended temporary retention of those exact image bytes to **2027-02-11**.

This PR tests whether the actual existing image can be run as a constrained, offline, read-only container with Linux kernel-enforced **cgroup-v2 memory and PID limits, CPU throttling, and tmpfs scratch quota**, and with a fail-closed external watchdog enforcing a deliberately much smaller synthetic **wall-clock deadline and cumulative cgroup CPU-use threshold**. It **does not run the PCC kernels or estimate the real UCI 50,000-iteration memory/CPU/time needs**.

## Exact source environment; no floating rebuild

Import image **only** from the previously copied P2-01J [successful run #43531054368](https://github.com/fbreve/pypcc/actions/runs/43531054368), exact artifact `14087551705` (expires 2027-02-11). Recompute **actual whole `docker save` gzip SHA256** `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`, validate source-committed P2-01I image lock, `docker load` the existing bytes with no registry pull, then demand original image ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`. No new image build or Python reinstall.

All five **stand-alone Python standard-library resource probes** are mounted read-only from `experiments/p2_07e_cgroup_probe_no_pcc.py` into the exact container, invoked by an explicit original CPython3.12.16 `--entrypoint` override. The original image's primary entrypoint is an import-only verifier; the test probe does **not import PCC, scikit-learn, NumPy, Numba or Cython**, never calls `fit_predict`, and contains no algorithm or datasets. The whole container root filesystem is `--read-only`, network is `none`, user `65534:65534`, all capabilities dropped, no-new-privileges active, no external filesystem write mounts. Its only scratch volume is `--tmpfs /tmp:rw,nosuid,nodev,noexec,mode=1777,...`. Every container has `--pids-limit 128`, `--cpus=1` and `--memory-swap` equal to `--memory`, prohibiting memory swap beyond the specified cgroup memory hard cap.

## Five actual kernel/parent fail-condition experiments

| Case | Stand-alone probe | Required verified result |
| --- | --- | --- |
| Baseline | Read real `/sys/fs/cgroup/{memory.max,memory.swap.max,memory.peak,memory.events,cpu.max,cpu.stat,pids.max}`, plus mounted tmpfs size | **2 GiB** cgroup memory.max, **0** additional swap, CPU quota at most one core, pids max 128, tmpfs ≤64 MiB, clean exit 0 |
| Forced wall | Sleep 60 s, synthetic parent cap **1.1s** | Parent `docker kill` on elapsed wall-time; container exits nonzero, no OOM |
| Forced aggregate CPU | Spin up to 60 process CPU seconds, sampled CPU limit **0.65 CPU seconds** | Parent reads actual cgroup-v2 `cpu.stat usage_usec` via `docker exec` and kills container above tiny threshold; nonzero exit, no OOM |
| Forced memory | Allocate and touch **256 MiB** inside container with **128 MiB** hard memory.max (swap equal to memory) | **Actual kernel `OOMKilled=true`** from Docker inspect; exit 137/9 |
| Forced scratch | Write 8 MiB inside **1 MiB** kernel-backed tmpfs | Actual Python `ENOSPC`, exit 42 and source-free diagnostic, no OOM |

The CPU quota (`--cpus=1`) **throttles instantaneous CPU use but does not enforce a total per-fit CPU-time budget**. The sampled parent loop additionally reads cgroup CPU usage counters and kills on a small experimental threshold. These readouts include the CPU cost of `docker exec` probes, so they can **conservatively overcount** a synthetic child workload; they are not yet a validated continuous total-CPU accounting service. Polling at ~120ms plus Docker command overhead can miss fast changes; cgroup hard memory/tmpfs limits are the kernel authority, independent of polling. The final production-grade 72-case supervisor is **not supplied or approved**.

The output is restricted to five statuses, bounded timings, source-free CPU/memory/scratch counters and a ten-line append-only `STARTED` / `EXPECTED_RESULT` ledger. A failure produces an append-only `FAILED` row; a case is not retried or replaced. Docker containers are removed in `finally`, and the test checks that no named container is left. Every operation is limited to a random unique `pypcc-p207e-*-` prefix.

## Scope boundary and next gate

Expected CI: six source-only adversarial protocol tests on Linux and Windows, followed by a separate Linux Docker cgroup-v2 job consuming the **already archived** image without uploading another ~291 MB copy. CI must fail if actual kernel OOM, tmpfs ENOSPC, or sampled CPU/wall termination are not detected; no fake PASS or substitute successful exit permitted.

Success would establish that the **existing** exact CPython/Cython image can be started with OS-enforced limits and real synthetic kill paths, more relevant to final host supervision than earlier bare-process checks. It **would not** establish the resource ceiling for any of the 72 original UCI fits, select the final execution host, produce an immutable full scientific runner image, validate Numba JIT execution resources, prove an upper limit under real kNN graph degree distributions, approve the P2-07C budget, or seal P2-08.

**Mandatory final states even after CI PASS:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. No production merge.
