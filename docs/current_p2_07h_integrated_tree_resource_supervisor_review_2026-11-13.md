# PyPCC — P2-07H: integrated six-mode container resource + process-tree supervisor, **NO PCC** (2026-11-13)

**Research-only stacked DRAFT, based on P2-07G #281.** Frozen P1 scientific SHA256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`; all 72 original UCI cases remain `NOT_RUN_NO_APPROVAL`. No production source edits or merge, no UCI data, native PCC import or compilation, graph/mask building, synthetic or real **PCC** propagation, or P3 execution permission.

## Experiment

P2-07E #279 verified five individual synthetic Linux kernel/host quotas. P2-07G #281 verified the Linux cleanup of four independent source-free Python root → child → grandchild trees (including a new process session, SIGTERM-resistant descendants and original PID-1 crash). P2-07F #280 verified the original 72-case P1 administrative case ordering, no retries and a still-unapproved operational budget. **P2-07H tests the interaction of those mechanisms in ONE host supervisor** on the **same exactly exported OCI image**, using a single fixed execution/monitoring/cleanup path for all six artificial three-process workers, with a barrier preventing the synthetic work until the three host PIDs and kernel quotas are verified.

The image is read from the existing P2-01J Actions artifact `14087551705` (expires 2027-02-11), Docker-save gzip SHA256 **`008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`**, Docker config ID **`sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`**. Only offline `docker load`; **no pull, build, image re-export, pip install or network**. The original P2-07F metadata-only report (canonical SHA `d6c7fedc5abb3bf48395a3b223bb1112c11ba6a21593075c55e4a02afabd1328`) and original P2-07G process-cleanup source-free receipt are independently validated before running any probe.

The only worker executable is `experiments/p2_07h_unified_synthetic_three_process_fixture_no_pcc.py`, a mounted read-only **standard-library-only** fixture in the original OCI. The original image is import-only; the supervisor overrides its entrypoint strictly to run the artificial fixture. Each container uses `--network none --read-only --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --cpus 1 --memory-swap` equal to `--memory`; the **only writable filesystem is bounded tmpfs /tmp**. Exact Linux original host PID+starttime identities are obtained from `docker top` and `/proc/<pid>/stat`, with each PID confirmed to share the same cgroup. The parent releases a single `/tmp/p207h-go` file only after verifying three fixed processes and the actual cgroup `memory.max`, `memory.swap.max=0`, `pids.max=128`, `cpu.max` and tmpfs available space.

## Integrated source-free scenarios

| Three-process scenario | Small deliberately artificial guardrail | Expected proof |
| --- | --- | --- |
| `normal` | 2 GiB RAM, no swap, 64 MiB tmpfs; three children exit after host release | Docker exit **0**, all **three** originally observed host PIDs gone |
| `wall` | Same hard caps, host wall watchdog **1.20 s** after release | Forced nonzero stop and full three-PID cleanup |
| `cpu` | Same hard caps, host cumulative cgroup `cpu.stat usage_usec` threshold **0.75 CPU-second** after release | Forced nonzero stop, kernel CPU quota still ≤1 CPU, all three original PIDs gone |
| `memory` | Actual kernel **128 MiB** `memory.max`, swap=0; grandchild touches **256 MiB** | Genuine kernel cgroup `memory.events oom_kill` increment (or container `OOMKilled`), followed by full tree stop and cleanup |
| `scratch` | Actual **1 MiB** kernel tmpfs; grandchild attempts 8 MiB | Real `ENOSPC` reported from grandchild and full tree stop/cleanup |
| `crash` | Same hard caps; synthetic root exits **23** while children exist | Exit 23 preserved; container cgroup descendants all disappear |

Every scenario uses the **same** host supervision loop for cgroup use accounting and fail-closed termination, then requires **`docker rm -f`, no inspectable container, and disappearance of all three original host (PID,starttime) identities** before proceeding. Total: **18 actual original host process identities** must disappear across six cases. The memory case tests an especially relevant failure mode: the kernel might kill only a **grandchild**, leaving PID1 alive; the parent monitors `memory.events oom_kill` and must kill the **whole tree**, rather than reporting success when the grandchild alone dies.

The kernel's memory/swap/tmpfs/pid quotas are hard bounds, but **CPU quota is an instantaneous rate bound, not a cumulative CPU-seconds limit**. The host CPU watcher is sampled (and its `docker exec` counter reads also contribute CPU usage), so it may overcount workload CPU and miss short transient events. The wall guardrail is likewise sampled. Neither the chosen 0.75 CPU-s / 1.20 wall-s thresholds nor the 128MiB/1MiB failure injections are the proposed scientific P2-07C limits for real datasets; they are artificially small so each failure mode is exercised predictably.

## Evidence and fail-closed behavior

The integrated host supervisor appends only **five bounded, JSON-lines and SHA-chained, `fsync`ed events per scenario**, totaling 30 for success: `STARTED`, `TREE_AND_HARD_CAPS_VERIFIED`, `RELEASED_SYNTHETIC_BARRIER`, `TERMINATION_DECISION_RECORDED`, `ALL_PIDS_AND_CONTAINER_GONE`. On any error it adds `FAIL_CLOSED`, invokes Docker cleanup unconditionally and fails the CI **without retrying that case or executing the next one**. Logs contain only artificial modes, quotas, counters, exit statuses and SHA digests. This hash chain detects accidental alteration, not an adversary capable of rewriting it and recomputing hashes.

Eleven adversarial source-only tests run on **Linux and Windows** (false P2 authorization, missing OOM/ENOSPC, wrong cgroup quotas, surviving child/grandchild, malformed SHA journals and forbidden worker identities). The **actual** OS/process-tree integrated tests run **only on Linux cgroup v2**; Windows source-only checks **do not prove** Windows Job Object process termination or resource quotas. The original production regression is independent and must complete before the scientific tooling is described as CI-green.

## Actual integrated CI evidence

Successful [technical workflow #43547115492](https://github.com/fbreve/pypcc/actions/runs/43547115492) completed **3/3 PASS**: **11/11 source-only adversarial tests per OS** on Linux and Windows, plus all six **actual** Linux original-OCI synthetic tree/quota modes, checked by a separate independent report-and-journal gate. The previous two failing attempts were deliberately fail-closed and left failure-only artifact evidence; the fixes addressed (1) cgroup counter command syntax and (2) container exit versus concurrent `docker exec` races, including the case where the Linux kernel OOM kills the container and Docker returns empty stderr. A counter failure is accepted **only when a new exact-container Docker inspection finds it stopped**; an unexpected live-container failure still blocks.

| Mode | Actual source-free termination reason | Root exit | Three host PID/starttime identities gone |
| --- | --- | ---: | :---: |
| `normal` | Explicit clean three-worker completion | 0 | PASS |
| `wall` | Sampled host wall watchdog | 137 | PASS |
| `cpu` | Sampled host cgroup CPU usage ≥0.75 CPU-s | 137 | PASS |
| `memory` | Genuine kernel cgroup OOM while grandchild allocates | 137 | PASS |
| `scratch` | Genuine tmpfs `ENOSPC` from grandchild | 137 | PASS |
| `crash` | Synthetic root stops with code 23 | 23 | PASS |

One original source-free archive uploaded, **not a duplicate Docker image**: artifact **`14094069243`**, enclosing GitHub ZIP digest `81b1ac17d9798bf45fa708a86760e05318f7bd63ce9e705764a94d9844a29e40`, expires **2026-12-14T02:35:56Z**. The 30-event append-only journal's final SHA256 is **`02a1788eea786cf041c21bc593bc10697539e652685cd174e9d3f4c354fb54d7`**. The source-free report records all **18 original host identities disappeared**, the **real kernel OOM flag**, the **real tmpfs ENOSPC flag**, and **zero PCC fits**. Last sampled CPU `0.779799 CPU-s` in the CPU-threshold mode is measured cgroup usage **including the Docker monitoring subprocess overhead**; it must not be compared to scientific worker CPU usage or treated as a real-fit bound. The last memory-peak sample before kernel OOM is **not** the actual allocation peak; the kernel OOM event is the authoritative failure proof.

## Explicit remaining blockers

Even six successful integrated artificial three-process tests do **not** prove that the real P1 experiment runner exists in this OCI image, that 72 original UCI graphs fit the proposed 2GiB/1800s/3600CPU-s/1GiB scratch limits, that all real native Numba/Cython dependencies or source-build compiler/OS SDK closures are sealed, or that a real worker's external subprocesses cannot escape the selected resource control. The durable local/NAS archival destination is still not selected, although an exact binary image is currently temporarily preserved to February 2027. No 72-case scientific launcher is called in this stage.

**Never infer scientific P2 GO:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. A separately reviewed, exact final P2-08 SHA plus explicit maintainer authorization is required before any of the 72 original real-data PCC fits.
