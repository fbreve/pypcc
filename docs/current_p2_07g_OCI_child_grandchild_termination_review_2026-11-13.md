# PyPCC — P2-07G host-confirmed OCI descendant cleanup and fail-closed journal (2026-11-13)

**Stacked research DRAFT on #280.** Original P1 SHA256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. No production change, no master merge. The full scientific P2 execution preflight SHA is **null**, `P2_execution_authorized=false` and **all original 72 real PCC cases remain `NOT_RUN_NO_APPROVAL`**. This stage performs **zero synthetic or real PCC fits, zero UCI dataset downloads or rows, zero graph/mask materialization, and zero native kernel/JIT execution**.

## Why P2-07G exists

The genuine P2-07E cgroup-v2 Linux tests #279 proved that kernel memory/CPU quota/tmpfs limits and host CPU/wall watcher can terminate a **single synthetic Python process** in the original content-identified OCI image. P2-07F #280 then verified all original 72 P1 case identifiers, strict original ordering, draft budget arithmetic and fake stop positions, without scientific execution. Those steps do not yet prove that a failed worker leaves **no living Python child/grandchild process** or that a parent-process death cannot leave one working in the background.

P2-07G tests **four bounded, fully artificial, three-process trees**, each started inside the EXACT same previously saved OCI image from P2-01J/01I. Each tree has root → child → grandchild, created only with the Python standard library; there are no particle kernels, original UCI data, graph edges/masks, research processes or scientific runner. The original image `docker save` gzip SHA256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8` and loaded Docker config ID `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55` must be recomputed/verified. The image is pulled from existing temporary artifact #14087551705 (currently expires 2027-02-11); no Dockerfile rebuild/registry pull or new large artifact duplication.

The fixture `experiments/p2_07g_synthetic_descendant_tree_no_pcc.py` is mounted **read-only** into that original image using a strict `--entrypoint` override. Containers run `--network none --read-only --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --memory 2g --memory-swap 2g --cpus 1`, with only **64MiB** of writable tmpfs. No host directory is mounted writable.

## Four actual descendant failure modes

| Fixed source-free test | Failure injected | Evidence needed for PASS |
| --- | --- | --- |
| `normal_tree` | Host `docker kill` after root, child and grandchild are observed | Container exits nonzero and all three **original host PID + kernel process starttime** identities disappear |
| `detached_tree` | Child and grandchild create a new Linux session, but remain within the same Docker cgroup; host kills the container | All three original process identities were observed in **one Linux cgroup** before termination and all three disappear |
| `stubborn_tree` | All artificial Python processes explicitly ignore `SIGTERM`; host calls `docker stop --time 1` | Docker escalates to termination and all three original host identities disappear |
| `synthetic_crash` | After a confirmed host tree snapshot, supervisor writes a synthetic marker; PID-1 Python root exits with **status 23** while descendants are alive | Docker/container runtime terminates remaining descendants; no observed original host process remains and original root error code is preserved |

The Linux host obtains actual PIDs using `docker top -eo pid,ppid,comm`, verifies all processes belong to the **same cgroup** using `/proc/<pid>/cgroup`, and reads each process's original `/proc/<pid>/stat` field 22 (`starttime`) before termination. Following container termination and `docker rm -f`, it rechecks each original **(PID,starttime)** pair until gone. This avoids a false survivor conclusion caused by a host PID being recycled for a different process. **A zombie with the same original identity remains a failure until reaped.** It also checks `docker inspect` can no longer find the removed container. The host cannot claim all possible future subprocess escape mechanisms are impossible; only **the twelve actual, observed processes** in these four controlled scenarios are proven absent.

On any error, the supervisor writes a bounded `FAIL_CLOSED` record and unconditionally runs Docker cleanup in `finally`, then raises a CI failure. **No retry or alternate case** is allowed. Recording an administrative failure is not a grant to proceed to the next real case.

## Journal and independent tests

Every case creates a five-event, flush-and-`fsync`, **append-only** SHA-256 chained JSON-lines journal: `STARTED`, `THREE_DESCENDANTS_SAME_CGROUP`, `TERMINATION_REQUESTED`, `ROOT_EXIT_NONZERO`, `ALL_ORIGINAL_PROCESS_IDENTITIES_GONE`. The 20-record successful proof includes only case mode, bounded process counts and exit codes, fixed no-PCC flags and previous/current hashes. This is a deterministic **integrity chain, not a cryptographic signature**. Failure appends `FAIL_CLOSED`; a truncated journal **cannot** pass the full result gate. No raw UCI features, labels, masks, graph adjacency, native RNG states or numerical propagation arrays are created or exported.

Nine adversarial source-only tests run on Linux and Windows and reject forbidden image-source drift, wrong container isolation, extra/unexpected PID, one surviving descendant, reuse of PIDs, missing/truncated/tampered journal, forged scientific fit or a false final P2 authorization. The actual descendant/OS proof is **Linux-only**; Windows CI tests the source-contract and must not be described as a Windows Job Object/whole-tree kill proof. The Linux job independently imports source-free P2-07F #280 metadata artifact, whose exact report SHA256 is `d6c7fedc5abb3bf48395a3b223bb1112c11ba6a21593075c55e4a02afabd1328`, and verifies it still reports all 72 scientific cases NOT_RUN.

## Remaining execution blockers

Even 4/4 real *artificial* descendant cleanup PASS would **not** prove the final original PCC runner has equivalent process-tree properties, durable host storage, real UCI 50k memory/CPU time/disk budgets, acceptance of interruptions and baseline/observed exact parity under every real graph degree distribution, a source-reproducible CPython+SDK+OS toolchain, or an approved final P2-08 execution manifest.

These fields must remain exactly **false**: `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. Keep as stacked DRAFT research PR, no merge to production and **no scientific P3 launch**.
