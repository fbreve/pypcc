# PyPCC P2-01I — immutable EXACT previously exported OCI binary runtime lock (2026-11-13)

**Research only; stacked on #276; NO P2 execution authority and zero official UCI PCC fits.**

P2-01H #276 demonstrated an important but narrower capability: the exact exported Ubuntu26/CPython 3.12.16/13-wheel/two-NEW-Cython binary image could be loaded and imported on a **second separate CI host**, both in an offline, read-only Docker container with no propagation or dataset access. The whole exported image was preserved as Actions artifact 14081565237 from passing run 43516085668. However, repeated *fresh* builds from the same floating `ubuntu:26.04` discovery and CPython patch had **different copied Python runtime-tree SHA256**, so a Dockerfile recipe alone did not demonstrate an identical binary rebuild.

P2-01I addresses that gap honestly by **pinning the exact known-good binary image bytes** in the versioned repository manifest `docs/current_p2_01i_exact_OCI_binary_replay_lock_2026-11-13.json`. The future execution-image candidate must reference this exact exported artifact by content hash, not re-run `docker pull ubuntu:26.04` or silently rebuild a replacement.

## Binary lock (NOT an execution approval SHA)

| Evidence | Exact frozen value |
| --- | --- |
| Passing technical source workflow | `43516085668` (PR #276) |
| OCI base manifest ref | `ubuntu@sha256:065aa8626841f16057b78b027b464cd59c4fac36c204c72dcbf48d66f5fa8a96` |
| Original exported image config ID | `sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55` |
| Complete `docker save` gzip image SHA256 | `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8` |
| Exact copied CPython 3.12.16 tree SHA256 | `cd4be74d169340445a87c34c1dace8e92598ab87b1625ea52ee8d31bad1590a7` |
| Original host image candidate differing CPython tree | `496e9d7318d30d9300c949b75cf1709d65a73c2281289ee8786aaddb777f455d` (earlier valid run, NOT identical rebuild) |
| New original Cython production module | `cec70629e10b1f6e9f53274d571a52ad2177d31b89d2c189840f3abda2944a8a` |
| New isolated research Cython module | `1ea9f1c151df795071b3fe449df2a3e35ab72bb79156b073df9f8297c66d899f` |
| 13 exact later candidate wheels | Individually hashed in archived #264 and image manifest |
| Original source-only helpers | `pcc_metrics.py`, `pcc_validation.py`, original production Git blobs |
| Selected additional native library | Separately SHA-pinned original host `libffi.so.8`, **not part of minimal Ubuntu base** |

The image archive is accessible within exact GitHub Actions artifact `14081565237`, GitHub ZIP digest `cbca5754de9b32048254b0875bffb30ac3bd6510df3168eeff6fb44fe203dead`. The exact independent second-host receipt artifact is `14080014146`. **Both are temporary** with expiry **2026-12-13**, so the project must agree on a durable private archive or registry before these hashes can serve as long-term execution inputs. Do not publish or push the environment to an external registry implicitly.

## Verification contract

The new job **does not build Docker images**. On Linux and Windows, pure-stdlib adversarial tests reject wrong source run, OCI archive digest, config ID, CPython/native SHA, fake claim of source-reproducible compiler/SDK closure, floating tag use, and any attempt to turn this lock into P2 approval. On an independent Ubuntu runner, the workflow:

1. Downloads **only** the original P2-01H exported OCI archive and separate-host safe import receipt from exact run #43516085668 (no original UCI rows or PCC datasets).
2. Independently calculates SHA256 over the actual full `docker save` gzip bytes and compares original import-only receipts from two separate CI hosts to the exact source-committed reference lock.
3. Performs `docker load` of that **existing archive** with no registry pull; requires the **exact same image config ID**, linux/amd64 architecture.
4. Launches the **same** archived image `--network none --read-only --cap-drop ALL --security-opt no-new-privileges --memory 2g`; runs only its embedded import-only manifest/tree/native ELF byte checker and compares it with the prior two imports.
5. Archives only safe per-host receipt hashes and false-GO status. The large OCI archive is **not copied into another GitHub artifact**, avoiding redundant 290 MB storage.

This is **image-binary replay reproducibility**, not deterministic rebuilding of CPython/SDK from source, a complete frozen scientific PCC runner, a durable full OS/toolchain build, or a real 50,000-iteration CPU/RSS/disk guarantee. A future scientific environment needs original PCC code at its P1 pinned Git revision, approved real graph/seed/mask inputs and a resource supervisor, followed by the **separate explicit approval of an exact completed P2-08 execution SHA**. This PR itself does none of those.

**Non-negotiable STOPs:** `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. Source P1 scientific SHA remains `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. Keep DRAFT, no merge to master.
