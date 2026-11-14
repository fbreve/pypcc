# PyPCC — P2-01J: verified interim archive and opt-in durable local/NAS backup (2026-11-13)

**Stacked research DRAFT on P2-01I #277. No production changes, no master merge, no PCC fits.**

## Purpose and original source bytes

The existing P2-01I #277 proves that a particular **already exported OCI/Docker binary image** can be moved to independent hosts without rebuilding Python or any Cython extension. This experiment addresses the **expiration risk**, but separates **temporary GitHub Actions retention** from actual *durable private backup*.

Original full archived OCI image source: P2-01H [successful run #43516085668](https://github.com/fbreve/pypcc/actions/runs/43516085668), artifact **14081565237**, current expiration **2026-12-13**. Its actual `p2-01h-exact-runtime-image.tar.gz` payload has **SHA256 `008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`** and its Docker config ID is **`sha256:b97f87c6698a3404912807ae0b7d4659b86012eb95a2f26a67b25b4274e77e55`**. The source-committed canonical #277 P2-01I lock SHA is `7fa90329779980f1b8739428796617a2da1ba872dac2e4c02d23fb07db3f5340`.

The original image was already proven to import two **NEW** P2-01F Cython extensions, 13 exact later candidate wheel distributions, and only the two original Python native import helpers on a pinned Ubuntu26 OCI base, with separately SHA-verified host `libffi.so.8`. **No official UCI source, graph, mask, PCC model runner or actual fit is included.** This stage reuses exact pre-existing bytes and does not execute Docker, PCC propagation or dataset code.

## Part A: optional safe temporary retention extension — same repository, no registry push

The workflow `research-p2-01j-exact-OCI-interim-archive-no-pcc.yml` runs six pure-stdlib adversarial contract tests on **Linux and Windows**, then **attempts** to query the repository's actual `actions/permissions/artifact-and-log-retention` policy. **Observed first CI run: the restricted `GITHUB_TOKEN` received HTTP 403**, so the workflow must NEVER describe the repo setting as verified. In that case it submits **90 days as an explicitly unverified REQUEST only**, lets the GitHub upload action enforce the account setting (or fail), and verifies the resulting artifact's real expiry afterward via the Actions artifact response. If that metadata cannot be retrieved, the procedure cannot claim the requested duration. It calculates a new upload retention of **min(90 days, repository configured days, repository/owner maximum)**. Public repositories cannot exceed 90 days, and the repository's existing policy can be shorter. Crucially, changing this setting **does not extend the older artifact**. A new GitHub artifact is the only permitted temporary copy; no false promise of perpetual storage or hidden visibility on a public repository.

Only after re-verifying the **exact whole OCI tar.gz SHA**, correct originally archived config ID, frozen original P1/P2-01I lock, and the *original first and separate second host import receipts*, CI uploads **one new identical binary** plus a small source-free receipt. It uses `compression-level: 0` to avoid needless CPU for an already compressed image. A **second separate CI runner** downloads this just-created Actions artifact, rechecks the exact same whole-file SHA, receipt and the frozen **false-GO** flags.

**Important:** Actions storage counts toward potential account costs. This creates at most one new ~291MB temporary archive per completed workflow run, and the workflow is **not scheduled** to renew indefinitely. Re-running the PR workflow can create additional copies; avoid unnecessary re-runs. Its actual artifact ID/expiry must be confirmed in the successful Actions run. If repository retention configuration permits only 30 days, this step cannot extend to 90: **do not report 90 days without evidence**. To allow up to 90 days, repo owner can change Actions artifact retention settings (which affect only subsequently created artifacts), subject to repository and organization policy.

This workflow performs **no release creation, GitHub Package/Container Registry push, private storage upload, repo settings change, Docker build/pull, UCI download, graph/mask generation, synthetic fit or real PCC fit**.

## Confirmed temporary retention and bounded storage behavior

The original repo-setting endpoint remained unauthorized for `GITHUB_TOKEN` (HTTP 403), but the GitHub Actions upload accepted the proposed **90-day retention** and its actual artifact metadata confirms expiry. Successful P2-01J [run #43531054368](https://github.com/fbreve/pypcc/actions/runs/43531054368) completed **4/4 PASS**: six fail-closed tests each on Linux and Windows, the original byte-content/receipt audit, one Actions upload, and independent new-artifact download+SHA verification. Newly verified artifact **14087551705**, ZIP digest `aa19e9e5183da1d71ed1c4ac9221c12fd3b23e87f7abaaa6545cd8f9dd5271cb`, expires **2027-02-11T23:59:18Z**. The original underlying `docker save` gzip SHA remains **`008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8`**; the GitHub artifact ZIP digest is a different enclosing archive, NOT a changed OCI payload.

One earlier successful rearchive CI run #43531028233 also created the same ~291 MB payload as artifact **14087641451**, expiring **2027-02-11T23:59:04Z**. Multiple copies can consume extra billable Actions storage. **Consequently, the large download/upload/second-host jobs are now restricted to explicit `workflow_dispatch` only**, while ordinary PR changes run only the small source-only contract tests; do not create repeated ~291MB copies inadvertently. Both prior copies remain governed by their GitHub expiry, not immutable permanent storage. This branch does not delete either copy without an explicit decision.

## Part B: durable copy only at a user-selected local or NAS location

The optional PowerShell script `scripts/p2_01j_archive_original_image_locally.ps1` is **not run by CI** and will not choose a destination on the user's behalf. It requires an **existing, explicitly passed destination directory** and an authenticated GitHub CLI (`gh`). It downloads the **extended-retention artifact 14087551705** (successful workflow #43531054368, expires **2027-02-11T23:59:18Z**) using its exact workflow run/name, validates the SHA256 of the **unchanged original OCI archive payload** before and after copying, refuses to overwrite an existing different image, and writes a small JSON receipt beside the archive.

Example for a Windows PC with an existing local backup directory:

```powershell
.\scripts\p2_01j_archive_original_image_locally.ps1 -Destination 'D:\ResearchBackups\PyPCC'
.\scripts\p2_01j_archive_original_image_locally.ps1 -Destination 'D:\ResearchBackups\PyPCC' -VerifyExistingOnly
```

The exact same command accepts an existing NAS UNC path, e.g. `\\nas\backup\PyPCC`, if the user chooses to store it there. **This is an example, not a claim that those paths exist.** No backup to a local PC, NAS, cloud drive, private registry or release has been done automatically. Durable preservation requires running the helper at the chosen storage destination and confirming a second independent copy/backup policy. The archived binary contains third-party software and should not be published to an external public release/registry without a separate license/distribution review.

## Limits and scientific gate

- A SHA256-stable OCI **binary archive** is not a source-reproducible rebuild of CPython, compiler, headers, SDK, glibc or Numba JIT; P2-01 final environment closure remains pending.
- Resource feasibility of original real-data graphs for the proposed 72×50,000-iteration scientific executions is not established by import-only image tests, even if the container is constrained to 2 GiB on replay.
- The P2-07C candidate resource policy is not approved. The separate final P2-08 execution manifest SHA is **null**, and only exact-SHA explicit maintainer approval can authorize P3 real fits.
- Final states always `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`.

The next decision after this temporary copy is **which controlled, durable local/private destination** should hold the already verified image before original/source GitHub artifacts expire. Keep this research PR DRAFT and stacked; it should not be merged automatically.
