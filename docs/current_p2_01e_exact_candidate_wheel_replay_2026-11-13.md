# PyPCC P2-01E — exact archived candidate wheel replay + original Cython import (2026-11-13)

**Research-only stacked DRAFT, not P2 execution approval.** This step closes an important *candidate environment* gap discovered after P2-01D/P2-07B: the earlier P2-01C native binaries were preserved but their **original installed distribution wheel bytes were not archived**. P2-01D subsequently downloaded and archived a **new, 13-wheel candidate set** for each platform. P2-01E reuses those *exact subsequent bytes*, without downloading replacements or rebuilding either existing native module.

P1 scientific protocol remains SHA256 `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. Production is untouched; no merge into `master`; all P2/P3 real scientific PCC fits remain unauthorized.

## 1. Source artifacts, with correct provenance boundaries

Previously archived **candidate wheel bytes** (NOT historical P2-01C installed wheel identities), GitHub Actions P2-01D run [#43447245129](https://github.com/fbreve/pypcc/actions/runs/43447245129):

| Runner | Candidate wheelhouse artifact ID | GitHub artifact ZIP digest | Exact Python patch |
| --- | --- | --- | --- |
| Linux | `14049265732` | `cb79a5637face6afd3a72ea2b05d4b3dc3ce6a52b3bc4ebab61dc8c14fbfc96d` | CPython **3.12.16** |
| Windows | `14047753965` | `216f92e5858d29a771ea2ed7bfca217899a7125dd04ac7a721ba7609cf69b501` | CPython **3.12.10** |

Previously archived **original native compiled Cython bytes**, GitHub Actions P2-01C run [#43434870932](https://github.com/fbreve/pypcc/actions/runs/43434870932), Linux artifact ID `14041882611`, Windows artifact ID `14043931061`. The original build was at the synthetic GitHub PR-merge commit `24990f1e48c12cf2afd1d18bbf42f3de6cd1944c`, not merely the #262 PR head.

All these GitHub artifacts are temporary (30-day retention, expected expiration around **2026-12-13**). The P2-01D archive's GitHub ZIP checksum is retained as source metadata here; the GitHub artifact download verifies its own transmitted bytes. P2-01E additionally independently hashes every extracted wheel against its exact P2-01D SHA-256 inventory.

## 2. Install from those exact bytes, in **offline package mode**

1. Independently confirm the unchanged P1 science, original production Git blobs and source policy; pin CPython patch and x86-64 architecture separately for each OS.
2. Download **only the exact P2-01D archived wheel artifact**, which includes 13 complete `.whl` files and its published SHA/filename/byte-length inventory.
3. Verify exact wheel names, sizes, contents, ZIP paths, embedded `METADATA` names and versions; refuse any new/missing/replaced wheel file. Check all 13 pinned versions are unchanged.
4. Build a separate hash-locked `requirements.txt` with one `package==version --hash=sha256:...` line per archived wheel, then install **only from the downloaded local wheelhouse** via `pip install --no-index --no-deps --require-hashes --only-binary=:all:`. No PyPI dependency download, no wheel resolution from the public internet and no unpinned package build.
5. Revalidate installed package versions, all wheel tag compatibility, and `pip check`. Bootstrap `pip`, the CPython distribution and the GitHub hosted OS are observed but **not fully sealed as hashed compiler/SDK/OS images**.
6. Download the original old P2-01C Cython production/research binaries. Verify their original file SHA256/length and Python patch/extension ABI. Import the exact originals without rebuild or invoking propagation, then inspect their ELF/PE static import names (not host-library byte identity).
7. Archive metadata and hashes/lock only, **not another full copy of the wheelhouse or native bytes**. A third job verifies completeness across the two OS, without expecting equal native binary bytes or assuming identical compilers or runner OS images.

**No source data, artificial PCC fit, 50000-iteration study, graph, mask, observer pass, or backend propagation is performed in P2-01E.** This is specifically an environment *byte replay and native import* gate.

## 3. What P2-01E can and cannot conclude

**Can conclude when CI passes:** platform-specific *P2-01D candidate wheel bytes* are exactly reusable under the same pinned CPython 3.12 patch; 13 distributions can be installed from hash-locked offline inputs; original **P2-01C native Cython binary bytes** import with matching recorded ABI/Python patch; the production Git blobs/P1 SHA are still unchanged.

**Cannot conclude from old archives:** which precise wheel byte files P2-01C originally installed, which bytes of GCC/MSVC/Windows SDK produced those native binaries, the originally resolved shared system library byte IDs, a complete portable toolchain/ABI lock, final Numba PCC native specialization stability, P2-07 resource approval, or the performance/accuracy of any real PCC run. A new independently built execution environment will need its own exact compiled binary/toolchain evidence; **never relabel an old binary import as a new reproducible native compilation**.

Regardless of a green CI, these flags remain intentionally false:
`P2_01_final_execution_environment_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_08_signed_technical_manifest_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `PCC_real_data_fits_executed=0`.

**Next:** decide the durable execution-host/toolchain image and system DLL/ELF dependency hash capture for a *new* reproducibly built native candidate; independently review CPU/RSS/disk/resource budgets and the final immutable P2 technical SHA before requesting explicit human approval. No automatic P3.
