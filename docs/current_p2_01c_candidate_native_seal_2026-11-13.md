# P2-01C — candidate Python 3.12 package pins and exact Cython artifact retention (2026-11-13)

**Scope:** technical research only; this is **not** the P2 scientific execution preflight or authorization. Stacked on draft PR #261, itself stacked on frozen P1 #258. The approved P1 scientific SHA-256 remains `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. Production sources are unmodified.

Prior evidence: at the exact head of #261 (`d69985819914e52ce6cae619bf9915cafc29fa87`), the source check, native build/import smoke and unchanged regression each completed **4/4** Linux/Windows × Python 3.10/3.12 jobs. The original native jobs retained JSON hashes, **not** compiled binary bytes. They also invoked only a scalar Numba JIT demonstration, never the compiled PCC kernel.

## Candidate—not approved—technical environment

This iteration narrows the technical build matrix to **Python 3.12 on Linux and Windows**. The [candidate requirements](../experiments/p2_01c_candidate_py312_requirements.txt) pin direct/build dependencies **and the observed transitive packages** from the successful 2026-11-13 Python 3.12 logs, including NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Numba 0.68.1, llvmlite 0.50.0 and Cython 3.3.1. A native build uses `pip install --no-build-isolation --no-deps -e .` after the candidate packages have been installed, avoiding implicit, independently resolved Python build dependencies. The helper checks every listed package's installed metadata version, writes `pip freeze --all`, and records OS, Python patch version, SOABI, compiler metadata and CI run identifiers.

**Limitations:** this is an explicit version-pinned candidate set, **not a bitwise wheel-hash lock or pinned Python patch/runtime image**. The runner OS images and GitHub Actions action tags are not immutable. On Windows the compiler/SDK may need additional identification beyond what Python sysconfig reports. The observed compiler/flags are *descriptive*, not a complete ABI/architecture lock.

## Native bytes (no PCC fits)

The workflow independently checks Git source hashes, retrieves each observer from its original immutable PR head, compiles production `pcc_step` and isolated `pcc_step_live_research`, performs the previous P2-01B import/smoke, then **copies the exact imported `.so` / `.pyd` files** into `p2-01c-seal/native/`. It independently compares file length and SHA-256 to the freshly generated P2-01B report, and verifies the copied bytes a second time. A complete provenance JSON, native smoke JSON, source audit, candidate requirements, and full pip freeze accompany the binaries.

The GitHub Actions artifact has **30-day retention**. Downloading/reusing the exact artifacts later requires digest verification and separate authorization; an automatically rebuilt binary is **not interchangeable**, even when file sources and package versions match. A matrix member's artifact is platform- and Python-ABI-specific.

The Numba smoke remains scalar-only: **no Numba PCC dispatcher signatures, cache objects or LLVM kernel specialization have been sealed.** Synthetic observer-vs-production parity is **not** performed here; keep it for the independent P2-06 stage. The Python NumPy observer has no separate native binary; its exact Python source and installed NumPy package version are checked, not a unique compiled observer module.

## Acceptance boundaries

- **Allowed:** pinned-source fetch and audit; dependency install; isolated compilation and *import*; scalar-only Numba JIT; artifact SHA verification; stdlib-only tamper tests; preserving files in CI.
- **Forbidden:** real source datasets; building actual graphs/masks; PCC propagation (including synthetic); changes to any production file; merging this branch to master; inferring P2 authorization from the scientific P1 SHA.
- Expected report fields: `P2_01_complete_gate_pass=false`, `P2_06_observer_parity_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `PCC_fits_executed=0`.

**Status:** workflow and code prepared. CI results and actual artifact IDs must be verified before claiming P2-01C PASS. After that, choose the final execution OS/Python patch and compiler/SDK, freeze wheel hashes and Numba signatures, validate native same-backend synthetic parity separately, and continue P2-02 through P2-08. No real fit before a newly approved exact P2 execution SHA.
