# P2-06 candidate — within-backend synthetic parity using exact archived P2-01C binaries

**Status:** new, deliberately separate research gate. Not a P2 execution preflight/GO. Scientific P1 SHA `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166` stays frozen; no production files are modified, no real-data fits are permitted.

## Immutable inputs and expected validation

- Native **production `pcc_step` and research `pcc_step_live_research` bytes are downloaded from the exact P2-01C run [#43434870932](https://github.com/fbreve/pypcc/actions/runs/43434870932)**, rather than rebuilt. Artifact IDs: Linux `14041882611`, Windows `14043931061`. Both are scheduled to expire 2026-12-13.
- The previous P2-01C job was a pull-request build; its **actual checked-out Git commit is the synthetic merge** `24990f1e48c12cf2afd1d18bbf42f3de6cd1944c`, not branch head `fa71adb7c4defa54de9d233de4b0bc84544d1c24`. Both identities are distinguished, not conflated.
- Verify the archived P2-01B report SHA, each Cython binary SHA-256 and length, ELF/PE x86-64 header, platform, exact CPython patch version and `EXT_SUFFIX`, and every candidate Python package version. Import exactly the restored production/research binaries and verify they remain unchanged after the tests.
- Retrieve **unaltered original research observers AND test sources** by immutable Git commits: NumPy PR #253 `6d908e7`, Numba PR #255 `4c8e6df`, Cython PR #256 `af34f50`. Verify the original test Git blob hashes, not copies rewritten for this new study.
- Run original within-backend synthetic suites on tiny, constructed 8-node undirected graphs (including zero iterations, early-stop/stuck scenarios, prefix state, read-only capture ownership, full model postprocessing, RNG continuation and Cython native cache paths). The suites are **not** cross-backend trajectory equality tests.
- After passing, collect actual compiled Numba observer/production particle-kernel and rowmax dispatcher signatures and observational LLVM IR digests. Unlike the previous scalar-only smoke, this genuinely specializes the Numba PCC kernel on synthetic arrays. The JIT specializations and IR are recorded, **not** approved as a final native execution lock.

## Interpret results accurately

This workflow **does execute small synthetic PCC propagations**, because otherwise no genuine within-backend parity can be tested. It never reads a real dataset, makes a real experimental graph or selects a real label mask. A synthetic `fit_predict` test is not a scientific P2 real-data fit. Never report *zero PCC propagation* for this stage: report **zero real-data fits** only.

Even if Linux and Windows both pass, this would prove only **candidate-runner synthetic within-backend parity** for the exact archived Cython bytes and tested Numba specializations. It does **not** complete the P2-06 real-execution gate until the final P2 environment/source versions, ABI/toolchain and full machine/resource preflight are approved. No automatically rebuilt native module can silently replace the archived one. Binary artifact expiration is a real continuity blocker.

**Required invariants:** `P2_06_full_execution_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`. Existing P2-01C archived binaries are untouched; no merge into `master`.
