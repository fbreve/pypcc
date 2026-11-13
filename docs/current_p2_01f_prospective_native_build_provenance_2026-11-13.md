# P2-01F — prospective NEW native build toolchain byte ledger (2026-11-13)

**Only a research candidate.** The scientific P1 hash remains `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`. This is stacked on #270, with the original production and research source Git blobs frozen, no merge to `master` and **zero real or synthetic PCC fits in this P2-01F workflow**.

## Exact inputs and builds

This is **not an attempt to retroactively prove or reconstruct the 2026-11-13 P2-01C historical native build**. P2-01C #262 preserved exact original `.so/.pyd` bytes but did not preserve original installed wheel or full compiler/SDK byte closure. The following work independently creates **NEW** native binaries from the same frozen code, using the 13 *later* archived P2-01D candidate wheel files, whose SHA/metadata and local offline installation were verified in P2-01E #270.

The workflow pins Linux **CPython 3.12.16** and Windows **CPython 3.12.10**, SHA-verifies each archived candidate wheel and installs all 13 wheels with `--no-index --no-deps --require-hashes --only-binary=:all:`. It independently checks exact P1 approval, original production Git source blobs and three original isolated observers + Cython generator blobs. It invokes:

- **Original production**: `pip install --no-index --no-deps --no-build-isolation --verbose --editable .`, using the *unchanged* original `setup.py`, `pcc_step.pyx`, Cython flags and original source hotloops.
- **Original research isolated extension**: `python -m experiments.build_live_cython`, using the byte-exact original generator from #256, **not** changing the production Cython source.
- Original P2-01B native import/scalar-only Numba smoke and P2-01C native byte-sealing scripts are then reused to hash/import **these newly built binaries**. There is no invocation of `fit_predict`, particle propagation, official UCI loading, mask sampling, or graph construction.

Every new native build stores the actual stdout/stderr compile/link command lines and **requires a real compile and link command for each of the two modules**. A missing build line fails closed rather than recording a guessed version. It SHA-hashes the compiler and linker executable files actually named in the logs (GCC driver, selected GNU tool subprocesses or MSVC `cl.exe` and `link.exe`), while recording their paths, version banners, actual invoked compile/link arguments, exact newly built .so/.pyd bytes, architecture, Python ABI and static ELF/PE imported library names. Windows records **candidate named SDK/UCRT/Python import libraries physically found in the linker search paths**, with SHA; this does **not** prove all those files were selected by the linker. Linux records basic GNU tool files and ELF imported names, without claiming a complete resolved loader-DSO byte closure.

**These newly compiled .so/.pyd bytes are preserved in a separate named P2-01F GitHub Actions artifact** and are not confused with nor substituted for the archived P2-01C originals. Their equality with the historical binaries is neither assumed nor required.

## What passing P2-01F means

The new candidate's package *inputs*, compile/link *commands* and some invoked compiler/linker executable bytes have concrete, inspectable SHA-256 evidence. Original scientific and production source identities remain unchanged. All output is candidate-specific, with 30-day Actions retention, and may be vulnerable to GitHub-hosted runner image updates. An artifact ZIP is **not** a durable complete OS/toolchain image.

**Remaining limitations:** full compiler DLL/internal tools, entire included header trees, final linker-selected library input closure, dynamically loaded system DLL/ELF DSO byte identities, reproducibility in another *fully pinned execution environment*, actual native Numba PCC specialization and 50k-run resources are not proven. The actual GitHub-hosted Windows/Linux image is recorded rather than content-addressably pinned. No cross-OS binary identity is expected.

Passing this CI can set **`new_native_exact_source_smoke_proved=true`** and **`both_OS_new_compilation_and_invoked_compiler_hashes_captured=true`**, but it **must** preserve `complete_compiler_SDK_system_include_and_binary_link_closure_sealed=false`, `P2_01_final_execution_toolchain_lock_pass=false`, `P2_07_full_resources_gate_pass=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`.

Before any P3 real fits, separately review a durable execution host/image and whole 72-case RAM/CPU/time/log budget, then seal the exact P2-08 technical manifest with **explicit maintainer approval**. No automatic P2 authorization or merge.

**CI status:** scripts and workflow prepared; candidate PASS only if Linux, Windows and the independent two-OS report are all successful.
