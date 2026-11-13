# P2-01D — historical build-log provenance and exact archived binary static linkage (2026-11-13)

This is a **research-only forensic supplement** stacked on #263, #262, #261 and the frozen P1 protocol #258. It does **not** recompile the original two Cython extensions or run synthetic/real PCC. Scientific SHA from P1 remains `f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166`.

## What is actually known from the original P2-01C compilation

The decoded build logs of [original run #43434870932](https://github.com/fbreve/pypcc/actions/runs/43434870932) directly show the isolated research extension's C compilation and link invocations. They identify:

| OS / Python | Original Cython research compile/link evidence |
| --- | --- |
| Linux / CPython **3.12.16** | **GCC 15.2.0 (Ubuntu 15.2.0-16ubuntu1)**; observed `-fno-strict-overflow -Wsign-compare -DNDEBUG -g -O3 -Wall -fPIC`, and `gcc -shared` with Python library rpath |
| Windows / CPython **3.12.10** | **MSVC tools directory 14.51.36231** (Visual Studio 18 Enterprise, `HostX86\x64`); observed `/O2 /GL /MD`, `link.exe /LTCG /DLL` and Windows SDK include/lib directories **10.0.26100.0** |

Machine-readable provenance: [original build-log facts](current_p2_01d_original_build_log_facts_2026-11-13.json). The original logs **do not prove** the executable/compiler object bytes or the original wheel bytes, nor were all original production extension build commands preserved in this facts table. The original PR's CI checked out **merge commit** `24990f1e48c12cf2afd1d18bbf42f3de6cd1944c`, not PR head `fa71adb7c4defa54de9d233de4b0bc84544d1c24`. Do not substitute the current runner/compiler environment for those historical build facts.

## Read-only binary inspection

The P2-01D workflow downloads the **same original P2-01C compiled Cython binaries** from run #43434870932:

- Linux artifact ID `14041882611`;
- Windows artifact ID `14043931061`.

It independently verifies the P2-01C report, exact original native lengths and SHA-256, operating-system architecture, CPython patch version, extension ABI suffix and candidate installed package versions. A small standard-library ELF64 parser reads `DT_NEEDED`, `RPATH`, `RUNPATH` and optional GNU build IDs; a PE32+ parser reads the imported DLL names. These parsers do **not** load or execute the binaries (unlike `ldd`) and have adversarial/truncated-byte unit tests. This records **static dependency names**; it does not hash/approve the resolved host `.so` or DLL images.

## Newly downloaded candidate wheel archive — not retrospective proof

A separate `pip download --only-binary=:all: --no-deps` step fetches **new wheels** matching the 13 Python 3.12 package version pins. A newly provisioned runner installs only from those local wheels (`--no-index --find-links`), checks installed versions and `pip check`, then hashes, checks the distribution name/version from both wheel filenames and embedded `METADATA`, and validates supported Python/ABI/platform tags. The CI preserves all 13 exact wheel bytes and a SHA-256 manifest for **this new candidate environment**.

**Critical limitation:** neither `pip freeze` nor a fresh `pip download` proves that P2-01C originally installed those exact wheel bytes. Retrospective identity of its installed wheels cannot be established from the previously archived materials. These new wheel hashes are **candidate input artifacts for an independently reviewed future execution environment**, not the P2-01C wheel lock. Original Cython bytes remain untouched; do not equate a rebuilt module with them.

## Scope and gates

No PCC propagations or synthetic model fits occur in **P2-01D itself** (P2-06 already separately tested 21/21 original synthetic parity cases per OS using the archived binaries). No external datasets, no graph/mask generation, no real scientific fits. All production files remain untouched. The new PR remains **DRAFT**, stacked; no merge into master.

Even with a successful Linux/Windows audit, **complete P2-01 and final P2-06 are not approved** until all execution toolchain, Python/SDK/system dependencies, wheel/Numba specialization and resource budgets are frozen in a new, compatible environment and separately reviewed. Expected permanent false fields: `original_dependency_wheel_bytes_sealed=false`, `original_compiler_toolchain_byte_hashes_sealed=false`, `system_DLL_or_SO_dependency_bytes_sealed=false`, `P2_execution_preflight_sha256=null`, `P2_execution_authorized=false`, `real_PCC_fits_executed=0`.

Source artifacts are subject to GitHub Actions 30-day retention (P2-01C binary artifacts currently through 2026-12-13); a long-term scientific archive requires a new explicit preservation decision. No automatic assumption of permanent availability.
