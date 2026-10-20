# Assembly comparison — GCC vs MSVC

This diagnostic compares artifacts from workflow run 39480965142 (PR #48), generated from pinned Cython 3.0.12 and NumPy 2.2.6. The source SHA-256 matches across operating systems for each variant.

## Static instruction counts

| Compiler | Variant | Parsed instructions | Jump instructions |
|---|---|---:|---:|
| GCC -O3 | baseline | 339 | 44 |
| GCC -O3 | fused | 341 | 48 |
| MSVC /O2 | baseline | 474 | 43 |
| MSVC /O2 | fused | 533 | 54 |

For GCC, the counted function is `__pyx_f_8pcc_step_pcc_step_sequential.isra.0`, from its function label through its `.size` directive. For MSVC, the counted function is `__pyx_f_8pcc_step_pcc_step_sequential PROC` through `ENDP`. Instruction lines were parsed from GCC assembly mnemonics and MSVC address/hex-byte listings. The MSVC parser counts the first line of each encoded instruction. These counts are not directly comparable *between* compilers due to different function specialization and code generation.

The fused variant increases counted instructions by 2 (+0.6%) in GCC and by 59 (+12.4%) in MSVC; jump instructions increase by 4 and 11 respectively. This is consistent with prior matched benchmarks showing Linux benefits without consistent Windows benefits, but static instruction counts **do not establish causation**. In particular, the extra instructions may be in cold blocks, and the dynamically executed path may be shorter despite larger code size.

## Generated C equivalence

Raw generated C differs between operating systems. Previous comparison found path references to NumPy `.pxd` files in Cython source comments, not differing executable C statements. The extracted Cython source marker sequence is identical between Linux and Windows for each variant (baseline: 196 markers; fused: 204 markers).

## Next experiment

Instrument the *hot unlabeled-node update loop* with hardware performance counters (retired instructions, branch misses, cache misses) in a local Linux run and Windows ETW or equivalent profiling, or create an isolated microbenchmark with identical distributions and compiler builds. Do not infer runtime performance from the static counts alone.

Artifacts: https://github.com/fbreve/pypcc/actions/runs/39480965142
