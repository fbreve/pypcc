"""Inspect compiled Numba PCC kernels without changing production execution.

Runs a deterministic warmup and reports JIT signatures, LLVM IR size and
native assembly size. These are code-generation diagnostics, NOT phase timings.
Use alongside workload benchmarks to decide whether low-level profiling is warranted.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_numba_revision import run
from pcc_numba import _pcc_step_numba_sequential, _pcc_calc_mmpot_sequential


def inspect(dispatcher):
    output = []
    for signature in dispatcher.signatures:
        llvm = dispatcher.inspect_llvm(signature)
        asm = dispatcher.inspect_asm(signature)
        output.append({
            "signature": str(signature),
            "llvm_lines": len(llvm.splitlines()),
            "assembly_lines": len(asm.splitlines()),
            "llvm_branches": sum(
                line.lstrip().startswith(("br ", "switch "))
                for line in llvm.splitlines()
            ),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--classes", type=int, default=8)
    args = parser.parse_args()
    if args.classes < 2 or args.nodes < 4 * args.classes:
        parser.error("nodes must be >= 4 * classes and classes >= 2")
    run(args.nodes, args.classes, 1)
    print(json.dumps({
        "nodes": args.nodes,
        "classes": args.classes,
        "step": inspect(_pcc_step_numba_sequential),
        "mmpot": inspect(_pcc_calc_mmpot_sequential),
        "note": "Instruction/IR counts are not runtime cost or phase timings."
    }, indent=2))


if __name__ == "__main__":
    main()
