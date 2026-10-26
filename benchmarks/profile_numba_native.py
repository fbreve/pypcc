"""Long-running deterministic Numba workload for external native profilers.

Run under Linux perf, for example:
  NUMBA_ENABLE_PROFILING=1 python benchmarks/profile_numba_native.py --seconds 15
or:
  NUMBA_ENABLE_PROFILING=1 perf record -g --call-graph dwarf -- \
    python benchmarks/profile_numba_native.py --seconds 15
Then inspect with 'perf report'. Requires host permission for perf events.

The timer controls how many full-fit calls run; do not use these timings as
cross-revision performance evidence. No production code is instrumented.
"""
import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_numba_revision import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2048)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.classes < 2 or args.nodes < 4 * args.classes or args.iterations < 1 or args.seconds <= 0:
        parser.error("Invalid profiling configuration")
    run(args.nodes, args.classes, 1)  # JIT warmup before sampling workload.
    deadline = perf_counter() + args.seconds
    count = 0
    while perf_counter() < deadline:
        run(args.nodes, args.classes, args.iterations)
        count += 1
    print(f"Completed {count} deterministic full-fit runs")


if __name__ == "__main__":
    main()
