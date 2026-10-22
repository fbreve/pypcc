"""Benchmark master against the experimental fused Cython implementation.

Each implementation runs in its own process and source tree. Timings include only
pcc_propagate, not graph construction or module import.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.source).resolve()))
sys.path.insert(0, str(Path(args.source).resolve() / "benchmarks"))

from graph_topology import build
from pcc_step import pcc_propagate

rows = []
for degree in (2, 8, 32):
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.75):
        for seed in (101, 202, 303, 404, 505):
            samples = []
            last_state = None
            for rep in range(7):
                inputs, scratch, weights = build(degree, fraction, 512, "random", "random", seed)
                t0 = time.perf_counter_ns()
                pcc_propagate(*inputs, 1.0, 2.0, *scratch, weights, 300, 0, 10, 1)
                samples.append((time.perf_counter_ns() - t0) / 1e6)
                state = [np.asarray(x).tolist() for x in (inputs[7], inputs[9], inputs[10], inputs[11], inputs[12])]
                if last_state is not None and state != last_state:
                    raise AssertionError(f"Non-deterministic result: {degree}, {fraction}, {seed}")
                last_state = state
            rows.append({"degree": degree, "fraction": fraction, "seed": seed,
                         "median_ms": statistics.median(samples), "state": last_state})
Path(args.output).write_text(json.dumps(rows), encoding="utf-8")
print("CASES", len(rows))
