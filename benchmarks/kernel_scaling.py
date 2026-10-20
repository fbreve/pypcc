"""Diagnostic Cython kernel scaling benchmark: particles, classes and graph size."""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcc_step import pcc_propagate

# Each series changes one dimension while holding the others fixed.
CASES = [
    ("particles", 2048, 8, p) for p in (64, 256, 1024, 2048)
] + [
    ("classes", 2048, c, 512) for c in (2, 4, 8, 16)
] + [
    ("nodes", n, 8, 512) for n in (512, 2048, 8192, 16384)
]
ITERATIONS = 300
REPEATS = 5

def build(n, c, p):
    # Two-neighbor ring: constant degree, predictable allocation and valid indices.
    ids = np.arange(n, dtype=np.int64)
    neighbors = np.column_stack(((ids - 1) % n, (ids + 1) % n))
    degree = np.full(n, 2, dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    positions = np.arange(p, dtype=np.int64) % n
    classes = np.arange(p, dtype=np.int64) % c
    strength = np.ones(p, dtype=np.float64)
    distance = np.zeros((n, p), dtype=np.uint8)
    dominance = np.full((n, c), 1.0 / c, dtype=np.float64)
    ownership = np.zeros((n, c), dtype=np.float64)
    args = [neighbors, degree, labels, 0.5, 0.1, c, np.zeros(c),
            positions, classes, strength, distance, dominance, ownership]
    scratch = [np.zeros(c), np.zeros(c), np.zeros(2), np.zeros(2),
               np.zeros(2), np.zeros(2)]
    return args, scratch, np.ones(257, dtype=np.float64)

def measure(n, c, p):
    samples = []
    for _ in range(REPEATS):
        args, scratch, weights = build(n, c, p)
        start = time.perf_counter_ns()
        pcc_propagate(*args, 1.0, 2.0, *scratch, weights,
                      ITERATIONS, 0, 10, 1)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples

results = []
for dimension, n, c, p in CASES:
    samples = measure(n, c, p)
    results.append(dict(dimension=dimension, nodes=n, classes=c,
                        particles=p, iterations=ITERATIONS,
                        samples_ms=samples, median_ms=statistics.median(samples),
                        ns_per_particle_iteration=statistics.median(samples)*1e6/(p*ITERATIONS)))
output = dict(python=sys.version, numpy=np.__version__, results=results)
Path("kernel-scaling-result.json").write_text(json.dumps(output, indent=2))
print("| Variable | Nodes | Classes | Particles | Median ms | ns / particle-iteration |")
print("|---|---:|---:|---:|---:|---:|")
for r in results:
    print(f'| {r["dimension"]} | {r["nodes"]} | {r["classes"]} | {r["particles"]} | {r["median_ms"]:.3f} | {r["ns_per_particle_iteration"]:.1f} |')
