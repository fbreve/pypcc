"""Measure Cython propagation sensitivity to graph degree and labeled fraction.

Fresh deterministic inputs per repeat. No production kernel changes.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcc_step import pcc_propagate

NODES = 2048
CLASSES = 8
ITERATIONS = 300
REPEATS = 7
CASES = (
    [("degree", d, 0.0, 512) for d in (2, 4, 8, 16, 32)]
    + [("labeled", 8, f, 512) for f in (0.0, 0.1, 0.25, 0.5, 0.75)]
    + [("particles", 8, 0.25, p) for p in (64, 256, 512, 1024, 2048)]
)

def build(degree, fraction, particles):
    # Symmetric circulant graph: exactly 'degree' distinct neighbors.
    ids = np.arange(NODES, dtype=np.int64)
    offsets = np.array([v for k in range(1, degree // 2 + 1) for v in (-k, k)])
    neighbors = (ids[:, None] + offsets[None, :]) % NODES
    degrees = np.full(NODES, degree, dtype=np.int64)
    labels = np.full(NODES, -1, dtype=np.int64)
    count = round(NODES * fraction)
    labels[:count] = np.arange(count, dtype=np.int64) % CLASSES
    positions = np.arange(particles, dtype=np.int64) % NODES
    classes = np.arange(particles, dtype=np.int64) % CLASSES
    strength = np.ones(particles, dtype=np.float64)
    distance = np.zeros((NODES, particles), dtype=np.uint8)
    dominance = np.full((NODES, CLASSES), 1.0 / CLASSES, dtype=np.float64)
    ownership = np.zeros((NODES, CLASSES), dtype=np.float64)
    args = [neighbors, degrees, labels, 0.5, 0.1, CLASSES,
            np.zeros(CLASSES), positions, classes, strength,
            distance, dominance, ownership]
    scratch = [np.zeros(CLASSES), np.zeros(CLASSES),
               np.zeros(degree), np.zeros(degree),
               np.zeros(degree), np.zeros(degree)]
    return args, scratch, np.ones(257, dtype=np.float64)

def measure(degree, fraction, particles):
    samples = []
    for _ in range(REPEATS):
        args, scratch, weights = build(degree, fraction, particles)
        start = time.perf_counter_ns()
        pcc_propagate(*args, 1.0, 2.0, *scratch, weights,
                      ITERATIONS, 0, 10, 1)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples

results = []
for dimension, degree, fraction, particles in CASES:
    samples = measure(degree, fraction, particles)
    median = statistics.median(samples)
    results.append(dict(dimension=dimension, nodes=NODES,
                        degree=degree, labeled_fraction=fraction,
                        particles=particles, median_ms=median,
                        samples_ms=samples,
                        ns_per_particle_iteration=median * 1e6 / (particles * ITERATIONS)))
Path("graph-topology-result.json").write_text(
    json.dumps(dict(python=sys.version, numpy=np.__version__,
                    iterations=ITERATIONS, results=results), indent=2))
print("| Dimension | Degree | Labeled | Particles | Median ms | ns/particle-iteration |")
print("|---|---:|---:|---:|---:|---:|")
for x in results:
    print(f'| {x["dimension"]} | {x["degree"]} | {x["labeled_fraction"]:.0%} | {x["particles"]} | {x["median_ms"]:.3f} | {x["ns_per_particle_iteration"]:.1f} |')
