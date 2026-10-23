"""Compare C, Fortran and positive-strided Cython memoryviews without kernel changes.

Run after building pcc_step: python benchmarks/benchmark_cython_layouts.py
Timing is exploratory: compare medians across repeated, independently reset fits.
"""
import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcc_step import pcc_propagate


def layout(array, kind):
    if kind == "C":
        return np.array(array, order="C", copy=True)
    if kind == "F":
        return np.array(array, order="F", copy=True)
    # Both axes have positive non-unit strides; keep backing storage alive.
    padded = np.empty((array.shape[0] * 2, array.shape[1] * 2), dtype=array.dtype)
    view = padded[::2, ::2]
    view[:] = array
    return view


def run(kind, n, q, d, iterations):
    c = 8
    idx = np.arange(n, dtype=np.int64)
    offsets = np.array([v for j in range(1, d // 2 + 1) for v in (-j, j)], dtype=np.int64)
    neighbors = layout((idx[:, None] + offsets) % n, kind)
    degree = np.full(n, d, dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    labels[::4] = np.arange(len(labels[::4])) % c
    positions = np.arange(q, dtype=np.int64) % n
    classes = np.arange(q, dtype=np.int64) % c
    strength = np.ones(q, dtype=np.float64)
    distance = layout(np.zeros((n, q), dtype=np.uint8), kind)
    dominance = layout(np.full((n, c), 1.0 / c), kind)
    ownership = layout(np.zeros((n, c)), kind)
    scratch = [np.zeros(c), np.zeros(c), np.zeros(d), np.zeros(d),
               np.zeros(d), np.zeros(d)]
    start = perf_counter()
    pcc_propagate(neighbors, degree, labels, 0.5, 0.1, c, np.zeros(c),
                  positions, classes, strength, distance, dominance, ownership,
                  1.0, 2.0, *scratch, np.ones(257), iterations, 0, 10, 1)
    elapsed = perf_counter() - start
    # Exact comparison, including the state used by subsequent iterations.
    state = (positions.copy(), strength.copy(), distance.copy(),
             dominance.copy(), ownership.copy())
    return elapsed, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=512)
    p.add_argument("--particles", type=int, default=128)
    p.add_argument("--degree", type=int, default=16)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=7)
    a = p.parse_args()
    if not (a.nodes > a.degree > 0 and a.degree % 2 == 0
            and a.particles > 0 and a.iterations > 0 and a.repeats > 0):
        p.error("Require nodes > even positive degree, and positive other parameters")

    samples = {kind: [] for kind in ("C", "F", "strided")}
    for repeat in range(a.repeats):
        reference = None
        # Rotate order to reduce systematic warm-up / frequency bias.
        kinds = ("C", "F", "strided")
        for kind in kinds[repeat % 3:] + kinds[:repeat % 3]:
            elapsed, state = run(kind, a.nodes, a.particles, a.degree, a.iterations)
            if reference is None:
                reference = state
            elif not all(np.array_equal(x, y) for x, y in zip(reference, state)):
                raise AssertionError(f"State mismatch for {kind}, repeat {repeat}")
            samples[kind].append(elapsed)
    for kind in ("C", "F", "strided"):
        print(f"{kind:8s} median_s={statistics.median(samples[kind]):.6f} "
              f"min_s={min(samples[kind]):.6f} repeats={a.repeats}")
    print("STATE_EQUIVALENCE_OK")


if __name__ == "__main__":
    main()
