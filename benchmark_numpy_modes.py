"""Standalone microbenchmark for PCC NumPy update modes.

Run: python benchmark_numpy_modes.py
Not a classification-quality benchmark.
"""
import time
import numpy as np
from pcc_numpy import pcc_step_numpy


def run(n_nodes=1000, n_particles=100, k=10, steps=50, seed=123):
    rng = np.random.default_rng(seed)
    neighbors = rng.integers(0, n_nodes, size=(n_nodes, k), dtype=np.int64)
    degrees = np.full(n_nodes, k, dtype=np.int64)
    labels = np.full(n_nodes, -1, dtype=np.int64)
    labels[:n_particles] = np.arange(n_particles) % 2
    classes = labels[:n_particles].copy()
    weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** 2
    base_dominance = np.full((n_nodes, 2), 0.5)
    base_dominance[:n_particles] = 0
    base_dominance[np.arange(n_particles), classes] = 1

    results = {}
    for mode in ("parallel", "sequential"):
        positions = np.arange(n_particles, dtype=np.int64)
        strength = np.ones(n_particles)
        distance = np.full((n_nodes, n_particles), 255, dtype=np.uint8)
        distance[positions, np.arange(n_particles)] = 0
        dominance = base_dominance.copy()
        owndeg = np.zeros((n_nodes, 2))
        np.random.seed(seed)
        start = time.perf_counter()
        for _ in range(steps):
            pcc_step_numpy(neighbors, degrees, labels, 0.5, 0.1, 2,
                           np.zeros(2), positions, classes, strength,
                           distance, dominance, owndeg,
                           dist_weights=weights, update_mode=mode)
        elapsed = time.perf_counter() - start
        assert np.all(np.isfinite(dominance))
        assert np.min(dominance) >= -1e-10
        np.testing.assert_allclose(dominance.sum(axis=1), 1, atol=1e-9)
        results[mode] = elapsed
        print(f"{mode}: {elapsed:.4f}s; {steps} iterations, "
              f"{n_nodes} nodes, {n_particles} particles, k={k}")
    print(f"sequential/parallel ratio: "
          f"{results['sequential']/results['parallel']:.2f}x")


if __name__ == "__main__":
    for n, p, k in ((100, 20, 5), (1000, 100, 10), (5000, 500, 10)):
        run(n_nodes=n, n_particles=p, k=k)
