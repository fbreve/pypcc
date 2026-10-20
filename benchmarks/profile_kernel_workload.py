"""Deterministic kernel workload for instruction profiling (not a timing benchmark)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcc_step import pcc_propagate

n, c, p = 2048, 8, 256
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
scratch = [np.zeros(c), np.zeros(c), np.zeros(2), np.zeros(2),
           np.zeros(2), np.zeros(2)]
pcc_propagate(neighbors, degree, labels, 0.5, 0.1, c, np.zeros(c),
              positions, classes, strength, distance, dominance, ownership,
              1.0, 2.0, *scratch, np.ones(257, dtype=np.float64),
              300, 0, 10, 1)
print("checksum", float(dominance.sum()), int(positions.sum()))
