"""Deterministic native CPU profiling workload; not a timing benchmark."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pcc_step import pcc_propagate

p = argparse.ArgumentParser()
p.add_argument("--nodes", type=int, default=512)
p.add_argument("--particles", type=int, default=128)
p.add_argument("--degree", type=int, default=16)
p.add_argument("--iterations", type=int, default=40)
a = p.parse_args()
assert a.degree > 0 and a.degree < a.nodes and a.degree % 2 == 0
n, q, d, c = a.nodes, a.particles, a.degree, 8
idx = np.arange(n, dtype=np.int64)
offsets = np.array([v for j in range(1, d // 2 + 1) for v in (-j, j)], dtype=np.int64)
neighbors = (idx[:, None] + offsets) % n
degree = np.full(n, d, dtype=np.int64)
labels = np.full(n, -1, dtype=np.int64)
labels[::4] = np.arange(len(labels[::4]), dtype=np.int64) % c
positions = np.arange(q, dtype=np.int64) % n
classes = np.arange(q, dtype=np.int64) % c
strength = np.ones(q, dtype=np.float64)
distance = np.zeros((n, q), dtype=np.uint8)
dominance = np.full((n, c), 1.0 / c, dtype=np.float64)
ownership = np.zeros((n, c), dtype=np.float64)
scratch = [np.zeros(c), np.zeros(c), np.zeros(d), np.zeros(d), np.zeros(d), np.zeros(d)]
if os.environ.get('PCC_CALLGRIND_ISOLATE') == '1':
    subprocess.run(['callgrind_control', '-i', 'on', str(os.getpid())], check=True)
pcc_propagate(neighbors, degree, labels, 0.5, 0.1, c, np.zeros(c),
              positions, classes, strength, distance, dominance, ownership,
              1.0, 2.0, *scratch, np.ones(257, dtype=np.float64),
              a.iterations, 0, 10, 1)
if os.environ.get("PCC_CALLGRIND_ISOLATE") == "1":
    subprocess.run(["callgrind_control", "-i", "off", str(os.getpid())], check=True)
print("PROFILE_DONE", n, q, d, a.iterations, float(dominance.sum()), int(positions.sum()))
