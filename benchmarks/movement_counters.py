"""Trace candidate movement decisions without changing the timed Cython kernel.

Runs a faithful *decision-only* replay of the xorshift selection and update
acceptance rules. This is NOT a replacement for the production implementation:
dominance and strengths evolve in the real kernel and affect future decisions.
We therefore validate the first step against pcc_step and only report counters
for that first step, where both start from identical state.
"""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.graph_topology import build
from pcc_step import pcc_step

MASK = 0xFFFFFFFF
def xorshift(state):
    x = state
    x ^= (x << 13) & MASK
    x ^= x >> 17
    x ^= (x << 5) & MASK
    return x & MASK

def trace_one_step(args, weights):
    neighbors, degree, labels, p_grd, delta_v, classes, _, positions, particle_classes, strength, distance, dominance, ownership = args
    state = 123456789
    counters = dict(greedy_attempts=0, greedy_selected=0, random_selected=0,
                    labeled_destinations=0, unlabeled_destinations=0,
                    accepted=0, rejected=0, skipped=0)
    for particle in range(len(positions)):
        current = int(positions[particle])
        if current < 0 or current >= len(labels) or degree[current] <= 0:
            counters["skipped"] += 1
            continue
        label = int(particle_classes[particle])
        if label < 0 or label >= classes:
            counters["skipped"] += 1
            continue
        state = xorshift(state)
        draw = state / 4294967296.0
        k = int(degree[current])
        greedy = False
        if draw < p_grd:
            counters["greedy_attempts"] += 1
            cumulative = np.empty(k, dtype=float)
            total = 0.0
            for i in range(k):
                nxt = int(neighbors[current, i])
                total += weights[distance[nxt, particle]] * dominance[nxt, label]
                cumulative[i] = total
            if total > 0:
                state = xorshift(state)
                threshold = state / 4294967296.0 * total
                choice = 0
                for i in range(k):
                    if threshold < cumulative[i]:
                        choice = i
                        break
                nxt = int(neighbors[current, choice])
                greedy = True
            else:
                state = xorshift(state)
                nxt = int(neighbors[current, state % k])
        else:
            state = xorshift(state)
            nxt = int(neighbors[current, state % k])
        counters["greedy_selected" if greedy else "random_selected"] += 1
        counters["labeled_destinations" if labels[nxt] != -1 else "unlabeled_destinations"] += 1
        if labels[nxt] == -1:
            step = strength[particle] * (delta_v / (classes - 1))
            reduction = np.minimum(dominance[nxt], step)
            dominance[nxt] -= reduction
            dominance[nxt, label] += reduction.sum()
        strength[particle] = dominance[nxt, label]
        if distance[current, particle] < 255:
            if distance[nxt, particle] > distance[current, particle] + 1:
                distance[nxt, particle] = distance[current, particle] + 1
        if not greedy:
            ownership[nxt, label] += strength[particle]
        if dominance[nxt, label] == dominance[nxt].max():
            positions[particle] = nxt
            counters["accepted"] += 1
        else:
            counters["rejected"] += 1
    return counters

def main():
    records = []
    for degree in (2, 8, 32):
        for fraction in (0.1, 0.25, 0.5):
            for layout in ("prefix", "random"):
                for placement in ("stratified", "random"):
                    for seed in (101, 202, 303, 404, 505):
                        args, scratch, weights = build(degree, fraction, 512, layout, placement, seed)
                        clone = [np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args]
                        counts = trace_one_step(args, weights)
                        pcc_step(*clone, 1.0, 2.0, prob=scratch[4],
                                 slices=scratch[5], dist_weights=weights)
                        fields = (7, 9, 10, 11, 12)
                        for index in fields:
                            if not np.allclose(args[index], clone[index], rtol=0, atol=1e-12):
                                raise AssertionError(f"Trace mismatch field {index}, degree={degree}, fraction={fraction}, layout={layout}, placement={placement}, seed={seed}")
                        records.append(dict(degree=degree, fraction=fraction, layout=layout,
                                            placement=placement, seed=seed, **counts))
    Path("movement-counters-result.json").write_text(json.dumps(records, indent=2))
    print("Validated", len(records), "first-step decision traces against Cython")
    for r in records:
        print(json.dumps(r, sort_keys=True))

if __name__ == "__main__":
    main()
