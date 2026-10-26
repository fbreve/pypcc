"""Controlled full-fit Numba workload profiling (diagnostic only).

Contrasts exercise different branches but do NOT isolate individual kernel costs:
model dynamics and memory access patterns change with each parameter.
Run: python benchmarks/profile_numba_workloads.py --nodes 2048 --classes 16
"""
import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcc import ParticleCompetitionAndCooperation


@njit
def seed_numba(seed):
    np.random.seed(seed)


def run(nodes, classes, iterations, greedy, delta_v, deltap):
    model = ParticleCompetitionAndCooperation(impl="numba")
    offsets = np.array([-2, -1, 1, 2], dtype=np.int64)
    neighbors = (np.arange(nodes, dtype=np.int64)[:, None] + offsets) % nodes
    degrees = np.full(nodes, 4, dtype=np.int64)
    labels = np.full(nodes, -1, dtype=np.int64)
    labeled = np.arange(0, nodes, 4, dtype=np.int64)
    labels[labeled] = np.arange(len(labeled)) % classes
    model.set_graph(neighbors, degrees, k_nn=4)
    np.random.seed(12345)
    seed_numba(12345)
    start = perf_counter()
    model.fit_predict(labels, p_grd=greedy, delta_v=delta_v,
                      deltap=deltap, max_iter=iterations, early_stop=False)
    return perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2048)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.classes < 2 or args.nodes < 4 * args.classes or args.repeats < 1:
        parser.error("Invalid nodes/classes/repeats")
    cases = [
        ("default", 0.5, 0.1, 1.0),
        ("random_only", 0.0, 0.1, 1.0),
        ("greedy_only", 1.0, 0.1, 1.0),
        ("no_dominance_transfer", 0.5, 0.0, 1.0),
        ("partial_strength_update", 0.5, 0.1, 0.5),
    ]
    # Warm each specialization and alternate cases to reduce order bias.
    for _, greedy, delta_v, deltap in cases:
        run(args.nodes, args.classes, 1, greedy, delta_v, deltap)
    samples = {name: [] for name, *_ in cases}
    for repeat in range(args.repeats):
        order = cases if repeat % 2 == 0 else list(reversed(cases))
        for name, greedy, delta_v, deltap in order:
            samples[name].append(run(args.nodes, args.classes, args.iterations,
                                     greedy, delta_v, deltap))
    print(json.dumps({
        "nodes": args.nodes, "classes": args.classes,
        "iterations": args.iterations, "repeats": args.repeats,
        "results": {name: {"median_s": statistics.median(values),
                           "samples_s": values} for name, values in samples.items()},
        "caution": "Parameter contrasts alter trajectories; not a kernel time breakdown."
    }, indent=2))


if __name__ == "__main__":
    main()
