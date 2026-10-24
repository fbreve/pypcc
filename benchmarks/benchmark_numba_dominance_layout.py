"""Measure Numba full-fit dominance layout over node/class scales.

Uses public fit API; changes only allocation order in a subclass, never production.
"""
import argparse
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


class COrderPCC(ParticleCompetitionAndCooperation):
    def _ParticleCompetitionAndCooperation__genNodes(self):
        nodes = super()._ParticleCompetitionAndCooperation__genNodes()
        nodes.dominance = np.ascontiguousarray(nodes.dominance)
        return nodes


def fit(order, n, classes, iterations):
    model_type = COrderPCC if order == "C" else ParticleCompetitionAndCooperation
    model = model_type(impl="numba")
    offsets = np.array([-2, -1, 1, 2], dtype=np.int64)
    neighbors = (np.arange(n, dtype=np.int64)[:, None] + offsets) % n
    degrees = np.full(n, len(offsets), dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    # Ensure each class has a representative while varying particle count moderately.
    labeled = np.arange(0, n, 4, dtype=np.int64)
    labels[labeled] = np.arange(len(labeled)) % classes
    model.set_graph(neighbors, degrees, k_nn=4)
    np.random.seed(12345)
    seed_numba(12345)
    start = perf_counter()
    prediction = model.fit_predict(labels, max_iter=iterations, early_stop=False)
    elapsed = perf_counter() - start
    state = (prediction.copy(), model.part.curnode.copy(), model.part.strength.copy(),
             model.part.dist_table.copy(), model.node.dominance.copy(),
             model.owndeg.copy())
    assert (model.node.dominance.flags.c_contiguous if order == "C"
            else model.node.dominance.flags.f_contiguous)
    return elapsed, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=1024)
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--repeats", type=int, default=7)
    a = p.parse_args()
    if a.nodes < 4 * a.classes or a.classes < 2 or a.iterations < 1 or a.repeats < 1:
        p.error("Require nodes >= 4 * classes, classes >= 2 and positive iterations/repeats")
    samples = {"C": [], "F": []}
    for rep in range(a.repeats):
        baseline = None
        for order in (("C", "F") if rep % 2 == 0 else ("F", "C")):
            elapsed, state = fit(order, a.nodes, a.classes, a.iterations)
            if baseline is not None:
                assert all(np.array_equal(x, y) for x, y in zip(baseline, state)), (a.nodes, a.classes, rep)
            baseline = state
            samples[order].append(elapsed)
    print(f"NODES {a.nodes} CLASSES {a.classes} ITERATIONS {a.iterations}", flush=True)
    for order in ("C", "F"):
        print(f"{order} median_s={statistics.median(samples[order]):.6f} "
              f"min_s={min(samples[order]):.6f} repeats={a.repeats}", flush=True)
    print("STATE_EQUIVALENCE_OK", flush=True)


if __name__ == "__main__":
    main()
