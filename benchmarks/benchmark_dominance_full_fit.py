"""Full-fit comparison of C and Fortran dominance allocations (experimental)."""
import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcc import ParticleCompetitionAndCooperation as PCC
from numba import njit

@njit
def seed_numba(seed):
    np.random.seed(seed)

original = PCC._ParticleCompetitionAndCooperation__genNodes

def fortran_nodes(self):
    nodes = original(self)
    nodes.dominance = np.asfortranarray(nodes.dominance)
    return nodes

def execute(backend, order, nodes, iterations):
    graph = (np.arange(nodes)[:, None] + np.array([-2, -1, 1, 2])) % nodes
    degrees = np.full(nodes, 4, dtype=np.int64)
    labels = np.full(nodes, -1, dtype=np.int64)
    labels[::8] = np.arange(len(labels[::8])) % 4
    PCC._ParticleCompetitionAndCooperation__genNodes = fortran_nodes if order == "F" else original
    try:
        model = PCC(impl=backend)
        model.set_graph(graph, degrees, k_nn=4)
        np.random.seed(12345)
        if backend == "numba":
            seed_numba(12345)
        start = perf_counter()
        result = model.fit_predict(labels, max_iter=iterations, early_stop=False)
        elapsed = perf_counter() - start
        state = (result.copy(), model.part.curnode.copy(),
                 model.part.strength.copy(), model.part.dist_table.copy(),
                 model.node.dominance.copy(), model.owndeg.copy())
        assert model.node.dominance.flags.f_contiguous if order == "F" else model.node.dominance.flags.c_contiguous
        return elapsed, state
    finally:
        PCC._ParticleCompetitionAndCooperation__genNodes = original

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    for backend in ("cython", "numba", "numpy"):
        times = {"C": [], "F": []}
        for repeat in range(args.repeats):
            reference = None
            for order in (("C", "F") if repeat % 2 == 0 else ("F", "C")):
                elapsed, state = execute(backend, order, args.nodes, args.iterations)
                if reference is None:
                    reference = state
                else:
                    assert all(np.array_equal(a, b) for a, b in zip(reference, state)), (backend, repeat)
                times[order].append(elapsed)
        print(f"BACKEND {backend} NODES {args.nodes} ITERATIONS {args.iterations}", flush=True)
        for order in ("C", "F"):
            print(f"{order} median_s={statistics.median(times[order]):.6f} repeats={args.repeats}", flush=True)
        print("STATE_EQUIVALENCE_OK", flush=True)

if __name__ == "__main__":
    main()
