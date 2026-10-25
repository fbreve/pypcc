"""Deterministic full-fit Numba timing and final-state fingerprint for cross-revision comparisons."""
import argparse
import hashlib
import json
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


def run(nodes, classes, iterations):
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
    prediction = model.fit_predict(labels, max_iter=iterations, early_stop=False)
    elapsed = perf_counter() - start
    state = (prediction, model.part.curnode, model.part.strength,
             model.part.dist_table, model.node.dominance, model.owndeg)
    digest = hashlib.sha256()
    for array in state:
        arr = np.asarray(array)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return elapsed, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.classes < 2 or args.nodes < 4 * args.classes or args.repeats < 1:
        parser.error("Invalid nodes/classes/repeats")
    # Compile and warm up before collecting timings.
    run(args.nodes, args.classes, 1)
    samples = [run(args.nodes, args.classes, args.iterations) for _ in range(args.repeats)]
    digests = {digest for _, digest in samples}
    if len(digests) != 1:
        raise AssertionError("Non-deterministic final state across repetitions")
    print(json.dumps({"nodes": args.nodes, "classes": args.classes,
                      "iterations": args.iterations,
                      "times_s": [elapsed for elapsed, _ in samples],
                      "sha256": samples[0][1]}))


if __name__ == "__main__":
    main()
