"""Prospective original-Git-blob comparison for set_graph installation overhead.

Run from repository root on Linux/Windows Python 3.12 (see protocol).
This is experiment code, NOT part of PCC public/runtime API.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

import numpy as np

from pcc import ParticleCompetitionAndCooperation as Candidate
from pcc_graph import build_knn_graph

BASE = "ababc48986b40f083d5930e922f844aa2ea418b2"
SEED = 34007
SIZES = ((150, 20), (569, 25), (1797, 35),
         (30000, 35), (30000, 100), (100000, 35))
OLD = '''        for row, degree in zip(neib_list, new_neib_qt):
            validate_finite("neib_list", row[:degree])
        # Convert only active entries: padding may contain NaN, infinity,
        # strings or arbitrary sentinels and is never read by the kernels.
        active = np.arange(width)[None, :] < new_neib_qt[:, None]
        new_neib_list = np.zeros((n_nodes, width), dtype=np.int64)
        self._check_int64_conversion_range("neib_list", neib_list[active])
        new_neib_list[active] = neib_list[active].astype(np.int64)
        if np.any(new_neib_list[active] < 0) or np.any(new_neib_list[active] >= n_nodes):
            raise ValueError("active neighbor indices must refer to existing nodes")'''
NEW = '''        # Validate and convert only active entries. The existing dense
        # mask is reused instead of performing one Python call per node.
        # Unused padding may contain NaN, infinity or arbitrary sentinels.
        active = np.arange(width)[None, :] < new_neib_qt[:, None]
        active_neighbors = neib_list[active]
        validate_finite("neib_list", active_neighbors)
        self._check_int64_conversion_range("neib_list", active_neighbors)
        converted_neighbors = active_neighbors.astype(np.int64, copy=False)
        if np.any(converted_neighbors < 0) or np.any(converted_neighbors >= n_nodes):
            raise ValueError("active neighbor indices must refer to existing nodes")
        new_neib_list = np.zeros((n_nodes, width), dtype=np.int64)
        new_neib_list[active] = converted_neighbors'''


def sha(a):
    a = np.asarray(a)
    return hashlib.sha256(a.tobytes(order="C") + str(a.shape).encode()
                          + str(a.dtype).encode()).hexdigest()


def baseline():
    source_bytes = subprocess.check_output(
        ["git", "show", BASE + ":pcc.py"])
    original = source_bytes.decode("utf-8").replace("\r\n", "\n")
    current = Path("pcc.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    if original.count(OLD) != 1 or original.replace(OLD, NEW) != current:
        raise RuntimeError("production pcc.py does not equal the single preregistered candidate block")
    mod = types.ModuleType("pcc_set_graph_git_baseline")
    sys.modules[mod.__name__] = mod
    exec(compile(original, BASE + ":pcc.py", "exec"), mod.__dict__)
    return mod.ParticleCompetitionAndCooperation, hashlib.sha256(source_bytes).hexdigest()


def fixture(n, w, rng):
    degree = rng.integers(1, w + 1, size=n, dtype=np.int64)
    graph = rng.integers(0, n, size=(n, w), dtype=np.int64)
    graph[np.arange(w)[None, :] >= degree[:, None]] = -1
    return graph, degree


def check_exact(reference, candidate, graph, degree):
    a = reference(impl="numpy")
    b = candidate(impl="numpy")
    a.set_graph(graph, degree)
    b.set_graph(graph, degree)
    for name in ("neib_list", "neib_qt"):
        arr = getattr(a, name)
        result = getattr(b, name)
        if arr.dtype != result.dtype or arr.shape != result.shape or not np.array_equal(arr, result):
            raise AssertionError(name + ": full original-vs-candidate graph mismatch")
    if a.k_nn != b.k_nn:
        raise AssertionError("k_nn mismatch")
    if np.shares_memory(graph, b.neib_list) or np.shares_memory(degree, b.neib_qt):
        raise AssertionError("public copy/ownership changed")
    return sha(a.neib_list), sha(a.neib_qt)


def timed(cls, graph, degree):
    m = cls(impl="numpy")
    start = time.perf_counter_ns()
    m.set_graph(graph, degree)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return elapsed


def peak(cls, graph, degree):
    gc.collect()
    tracemalloc.start()
    try:
        m = cls(impl="numpy")
        m.set_graph(graph, degree)
        _, bytes_peak = tracemalloc.get_traced_memory()
        return bytes_peak
    finally:
        tracemalloc.stop()


def adverse(reference, candidate):
    from decimal import Decimal
    cases = [
        (np.array([[1., np.nan], [0., np.inf]]), np.ones(2), True),
        (np.array([[1, "bad"], [0, None]], dtype=object), np.ones(2), True),
        (np.array([[Decimal("NaN")], [0]], dtype=object), np.ones(2), False),
        (np.array([[np.inf], [0.]]), np.ones(2), False),
        (np.array([[2**64 - 1], [0]], dtype=np.uint64), np.ones(2), False),
        (np.array([[-1], [0]], dtype=np.int64), np.ones(2), False),
        (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), True),
        (np.empty((2, 0), dtype=np.int64), np.zeros(2, dtype=np.int64), True),
    ]
    for graph, degree, valid in cases:
        outcomes = []
        for cls in (reference, candidate):
            model = cls(impl="numpy")
            model.set_graph(np.array([[1], [0]]), np.ones(2, dtype=np.int64))
            old = (model.neib_list, model.neib_qt, model.k_nn)
            try:
                model.set_graph(graph, degree)
                outcomes.append(("ok", sha(model.neib_list), sha(model.neib_qt)))
                if not valid:
                    raise AssertionError("accepted a designated invalid graph")
            except (TypeError, ValueError, OverflowError) as exc:
                if valid:
                    raise
                if model.neib_list is not old[0] or model.neib_qt is not old[1] or model.k_nn != old[2]:
                    raise AssertionError("invalid graph modified installed state")
                outcomes.append(("error", type(exc).__name__, str(exc)))
        if outcomes[0] != outcomes[1]:
            raise AssertionError("different original/candidate public behavior: " + str(outcomes))


def native_pairs(reference, candidate):
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
    from sklearn.preprocessing import StandardScaler
    cases = [("Iris", load_iris), ("Wine", load_wine),
             ("Breast Cancer", load_breast_cancer), ("Digits", load_digits)]
    expected = dict(numpy="pcc_numpy", numba="pcc_numba", cython="pcc_step")
    results = []
    for dataset, loader in cases:
        X, y = loader(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        graph, degree = build_knn_graph(X, k_nn=10)
        labels = np.full(y.size, -1, dtype=np.int64)
        for value in np.unique(y):
            labels[np.flatnonzero(y == value)[:3]] = int(value)
        for backend, module in expected.items():
            a, b = reference(impl=backend), candidate(impl=backend)
            a.set_graph(graph, degree, k_nn=10)
            b.set_graph(graph, degree, k_nn=10)
            if a._get_backend_fn().__module__ != module or b._get_backend_fn().__module__ != module:
                raise AssertionError("native backend fallback " + backend)
            predictions = []
            for model in (a, b):
                result = model.fit_predict(labels, max_iter=101, early_stop=False, seed=34017)
                predictions.append(tuple(sha(z) for z in (
                    result, model.node.dominance, model.part.curnode,
                    model.part.strength, model.part.dist_table, model.owndeg)))
            if predictions[0] != predictions[1]:
                raise AssertionError("native final state mismatch: " + dataset + "/" + backend)
            results.append(dict(dataset=dataset, backend=backend, state_hashes=predictions[0]))
    return results


def main():
    Reference, original_sha = baseline()
    adverse(Reference, Candidate)
    rng = np.random.default_rng(SEED)
    rows = []
    for n, w in SIZES:
        graph, degree = fixture(n, w, rng)
        graph_sha, degree_sha = check_exact(Reference, Candidate, graph, degree)
        for cls in (Reference, Candidate):
            timed(cls, graph, degree)  # separate untimed warm-up per size/method
        timings = []
        for i in range(25):
            order = (0, 1) if i % 2 else (1, 0)
            row = [None, None]
            for index in order:
                row[index] = timed((Reference, Candidate)[index], graph, degree)
            timings.append(row)
        memories = [[peak(cls, graph, degree) for cls in (Reference, Candidate)]
                    for _ in range(5)]
        if min(x for pair in timings for x in pair) <= 0:
            raise AssertionError("non-positive CPU time")
        if min(x for pair in memories for x in pair) <= 0:
            raise AssertionError("non-positive traced peak")
        rows.append(dict(nodes=n, width=w, graph_sha=graph_sha,
                         degree_sha=degree_sha, timings_ms=timings,
                         traced_peaks_bytes=memories))
        print(f"{n}x{w}: paired CPU saving "
              f"{100*(1-statistics.median([b/a for a,b in timings])):.2f}%",
              flush=True)
    native = native_pairs(Reference, Candidate)
    result = dict(schema=1, baseline=BASE, original_blob_sha256=original_sha,
                  seed=SEED, platform=sys.platform, hostname=platform.node(),
                  cpu=platform.processor(), python=sys.version,
                  numpy=np.__version__, cases=rows, adverse_count=8,
                  native=native)
    outfile = Path(os.environ.get("PCC_SET_GRAPH_RESULT", "set_graph_validation_result.json"))
    outfile.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote", outfile, "with", len(rows), "sizes and", len(native), "native pairs")


if __name__ == "__main__":
    main()
