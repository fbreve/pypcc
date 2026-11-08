"""Native original-master comparison of single-hunk __genNodes one-hot initialization."""
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
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation as Candidate
from pcc_graph import build_knn_graph

BASE = "7e589e2c9097a507d636b55c6269c0e3223e8343"
SIZES = ((150, 3), (1797, 10), (30000, 10),
         (30000, 50), (100000, 50), (100000, 2))
LAYOUT_BACKENDS = ("numpy", "numba")
ALL_BACKENDS = ("numpy", "numba", "cython")
DATASETS = (("Iris", load_iris), ("Wine", load_wine),
            ("Breast Cancer", load_breast_cancer), ("Digits", load_digits))
LABEL_SEEDS = (35027, 35053)
OLD = """        for i in range(self.c):
            dominance[self.mapped_labels == i, i] = 1.0"""
NEW = """        # Assign all labeled one-hot entries without rescanning every node
        # once per class. Preserve the existing dominance array and layout.
        dominance[np.flatnonzero(labeled_idx), self.mapped_labels[labeled_idx]] = 1.0"""


def array_sha(value):
    a = np.asarray(value)
    return hashlib.sha256(json.dumps([a.shape, a.dtype.str]).encode()
                          + np.ascontiguousarray(a).tobytes()).hexdigest()


def reference_class():
    original_bytes = subprocess.check_output(["git", "show", BASE + ":pcc.py"])
    original = original_bytes.decode("utf-8").replace("\r\n", "\n")
    current = Path("pcc.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    if original.count(OLD) != 1 or original.replace(OLD, NEW) != current:
        raise RuntimeError("source diff violates the sole precommitted __genNodes block")
    mod = types.ModuleType("original_pcc_nodes_onehot")
    sys.modules[mod.__name__] = mod
    exec(compile(original, BASE + ":pcc.py", "exec"), mod.__dict__)
    return mod.ParticleCompetitionAndCooperation, hashlib.sha256(original_bytes).hexdigest()


def prepare_model(cls, mapped, classes, backend):
    m = cls(impl=backend)
    m.neib_list = np.empty((len(mapped), 1), dtype=np.int64)
    m.mapped_labels = mapped
    m.c = classes
    fn = m._get_backend_fn()
    expected = {"numpy": "pcc_numpy", "numba": "pcc_numba", "cython": "pcc_step"}[backend]
    if fn.__module__ != expected:
        raise RuntimeError("native backend fallback: " + backend)
    return m


def node_state(model):
    nodes = model._ParticleCompetitionAndCooperation__genNodes()
    return nodes, (array_sha(nodes.dominance), array_sha(nodes.label),
                   array_sha(model.owndeg))


def node_time(cls, mapped, classes, backend):
    model = prepare_model(cls, mapped, classes, backend)
    start = time.perf_counter_ns()
    nodes, hashes = node_state(model)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return elapsed, hashes, (nodes.dominance.strides, nodes.dominance.flags.c_contiguous,
                              nodes.dominance.flags.f_contiguous,
                              model.owndeg.strides, model.owndeg.flags.f_contiguous)


def node_memory(cls, mapped, classes, backend):
    model = prepare_model(cls, mapped, classes, backend)
    gc.collect()
    tracemalloc.start()
    try:
        nodes, hashes = node_state(model)
        _, peak = tracemalloc.get_traced_memory()
        return peak, hashes
    finally:
        tracemalloc.stop()


def micro(reference):
    rng = np.random.default_rng(35011)
    rows = []
    for n, classes in SIZES:
        mapped = np.full(n, -1, dtype=np.int64)
        homes = rng.choice(n, size=3 * classes, replace=False)
        mapped[homes] = np.repeat(np.arange(classes, dtype=np.int64), 3)
        mapped_sha = array_sha(mapped)
        for backend in LAYOUT_BACKENDS:
            a = node_time(reference, mapped, classes, backend)
            b = node_time(Candidate, mapped, classes, backend)
            if a[1:] != b[1:]:
                raise AssertionError(f"changed complete node arrays/layout {n} {classes} {backend}")
            samples = []
            for i in range(20):
                order = (1, 0) if i % 2 == 0 else (0, 1)
                pair = [None, None]
                for j in order:
                    elapsed, hashes, layout = node_time((reference, Candidate)[j],
                                                        mapped, classes, backend)
                    if hashes != a[1] or layout != a[2]:
                        raise AssertionError("non-deterministic node initialization")
                    pair[j] = elapsed
                samples.append(pair)
            memories = []
            for i in range(4):
                order = (1, 0) if i % 2 == 0 else (0, 1)
                pair = [None, None]
                for j in order:
                    peak, hashes = node_memory((reference, Candidate)[j],
                                               mapped, classes, backend)
                    if hashes != a[1]:
                        raise AssertionError("tracemalloc node-array mismatch")
                    pair[j] = peak
                memories.append(pair)
            rows.append(dict(nodes=n, classes=classes, backend=backend,
                             mapped_sha=mapped_sha, array_hashes=a[1],
                             layout=[list(a[2][0]), a[2][1], a[2][2],
                                     list(a[2][3]), a[2][4]],
                             timings_ms=samples, traced_peaks_bytes=memories))
            print(f"{n}x{classes}/{backend}: whole-init CPU saving "
                  f"{100*(1-statistics.median([y/x for x,y in samples])):+.2f}%", flush=True)
    return rows


def fit_with_metrics(cls, neighbors, degrees, labels, backend, early_stop, seed):
    m = cls(impl=backend)
    m.set_graph(neighbors, degrees, k_nn=10)
    underlying = m._get_backend_fn()
    expected = {"numpy": "pcc_numpy", "numba": "pcc_numba", "cython": "pcc_step"}[backend]
    if underlying.__module__ != expected:
        raise RuntimeError("native fallback " + backend)
    phase = {}
    def timed_backend(*args, **kwargs):
        kwargs["return_info"] = True
        entered = time.perf_counter_ns()
        info = underlying(*args, **kwargs)
        exited = time.perf_counter_ns()
        phase.update(info)
        phase["entered_ns"] = entered
        phase["exited_ns"] = exited
        return None
    # Crucial: original benchmark runner.measure *does not* preserve this name.
    # __genNodes tests this exact function name to choose C layout for Numba >=8 classes.
    timed_backend.__name__ = underlying.__name__
    m._get_backend_fn = lambda: timed_backend
    start = time.perf_counter_ns()
    predictions = m.fit_predict(labels, max_iter=1000, early_stop=early_stop,
                                es_chk=10, seed=seed)
    end = time.perf_counter_ns()
    if "entered_ns" not in phase:
        raise AssertionError("timed backend not invoked")
    expected_order = ("C" if backend == "numba" and len(np.unique(labels[labels != -1])) >= 8
                      else "F")
    flag = (m.node.dominance.flags.c_contiguous if expected_order == "C"
            else m.node.dominance.flags.f_contiguous)
    if not flag:
        raise AssertionError("incorrect native dominance C/F layout")
    arrays = {
        "predictions": predictions,
        "dominance": m.node.dominance,
        "positions": m.part.curnode,
        "strengths": m.part.strength,
        "distances": m.part.dist_table,
        "owndeg": m.owndeg,
    }
    hashes = {k: array_sha(v) for k,v in arrays.items()}
    rng_sha = None
    if backend == "numpy":
        state = np.random.get_state()
        rng_sha = hashlib.sha256(state[1].tobytes()
                                 + repr((state[0],state[2],state[3],state[4])).encode()).hexdigest()
    timings = dict(fit_predict_s=(end-start)/1e9,
                   initialization_and_dispatch_s=(phase["entered_ns"]-start)/1e9,
                   propagation_s=(phase["exited_ns"]-phase["entered_ns"])/1e9,
                   postprocessing_s=(end-phase["exited_ns"])/1e9)
    return dict(state_hashes=hashes, info={k:v for k,v in phase.items()
                                         if k not in ("entered_ns", "exited_ns")},
                backend=backend, backend_identity=f"{underlying.__module__}.{underlying.__name__}",
                layout=expected_order, rng_sha=rng_sha, timings=timings)


def real_native(reference):
    rows=[]
    for dataset, loader in DATASETS:
        raw, truth=loader(return_X_y=True)
        x=StandardScaler().fit_transform(raw)
        neighbors, degrees=build_knn_graph(x, k_nn=10, n_jobs=1)
        graph_sha=(array_sha(neighbors), array_sha(degrees))
        for label_seed in LABEL_SEEDS:
            rng=np.random.default_rng(label_seed)
            labels=np.full(len(truth), -1, dtype=np.int64)
            for cls in np.unique(truth):
                homes=rng.choice(np.flatnonzero(truth==cls), size=3, replace=False)
                labels[homes]=int(cls)
            label_sha=array_sha(labels)
            for backend in ALL_BACKENDS:
                # Precompile identical native argument shapes before timed comparisons.
                if backend=="numba":
                    for cls in (reference, Candidate):
                        fit_with_metrics(cls, neighbors, degrees, labels, backend, False, 35059)
                for early_stop in (False, True):
                    order=(0,1) if label_seed==LABEL_SEEDS[0] else (1,0)
                    pair=[None,None]
                    for idx in order:
                        pair[idx]=fit_with_metrics((reference,Candidate)[idx],
                                                   neighbors,degrees,labels,
                                                   backend,early_stop,35059)
                    a,b=pair
                    if any(a[field]!=b[field] for field in
                           ("state_hashes", "info", "backend_identity", "layout", "rng_sha")):
                        raise AssertionError("full native parity FAIL "
                                             +repr((dataset,label_seed,backend,early_stop)))
                    rows.append(dict(dataset=dataset,label_seed=label_seed,
                                     label_sha=label_sha,graph_sha=graph_sha,
                                     backend=backend,early_stop=early_stop,
                                     state_hashes=a["state_hashes"],
                                     info=a["info"],backend_identity=a["backend_identity"],
                                     layout=a["layout"],rng_sha=a["rng_sha"],
                                     reference_timings=a["timings"],
                                     candidate_timings=b["timings"]))
        print(f"{dataset}: completed {2*3*2} exact native state-pairs",flush=True)
    return rows


def main():
    original, blob_sha=reference_class()
    results = dict(schema=1, base_commit=BASE, git_blob_sha256=blob_sha,
                   head_commit=subprocess.check_output(["git", "rev-parse","HEAD"],
                                                       text=True).strip(),
                   os=sys.platform, platform=platform.platform(), cpu=platform.processor(),
                   python=sys.version, numpy=np.__version__, micro=micro(original),
                   real=real_native(original))
    path=Path(os.environ.get("PCC_ONEHOT_RESULT","nodes_onehot_result.json"))
    path.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Complete original-vs-candidate result:",path, len(results["micro"]),
          "micro cases and",len(results["real"]),"native state pairs",flush=True)

if __name__=="__main__":
    main()
