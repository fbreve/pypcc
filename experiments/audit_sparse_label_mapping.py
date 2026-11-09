"""Frozen source-block micro and exact actual-PCC native label-mapping comparison.

Only experimental helper code lives here; no changes to PCC native backends.
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
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation as Candidate
from pcc_graph import build_knn_graph

BASE = "a2fd27adf53155f378f12eeeb7cf67ec0625e195"
ALL_BACKENDS = ("numpy", "numba", "cython")
DATASETS = (("Iris", load_iris), ("Wine", load_wine),
            ("Breast Cancer", load_breast_cancer), ("Digits", load_digits))
LABEL_SEEDS = (37027, 37053)
CASES = (
    (150, 3, "sparse"), (569, 2, "sparse"),
    (1023, 10, "sparse"), (1024, 10, "sparse"),
    (1797, 10, "sparse"), (30000, 10, "sparse"),
    (30000, 50, "sparse"), (100000, 50, "sparse"),
    (100000, 2, "sparse"), (30000, 50, "half"),
    (30000, 10, "half"), (100000, 50, "ninety"),
)
OLD = """            self.mapped_labels = np.full(self.labels.shape, -1, dtype=np.int64)
            for lbl, idx in self.label_to_idx.items():
                self.mapped_labels[self.labels == lbl] = idx"""
NEW = """            self.mapped_labels = np.full(self.labels.shape, -1, dtype=np.int64)
            if self.labels.size < 1024:
                for lbl, idx in self.label_to_idx.items():
                    self.mapped_labels[self.labels == lbl] = idx
            else:
                # Scan once for known homes; map only those labels in sorted
                # unique_labels. The padded graph and fit backends stay intact.
                known_rows = np.flatnonzero(self.labels != -1)
                self.mapped_labels[known_rows] = np.searchsorted(
                    self.unique_labels, self.labels[known_rows]
                )"""


def array_sha(value):
    a = np.asarray(value)
    return hashlib.sha256(json.dumps([a.shape, a.dtype.str]).encode()
                          + np.ascontiguousarray(a).tobytes()).hexdigest()


def original_class_and_blocks():
    baseline_bytes = subprocess.check_output(["git", "show", BASE + ":pcc.py"])
    baseline = baseline_bytes.decode("utf-8").replace("\r\n", "\n")
    current = Path("pcc.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    if baseline.count(OLD) != 1 or baseline.replace(OLD, NEW) != current:
        raise AssertionError("source guard: more than exact mapped-label hunk was changed")
    mod = types.ModuleType("original_pcc_mapping")
    sys.modules[mod.__name__] = mod
    exec(compile(baseline, BASE + ":pcc.py", "exec"), mod.__dict__)
    blocks = []
    for block, name in ((OLD, "original"), (NEW, "candidate")):
        namespace = {"np": np}
        exec(compile("def mapped_block(self):\n" + block + "\n", name, "exec"),
             namespace)
        blocks.append(namespace["mapped_block"])
    return mod.ParticleCompetitionAndCooperation, blocks, hashlib.sha256(baseline_bytes).hexdigest()


def case_inputs(n, classes, density):
    # Same deterministic fixtures on Linux/Windows. Nonconsecutive negative labels,
    # preserving the excluded sentinel -1 and exact original np.unique order.
    rng = np.random.default_rng(37011 + n + classes + {"sparse": 0, "half": 1, "ninety": 2}[density])
    labels = np.full(n, -1, dtype=np.int64)
    count = 3 * classes if density == "sparse" else (
        n//2 if density == "half" else int(n*0.9))
    homes = rng.choice(n, size=count, replace=False)
    source_labels = (np.arange(classes, dtype=np.int64) * 371) - 1001111
    labels[homes] = source_labels[np.arange(count) % classes]
    uniques = np.unique(labels)
    uniques = uniques[uniques != -1]
    if uniques.size != classes:
        raise AssertionError("missing actual class")
    mapping = {lbl: idx for idx, lbl in enumerate(uniques)}
    reference = np.full(n, -1, dtype=np.int64)
    for key, idx in mapping.items():
        reference[labels == key] = idx
    return labels, uniques, mapping, reference


def new_box(labels, uniques, mapping):
    return types.SimpleNamespace(labels=labels, unique_labels=uniques,
                                 label_to_idx=mapping, mapped_labels=None)


def verify_mapping(box, expected, labels):
    actual = box.mapped_labels
    if actual is None:
        raise AssertionError("mapped array not installed")
    if actual.dtype != np.int64 or actual.shape != labels.shape:
        raise AssertionError("dtype/shape changed")
    if not actual.flags.c_contiguous or not actual.flags.writeable:
        raise AssertionError("mapped layout/ownership changed")
    if np.shares_memory(actual, labels) or not np.array_equal(actual, expected):
        raise AssertionError("incorrect mappings or input alias")
    return array_sha(actual)


def timed_mapping(fn, box):
    box.mapped_labels = None  # avoid counting previous result in current peak
    started = time.perf_counter_ns()
    fn(box)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return elapsed_ms


def peak_mapping(fn, box):
    box.mapped_labels = None
    gc.collect()
    tracemalloc.start()
    try:
        fn(box)
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def micro(blocks):
    results = []
    for n, classes, density in CASES:
        labels, uniques, mapping, expected = case_inputs(n, classes, density)
        boxes = [new_box(labels, uniques, mapping), new_box(labels, uniques, mapping)]
        for i in (0, 1):
            for _ in range(3):
                timed_mapping(blocks[i], boxes[i])
                verify_mapping(boxes[i], expected, labels)
        pairs = []
        for i in range(25):
            order = (0, 1) if i % 2 else (1, 0)
            pair = [None, None]
            for j in order:
                pair[j] = timed_mapping(blocks[j], boxes[j])
                verify_mapping(boxes[j], expected, labels)
            pairs.append(pair)
        peak_pairs = []
        for i in range(5):
            order = (0, 1) if i % 2 else (1, 0)
            pair = [None, None]
            for j in order:
                pair[j] = peak_mapping(blocks[j], boxes[j])
                verify_mapping(boxes[j], expected, labels)
            peak_pairs.append(pair)
        saving = 100*(1-statistics.median([b/a for a,b in pairs]))
        print(f"{n}x{classes} {density}: mapping CPU {saving:+.2f}%; "
              f"tracemalloc peak reference/candidate "
              f"{statistics.median(row[0] for row in peak_pairs):.0f}/"
              f"{statistics.median(row[1] for row in peak_pairs):.0f} B", flush=True)
        results.append(dict(nodes=n, classes=classes, density=density,
                            labels_sha=array_sha(labels), uniques_sha=array_sha(uniques),
                            mapped_sha=array_sha(expected), homes=int((labels != -1).sum()),
                            timings_ms=pairs, traced_peak_bytes=peak_pairs))
    return results


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
                        fit_with_metrics(cls, neighbors, degrees, labels, backend, False, 37059)
                for early_stop in (False, True):
                    order=(0,1) if label_seed==LABEL_SEEDS[0] else (1,0)
                    pair=[None,None]
                    for idx in order:
                        pair[idx]=fit_with_metrics((reference,Candidate)[idx],
                                                   neighbors,degrees,labels,
                                                   backend,early_stop,37059)
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
    original, blocks, original_git_sha = original_class_and_blocks()
    result = dict(schema=1, base_commit=BASE, original_pcc_sha256=original_git_sha,
                  head_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  os=sys.platform, host=platform.platform(), cpu=platform.processor(),
                  python=sys.version, numpy=np.__version__,
                  micro=micro(blocks), real=real_native(original))
    path=Path(os.environ.get("PCC_LABEL_MAP_RESULT", "label_mapping_result.json"))
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Complete original-vs-candidate label mapping audit:", path,
          len(result["micro"]), "micro cases and", len(result["real"]),
          "native original/candidate fit pairs", flush=True)


if __name__=="__main__":
    main()
