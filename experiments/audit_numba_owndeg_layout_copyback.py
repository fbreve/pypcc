"""Source-exact Numba owndeg C-native, F-copyback complete-fit A/B study.

Frozen protocol: experiments/numba_owndeg_layout_copyback_protocol.md.
Original propagation, graph, particle RNG, stopping and all backends unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
import types

import numpy as np
from numba import njit
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from benchmarks.fixtures import array_hash
from benchmarks.runner import expected_backend
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph

BASE = "4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
PROTECTED = ("pcc.py", "pcc_graph.py", "pcc_numba.py", "pcc_numpy.py",
             "pcc_step.pyx", "pcc_validation.py", "setup.py")
OLD_OWN = """        self.owndeg = np.zeros(
            shape=(n_nodes, self.c),
            dtype=np.float64,
            order='F',
        )"""
NEW_OWN = OLD_OWN.replace("order='F'", "order='C' if use_c_order else 'F'")
OLD_SUM = "        row_sums = np.sum(self.owndeg, axis=1, keepdims=True)"
NEW_SUM = (
    "        # Copy only temporary true-Numba C-order owndeg back to original F.\n"
    "        if self.owndeg.flags.c_contiguous and self.owndeg.shape[1] >= 8:\n"
    "            self.owndeg = np.asfortranarray(self.owndeg)\n"
    "        row_sums = np.sum(self.owndeg, axis=1, keepdims=True)"
)
CASES = ("wine", "breast_cancer", "digits",
         "ring_20000_16", "ring_60000_32", "ring_100000_50")
REAL = ("wine", "breast_cancer", "digits")
BACKENDS = ("numpy", "numba", "cython")
MASK_SEEDS = (85903, 86011)
STATE_KEYS = ("predictions", "dominance", "positions", "strengths",
              "distances", "owndeg", "node_labels", "mapped_labels")
PARAMS = dict(p_grd=.5, delta_v=.1, deltap=1., dexp=2., es_chk=200, seed=29)
PAIRS = 5
MEMORY_CASES = (("ring_100000_50", "numba", 0, False),
                ("ring_100000_50", "numba", 2500, False),
                ("digits", "numba", 0, False),
                ("digits", "numba", 1000, False),
                ("digits", "numba", 1000, True))


@njit
def next_numba():
    return np.random.random()


def original_git_source():
    root = Path(__file__).resolve().parents[1]
    for path in PROTECTED:
        blob = subprocess.check_output(["git", "show", f"{BASE}:{path}"])
        current = (root / path).read_bytes()
        assert current.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"), path
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD", "--", *PROTECTED],
        text=True).strip()
    assert not changed, f"protected real production changed: {changed}"
    blob = subprocess.check_output(["git", "show", f"{BASE}:pcc.py"])
    return blob


def source_derived_candidate(original):
    src = original.decode("utf-8")
    assert src.count(OLD_OWN) == 1 and src.count(OLD_SUM) == 1
    altered = src.replace(OLD_OWN, NEW_OWN, 1).replace(OLD_SUM, NEW_SUM, 1)
    restored = altered.replace(NEW_OWN, OLD_OWN, 1).replace(NEW_SUM, OLD_SUM, 1)
    assert restored == src, "candidate not two literal Git source replacements"
    module = types.ModuleType("_pcc_numba_owndeg_layout_science")
    module.__file__ = "source_exact/numba_owndeg_candidate_pcc.py"
    sys.modules[module.__name__] = module
    exec(compile(altered, module.__file__, "exec"), module.__dict__)
    return module.ParticleCompetitionAndCooperation, hashlib.sha256(altered.encode()).hexdigest()

def actual_dispatch_branch_guard(cls):
    """Verify actual candidate native C layout, other backends F, final F."""
    n, c = 160, 10
    rows=np.arange(n,dtype=np.int64)
    graph=np.column_stack(((rows-1)%n,(rows+1)%n))
    degrees=np.full(n,2,dtype=np.int64)
    labels=np.full(n,-1,dtype=np.int64)
    labels[2:2+c]=np.arange(c,dtype=np.int64)
    findings={}
    for backend in BACKENDS:
        m=make_model(cls,backend,graph,degrees,2)
        o=make_model(PCC,backend,graph,degrees,2)
        assert m._get_backend_fn() is expected_backend(backend)
        for obj in (m,o):
            obj.c=c
            obj.mapped_labels=labels.copy()
        cm=m._ParticleCompetitionAndCooperation__genNodes()
        co=o._ParticleCompetitionAndCooperation__genNodes()
        want_c=backend=="numba"
        assert bool(m.owndeg.flags.c_contiguous)==want_c,backend
        assert bool(m.owndeg.flags.f_contiguous)==(not want_c),backend
        assert bool(o.owndeg.flags.f_contiguous)
        assert array_hash(cm.dominance)==array_hash(co.dominance)
        pc=fit(m,labels,0,False)
        po=fit(o,labels,0,False)
        assert fingerprints(actual_state(m,pc))==fingerprints(actual_state(o,po))
        assert m.owndeg.flags.f_contiguous and not m.owndeg.flags.c_contiguous
        findings[backend]=[want_c,not want_c]
    return findings

def micro_functional():
    """Float64 original F -> native C -> restored F: exact numeric bytes."""
    rng=np.random.default_rng(86057)
    checked=0
    for n in (0,1,3,1024,30000):
        for classes in (0,2,10,50):
            for order in ("C","F"):
                arr=np.array(rng.standard_normal((n,classes)),dtype=np.float64,order=order)
                if n and classes:
                    arr[0,0]=-0.0
                    if n>1: arr[1,0]=np.nan
                    if n>2: arr[2,0]=np.inf
                for _ in range(5):
                    restored=np.asfortranarray(np.array(arr,order="C",copy=True))
                    assert np.array_equal(arr,restored,equal_nan=True)
                    assert array_hash(arr)==array_hash(restored),(n,classes,order)
                    checked+=1
    assert checked==200
    return checked

def masks_for(truth):
    rows = []
    for seed in MASK_SEEDS:
        rng = np.random.default_rng(seed)
        mask = np.full(len(truth), -1, dtype=np.int64)
        for cls in np.unique(truth):
            positions = np.flatnonzero(truth == cls)
            assert len(positions) >= 4
            mask[rng.choice(positions, size=3, replace=False)] = cls
        rows.append(mask)
    assert array_hash(rows[0]) != array_hash(rows[1])
    return rows


def fixture(case):
    if case in REAL:
        data = getattr(datasets, "load_" + case)()
        truth = np.asarray(data.target, dtype=np.int64)
        features = StandardScaler().fit_transform(data.data)
        graph, degrees = build_knn_graph(features, k_nn=10, n_jobs=1)
        k = 10
    else:
        _, n, classes = case.split("_")
        n, classes = int(n), int(classes)
        truth = np.arange(n, dtype=np.int64) % classes
        rows = np.arange(n, dtype=np.int64)
        graph = np.column_stack(((rows-1) % n, (rows+1) % n))
        degrees = np.full(n, 2, dtype=np.int64)
        k = 2
        # Undirected reciprocal ring, not directed kNN.
        assert np.array_equal(graph[graph[:, 0], 1], rows)
        assert np.array_equal(graph[graph[:, 1], 0], rows)
    masks = masks_for(truth)
    assert len(np.unique(truth)) >= 2
    return graph, degrees, masks, k


def make_model(cls, backend, graph, degree, k):
    model = cls(impl=backend)
    model.set_graph(graph, degree, k_nn=k)
    assert model._get_backend_fn() is expected_backend(backend)
    return model


def actual_state(model, predicted):
    return (predicted, model.node.dominance, model.part.curnode,
            model.part.strength, model.part.dist_table, model.owndeg,
            model.node.label, model.mapped_labels)


def fingerprints(state):
    return dict(zip(STATE_KEYS, (array_hash(a) for a in state)))


def fit(model, mask, iterations, stop):
    return model.fit_predict(mask, max_iter=iterations,
                             early_stop=stop, **PARAMS)


def batch_size(case,steps):
    if case in REAL:
        return 64 if steps>=1000 else 128
    return {"ring_20000_16":16,"ring_60000_32":8,
            "ring_100000_50":8}[case]

def timed_batch(model, masks, iterations, stop, n):
    objects = []
    start_cpu = time.process_time_ns()
    start_wall = time.perf_counter_ns()
    for i in range(n):
        ans = fit(model, masks[i % 2], iterations, stop)
        objects.append(actual_state(model, ans))
    elapsed_wall = time.perf_counter_ns() - start_wall
    elapsed_cpu = time.process_time_ns() - start_cpu
    # Hash, object checks and any native diagnostics are AFTER both clocks.
    hashes = [fingerprints(st) for st in objects]
    for left, right in zip(objects[:-1], objects[1:]):
        for j in (0, 1, 4, 5, 6, 7):
            assert left[j] is not right[j], "successive fits illegally alias state"
    layout = [(bool(st[1].flags.c_contiguous), bool(st[1].flags.f_contiguous),
               bool(st[5].flags.c_contiguous), bool(st[5].flags.f_contiguous))
              for st in objects]
    assert all(not l[2] and l[3] for l in layout), "original final owndeg F layout lost"
    return dict(cpu_ns=elapsed_cpu, wall_ns=elapsed_wall,
                states=hashes, dominance_layout=layout)


def rng_endpoint(backend):
    if backend == "numba":
        return float(next_numba())
    if backend == "numpy":
        return float(np.random.random())
    return None


def replay_with_real_stop_info(model, mask, iterations, stop):
    actual = model._get_backend_fn()
    storage = {}

    def backend(*args, **kwargs):
        assert "return_info" not in kwargs
        storage["info"] = actual(*args, **kwargs, return_info=True)
        return storage["info"]

    # Preserve original __genNodes C-order layout test for true native Numba.
    backend.__name__ = actual.__name__
    assert "_get_backend_fn" not in model.__dict__
    model._get_backend_fn = lambda: backend
    try:
        prediction = fit(model, mask, iterations, stop)
    finally:
        del model._get_backend_fn
    assert isinstance(storage.get("info"), dict)
    return fingerprints(actual_state(model, prediction)), storage["info"], rng_endpoint(model.impl)


def true_replay(a, b, masks, iterations, stop, sha):
    evidence = []
    for k, mask in enumerate(masks):
        ah, ai, ar = replay_with_real_stop_info(a, mask, iterations, stop)
        bh, bi, br = replay_with_real_stop_info(b, mask, iterations, stop)
        assert ah == bh == sha[k], "paired original-vs-candidate replay state mismatch"
        assert ai == bi, "native true stopping metrics mismatch"
        assert ar == br, "true next original RNG mismatch"
        assert a.node.dominance.flags.c_contiguous == b.node.dominance.flags.c_contiguous
        assert a.node.dominance.flags.f_contiguous == b.node.dominance.flags.f_contiguous
        assert a.owndeg.flags.f_contiguous and b.owndeg.flags.f_contiguous
        evidence.append(dict(stop_info=ai, next_rng=ar))
    # Invalid seed is rejected BEFORE mutations and preserves previous result ownership.
    for m in (a, b):
        old = actual_state(m, masks[-1])
        sha_before = fingerprints(old)
        try:
            fit(m, masks[-1], iterations, stop=False) if False else m.fit_predict(
                masks[-1], max_iter=iterations, early_stop=stop, seed=-1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed not rejected")
        after = actual_state(m, old[0])
        assert all(left is right for left, right in zip(old[1:], after[1:]))
        assert fingerprints(after) == sha_before
    return evidence


def cases_and_modes(smoke=False):
    if smoke:
        return [("wine", "numba", 0, False)]
    out = []
    for name in CASES:
        for backend in BACKENDS:
            out.extend((name, backend, i, False) for i in (0, 5))
            if name in REAL and backend in ("numba", "cython"):
                out.extend((name, backend, 1000, stop) for stop in (False, True))
    out.extend((name, "numba", 2500, False) for name in CASES if name not in REAL)
    assert len(out) == 51, len(out)
    return out


def one_cohort(cls, case, backend, iterations, stop, smoke=False):
    graph, degree, masks, k = fixture(case)
    orig = make_model(PCC, backend, graph, degree, k)
    cand = make_model(cls, backend, graph, degree, k)
    # Numba compilation + loaded cache and diagnostic RNG helper off-clock.
    if backend == "numba":
        next_numba()
        for m in (orig, cand):
            fit(m, masks[0], 2, False)
    reps = 1 if smoke else PAIRS
    n = 8 if smoke else batch_size(case,iterations)
    rows = []
    for rep in range(reps):
        if rep % 2 == 0:
            a = timed_batch(orig, masks, iterations, stop, n)
            b = timed_batch(cand, masks, iterations, stop, n)
        else:
            b = timed_batch(cand, masks, iterations, stop, n)
            a = timed_batch(orig, masks, iterations, stop, n)
        assert a["states"] == b["states"], "bitwise native state deviation"
        assert a["dominance_layout"] == b["dominance_layout"], "C/F layout deviation"
        if backend == "numba" and len(masks):
            assert all(x[0] for x in a["dominance_layout"]) if case.endswith("_50") else True
        replay = true_replay(orig, cand, masks, iterations, stop, a["states"])
        rows.append(dict(rep=rep, original_cpu_ns=a["cpu_ns"],
                         candidate_cpu_ns=b["cpu_ns"],
                         original_wall_ns=a["wall_ns"],
                         candidate_wall_ns=b["wall_ns"],
                         state_shas=a["states"], layout=a["dominance_layout"],
                         native_replay=replay))
        print(f"{case} {backend} steps={iterations} stop={stop} pair={rep+1}/{reps} "
              f"original_cpu_ms={a['cpu_ns']/1e6:.3f} "
              f"candidate_cpu_ms={b['cpu_ns']/1e6:.3f}", flush=True)
    return dict(case=case, backend=backend, steps=iterations, early_stop=stop,
                graph_sha=array_hash(graph), degree_sha=array_hash(degree),
                label_shas=[array_hash(m) for m in masks],
                batch_fits=n, pairs=rows)


def peak_memory_trial(cls, case, backend, iterations, stop):
    graph, degree, masks, k = fixture(case)
    records = []
    # Each sample uses FRESH model, same original graph values in both arms.
    for rep in range(3):
        original = make_model(PCC, backend, graph, degree, k)
        candidate = make_model(cls, backend, graph, degree, k)
        def one_fit(model):
            tracemalloc.start()
            tracemalloc.reset_peak()
            ans = fit(model, masks[rep % 2], iterations, stop)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return int(peak), fingerprints(actual_state(model, ans))
        if rep % 2:
            candidate_peak, candidate_sha = one_fit(candidate)
            original_peak, original_sha = one_fit(original)
        else:
            original_peak, original_sha = one_fit(original)
            candidate_peak, candidate_sha = one_fit(candidate)
        assert original_sha == candidate_sha, "memory cohort native state changed"
        records.append(dict(rep=rep, original_peak_bytes=original_peak,
                            candidate_peak_bytes=candidate_peak))
    return dict(case=case, backend=backend, steps=iterations,
                early_stop=stop, samples=records)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True)
    args=p.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        assert os.environ.get(key) == "1", f"thread setting {key} must be 1"
    blob=original_git_source()
    candidate, candidate_sha=source_derived_candidate(blob)
    micro=micro_functional()
    dispatch_shapes=actual_dispatch_branch_guard(candidate)
    cohorts=[]
    for case,backend,steps,stop in cases_and_modes(args.smoke):
        cohorts.append(one_cohort(candidate, case, backend, steps, stop, args.smoke))
    memory=[]
    if not args.smoke:
        for case,backend,steps,stop in MEMORY_CASES:
            memory.append(peak_memory_trial(candidate,case,backend,steps,stop))
    raw=dict(schema=1, source_base=BASE, original_sha=hashlib.sha256(blob).hexdigest(),
             candidate_sha=candidate_sha, os=sys.platform, platform=platform.platform(),
             python=sys.version, cases=CASES, micro_functional_checks=micro,
             exact_production_source=True, dispatch_branch_shapes=dispatch_shapes,
             pairs=1 if args.smoke else PAIRS,
             cohorts=cohorts, memory=memory)
    path=Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"SOURCE EXACT NUMBA OWNDEG LAYOUT SCIENCE COMPLETE {sys.platform}: "
          f"{len(cohorts)} cohorts, {micro} standalone correctness checks",flush=True)


if __name__ == "__main__":
    main()
