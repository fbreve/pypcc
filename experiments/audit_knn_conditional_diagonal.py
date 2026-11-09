"""Actual frozen kNN Git-blob baseline versus single conditional diagonal change."""
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pcc import ParticleCompetitionAndCooperation as Candidate
from pcc_graph import build_knn_graph as optimized_knn

BASE="c1073ae53ea785031de9e474b678e94f44a2124d"
OLD="""    adj.setdiag(0)                          # remove self-loops (safety)
    adj.eliminate_zeros()"""
NEW="""    if np.any(rows == cols):
        adj.setdiag(0)                      # remove actual self-loops
        adj.eliminate_zeros()"""
REAL=(("iris",load_iris),("wine",load_wine),
      ("breast_cancer",load_breast_cancer),("digits",load_digits))
BACKENDS=("numpy","numba","cython")
LABEL_SEEDS=(42027,42053)
K_VALUES=(1,3,10)
PROTECTED=("pcc.py","pcc_numpy.py","pcc_numba.py",
           "pcc_step.pyx","pcc_validation.py","setup.py","benchmarks")


def array_sha(value):
    a=np.asarray(value)
    return hashlib.sha256(json.dumps([a.shape,a.dtype.str]).encode()
                          +np.ascontiguousarray(a).tobytes()).hexdigest()


def original_source():
    raw=subprocess.check_output(["git","show",BASE+":pcc_graph.py"])
    original=raw.decode("utf-8").replace("\r\n","\n")
    current=Path("pcc_graph.py").read_text(encoding="utf-8").replace("\r\n","\n")
    if original.count(OLD)!=1 or original.replace(OLD,NEW)!=current:
        raise AssertionError("candidate graph source not exactly frozen single hunk")
    changed=subprocess.check_output(["git","diff","--name-only",BASE,"HEAD","--",
                                     *PROTECTED],text=True)
    if changed.strip():
        raise AssertionError("unexpected production/backend/benchmark changes: "+changed)
    mod=types.ModuleType("actual_prechange_pcc_graph")
    sys.modules[mod.__name__]=mod
    exec(compile(original,BASE+":pcc_graph.py","exec"),mod.__dict__)
    return mod.build_knn_graph,hashlib.sha256(raw).hexdigest()


def fixtures():
    for name,loader in REAL:
        x,truth=loader(return_X_y=True)
        x=np.ascontiguousarray(StandardScaler().fit_transform(x),dtype=np.float64)
        for k in K_VALUES:
            yield dict(name=name,k=k,x=x,truth=truth,synthetic=False)
    for n,d,name in ((512,8,"duplicate512"),(2048,16,"duplicate2048")):
        rng=np.random.default_rng(44001+n)
        x=rng.normal(size=(n,d))
        x[13:55]=x[0]
        x[60:110]=0.
        x[111:145]=x[3]
        x=np.ascontiguousarray(x,dtype=np.float64)
        for k in K_VALUES:
            yield dict(name=name,k=k,x=x,truth=None,synthetic=True)
    x=np.zeros((64,4),dtype=np.float64)
    for k in K_VALUES:
        yield dict(name="all_identical64",k=k,x=x,truth=None,synthetic=True)


def original_self_count(x,k):
    # One metadata-only source-equivalent neighbor query OUTSIDE timed paths.
    neighbors=NearestNeighbors(n_neighbors=k+1,algorithm="auto",
                               metric="minkowski",p=2,n_jobs=1).fit(x)
    cols=neighbors.kneighbors(x,return_distance=False)[:,1:]
    return int(np.sum(cols==np.arange(len(x),dtype=np.int64)[:,None]))


def exact_pair(a,b):
    assert len(a)==2 and len(b)==2
    for one,two in zip(a,b):
        if (one.shape!=two.shape or one.dtype!=two.dtype or
                not one.flags.c_contiguous or not two.flags.c_contiguous or
                one.dtype!=np.int64 or not one.flags.writeable or
                not two.flags.writeable or not np.array_equal(one,two)):
            raise AssertionError("original/candidate full sorted padded graph differs")
    neibs,degs=a
    if neibs.ndim!=2 or degs.ndim!=1 or neibs.shape[0]!=len(degs):
        raise AssertionError("graph layout changed")
    active=np.arange(neibs.shape[1])[None,:]<degs[:,None]
    if np.any(degs<0) or np.any(degs>neibs.shape[1]):
        raise AssertionError("invalid graph degrees")
    if np.any(neibs[~active]!=-1):
        raise AssertionError("missing original -1 padding")
    if np.any(neibs[active]<0) or np.any(neibs[active]>=len(degs)):
        raise AssertionError("invalid original graph")
    return [array_sha(neibs),array_sha(degs)]


def timed(fn,x,k):
    t=time.perf_counter_ns()
    out=fn(x,k_nn=k,n_jobs=1)
    return (time.perf_counter_ns()-t)/1e6,out


def tracked(fn,x,k):
    gc.collect()
    tracemalloc.start()
    try:
        out=fn(x,k_nn=k,n_jobs=1)
        _,peak=tracemalloc.get_traced_memory()
        return peak,out
    finally:
        tracemalloc.stop()


def measured(original,case):
    name,k,x=case["name"],case["k"],case["x"]
    src_sha=array_sha(x)
    ref=original(x,k_nn=k,n_jobs=1)
    cand=optimized_knn(x,k_nn=k,n_jobs=1)
    graph_hash=exact_pair(ref,cand)
    self_count=original_self_count(x,k)
    before=array_sha(x)
    for func in (original,optimized_knn):
        for _ in range(2):
            _,out=timed(func,x,k)
            exact_pair(ref,out)
    pairs=[]
    for i in range(25):
        row=[None,None]
        for idx in ((0,1) if i%2 else (1,0)):
            ms,out=timed((original,optimized_knn)[idx],x,k)
            exact_pair(ref,out)
            row[idx]=ms
        pairs.append(row)
    peaks=[]
    for i in range(5):
        row=[None,None]
        for idx in ((0,1) if i%2 else (1,0)):
            peak,out=tracked((original,optimized_knn)[idx],x,k)
            exact_pair(ref,out)
            row[idx]=peak
        peaks.append(row)
    if array_sha(x)!=before:
        raise AssertionError("graph builder modified source samples")
    saving=100*(1-statistics.median([b/a for a,b in pairs]))
    print(f"{name} k={k} self_edges={self_count}: full graph CPU "
          f"{saving:+.2f}%, tracemalloc original/candidate "
          f"{statistics.median(v[0] for v in peaks):.0f}/"
          f"{statistics.median(v[1] for v in peaks):.0f} B",flush=True)
    return dict(name=name,k=k,synthetic=case["synthetic"],
                nodes=len(x),features=x.shape[1],
                input_sha=src_sha,ordered_graph_sha=graph_hash,
                original_self_count=self_count,
                timings_ms=pairs,peak_bytes=peaks),ref,cand


def validate_invalid(original):
    x=np.arange(24,dtype=np.float64).reshape((12,2))
    cases=(("bad_shape",x.reshape(-1),3),
           ("zero_k",x,0),("negative_k",x,-1),
           ("equal_node_count",x,len(x)),
           ("float_k",x,3.5))
    results={}
    for name,data,k in cases:
        out=[]
        for f in (original,optimized_knn):
            try:
                f(data,k_nn=k,n_jobs=1)
            except Exception as e:
                out.append((type(e).__name__,str(e)))
            else:
                raise AssertionError("invalid original graph input unexpectedly accepted")
        if out[0]!=out[1]:
            raise AssertionError("changed public invalid input exception: "+repr(name))
        results[name]=out[0]
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
    # Crucial: old benchmark runner.measure once failed to preserve this name (#203 fixed).
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



def real_native(original, specs):
    rows=[]
    for case in specs:
        if case["synthetic"] or case["k"]!=10:
            continue
        name,x,truth=case["name"],case["x"],case["truth"]
        ref=original(x,k_nn=10,n_jobs=1)
        new=optimized_knn(x,k_nn=10,n_jobs=1)
        exact_pair(ref,new)
        for label_seed in LABEL_SEEDS:
            rng=np.random.default_rng(label_seed)
            labels=np.full(len(truth),-1,dtype=np.int64)
            for cls in np.unique(truth):
                homes=rng.choice(np.flatnonzero(truth==cls),size=3,replace=False)
                labels[homes]=int(cls)
            label_hash=array_sha(labels)
            for backend in BACKENDS:
                for stop in (False,True):
                    pair=[]
                    for graph in (ref,new):
                        pair.append(fit_with_metrics(
                            Candidate,graph[0],graph[1],labels,backend,stop,42059))
                    a,b=pair
                    for key in ("state_hashes","info","backend_identity","layout","rng_sha"):
                        if a[key]!=b[key]:
                            raise AssertionError("six-state native difference "
                                                 +repr((name,label_seed,backend,stop,key)))
                    rows.append(dict(dataset=name,label_seed=label_seed,
                                     backend=backend,early_stop=stop,
                                     label_sha=label_hash,
                                     original_graph_sha=[array_sha(v) for v in ref],
                                     states=a["state_hashes"],info=a["info"],
                                     backend_identity=a["backend_identity"],
                                     layout=a["layout"],rng_sha=a["rng_sha"]))
        print(f"{name}: 12 exact native fitted graph pairs",flush=True)
    return rows


def main():
    original,source_sha=original_source()
    specs=list(fixtures())
    samples=[]
    for case in specs:
        record,_,_=measured(original,case)
        samples.append(record)
    errors=validate_invalid(original)
    native=real_native(original,specs)
    out=dict(schema=1,base_commit=BASE,
             head_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             platform=sys.platform,os=platform.platform(),cpu=platform.processor(),
             python=sys.version,numpy=np.__version__,
             scipy=__import__("scipy").__version__,
             sklearn=__import__("sklearn").__version__,
             git_original_graph_source_sha256=source_sha,
             env={v:os.getenv(v) for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                                          "MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
             samples=samples,invalid=errors,native=native)
    Path("knn_diagonal_result.json").write_text(
        json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("COMPLETE kNN conditional diagonal original-vs-candidate:",
          len(samples),"graph fixtures, 525 complete timing and 105 trace pairs,",
          len(native),"exact native fitted pairs",flush=True)


if __name__=="__main__":
    main()
