"""Source-pinned observational stages of current knn graph + public installation APIs."""
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
import types

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from pcc import ParticleCompetitionAndCooperation as PCC

BASE = "c1073ae53ea785031de9e474b678e94f44a2124d"
SAMPLE_PAIRS = 20
INSTALL_REPS = 10
BACKENDS = ("numpy", "numba", "cython")
REAL = (("iris", load_iris), ("wine", load_wine),
        ("breast_cancer", load_breast_cancer), ("digits", load_digits))
STAGE_NAMES = ("validation", "nearest_fit", "kneighbors", "directed_csr",
               "symmetrize_csr", "dense_materialization")
PROTECTED = ("pcc.py", "pcc_graph.py", "pcc_numpy.py", "pcc_numba.py",
             "pcc_step.pyx", "pcc_validation.py", "setup.py", "benchmarks")


def digest(value):
    arr = np.asarray(value)
    return hashlib.sha256(
        json.dumps([arr.shape, arr.dtype.str]).encode()
        + np.ascontiguousarray(arr).tobytes()
    ).hexdigest()


def guarded_modules():
    # The actual untouched source is the only basis for BOTH tested variants.
    raw = subprocess.check_output(["git", "show", BASE+":pcc_graph.py"])
    original = raw.decode("utf-8").replace("\r\n", "\n")
    current = Path("pcc_graph.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    if original != current:
        raise AssertionError("production graph source diverged from frozen Git blob")
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD", "--", *PROTECTED],
        text=True,
    )
    if diff.strip():
        raise AssertionError("protected production source or benchmark modified: "+diff)
    stamps = (
        ("    data = np.asarray(data)\n",
         "    _g0 = time.perf_counter_ns()\n"),
        ("    nbrs = NearestNeighbors(\n",
         "    _g1 = time.perf_counter_ns()\n"),
        ("    # indices[i, 0] is the query node itself; drop it\n",
         "    _g2 = time.perf_counter_ns()\n"),
        ("    # --- vectorized symmetrization via sparse matrix ---\n",
         "    _g3 = time.perf_counter_ns()\n"),
        ("    adj = adj + adj.T                       # symmetric\n",
         "    _g4 = time.perf_counter_ns()\n"),
        ("    # Convert back to dense neighbor lists\n",
         "    _g5 = time.perf_counter_ns()\n"),
        ("    return neib_list, neib_qt\n",
         """    _g6 = time.perf_counter_ns()
    _stage_sink.append({
        "parts_ns": [_g1-_g0, _g2-_g1, _g3-_g2,
                     _g4-_g3, _g5-_g4, _g6-_g5],
        "internal_ns": _g6-_g0,
        "nearest_method": str(getattr(nbrs, "_fit_method", "unknown")),
    })
"""),
    )
    # Guard that every insertion is within the *first* actual knn builder.
    front, back = original.split("\ndef build_graph_from_edge_index", 1)
    observed = front
    for anchor, insert in stamps:
        if observed.count(anchor) != 1:
            raise AssertionError("expected one graph phase anchor: "+repr(anchor))
        observed = observed.replace(anchor, insert+anchor, 1)
    observed += "\ndef build_graph_from_edge_index"+back
    sink=[]
    source_modules=[]
    for name, code in (("base_graph_source", original),
                       ("stamped_graph_source", observed)):
        mod=types.ModuleType(name)
        mod.__dict__.update(time=time, _stage_sink=sink)
        exec(compile(code, BASE+":pcc_graph.py/"+name, "exec"), mod.__dict__)
        source_modules.append(mod)
    return source_modules, sink, hashlib.sha256(raw).hexdigest()


def fixtures():
    for name, loader in REAL:
        x, truth = loader(return_X_y=True)
        x = StandardScaler().fit_transform(x)
        x = np.ascontiguousarray(x, dtype=np.float64)
        for k in (3,10):
            yield dict(name=name,k=k,x=x,truth=truth,adversarial=False)
    for n,d,name in ((512,8,"duplicate512"),(2048,16,"duplicate2048")):
        rng=np.random.default_rng(44001+n)
        x=rng.normal(size=(n,d))
        x[13:55]=x[0]
        x[60:110]=0.0
        x[111:145]=x[3]
        # Fully tied duplicate rows: do not change the historical sklearn
        # assumption that the first nearest index is the query itself.
        x=np.ascontiguousarray(x,dtype=np.float64)
        for k in (3,10):
            yield dict(name=name,k=k,x=x,truth=None,adversarial=True)


def graph_equal(a,b):
    if len(a)!=2 or len(b)!=2:
        raise AssertionError("graph output arity changed")
    if (a[0].dtype!=np.int64 or a[1].dtype!=np.int64
            or a[0].ndim!=2 or a[1].ndim!=1
            or not a[0].flags.c_contiguous or not a[1].flags.c_contiguous):
        raise AssertionError("dense public int64 output layout changed")
    if (a[0].shape!=b[0].shape or a[1].shape!=b[1].shape or
            not np.array_equal(a[0],b[0]) or
            not np.array_equal(a[1],b[1])):
        raise AssertionError("original vs observed graph arrays differ")
    if len(a[1])!=len(a[0]):
        raise AssertionError("invalid graph length")
    if np.any(a[1]<0) or np.any(a[1]>a[0].shape[1]):
        raise AssertionError("graph contains invalid degrees")
    active = np.arange(a[0].shape[1])[None,:] < a[1][:,None]
    if np.any(a[0][~active]!=-1):
        raise AssertionError("graph padding changed")
    if np.any(a[0][active]<0) or np.any(a[0][active]>=len(a[1])):
        raise AssertionError("out-of-range active neighbors")
    return (digest(a[0]), digest(a[1]))


def elapsed(call):
    start=time.perf_counter_ns()
    result=call()
    return (time.perf_counter_ns()-start)/1e6, result


def stage_case(spec, modules, sink):
    name,k,x,truth,bad = (spec[a] for a in ("name","k","x","truth","adversarial"))
    original=modules[0].build_knn_graph
    observed=modules[1].build_knn_graph
    # Both public functions are compiled from actual frozen Git code.
    ref=original(x,k_nn=k,n_jobs=1)
    seen=observed(x,k_nn=k,n_jobs=1)
    assert len(sink)==1
    sink.pop()
    digests=graph_equal(ref,seen)
    inputs=digest(x)
    vals=[]
    stages=[]
    methods=[]
    for i in range(SAMPLE_PAIRS):
        pair=[None,None]
        order=(0,1) if i%2==0 else (1,0)
        for idx in order:
            elapsed_ms, candidate=elapsed(
                lambda fn=(original,observed)[idx]: fn(x,k_nn=k,n_jobs=1))
            graph_equal(ref,candidate)
            pair[idx]=elapsed_ms
            if idx==1:
                if len(sink)!=1:
                    raise AssertionError("expected exactly one cloned stage sample")
                stage=sink.pop()
                if (len(stage["parts_ns"])!=6 or
                        min(stage["parts_ns"])<=0 or
                        sum(stage["parts_ns"])!=stage["internal_ns"]):
                    raise AssertionError("invalid source-cloned clock accounting")
                if any(value>elapsed_ms*1e6*1.01 for value in stage["parts_ns"]):
                    raise AssertionError("stage exceeds observed function elapsed")
                stages.append(stage)
                methods.append(stage["nearest_method"])
        vals.append(pair)
    if len(vals)!=20 or len(stages)!=20 or len(set(methods))!=1:
        raise AssertionError("unbalanced source-paired samples")
    # Installation API 1: caller has an existing graph, set_graph *copies*
    # the dense neighbor matrix, and changes unused -1 padding to zero.
    q=ref[1]
    neibs=ref[0]
    mask=np.arange(neibs.shape[1])[None,:]<q[:,None]
    expected=np.zeros_like(neibs)
    expected[mask]=neibs[mask]
    previous_hash=(digest(neibs),digest(q))
    def install():
        m=PCC(impl="numpy")  # constructor intentionally OUTSIDE timer
        elapsed_ms,_=elapsed(lambda:m.set_graph(neibs,q,k_nn=k))
        return elapsed_ms,m
    def check_install(m):
        if (not np.array_equal(m.neib_list,expected)
                or not np.array_equal(m.neib_qt,q)
                or m.neib_list.dtype!=np.int64
                or m.neib_qt.dtype!=np.int64
                or not m.neib_list.flags.c_contiguous
                or np.shares_memory(m.neib_list,neibs)
                or np.shares_memory(m.neib_qt,q)):
            raise AssertionError("set_graph active neighbors, copy, or padding changed")
    # Installation API 2: build_graph does NOT call set_graph; it directly
    # installs the original neighbor-array output and reuses its data object.
    def public_build():
        m=PCC(impl="numpy")  # constructor intentionally OUTSIDE timer
        elapsed_ms,_=elapsed(lambda:m.build_graph(x,k_nn=k,n_jobs=1))
        return elapsed_ms,m
    def check_build(m):
        if not np.array_equal(m.neib_list,neibs) or not np.array_equal(m.neib_qt,q):
            raise AssertionError("public build_graph returned a different graph")
        if m.data is not x or m.k_nn!=k:
            raise AssertionError("public build_graph metadata changed")
    _,m=install(); check_install(m)
    _,m=public_build(); check_build(m)
    set_times=[]; build_times=[]
    for i in range(INSTALL_REPS):
        if i%2:
            a,ma=install(); check_install(ma)
            b,mb=public_build(); check_build(mb)
        else:
            b,mb=public_build(); check_build(mb)
            a,ma=install(); check_install(ma)
        set_times.append(a)
        build_times.append(b)
    if previous_hash != (digest(neibs),digest(q)):
        raise AssertionError("set_graph mutated graph inputs")
    saved=dict(name=name,k=k,adversarial=bad,input_sha=inputs,
               graph_sha=list(digests),nodes=len(x),features=x.shape[1],
               edges=int(q.sum()),width=neibs.shape[1],method=methods[0],
               original_instrumented_pairs_ms=vals,
               stamped_stages_ns=stages,set_graph_ms=set_times,
               public_build_graph_ms=build_times)
    summary=[statistics.median([v["parts_ns"][j]/1e6 for v in stages])
             for j in range(6)]
    print(f"{name} k={k} method={methods[0]}: "
          f"fit/query/CSR/direct+sym/dense phase medians ms "
          f"{[round(s,3) for s in summary]}, "
          f"source original={statistics.median(v[0] for v in vals):.3f} ms, "
          f"set_graph={statistics.median(set_times):.3f} ms, "
          f"build_graph={statistics.median(build_times):.3f} ms",flush=True)
    return saved,ref,seen


def native_fit(neighbors,degrees,truth,backend):
    rng=np.random.default_rng(41021)
    labels=np.full(len(truth),-1,dtype=np.int64)
    for c in np.unique(truth):
        homes=rng.choice(np.flatnonzero(truth==c),size=3,replace=False)
        labels[homes]=int(c)
    m=PCC(impl=backend)
    m.set_graph(neighbors,degrees,k_nn=10)
    real=m._get_backend_fn()
    expected={"numpy":"pcc_numpy.pcc_propagate_numpy",
              "numba":"pcc_numba.pcc_propagate_numba",
              "cython":"pcc_step.pcc_propagate"}[backend]
    identity=real.__module__+"."+real.__name__
    if identity!=expected:
        raise AssertionError("native backend fallback "+identity)
    info={}
    def proxy(*args,**kwargs):
        kwargs["return_info"]=True
        info.update(real(*args,**kwargs))
        return None
    proxy.__name__=real.__name__  # preserve native C-order Numba >=8 classes
    m._get_backend_fn=lambda:proxy
    pred=m.fit_predict(labels,max_iter=100,early_stop=False,es_chk=10,seed=41029)
    layout=("C" if m.node.dominance.flags.c_contiguous else "F")
    if backend=="numba" and len(np.unique(truth))>=8 and layout!="C":
        raise AssertionError("Numba dominance must retain native C-order")
    if (info.get("effective_iterations")!=100 or
            info.get("stop_reason")!="max_iter"):
        raise AssertionError("incorrect native 100-iteration semantics")
    arrays={"predictions":pred,"dominance":m.node.dominance,
            "positions":m.part.curnode,"strengths":m.part.strength,
            "distances":m.part.dist_table,"owndeg":m.owndeg}
    return dict(backend_identity=identity,layout=layout,
                info=info,state_hashes={key:digest(val) for key,val in arrays.items()},
                labels_sha=digest(labels))


def main():
    mods,sink,source_sha=guarded_modules()
    records=[]
    native=[]
    for case in fixtures():
        record,original,observed=stage_case(case,mods,sink)
        records.append(record)
        if not case["adversarial"] and case["k"]==10:
            for backend in BACKENDS:
                a=native_fit(*original,case["truth"],backend)
                b=native_fit(*observed,case["truth"],backend)
                if a!=b:
                    raise AssertionError("native original/stamped six-array mismatch "
                                         +repr((case["name"],backend)))
                native.append(dict(dataset=case["name"],backend=backend,**a))
            print(f"{case['name']}: 3 exact same-backend native graph parity pairs",
                  flush=True)
    if sink:
        raise AssertionError("orphan instrumented stage data")
    result=dict(schema=1,base_commit=BASE,
                head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                original_source_sha256=source_sha,
                platform=sys.platform,system=platform.platform(),
                cpu=platform.processor(),python=sys.version,numpy=np.__version__,
                scipy=__import__("scipy").__version__,
                sklearn=__import__("sklearn").__version__,
                env={var:os.getenv(var) for var in
                     ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                      "MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
                stage_names=list(STAGE_NAMES),samples=records,native=native)
    path=Path("graph_phase_result.json")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")
    print("PASS complete graph stages:",len(records),"fixtures,",
          SAMPLE_PAIRS*len(records),"exact original/instrumented pairs,",
          INSTALL_REPS*len(records),"set_graph and build_graph calls each,",
          len(native),"native parity pairs",flush=True)


if __name__=="__main__":
    main()
