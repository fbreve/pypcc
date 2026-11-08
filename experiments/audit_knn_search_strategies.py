"""Isolated sklearn search-engine audit; original PCC graph builder unchanged.

The original master pcc_graph.py Git blob is compiled in a private module.
Three alternatives replace *only* the literal sklearn algorithm argument.
All graph and timing outcomes, including incompatibilities, are retained.
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
import types

import numpy as np
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from benchmarks.fixtures import Case, array_hash, select_labels
from benchmarks.runner import measure

MASTER="8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128"
DATASETS=("iris","wine","breast_cancer","digits")
K_VALUES=(1,3,10)
METHODS=("auto","brute","kd_tree","ball_tree")
ALTERNATIVES=METHODS[1:]
REPETITIONS=20
PHASE_REPETITIONS=10
SEMANTIC_SEED=28109
SEMANTIC_CAP=1000
SEMANTIC_BACKENDS=("numba","cython")
SYNTHETIC_NAMES=("duplicate_pairs","rectangular_grid","fortran_noncontiguous",
                 "constant_ties","equidistant_chain")


def guard(condition,message):
    if not condition:
        raise AssertionError(message)


def repo_root():
    return Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(
        ["git",*args],cwd=repo_root(),text=True).strip()


def load_original_engines():
    untouched=("pcc.py","pcc_numpy.py","pcc_numba.py","pcc_step.pyx",
               "pcc_graph.py","pcc_validation.py","benchmarks","setup.py")
    changed=git("diff","--name-only",MASTER,"HEAD","--",*untouched)
    guard(not changed,"Original PCC or benchmark code changed: "+changed)
    source=git("show",f"{MASTER}:pcc_graph.py")
    literal='algorithm="auto"'
    guard(source.count(literal)==1,
          "Original source must contain exactly one search algorithm literal")
    modules={}
    source_hashes={}
    for method in METHODS:
        rewritten=source if method=="auto" else source.replace(
            literal,f'algorithm="{method}"')
        mod=types.ModuleType(f"_original_graph_knn_{method}")
        exec(compile(rewritten,
                     f"{MASTER}:pcc_graph.py[{method}]","exec"),
             mod.__dict__)
        modules[method]=mod
        source_hashes[method]=hashlib.sha256(
            rewritten.encode("utf-8")).hexdigest()
    return modules,source_hashes


def inputs(name):
    loader={
        "iris":datasets.load_iris,"wine":datasets.load_wine,
        "breast_cancer":datasets.load_breast_cancer,
        "digits":datasets.load_digits,
    }[name]
    raw=loader()
    x=StandardScaler().fit_transform(
        np.asarray(raw.data,dtype=np.float64))
    truth=np.asarray(raw.target,dtype=np.int64)
    return x,truth


def graph_hash(g):
    return {"neighbors":array_hash(g[0]),"degrees":array_hash(g[1])}


def edge_set(g):
    arr,deg=g
    return {(min(i,int(j)),max(i,int(j)))
            for i in range(len(deg))
            for j in arr[i,:int(deg[i])] if i!=j}


def difference(reference,candidate):
    a,da=reference
    b,db=candidate
    exact=(a.shape==b.shape and a.dtype==b.dtype and
           da.shape==db.shape and da.dtype==db.dtype and
           np.array_equal(a,b) and np.array_equal(da,db))
    missing=edge_set(reference)-edge_set(candidate)
    extra=edge_set(candidate)-edge_set(reference)
    n=max(len(da),len(db))
    if a.shape!=b.shape:
        diff_rows=n
    else:
        diff_rows=int(np.count_nonzero(np.any(a!=b,axis=1) | (da!=db)))
    return dict(exact=bool(exact),different_rows=diff_rows,
                missing_edges=len(missing),extra_edges=len(extra),
                reference=graph_hash(reference),
                candidate=graph_hash(candidate))


def timed_pair(reference,candidate,rep):
    if rep%2==0:
        first,second=reference,candidate
    else:
        first,second=candidate,reference
    start=time.perf_counter_ns()
    a=first()
    middle=time.perf_counter_ns()
    b=second()
    end=time.perf_counter_ns()
    return ((a,b,middle-start,end-middle) if rep%2==0 else
            (b,a,end-middle,middle-start))


def engine_raw_order(x,k,method):
    m=NearestNeighbors(n_neighbors=k+1,algorithm=method,
                       metric="minkowski",p=2,n_jobs=1).fit(x)
    indices=m.kneighbors(x,return_distance=False)[:,1:]
    return m._fit_method,indices


def search_phases(x,k,method):
    rows=[]
    for rep in range(PHASE_REPETITIONS):
        t0=time.perf_counter_ns()
        m=NearestNeighbors(n_neighbors=k+1,algorithm=method,
                           metric="minkowski",p=2,n_jobs=1).fit(x)
        t1=time.perf_counter_ns()
        idx=m.kneighbors(x,return_distance=False)[:,1:]
        t2=time.perf_counter_ns()
        guard(idx.shape==(len(x),k),"Unexpected sklearn raw query shape")
        rows.append(dict(repeat=rep,selected_method=m._fit_method,
                         fit_ns=t1-t0,query_ns=t2-t1))
    return rows


def synthetic_inputs():
    grid=np.array([[i,j] for i in range(3) for j in range(4)],
                  dtype=np.float64)
    f=np.asfortranarray(
        np.arange(12*4,dtype=np.float64).reshape(12,4))
    duplicates=np.array(
        [[0,0],[0,0],[0,0],[1,0],[1,0],[0,1],
         [0,1],[1,1],[1,1],[2,0],[0,2],[2,2]],dtype=float)
    return {
        "duplicate_pairs":duplicates,
        "rectangular_grid":grid,
        "fortran_noncontiguous":f,
        "constant_ties":np.zeros((12,3),dtype=float),
        "equidistant_chain":np.arange(12,dtype=float).reshape(-1,1),
    }


def real_conditions(modules):
    data_rows=[]
    graph_rows=[]
    phase_rows=[]
    caches={}
    for dataset in DATASETS:
        x,truth=inputs(dataset)
        caches[dataset]=(x,truth)
        xhash=array_hash(x)
        for k in K_VALUES:
            builders={
                method:(lambda method=method:modules[method].build_knn_graph(
                    x,k_nn=k,n_jobs=1))
                for method in METHODS}
            original=builders["auto"]()
            orig_hash=graph_hash(original)
            actual_methods={}
            raw_orders={}
            for method in METHODS:
                chosen,idx=engine_raw_order(x,k,method)
                actual_methods[method]=chosen
                raw_orders[method]=idx
                phase_rows.extend(
                    dict(dataset=dataset,k=k,method=method,**row)
                    for row in search_phases(x,k,method))
            guard(
                all(row["selected_method"]==actual_methods[row["method"]]
                    for row in phase_rows if
                    row["dataset"]==dataset and row["k"]==k),
                "Selected sklearn search engine changed within case")
            for candidate in ALTERNATIVES:
                first=builders[candidate]()
                check=difference(original,first)
                raw_differ=int(np.count_nonzero(np.any(
                    raw_orders["auto"]!=raw_orders[candidate],axis=1)))
                data_rows.append(dict(
                    dataset=dataset,k=k,method=candidate,
                    original_method=actual_methods["auto"],
                    candidate_method=actual_methods[candidate],
                    input_sha256=xhash,
                    raw_neighbor_different_rows=raw_differ,
                    **check))
                for rep in range(REPETITIONS):
                    a,b,ref_ns,cand_ns=timed_pair(
                        builders["auto"],builders[candidate],rep)
                    check_again=difference(a,b)
                    guard(graph_hash(a)==orig_hash and
                          check_again==check,
                          f"Original graph or candidate graph was nondeterministic: "
                          f"{dataset}/{k}/{candidate}/{rep}")
                    graph_rows.append(dict(
                        dataset=dataset,k=k,method=candidate,
                        repeat=rep,
                        first="auto" if rep%2==0 else candidate,
                        reference_ns=ref_ns,candidate_ns=cand_ns,
                        output_exact=check["exact"],
                        reference_sha256=check["reference"],
                        candidate_sha256=check["candidate"]))
            print(f"real {dataset} k={k}: auto resolved to "
                  f"{actual_methods['auto']}, exact compared with all 3 methods "
                  f"for {REPETITIONS} pairs",flush=True)
    return data_rows,graph_rows,phase_rows,caches


def synthetic_conditions(modules):
    rows=[]
    for name,x in synthetic_inputs().items():
        guard(name in SYNTHETIC_NAMES and len(x)==12,
              "Synthetic fixture unexpectedly changed")
        for k in K_VALUES:
            for p in (1,2):
                original=modules["auto"].build_knn_graph(
                    x,k_nn=k,p=p,n_jobs=1)
                for method in ALTERNATIVES:
                    result=modules[method].build_knn_graph(
                        x,k_nn=k,p=p,n_jobs=1)
                    rows.append(dict(
                        name=name,k=k,p=p,method=method,
                        input_sha256=array_hash(x),
                        **difference(original,result)))
    guard(len(rows)==len(SYNTHETIC_NAMES)*len(K_VALUES)*2*
          len(ALTERNATIVES),"Synthetic case count changed")
    return rows


def native_conditions(modules,caches,real_data):
    lookup={(r["dataset"],r["k"],r["method"]):r
            for r in real_data}
    native=[]
    for dataset in DATASETS:
        x,truth=caches[dataset]
        original=modules["auto"].build_knn_graph(
            x,k_nn=10,n_jobs=1)
        case=Case(dataset,len(x),x.shape[1],int(len(np.unique(truth))),
                  k_nn=10,labels_per_class=3,
                  data_seed=0,label_seed=SEMANTIC_SEED)
        labels=select_labels(truth,case)
        for method in ALTERNATIVES:
            key=(dataset,10,method)
            valid=lookup[key]["exact"]
            if not valid:
                for backend in SEMANTIC_BACKENDS:
                    native.append(dict(
                        dataset=dataset,k=10,method=method,backend=backend,
                        status="skipped_graph_mismatch",
                        graph_exact=False,seed=SEMANTIC_SEED))
                continue
            candidate=modules[method].build_knn_graph(
                x,k_nn=10,n_jobs=1)
            guard(difference(original,candidate)["exact"],
                  f"Graph changed before native test {key}")
            for backend in SEMANTIC_BACKENDS:
                args=dict(truth=truth,labels=labels)
                a=measure(dict(neighbors=original[0],degrees=original[1],
                               **args),case,backend,SEMANTIC_CAP,
                          False,0,steady_state=True)
                b=measure(dict(neighbors=candidate[0],degrees=candidate[1],
                               **args),case,backend,SEMANTIC_CAP,
                          False,0,steady_state=True)
                compared=("state_hashes","effective_iterations",
                          "stop_reason","accuracy_unlabeled","checks",
                          "stalled_checks","stop_max")
                guard(all(a.get(field)==b.get(field) for field in compared)
                      and len(a["state_hashes"])==6
                      and a["effective_iterations"]==SEMANTIC_CAP,
                      f"Native state mismatch: {dataset}/{method}/{backend}")
                native.append(dict(
                    dataset=dataset,k=10,method=method,
                    backend=backend,status="exact",
                    graph_exact=True,seed=SEMANTIC_SEED,
                    state_hashes=a["state_hashes"],
                    effective_iterations=a["effective_iterations"],
                    stop_reason=a["stop_reason"],
                    accuracy_unlabeled=a["accuracy_unlabeled"],
                    backend_identity=a["backend_identity"]))
        print(f"native {dataset}: all eligible 1000-step original "
              "Numba/Cython cases matched",flush=True)
    guard(len(native)==len(DATASETS)*len(ALTERNATIVES)*
          len(SEMANTIC_BACKENDS),"Incomplete native safety checks")
    return native


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    mods,sources=load_original_engines()
    inputs_rows,graphs,phases,cache=real_conditions(mods)
    synthetic=synthetic_conditions(mods)
    native=native_conditions(mods,cache,inputs_rows)
    guard(len(inputs_rows)==36 and len(graphs)==720 and
          len(phases)==480 and len(synthetic)==90 and
          len(native)==24,"Incomplete frozen search-engine experiment")
    report=dict(
        protocol="knn_search_strategy_v1",master=MASTER,
        head=git("rev-parse","HEAD"),source_sha256=sources,
        datasets=list(DATASETS),k_values=list(K_VALUES),
        algorithms=list(METHODS),repetitions=REPETITIONS,
        phase_repetitions=PHASE_REPETITIONS,
        semantic_seed=SEMANTIC_SEED,semantic_cap=SEMANTIC_CAP,
        semantic_backends=list(SEMANTIC_BACKENDS),
        synthetic_fixtures=list(SYNTHETIC_NAMES),
        host=dict(os=platform.platform(),python=sys.version,
                  cpu=platform.processor(),
                  numpy=np.__version__,
                  env={key:os.getenv(key) for key in
                       ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                        "MKL_NUM_THREADS","NUMBA_NUM_THREADS")}),
        real_cases=inputs_rows,full_graph_pairs=graphs,
        search_phase_rows=phases,synthetic_cases=synthetic,
        native_cases=native)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(
        report,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print("COMPLETE: 720 full graph pairs, 480 engine phase samples, "
          "90 tie/duplicate safety comparisons, 24 native semantics "
          "(exact or explicit invalid-graph skip)",flush=True)


if __name__=="__main__":
    main()
