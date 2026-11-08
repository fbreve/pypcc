"""Preregistered exact graph-array equivalence and small native speed pilot.

Original graph reference is the actual frozen 2026-11-07 production Git blob,
not a hand-written approximation. Importing this script has no benchmark cost.
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
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from benchmarks.fixtures import Case, array_hash, select_labels
from benchmarks.runner import measure
from pcc_graph import build_knn_graph, build_graph_from_edge_index

MASTER="8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128"
DATASETS=("iris","wine","breast_cancer","digits")
K_VALUES=(1,3,10)
ROUNDS=30
LABEL_SEEDS=(27103,27211)
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
SEMANTIC_CAP=1000
K=10


def check(condition,message):
    if not condition:
        raise AssertionError(message)


def repo_root():
    return Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(
        ["git",*args],cwd=repo_root(),text=True).strip()


def trusted_reference():
    # This is the unchanged pinned master blob, not a manually duplicated
    # implementation or an external plugin. Requires fetch-depth:0 in CI.
    reference=git("show",f"{MASTER}:pcc_graph.py")
    check("def build_knn_graph(" in reference and
          "def build_graph_from_edge_index(" in reference,
          "Frozen source does not contain original graph builders")
    module=types.ModuleType("_pypcc_frozen_original_graph")
    exec(compile(reference,f"{MASTER}:pcc_graph.py","exec"),module.__dict__)
    return module,hashlib.sha256(reference.encode("utf-8")).hexdigest()


def verify_source_edit():
    untouched=["pcc.py","pcc_numpy.py","pcc_numba.py","pcc_step.pyx",
               "pcc_validation.py","benchmarks","setup.py"]
    dirty=git("diff","--name-only",MASTER,"HEAD","--",*untouched)
    check(not dirty, "Candidate unexpectedly changed production kernel/benchmark: "+dirty)
    source=(repo_root()/"pcc_graph.py").read_text(encoding="utf-8")
    check(source.count("np.arange(adj.indices.size, dtype=np.int64)")==2,
          "Candidate must change exactly two ordered dense scatter sites")
    check("[np.arange(d, dtype=np.int64) for d in degrees]" not in source,
          "Candidate still uses original temporary-per-vertex arrays")
    return source


def inputs(dataset):
    if dataset not in DATASETS:
        raise ValueError("Unplanned dataset")
    raw={
        "iris":datasets.load_iris,
        "wine":datasets.load_wine,
        "breast_cancer":datasets.load_breast_cancer,
        "digits":datasets.load_digits,
    }[dataset]()
    x=StandardScaler().fit_transform(
        np.asarray(raw.data,dtype=np.float64))
    truth=np.asarray(raw.target,dtype=np.int64)
    return x,truth


def exact_arrays(a,b):
    check(len(a)==len(b)==2,"Graph must return neighbor list and degrees")
    for label,old,new in zip(("neighbors","degrees"),a,b):
        check(old.shape==new.shape and old.dtype==new.dtype and
              np.array_equal(old,new),
              f"Original and candidate differ in {label}")
    return dict(neighbors=array_hash(a[0]),
                degrees=array_hash(a[1]))


def symmetrized_original(x,k):
    # Same algorithm and tie behavior as frozen original pcc_graph.py;
    # used to measure the three phases, not to define equivalence oracle.
    n=len(x)
    t=time.perf_counter_ns()
    knn=NearestNeighbors(
        n_neighbors=k+1,algorithm="auto",metric="minkowski",
        p=2,n_jobs=1).fit(x)
    indices=knn.kneighbors(x,return_distance=False)[:,1:]
    t1=time.perf_counter_ns()
    rows=np.repeat(np.arange(n,dtype=np.int64),k)
    cols=indices.ravel().astype(np.int64)
    ones=np.ones(len(rows),dtype=np.int8)
    adj=csr_matrix((ones,(rows,cols)),shape=(n,n),dtype=np.int8)
    adj=adj+adj.T
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:]=1
    t2=time.perf_counter_ns()
    return adj,dict(neighbors_search_ns=t1-t,
                    sparse_reciprocal_ns=t2-t1)


def scatter_reference(adj,edge_index):
    n=adj.shape[0]
    degrees=np.diff(adj.indptr).astype(np.int64)
    width=int(degrees.max())
    if edge_index:
        width=max(width,1)
    result=np.full((n,width),-1,dtype=np.int64)
    rows=np.repeat(np.arange(n,dtype=np.int64),degrees)
    cols=np.concatenate([
        np.arange(int(d),dtype=np.int64) for d in degrees
    ]) if n>0 else np.empty(0,dtype=np.int64)
    if not edge_index or len(rows):
        result[rows,cols]=adj.indices.astype(np.int64)
    return result,degrees


def scatter_candidate(adj,edge_index):
    n=adj.shape[0]
    degrees=np.diff(adj.indptr).astype(np.int64)
    width=int(degrees.max())
    if edge_index:
        width=max(width,1)
    result=np.full((n,width),-1,dtype=np.int64)
    rows=np.repeat(np.arange(n,dtype=np.int64),degrees)
    cols=(np.arange(adj.indices.size,dtype=np.int64)-
          np.repeat(adj.indptr[:-1],degrees))
    if not edge_index or len(rows):
        result[rows,cols]=adj.indices.astype(np.int64)
    return result,degrees


def time_pair(first,second):
    t=time.perf_counter_ns()
    a=first()
    t1=time.perf_counter_ns()
    b=second()
    t2=time.perf_counter_ns()
    return a,b,t1-t,t2-t1


def paired_builds(reference):
    rows=[]
    phases=[]
    graphs={}
    for name in DATASETS:
        x,truth=inputs(name)
        for k in K_VALUES:
            ref_func=lambda: reference.build_knn_graph(
                x,k_nn=k,n_jobs=1)
            candidate_func=lambda: build_knn_graph(
                x,k_nn=k,n_jobs=1)
            exact_arrays(ref_func(),candidate_func())
            condition_rows=[]
            for rep in range(ROUNDS):
                first,second=(ref_func,candidate_func) if rep%2==0 else (
                    candidate_func,ref_func)
                a,b,tfirst,tsecond=time_pair(first,second)
                fingerprint=exact_arrays(a,b)
                control,trial=(tfirst,tsecond) if rep%2==0 else (
                    tsecond,tfirst)
                condition_rows.append(dict(
                    dataset=name,k=k,repeat=rep,
                    reference_ns=control,candidate_ns=trial,
                    hashes=fingerprint,first="reference" if rep%2==0 else "candidate"))
            check(len({json.dumps(r["hashes"],sort_keys=True)
                       for r in condition_rows})==1,
                  "Graph hashes changed across matched repeats")
            rows.extend(condition_rows)
            if k==K:
                graphs[name]=(x,truth,ref_func(),candidate_func())
                # Three profiling phases: warm up, then independently
                # measure actual original sklearn+CSR build and both
                # independent dense scatter methods on the same CSR.
                adj,_=symmetrized_original(x,k)
                exact_arrays(scatter_reference(adj,False),
                             scatter_candidate(adj,False))
                for rep in range(ROUNDS):
                    adj2,times=symmetrized_original(x,k)
                    check(np.array_equal(adj.indptr,adj2.indptr)
                          and np.array_equal(adj.indices,adj2.indices),
                          "Original CSR changed between phase repeats")
                    first,second=(scatter_reference,scatter_candidate) if rep%2==0 else (
                        scatter_candidate,scatter_reference)
                    a,b,ta,tb=time_pair(
                        lambda:first(adj,False),lambda:second(adj,False))
                    exact_arrays(a,b)
                    t0,t1=(ta,tb) if rep%2==0 else (tb,ta)
                    phases.append(dict(
                        dataset=name,k=k,repeat=rep,
                        **times,
                        reference_scatter_ns=t0,
                        candidate_scatter_ns=t1,
                        hashes=dict(neighbors=array_hash(a[0]),
                                    degrees=array_hash(a[1]))))
        print(f"graph {name} all {len(K_VALUES)*ROUNDS} exact pairs PASS",
              flush=True)
    return rows,phases,graphs


def semantic_pairs(graphs):
    # Fresh model and original backend every time; independent backend RNGs.
    # Existing runner verifies that the requested native extension is actual.
    checks=[]
    for name in DATASETS:
        x,truth,ref_graph,trial_graph=graphs[name]
        c=int(np.unique(truth).size)
        for seed in LABEL_SEEDS:
            case=Case(name,len(x),x.shape[1],c,k_nn=K,
                      labels_per_class=3,data_seed=0,label_seed=seed)
            labels=select_labels(truth,case)
            for backend in BACKENDS:
                for mode in MODES:
                    early_stop=mode=="early_stopping"
                    arr_a=dict(neighbors=ref_graph[0],degrees=ref_graph[1],
                               truth=truth,labels=labels)
                    arr_b=dict(neighbors=trial_graph[0],degrees=trial_graph[1],
                               truth=truth,labels=labels)
                    exact_arrays(ref_graph,trial_graph)
                    # Keep the original production benchmarking logic:
                    # Numba compiles on the short warm-up, never in a timed fit.
                    if backend=="numba":
                        measure(arr_a,case,backend,2,False,0,
                                steady_state=False)
                    a=measure(arr_a,case,backend,SEMANTIC_CAP,
                              early_stop,0,steady_state=True)
                    b=measure(arr_b,case,backend,SEMANTIC_CAP,
                              early_stop,0,steady_state=True)
                    fields=("state_hashes","effective_iterations",
                            "stop_reason","checks","stalled_checks",
                            "stop_max","accuracy_unlabeled")
                    check(all(a.get(f)==b.get(f) for f in fields),
                          f"Native trajectory differs: {name}/{seed}/{backend}/{mode}")
                    check(len(a["state_hashes"])==6,
                          "Original benchmark did not capture six final state hashes")
                    checks.append(dict(
                        dataset=name,label_seed=seed,backend=backend,mode=mode,
                        iterations_cap=SEMANTIC_CAP,
                        reference_effective_iterations=a["effective_iterations"],
                        candidate_effective_iterations=b["effective_iterations"],
                        reference_stop_reason=a["stop_reason"],
                        candidate_stop_reason=b["stop_reason"],
                        reference_heldout_accuracy=a["accuracy_unlabeled"],
                        candidate_heldout_accuracy=b["accuracy_unlabeled"],
                        state_hashes=a["state_hashes"],
                        native_info_exact=True,
                        six_state_hashes_exact=True,
                        reference_backend_identity=a["backend_identity"],
                        candidate_backend_identity=b["backend_identity"]))
        print(f"native {name}: all {len(LABEL_SEEDS)*len(BACKENDS)*len(MODES)} "
              "reference/candidate exact state pairs PASS",flush=True)
    check(len(checks)==4*2*3*2,"Incomplete preregistered native 48-pair audit")
    return checks


def edge_index_cases(ref):
    cases=[
        (0,np.empty((2,0),dtype=np.int64)),
        (5,np.empty((2,0),dtype=np.int64)),
        (8,np.array([[1,1,2,2,2,2,4,4,7,6],
                     [2,2,1,2,2,4,4,4,6,7]],dtype=np.int64)),
        (5,np.array([[0,0,1,2,3,3],[1,1,0,3,2,3]],dtype=np.int64)),
        (9,np.array([[3,3,3,3,3,3,3,5,7],
                     [2,2,4,4,5,7,8,3,3]],dtype=np.int64)),
    ]
    res=[]
    for i,(n,edges) in enumerate(cases):
        orig=ref.build_graph_from_edge_index(n,edges)
        cand=build_graph_from_edge_index(n,edges)
        hashes=exact_arrays(orig,cand)
        res.append(dict(case=i,num_nodes=n,
                        n_source_edges=int(edges.shape[1]),hashes=hashes,
                        neighbor_shape=list(orig[0].shape),
                        degree_shape=list(orig[1].shape)))
    return res


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    verify_source_edit()
    reference,reference_hash=trusted_reference()
    graph,phases,stored=paired_builds(reference)
    native=semantic_pairs(stored)
    edges=edge_index_cases(reference)
    report=dict(
        protocol="graph_dense_layout_v1",master=MASTER,
        head=git("rev-parse","HEAD"),original_graph_sha256=reference_hash,
        host=dict(
            os=platform.platform(),python=sys.version,
            cpu=platform.processor(),processor=platform.uname().processor,
            thread_env={key:os.environ.get(key)
                        for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                                    "MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
            numpy=np.__version__),
        datasets=list(DATASETS),k_values=list(K_VALUES),rounds=ROUNDS,
        label_seeds=list(LABEL_SEEDS),backends=list(BACKENDS),
        modes=list(MODES),native_cap=SEMANTIC_CAP,
        graph_pairs=graph,phase_rows=phases,
        semantic_pairs=native,edge_index_cases=edges,
        completeness=dict(graph_pairs=len(graph),
                          phase_rows=len(phases),
                          semantic_pairs=len(native),
                          edge_index_cases=len(edges),
                          all_equal=True))
    check(len(graph)==len(DATASETS)*len(K_VALUES)*ROUNDS
          and len(phases)==len(DATASETS)*ROUNDS
          and len(native)==len(DATASETS)*len(LABEL_SEEDS)*
                 len(BACKENDS)*len(MODES),
          "Incomplete exact graph or native semantic experiment")
    args.output.write_text(json.dumps(
        report,indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8")
    print(f"PASS exact graphs={len(graph)} + scatter phases={len(phases)} "
          f"+ 48 original native trajectories and {len(edges)} edge cases",
          flush=True)


if __name__=="__main__":
    main()
