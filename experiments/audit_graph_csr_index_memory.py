"""Exact original-master graph comparison with paired memory + native controls.

No changes to the model, kernels, RNG, library or output graph contract.
The actual reference is compiled verbatim from the frozen original Git blob.
"""
import argparse
import gc
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
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from benchmarks.fixtures import Case,array_hash,select_labels
from benchmarks.runner import measure
from pcc_graph import build_knn_graph,build_graph_from_edge_index

MASTER="5a8cab9978cf367b8e02341ac856f297b7f2c2c5"
DATASETS=("iris","wine","breast_cancer","digits")
KS=(1,3,10)
SYNTH=(
    "empty_10000","isolated_30000_1000","sparse_10000_1000",
    "medium_1000_3000","dense_1000_30000","dense_10000_100000",
    "dense_30000_100000","loops_1000_3000","duplicates_1000_3000",
)
GRAPH_REPS=20
MEMORY_REPS=5
LABEL_SEEDS=(28639,28711)
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
ITERATIONS=1000


def check(ok,message):
    if not ok:
        raise AssertionError(message)


def root():
    return Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(
        ["git",*args],cwd=root(),text=True).strip()


def original_reference():
    original=subprocess.check_output(
        ["git","show",f"{MASTER}:pcc_graph.py"],cwd=root(),text=True)
    actual=(root()/"pcc_graph.py").read_text(encoding="utf-8")
    changes=(
        ("np.arange(n_nodes, dtype=np.int64), degrees)",
         "np.arange(n_nodes, dtype=adj.indices.dtype), degrees)"),
        ("np.arange(num_nodes, dtype=np.int64), degrees2)",
         "np.arange(num_nodes, dtype=adj.indices.dtype), degrees2)"),
        ("np.arange(adj.indices.size, dtype=np.int64)",
         "np.arange(adj.indices.size, dtype=adj.indptr.dtype)"),
        ("neib_list[row_idx, col_idx] = adj.indices.astype(np.int64)",
         "neib_list[row_idx, col_idx] = adj.indices"),
        ("neib_list[row_idx2, col_idx2] = adj.indices.astype(np.int64)",
         "neib_list[row_idx2, col_idx2] = adj.indices"),
    )
    expected=original
    for old,new in changes:
        expected_count=2 if "adj.indices.size" in old else 1
        check(expected.count(old)==expected_count,
              f"Original source differs from frozen index-width edit: {old}")
        expected=expected.replace(old,new)
    check(actual==expected,
          "Candidate production source differs from exactly five authorized replacements")
    untouched=(
        "pcc.py","pcc_numpy.py","pcc_numba.py","pcc_step.pyx",
        "pcc_validation.py","benchmarks","setup.py",
    )
    check(not git("diff","--name-only",MASTER,"HEAD","--",*untouched),
          "Native kernels, packaging or existing benchmark changed")
    reference=types.ModuleType("_frozen_original_csr_indices")
    exec(compile(original,f"{MASTER}:pcc_graph.py","exec"),reference.__dict__)
    return reference,hashlib.sha256(original.encode("utf-8")).hexdigest()


def exact(a,b,where):
    check(len(a)==len(b)==2,f"Unexpected original graph result: {where}")
    for name,x,y in zip(("neighbors","degrees"),a,b):
        check(x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y),
              f"Original and candidate {name} differ: {where}")
    check(a[0].dtype==np.dtype("int64") and
          a[1].dtype==np.dtype("int64"),
          "Public graph arrays no longer int64")
    return dict(neighbors=array_hash(a[0]),degrees=array_hash(a[1]))


def real_data(name):
    data={
        "iris":datasets.load_iris,"wine":datasets.load_wine,
        "breast_cancer":datasets.load_breast_cancer,
        "digits":datasets.load_digits,
    }[name]()
    x=StandardScaler().fit_transform(
        np.asarray(data.data,dtype=np.float64))
    y=np.asarray(data.target,dtype=np.int64)
    return x,y


def to_edges(graph):
    neighbors,degrees=graph
    rows=np.repeat(np.arange(len(degrees),dtype=np.int64),degrees)
    valid=(np.arange(neighbors.shape[1],dtype=np.int64)[None,:]
           <degrees[:,None])
    cols=np.asarray(neighbors[valid],dtype=np.int64)
    check(len(rows)==len(cols) and not np.any(rows==cols),
          "Original undirected graph contains self loop")
    return np.ascontiguousarray(np.vstack((rows,cols)))


def synthetic_inputs():
    rng=np.random.default_rng(28631)
    def pairs(n,m):
        src=rng.integers(0,n,size=m,dtype=np.int64)
        dst=rng.integers(0,n,size=m,dtype=np.int64)
        dst=np.where(src==dst,(dst+1)%n,dst)
        return np.vstack((src,dst))
    data={
        "empty_10000":(10000,np.empty((2,0),dtype=np.int64)),
        "isolated_30000_1000":(30000,pairs(30000,1000)),
        "sparse_10000_1000":(10000,pairs(10000,1000)),
        "medium_1000_3000":(1000,pairs(1000,3000)),
        "dense_1000_30000":(1000,pairs(1000,30000)),
        "dense_10000_100000":(10000,pairs(10000,100000)),
        "dense_30000_100000":(30000,pairs(30000,100000)),
    }
    orig=pairs(1000,2900)
    loops=rng.integers(0,1000,size=100,dtype=np.int64)
    data["loops_1000_3000"]=(
        1000,np.hstack((orig,np.vstack((loops,loops)))))
    base=pairs(1000,750)
    data["duplicates_1000_3000"]=(
        1000,np.hstack((base,base,base[::-1],base[::-1])))
    check(tuple(data)==SYNTH,"Changed frozen nine synthetic cases")
    for name,(n,e) in data.items():
        check(e.ndim==2 and e.shape[0]==2 and
              ("loops" in name)==bool(np.any(e[0]==e[1])),
              f"Unexpected synthetic case self-loop structure: {name}")
    return data


def traced_one(function):
    gc.collect()
    tracemalloc.start()
    result=None
    try:
        result=function()
        current,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    check(current>0 and peak>=current,"Invalid traced allocation result")
    graph=dict(
        current_traced_bytes=current,
        peak_traced_bytes=peak,
        returned_bytes=result[0].nbytes+result[1].nbytes)
    check(graph["returned_bytes"]>0,
          "Invalid returned graph allocation")
    hashes=dict(neighbors=array_hash(result[0]),
                degrees=array_hash(result[1]))
    del result
    return graph,hashes


def run_case(case_id,kind,reference,candidate,input_sha,extra):
    # Each function computes complete original production graph.
    first_graph=reference()
    other_graph=candidate()
    fp=exact(first_graph,other_graph,case_id+"/baseline")
    # Original retained output bytes are part of the graph's fixed public API.
    output_bytes=first_graph[0].nbytes+first_graph[1].nbytes
    del first_graph,other_graph
    exact(reference(),candidate(),case_id+"/warmups")
    timings=[]
    for rep in range(GRAPH_REPS):
        first,second=(reference,candidate) if rep%2==0 else (
            candidate,reference)
        t0=time.perf_counter_ns()
        x=first()
        t1=time.perf_counter_ns()
        y=second()
        t2=time.perf_counter_ns()
        before,after=(x,y) if rep%2==0 else (y,x)
        check(exact(before,after,f"{case_id}/time{rep}")==fp,
              "Original timing graph hash drift")
        a,b=((t1-t0,t2-t1) if rep%2==0 else (t2-t1,t1-t0))
        check(a>0 and b>0,"Invalid wall-clock timing")
        timings.append(dict(
            case=case_id,repeat=rep,order="original_first"
            if rep%2==0 else "candidate_first",
            reference_ns=a,candidate_ns=b))
        del x,y,before,after
    # Crucial: memory measured in its own pass, not during timed calls.
    memory=[]
    for rep in range(MEMORY_REPS):
        functions=(reference,candidate) if rep%2==0 else (
            candidate,reference)
        first,hfirst=traced_one(functions[0])
        second,hsecond=traced_one(functions[1])
        orig,cand=(first,second) if rep%2==0 else (second,first)
        check(hfirst==hsecond==fp and
              orig["returned_bytes"]==cand["returned_bytes"]==output_bytes,
              f"Traced original graph differs in {case_id}/{rep}")
        memory.append(dict(
            case=case_id,repeat=rep,order="original_first"
            if rep%2==0 else "candidate_first",
            reference=orig,candidate=cand))
    condition=dict(case=case_id,kind=kind,input_sha256=input_sha,
                   original_hashes=fp,returned_bytes=output_bytes,
                   **extra)
    print(f"{case_id}: {GRAPH_REPS} full timing pairs + "
          f"{MEMORY_REPS} traced memory pairs exact",flush=True)
    return condition,timings,memory


def real_conditions(reference):
    cases,timings,memory=[],[],[]
    native_graphs={}
    for name in DATASETS:
        x,truth=real_data(name)
        for k in KS:
            original_g=reference.build_knn_graph(x,k_nn=k,n_jobs=1)
            edges=to_edges(original_g)
            exact(original_g,reference.build_graph_from_edge_index(
                len(x),edges),f"{name}/k{k}/roundtrip")
            for source in ("knn","edge_index"):
                key=f"{name}-k{k}-{source}"
                if source=="knn":
                    a=lambda x=x,k=k:reference.build_knn_graph(
                        x,k_nn=k,n_jobs=1)
                    b=lambda x=x,k=k:build_knn_graph(
                        x,k_nn=k,n_jobs=1)
                    input_sha=array_hash(x)
                else:
                    a=lambda e=edges,n=len(x):reference.build_graph_from_edge_index(n,e)
                    b=lambda e=edges,n=len(x):build_graph_from_edge_index(n,e)
                    input_sha=array_hash(edges)
                cond,t,m=run_case(key,"real",a,b,input_sha,
                                  dict(dataset=name,k=k,source=source,
                                       n_nodes=len(x),
                                       n_input_edges=int(edges.shape[1])))
                cases.append(cond);timings.extend(t);memory.extend(m)
                if k==10:
                    native_graphs[name,source]=(
                        x,truth,a(),b())
    return cases,timings,memory,native_graphs


def synthetic_conditions(reference):
    cases,timings,memory=[],[],[]
    for name,(n,e) in synthetic_inputs().items():
        a=lambda e=e,n=n:reference.build_graph_from_edge_index(n,e)
        b=lambda e=e,n=n:build_graph_from_edge_index(n,e)
        cond,t,m=run_case(name,"synthetic",a,b,array_hash(e),
                          dict(n_nodes=n,n_input_edges=int(e.shape[1]),
                               self_loops=bool(np.any(e[0]==e[1]))))
        cases.append(cond);timings.extend(t);memory.extend(m)
    return cases,timings,memory


def invalid_controls(reference):
    cases=(
        ("n0",0,np.empty((2,0),dtype=np.int64)),
        ("n0_has_edge",0,np.array([[0],[0]],dtype=np.int64)),
        ("bad_shape",5,np.empty((3,0),dtype=np.int64)),
        ("bad_negative_nodecount",-1,np.empty((2,0),dtype=np.int64)),
        ("bad_negative_vertex",5,np.array([[0,-1],[1,2]],dtype=np.int64)),
        ("bad_oob_vertex",5,np.array([[0,5],[1,2]],dtype=np.int64)),
        ("loops_only",4,np.array([[0,1,2,3],[0,1,2,3]],dtype=np.int64)),
        ("noncontiguous_edges",6,np.arange(20,dtype=np.int64).reshape(2,10)[:,::2]%6),
    )
    rows=[]
    for name,n,e in cases:
        outcomes=[]
        for f in (reference.build_graph_from_edge_index,
                  build_graph_from_edge_index):
            try:
                graph=f(n,e)
                result=dict(status="success",
                            hashes=dict(neighbors=array_hash(graph[0]),
                                        degrees=array_hash(graph[1])),
                            shape=list(graph[0].shape))
            except Exception as exc:
                result=dict(status="error",type=type(exc).__name__,
                            message=str(exc))
            outcomes.append(result)
        check(outcomes[0]==outcomes[1],
              f"Original edge-index invalid behavior differs for {name}")
        rows.append(dict(case=name,input_sha256=array_hash(e),
                         outcome=outcomes[0],exact=True))
    # Additional ties / noncontiguous feature arrays.
    x=np.array([[0,0],[0,0],[1,0],[1,0],[0,1],[0,1],
                [1,1],[1,1],[2,0],[0,2]],dtype=np.float64)
    for name,data in (("duplicate_ties",x),
                      ("fortran_ties",np.asfortranarray(x))):
        for k in (1,3,len(data)-1):
            for p in (1,2):
                graph0=reference.build_knn_graph(data,k_nn=k,p=p,n_jobs=1)
                graph1=build_knn_graph(data,k_nn=k,p=p,n_jobs=1)
                fp=exact(graph0,graph1,f"{name}/k{k}/p{p}")
                rows.append(dict(case=f"{name}-k{k}-p{p}",
                                 input_sha256=array_hash(data),
                                 outcome=dict(status="success",hashes=fp,
                                              shape=list(graph0[0].shape)),
                                 exact=True))
    check(len(rows)==20,"Frozen edge/tied input controls changed")
    return rows


def native_conditions(graphs):
    checks=[]
    for dataset in DATASETS:
        for source in ("knn","edge_index"):
            x,truth,orig,trial=graphs[dataset,source]
            exact(orig,trial,f"native/{dataset}/{source}")
            for seed in LABEL_SEEDS:
                case=Case(dataset,len(x),x.shape[1],
                          int(len(np.unique(truth))),k_nn=10,
                          labels_per_class=3,data_seed=0,label_seed=seed)
                labels=select_labels(truth,case)
                arr=dict(labels=labels,truth=truth)
                for backend in BACKENDS:
                    if backend=="numba":
                        measure(dict(neighbors=orig[0],degrees=orig[1],**arr),
                                case,backend,2,False,0,steady_state=False)
                    for mode in MODES:
                        stop=(mode=="early_stopping")
                        a=measure(dict(neighbors=orig[0],degrees=orig[1],**arr),
                                  case,backend,ITERATIONS,stop,0,steady_state=True)
                        b=measure(dict(neighbors=trial[0],degrees=trial[1],**arr),
                                  case,backend,ITERATIONS,stop,0,steady_state=True)
                        fields=("state_hashes","effective_iterations",
                                "stop_reason","checks","stalled_checks","stop_max",
                                "backend_identity","accuracy_unlabeled")
                        check(all(a.get(k)==b.get(k) for k in fields)
                              and len(a["state_hashes"])==6,
                              f"Original native states differ: {dataset}/{source}/{seed}/{backend}/{mode}")
                        checks.append(dict(
                            dataset=dataset,source=source,seed=seed,
                            backend=backend,mode=mode,
                            iteration_cap=ITERATIONS,
                            state_hashes=a["state_hashes"],
                            effective_iterations=a["effective_iterations"],
                            stop_reason=a["stop_reason"],
                            backend_identity=a["backend_identity"],
                            accuracy_unlabeled=a["accuracy_unlabeled"],
                            exact=True))
    check(len(checks)==96,"Missing original-backend native controls")
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    reference,orig_sha=original_reference()
    rc,rt,rm,graphs=real_conditions(reference)
    sc,st,sm=synthetic_conditions(reference)
    ctrl=invalid_controls(reference)
    native=native_conditions(graphs)
    check(len(rc)==24 and len(sc)==9 and
          len(rt)+len(st)==660 and len(rm)+len(sm)==165 and
          len(ctrl)==20 and len(native)==96,
          "Incomplete frozen graph/memory/native cohort")
    report=dict(
        protocol="csr_index_memory_v1",master=MASTER,
        head=git("rev-parse","HEAD"),original_source_sha256=orig_sha,
        host=dict(os=platform.platform(),python=sys.version,
                  cpu=platform.processor(),numpy=np.__version__,
                  thread_env={key:os.environ.get(key) for key in (
                      "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                      "MKL_NUM_THREADS","NUMBA_NUM_THREADS")}),
        datasets=list(DATASETS),ks=list(KS),synthetic=list(SYNTH),
        timing_repetitions=GRAPH_REPS,memory_repetitions=MEMORY_REPS,
        label_seeds=list(LABEL_SEEDS),backends=list(BACKENDS),
        modes=list(MODES),max_iter=ITERATIONS,
        conditions=rc+sc,full_builder_timings=rt+st,
        traced_memory_pairs=rm+sm,
        edge_and_tie_controls=ctrl,native_pairs=native)
    output=args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True,
                                 allow_nan=False)+"\n",
                      encoding="utf-8")
    print(f"COMPLETE: {len(rt)+len(st)} timing pairs, "
          f"{len(rm)+len(sm)} isolated memory pairs, "
          f"{len(native)} unchanged native pairs, {len(ctrl)} input controls",
          flush=True)


if __name__=="__main__":
    main()
