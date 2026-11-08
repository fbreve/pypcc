"""Prospective exact edge-index graph cleanup audit, based on latest master.

This file must not change original graph/kNN/native PCC semantics. The reference
builder is imported verbatim from the frozen production Git blob. All original/
candidate timing pairs, including regressions, are retained and validated.
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
from sklearn.preprocessing import StandardScaler

from benchmarks.fixtures import Case, array_hash, select_labels
from benchmarks.runner import measure
from pcc_graph import build_graph_from_edge_index

MASTER="35dc81caf0388be8ea1729837ea6d54358b53964"
REAL_DATASETS=("iris","wine","breast_cancer","digits")
K_VALUES=(1,3,10)
SYNTHETIC_CASES=("empty_1","empty_10000","isolated_30000_1000",
                 "sparse_1000_1000","sparse_10000_1000",
                 "medium_1000_3000","dense_1000_30000",
                 "dense_10000_100000","mixed_loops_1000_2000",
                 "repeated_reciprocal_1000_3000")
REPETITIONS=30
SYNTH_SEED=28311
LABEL_SEEDS=(28319,28447)
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
ITERATIONS=1000


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def root():
    return Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(
        ["git",*args],cwd=root(),text=True).strip()


def load_reference():
    original=subprocess.check_output(\n        ["git","show",f"{MASTER}:pcc_graph.py"],\n        cwd=root(),text=True)
    candidate=(root()/"pcc_graph.py").read_text(encoding="utf-8")
    old='''    adj = csr_matrix((ones, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.bool_)
    adj.setdiag(0)      # remove self-loops
    adj.eliminate_zeros()

    # Deduplicate: keep only binary (0/1) entries'''
    new='''    adj = csr_matrix((ones, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.bool_)
    # Boolean CSR has only nonzero entries; without input self-loops there
    # is no diagonal entry to clear or explicit zero to eliminate. Keep the
    # original cleanup unchanged for the self-loop case.
    if np.any(src == dst):
        adj.setdiag(0)      # remove self-loops
        adj.eliminate_zeros()

    # Deduplicate: keep only binary (0/1) entries'''
    require(original.count(old)==1 and candidate==original.replace(old,new),
            "Candidate production source must contain EXACTLY the frozen one-hunk edit")
    untouched=("pcc.py","pcc_numpy.py","pcc_numba.py","pcc_step.pyx",
               "pcc_validation.py","benchmarks","setup.py")
    require(not git("diff","--name-only",MASTER,"HEAD","--",*untouched),
            "Production model/benchmark source was altered")
    module=types.ModuleType("_original_pre_cleanup_edge_builder")
    exec(compile(original,f"{MASTER}:pcc_graph.py","exec"),module.__dict__)
    return module,hashlib.sha256(original.encode("utf-8")).hexdigest()


def exact_graph(original,candidate,context):
    require(len(original)==len(candidate)==2,
            f"Original graph output shape invalid: {context}")
    for name,a,b in zip(("neighbors","degrees"),original,candidate):
        require(a.shape==b.shape and a.dtype==b.dtype
                and np.array_equal(a,b),
                f"Full graph {name} differs from frozen master: {context}")
    return dict(neighbors=array_hash(original[0]),
                degrees=array_hash(original[1]))


def real_input(name,reference):
    raw={
        "iris":datasets.load_iris,
        "wine":datasets.load_wine,
        "breast_cancer":datasets.load_breast_cancer,
        "digits":datasets.load_digits,
    }[name]()
    x=StandardScaler().fit_transform(
        np.asarray(raw.data,dtype=np.float64))
    truth=np.asarray(raw.target,dtype=np.int64)
    return x,truth


def directed_graph_edges(graph):
    neighbors,degree=graph
    n=len(degree)
    rows=np.repeat(np.arange(n,dtype=np.int64),degree)
    eligible=(np.arange(neighbors.shape[1],dtype=np.int64)[None,:] <
              degree[:,None])
    cols=np.asarray(neighbors[eligible],dtype=np.int64)
    require(len(rows)==len(cols) and not np.any(rows==cols),
            "Original real undirected kNN graph contains a self-loop")
    return np.ascontiguousarray(np.vstack([rows,cols]))


def synthetic_inputs():
    rng=np.random.default_rng(SYNTH_SEED)
    def random_loop_free(n,m):
        src=rng.integers(0,n,size=m,dtype=np.int64)
        dst=rng.integers(0,n,size=m,dtype=np.int64)
        dst=np.where(src==dst,(dst+1)%n,dst)
        return np.vstack((src,dst))
    d={}
    d["empty_1"]=(1,np.empty((2,0),dtype=np.int64))
    d["empty_10000"]=(10000,np.empty((2,0),dtype=np.int64))
    d["isolated_30000_1000"]=(30000,random_loop_free(30000,1000))
    d["sparse_1000_1000"]=(1000,random_loop_free(1000,1000))
    d["sparse_10000_1000"]=(10000,random_loop_free(10000,1000))
    d["medium_1000_3000"]=(1000,random_loop_free(1000,3000))
    d["dense_1000_30000"]=(1000,random_loop_free(1000,30000))
    d["dense_10000_100000"]=(10000,random_loop_free(10000,100000))
    edges=random_loop_free(1000,1900)
    loops=rng.integers(0,1000,size=100,dtype=np.int64)
    d["mixed_loops_1000_2000"]=(
        1000,np.column_stack((edges,np.vstack((loops,loops)))))
    base=random_loop_free(1000,750)
    repeated=np.hstack((base,base,base[::-1],base[::-1]))
    d["repeated_reciprocal_1000_3000"]=(1000,repeated)
    require(tuple(d)==SYNTHETIC_CASES,"Synthetic cohort changed")
    sizes={
        "empty_1":(1,0),"empty_10000":(10000,0),
        "isolated_30000_1000":(30000,1000),
        "sparse_1000_1000":(1000,1000),
        "sparse_10000_1000":(10000,1000),
        "medium_1000_3000":(1000,3000),
        "dense_1000_30000":(1000,30000),
        "dense_10000_100000":(10000,100000),
        "mixed_loops_1000_2000":(1000,2000),
        "repeated_reciprocal_1000_3000":(1000,3000)}
    for name,(n,e) in d.items():
        require(e.shape==(2,sizes[name][1]) and n==sizes[name][0],
                "Wrong synthetic input size")
        require((np.any(e[0]==e[1]))==(
            name=="mixed_loops_1000_2000"),
            f"Unexpected synthetic self loop in {name}")
    return d


def paired_case(reference,candidate,case_id,n,edge_index,kind,traits=None):
    before=reference.build_graph_from_edge_index(n,edge_index)
    after=candidate(n,edge_index)
    fp=exact_graph(before,after,case_id)
    original=lambda:reference.build_graph_from_edge_index(n,edge_index)
    changed=lambda:candidate(n,edge_index)
    # One real unmeasured warmup per side.
    exact_graph(original(),changed(),case_id+"/warmup")
    comparisons=[]
    for repetition in range(REPETITIONS):
        first,second=(original,changed) if repetition%2==0 else (
            changed,original)
        t0=time.perf_counter_ns()
        g0=first()
        t1=time.perf_counter_ns()
        g1=second()
        t2=time.perf_counter_ns()
        orig,trial=(g0,g1) if repetition%2==0 else (g1,g0)
        tf_ref,tf_trial=(t1-t0,t2-t1) if repetition%2==0 else (
            t2-t1,t1-t0)
        require(exact_graph(orig,trial,case_id+f"/{repetition}")==fp,
                f"Graph changed between repetitions: {case_id}")
        require(tf_ref>0 and tf_trial>0,
                f"Invalid physical elapsed time: {case_id}/{repetition}")
        comparisons.append(dict(
            case=case_id,kind=kind,repeat=repetition,
            reference_ns=tf_ref,candidate_ns=tf_trial,
            order="reference_first" if repetition%2==0 else "candidate_first"))
    metadata=dict(
        case=case_id,kind=kind,num_nodes=n,
        num_directed_input_edges=int(edge_index.shape[1]),
        input_sha256=array_hash(edge_index),
        original_hashes=fp,
        self_loops_present=bool(np.any(edge_index[0]==edge_index[1])),
        **(traits or {}))
    return metadata,comparisons,before,after


def real_conditions(reference):
    cases=[]
    timings=[]
    native_graphs={}
    for name in REAL_DATASETS:
        x,truth=real_input(name,reference)
        for k in K_VALUES:
            graph=reference.build_knn_graph(x,k_nn=k,n_jobs=1)
            edges=directed_graph_edges(graph)
            ident=f"{name}-k{k}"
            meta,rows,original,candidate=paired_case(
                reference,build_graph_from_edge_index,ident,len(x),edges,
                "real",dict(dataset=name,k=k,data_sha256=array_hash(x),
                            target_sha256=array_hash(truth)))
            exact_graph(graph,original,
                        f"Original edge-index reconstruction of union kNN: {ident}")
            cases.append(meta)
            timings.extend(rows)
            if k==10:
                native_graphs[name]=(x,truth,original,candidate)
            print(f"real {ident}: {REPETITIONS} full exact graph pairs PASS",
                  flush=True)
    return cases,timings,native_graphs


def synthetic_conditions(reference):
    cases=[]
    timings=[]
    for name,(n,edges) in synthetic_inputs().items():
        meta,rows,_,_=paired_case(
            reference,build_graph_from_edge_index,name,n,edges,"synthetic")
        cases.append(meta)
        timings.extend(rows)
        print(f"synthetic {name}: {REPETITIONS} full exact graph pairs PASS",
              flush=True)
    return cases,timings


def invalid_and_empty_controls(reference):
    # Preserve documented and observed input-validation behavior. Compare
    # both the exception CLASS and error message (no silent relaxation).
    inputs=[
        ("empty_0",0,np.empty((2,0),dtype=np.int64)),
        ("empty_0_with_edges",0,np.array([[0],[0]],dtype=np.int64)),
        ("negative_nodes",-1,np.empty((2,0),dtype=np.int64)),
        ("out_of_bounds",3,np.array([[0,3],[1,2]],dtype=np.int64)),
        ("negative_index",3,np.array([[0,-1],[1,2]],dtype=np.int64)),
        ("wrong_shape",3,np.empty((3,0),dtype=np.int64)),
        ("nonintegral_cast",3,np.array([[0.,1.],[1.,2.]],dtype=np.float64)),
        ("only_self_loops",4,np.array([[0,1,2,3],[0,1,2,3]],
                                     dtype=np.int64)),
    ]
    res=[]
    for name,n,edges in inputs:
        outcomes=[]
        for f in (reference.build_graph_from_edge_index,
                  build_graph_from_edge_index):
            try:
                val=f(n,edges)
                outcomes.append(dict(status="ok",
                                     neighbors=array_hash(val[0]),
                                     degrees=array_hash(val[1])))
            except Exception as e:
                outcomes.append(dict(status="error",
                                     error_class=type(e).__name__,
                                     message=str(e)))
        require(outcomes[0]==outcomes[1],
                f"Invalid/empty source semantics differ: {name}: {outcomes}")
        res.append(dict(case=name,num_nodes=n,
                        input_sha256=array_hash(edges),
                        outcome=outcomes[0],exact=True))
    return res


def native_conditions(graphs):
    checks=[]
    for dataset in REAL_DATASETS:
        x,truth,reference,candidate=graphs[dataset]
        n,c=len(x),len(np.unique(truth))
        for seed in LABEL_SEEDS:
            case=Case(dataset,n,x.shape[1],c,k_nn=10,
                      labels_per_class=3,data_seed=0,label_seed=seed)
            labels=select_labels(truth,case)
            arrays=dict(truth=truth,labels=labels)
            for backend in BACKENDS:
                for mode in MODES:
                    early=mode=="early_stopping"
                    before=dict(neighbors=reference[0],
                                degrees=reference[1],**arrays)
                    after=dict(neighbors=candidate[0],
                               degrees=candidate[1],**arrays)
                    if backend=="numba":
                        measure(before,case,"numba",2,False,0,
                                steady_state=False)
                    a=measure(before,case,backend,ITERATIONS,
                              early,0,steady_state=True)
                    b=measure(after,case,backend,ITERATIONS,
                              early,0,steady_state=True)
                    fields=("state_hashes","effective_iterations",
                            "stop_reason","accuracy_unlabeled",
                            "checks","stalled_checks","stop_max",
                            "backend_identity")
                    require(all(a.get(field)==b.get(field)
                                for field in fields),
                            f"Native state or original stop metadata mismatch: "
                            f"{dataset}/{seed}/{backend}/{mode}")
                    require(len(a["state_hashes"])==6,
                            "Native state must compare six complete arrays")
                    checks.append(dict(
                        dataset=dataset,label_seed=seed,backend=backend,
                        mode=mode,iteration_cap=ITERATIONS,
                        effective_iterations=a["effective_iterations"],
                        stop_reason=a["stop_reason"],
                        backend_identity=a["backend_identity"],
                        accuracy_unlabeled=a["accuracy_unlabeled"],
                        state_hashes=a["state_hashes"],
                        complete_exact_parity=True))
        print(f"native {dataset}: {2*len(BACKENDS)*len(MODES)} "
              "reference/candidate state pairs PASS",flush=True)
    require(len(checks)==len(REAL_DATASETS)*len(LABEL_SEEDS)*
            len(BACKENDS)*len(MODES),"Native full cohort incomplete")
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    reference,sourcehash=load_reference()
    real,real_times,graphs=real_conditions(reference)
    synthetic,synthetic_times=synthetic_conditions(reference)
    validation=invalid_and_empty_controls(reference)
    native=native_conditions(graphs)
    require(len(real)==12 and len(synthetic)==10 and
            len(real_times)==360 and len(synthetic_times)==300 and
            len(native)==48 and len(validation)==8,
            "Frozen full benchmark conditions incomplete")
    env=dict(os=platform.platform(),python=sys.version,
             cpu=platform.processor(),numpy=np.__version__,
             thread_env={name:os.environ.get(name) for name in
                         ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                          "MKL_NUM_THREADS","NUMBA_NUM_THREADS")})
    report=dict(schema="edge_index_diagonal_v1",master=MASTER,
                head=git("rev-parse","HEAD"),
                original_source_sha256=sourcehash,
                real_datasets=list(REAL_DATASETS),k_values=list(K_VALUES),
                synthetic_cases=list(SYNTHETIC_CASES),
                repeats=REPETITIONS,label_seeds=list(LABEL_SEEDS),
                backends=list(BACKENDS),modes=list(MODES),
                max_iter=ITERATIONS,random_seed=SYNTH_SEED,
                environment=env,conditions=real+synthetic,
                graph_pairs=real_times+synthetic_times,
                native_pairs=native,edge_controls=validation)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(
        report,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print(f"ALL PASS: {len(real_times)+len(synthetic_times)} "
          f"alternating full graph pairs, {len(native)} exact native "
          f"pairs, {len(validation)} invalid/empty controls",flush=True)


if __name__=="__main__":
    main()
