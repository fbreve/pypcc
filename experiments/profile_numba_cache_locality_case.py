"""Capture original Numba fit with simulated D1/LL cache events (Linux only).

Windows runs the exact same genuine source as a native reference. Valgrind
simulated misses are NOT real hardware misses, memory bandwidth or CPU time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from numba import njit
import numpy as np
import numba
from benchmarks.fixtures import Case, REAL_DATASETS, prepare_fixture, array_hash
from pcc import ParticleCompetitionAndCooperation as PCC
import pcc_numba

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
SEEDS=(89021,89119)
POLICIES=(0.0,0.5,1.0)
PROTECTED=("pcc.py","pcc_numba.py","pcc_graph.py","pcc_numpy.py",
           "pcc_validation.py","pcc_step.pyx","benchmarks/fixtures.py")
STATES=("predictions","dominance","positions","strengths","distances",
        "owndeg","node_labels","mapped_labels")


@njit
def original_next_numba_rng():
    return np.random.random()


def hash_state(model,predicted):
    return {
        "predictions":array_hash(predicted),
        "dominance":array_hash(model.node.dominance),
        "positions":array_hash(model.part.curnode),
        "strengths":array_hash(model.part.strength),
        "distances":array_hash(model.part.dist_table),
        "owndeg":array_hash(model.owndeg),
        "node_labels":array_hash(model.node.label),
        "mapped_labels":array_hash(model.mapped_labels),
    }


def guard_production_files():
    root=Path(__file__).resolve().parents[1]
    modifications=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        cwd=root,text=True).strip()
    if modifications:
        raise AssertionError("original production source diverged: "+modifications)
    hashes={}
    for path in PROTECTED:
        raw=subprocess.check_output(["git","show",BASE+":"+path],cwd=root)
        disk=(root/path).read_bytes()
        if raw.replace(b"\r\n",b"\n")!=disk.replace(b"\r\n",b"\n"):
            raise AssertionError("source Git original not checked out: "+path)
        hashes[path]=hashlib.sha256(raw).hexdigest()
    return hashes


def make_model(arrays):
    x=PCC(impl="numba",update_mode="sequential")
    x.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
    if x._get_backend_fn() is not pcc_numba.pcc_propagate_numba:
        raise AssertionError("real production Numba backend not selected")
    return x


def fit(model,labels,policy,count=1000):
    return model.fit_predict(
        labels,seed=29,max_iter=count,early_stop=False,
        p_grd=policy,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)


def invoke_command(on):
    command=["callgrind_control","-i","on" if on else "off",str(os.getpid())]
    result=subprocess.run(command,text=True,capture_output=True,timeout=10)
    if result.returncode:
        raise RuntimeError("Callgrind collection switch failed "+
                           repr((command,result.stdout[-3000:],result.stderr[-3000:])))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=("wine","digits"),required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--policy",type=float,choices=POLICIES,required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--reference-only",action="store_true")
    args=p.parse_args()

    if args.reference_only != (sys.platform=="win32"):
        raise AssertionError("Linux must use true Valgrind; Windows reference is unprofiled")
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                 "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        if os.environ.get(name)!="1":
            raise AssertionError("all real original native threads must equal 1: "+name)
    if (os.environ.get("NUMBA_ENABLE_PROFILING")!="1" or
            os.environ.get("NUMBA_DEBUGINFO")!="1"):
        raise AssertionError("original-source Numba JIT profiling opt-in missing")
    source_hashes=guard_production_files()
    nodes,features,classes,_=REAL_DATASETS[args.dataset]
    case=Case(args.dataset,nodes,features,classes,k_nn=10,
              labels_per_class=3,data_seed=0,label_seed=args.seed)
    arrays,meta=prepare_fixture(case)
    if meta["labeled_nodes"]!=3*classes:
        raise AssertionError("not 3 original class-balanced homes per class")
    graph=arrays["neighbors"]
    degrees=arrays["degrees"]
    assert graph.dtype==degrees.dtype==np.int64 and graph.shape[0]==nodes
    for i,k in enumerate(degrees):
        assert len(set(graph[i,:k]))==k and np.all((0<=graph[i,:k])&(graph[i,:k]<nodes))
        assert np.all([i in graph[j,:degrees[j]] for j in graph[i,:k]]),(
            "not undirected reciprocal original kNN union",i)
    # Materialize Numba JIT before any Linux Callgrind counters. Unique JIT cache
    # for each condition is mandatory; no disk-loaded cached step signatures.
    warm=make_model(arrays)
    fit(warm,arrays["labels"],args.policy,count=2)
    if not pcc_numba._pcc_propagate_numba_sequential.signatures:
        raise AssertionError("real original Numba propagation not JIT compiled")
    native_entries={}
    native_signatures={}
    if not args.reference_only:
        for name,dispatcher in (
            ("_seed_numba_rng",pcc_numba._seed_numba_rng),
            ("_pcc_step_numba_sequential",pcc_numba._pcc_step_numba_sequential),
            ("_pcc_calc_mmpot_sequential",pcc_numba._pcc_calc_mmpot_sequential),
            ("_pcc_propagate_numba_sequential",pcc_numba._pcc_propagate_numba_sequential),
        ):
            entries=[]
            for signature,compiled in dispatcher.overloads.items():
                address=compiled.library.get_pointer_to_function(
                    compiled.fndesc.llvm_func_name)
                if not isinstance(address,int) or address<=0:
                    raise AssertionError("missing exact same-process native JIT pointer")
                entries.append(dict(signature=str(signature),address=hex(address),
                                    llvm_symbol=compiled.fndesc.llvm_func_name))
            native_entries[name]=entries
            native_signatures[name]=[str(x) for x in dispatcher.signatures]
        assert native_entries["_pcc_step_numba_sequential"],(
            "actual original Numba step unavailable in same-process LLVM JIT")

    model=make_model(arrays)
    if not args.reference_only:
        invoke_command(True)
    try:
        prediction=fit(model,arrays["labels"],args.policy,count=1000)
    finally:
        if not args.reference_only:
            invoke_command(False)
    observed=hash_state(model,prediction)
    rng_next=float(original_next_numba_rng())
    layout=("C" if model.node.dominance.flags.c_contiguous else "F")
    assert layout==("C" if classes>=8 else "F")
    assert model.owndeg.flags.f_contiguous
    assert model.part.dist_table.dtype==np.uint8
    assert np.array_equal(prediction[arrays["labels"]!=-1],
                          arrays["labels"][arrays["labels"]!=-1])
    # The backend wrapper is applied only AFTER the profiled full original fit.
    # Preserve the currently merged #228 original code identity branch.
    import pcc as prod
    ref=make_model(arrays)
    original_fn=pcc_numba.pcc_propagate_numba
    info={}
    def wrapped(*native_args,**kwargs):
        kwargs["return_info"]=True
        info.update(original_fn(*native_args,**kwargs))
        return None
    wrapped.__name__="pcc_propagate_numba"
    ref._get_backend_fn=lambda:wrapped
    assert prod.pcc_propagate_numba is original_fn
    prod.pcc_propagate_numba=wrapped
    try:
        independently_replayed=fit(ref,arrays["labels"],args.policy,count=1000)
    finally:
        prod.pcc_propagate_numba=original_fn
    replay=hash_state(ref,independently_replayed)
    replay_rng=float(original_next_numba_rng())
    if observed!=replay or rng_next!=replay_rng:
        raise AssertionError("real original Numba state or MT19937 changed under source replay")
    if (info["effective_iterations"]!=1000 or
            info["stop_reason"]!="max_iter" or info["checks"]!=0):
        raise AssertionError("not the original native 1000 fixed iterations")
    assert ref.node.dominance.flags.c_contiguous==(classes>=8)
    assert ref.owndeg.flags.f_contiguous
    result=dict(
        schema=1,base=BASE,head=subprocess.check_output(
            ["git","rev-parse","HEAD"],text=True).strip(),
        dataset=args.dataset,seed=args.seed,policy=args.policy,
        case_id=f"{args.dataset}-k10-labels{args.seed}-p{args.policy}",
        steps=1000,numba_jit_profiling=True,
        cache_simulation_model=dict(D1_bytes=32768,D1_ways=8,
                                    LL_bytes=8388608,LL_ways=16,
                                    line_bytes=64),
        linux_actual_valgrind=not args.reference_only,
        windows_uninstrumented_reference=args.reference_only,
        numpy=np.__version__,numba=numba.__version__,python=sys.version,
        machine=platform.machine(),system=platform.platform(),
        original_compiled_step_signatures=native_signatures.get(
            "_pcc_step_numba_sequential",[]),
        native_jit_exact_entry_addresses=native_entries,
        graph_sha=[array_hash(graph),array_hash(degrees)],
        labels_sha=array_hash(arrays["labels"]),
        fixture_sha=meta["fixture_sha256"],
        source_hashes=source_hashes,
        original_dense_padded_int64=True,
        native_layout=layout,owndeg_f_layout=True,
        distances_uint8=True,
        state_hashes=observed,rng_next=rng_next,
        original_stop=dict((x,info[x]) for x in (
            "effective_iterations","stop_reason","checks","stalled_checks","stop_max")),
        callgrind_pid=os.getpid() if not args.reference_only else None,
        isolated_profile=not args.reference_only)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("ORIGINAL_NUMBA_CACHE_LOCALITY_SOURCE_REPLAY_PASS",
          args.dataset,args.seed,args.policy,sys.platform,flush=True)


if __name__=="__main__":
    main()
