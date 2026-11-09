"""Profile exactly ONE original *compiled* Numba PCC fit via Linux Callgrind.

All model and graph setup/JIT compilation happens before native collection.
All checksums/RNG probes and seeded state/info replay happen AFTER collection.
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

BASE="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
STATES=("predictions","dominance","positions","strengths","distances","owndeg")


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
    }


def make_model(arrays):
    x=PCC(impl="numba",update_mode="sequential")
    x.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
    if x._get_backend_fn() is not pcc_numba.pcc_propagate_numba:
        raise AssertionError("real production Numba backend not selected")
    return x


def fit(model,labels,count=1000):
    return model.fit_predict(
        labels,seed=0,max_iter=count,early_stop=False,
        p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)


def invoke_command(on):
    command=["callgrind_control","-i","on" if on else "off",str(os.getpid())]
    result=subprocess.run(command,text=True,capture_output=True,timeout=10)
    if result.returncode:
        raise RuntimeError("Callgrind collection switch failed "+
                           repr((command,result.stdout[-3000:],result.stderr[-3000:])))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=("wine","digits"),required=True)
    p.add_argument("--seed",type=int,choices=(73013,73117),required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()

    if os.name!="posix" or sys.platform!="linux":
        raise AssertionError("Callgrind profiling is Linux-only by preregistration")
    required=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
              "MKL_NUM_THREADS","NUMBA_NUM_THREADS")
    if any(os.environ.get(key)!="1" for key in required):
        raise AssertionError("native Numba threads must be controlled at 1")
    if (os.environ.get("NUMBA_ENABLE_PROFILING")!="1" or
            os.environ.get("NUMBA_DEBUGINFO")!="1"):
        raise AssertionError("Numba JIT profiling and debuginfo must be enabled")
    nodes,features,classes,_=REAL_DATASETS[args.dataset]
    case=Case(args.dataset,nodes,features,classes,
              k_nn=10,labels_per_class=3,data_seed=0,label_seed=args.seed)
    arrays,meta=prepare_fixture(case)
    if meta["labeled_nodes"]!=3*classes:
        raise AssertionError("wrong real class-balanced labeled fixture")
    # Warm up all actual native JIT signatures BEFORE Callgrind is enabled.
    warm=make_model(arrays)
    fit(warm,arrays["labels"],count=2)
    if not pcc_numba._pcc_propagate_numba_sequential.signatures:
        raise AssertionError("original Numba propagator was not JIT compiled")
    # Only exact LLVM machine-code entry addresses from THIS profiled Python
    # interpreter may resolve anonymous Valgrind function start addresses.
    # Disallow heuristics based on neighboring addresses or source names.
    native_entries={}
    native_signatures={}
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
                raise AssertionError("Numba reported a non-address LLVM native entry")
            entries.append(dict(signature=str(signature),
                                address=hex(address),
                                llvm_symbol=compiled.fndesc.llvm_func_name))
        native_entries[name]=entries
        native_signatures[name]=[str(signature) for signature in dispatcher.signatures]
    if not native_entries["_pcc_propagate_numba_sequential"]:
        raise AssertionError("original complete Numba native JIT was not materialized")
    model=make_model(arrays)
    invoke_command(True)
    try:
        predicted=fit(model,arrays["labels"],count=1000)
    finally:
        invoke_command(False)
    recorded=hash_state(model,predicted)
    layout="C" if model.node.dominance.flags.c_contiguous else "F"
    if layout!=("C" if classes>=8 else "F"):
        raise AssertionError("profiling changed native class C/F layout")
    rng_next=float(original_next_numba_rng())

    # Independent true source native prefix replay and original metrics,
    # entirely outside Callgrind's native instruction collection.
    ref=make_model(arrays)
    original=ref._get_backend_fn()
    result={}
    def observer(*args,**kwargs):
        kwargs["return_info"]=True
        result.update(original(*args,**kwargs))
        return None
    observer.__name__=original.__name__
    ref._get_backend_fn=lambda:observer
    expected=fit(ref,arrays["labels"],count=1000)
    expected_states=hash_state(ref,expected)
    expected_rng_next=float(original_next_numba_rng())
    if recorded!=expected_states or rng_next!=expected_rng_next:
        raise AssertionError("Callgrind instrumented original native output or RNG differs")
    if (result["effective_iterations"]!=1000 or
            result["stop_reason"]!="max_iter" or result["checks"]!=0):
        raise AssertionError("expected original exactly 1000 fixed Numba iterations")

    payload=dict(
        schema=1,base=BASE,case_id=case.case_id,
        dataset=args.dataset,seed=args.seed,
        numpy=np.__version__,numba=numba.__version__,python=sys.version,
        machine=platform.machine(),system=platform.platform(),
        numba_jit_profiling=True,
        original_compiled_step_signatures=native_signatures[
            "_pcc_step_numba_sequential"],
        native_jit_exact_entry_addresses=native_entries,
        original_compiled_propagation_signatures=native_signatures[
            "_pcc_propagate_numba_sequential"],
        graph_sha=[
            array_hash(arrays["neighbors"]),array_hash(arrays["degrees"])],
        labels_sha=array_hash(arrays["labels"]),
        fixture_sha=meta["fixture_sha256"],source_sha=hashlib.sha256(
            Path(pcc_numba.__file__).read_bytes()).hexdigest(),
        original_graph_dense_int64=bool(
            arrays["neighbors"].dtype==np.int64),
        native_layout=layout,
        state_hashes=recorded,rng_next=rng_next,
        original_stop=dict((name,result[name]) for name in (
            "effective_iterations","stop_reason","checks",
            "stalled_checks","stop_max")),
        callgrind_pid=os.getpid(),isolated_profile=True)
    Path(args.out).write_text(
        json.dumps(payload,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print("NUMBA_CALLGRIND_NATIVE_PROFILE_PASS "
          +repr((case.case_id,layout,os.getpid())),flush=True)


if __name__=="__main__":
    main()
