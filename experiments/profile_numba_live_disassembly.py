"""Snapshot ORIGINAL live Numba JIT function bytes with SAME-RUN Callgrind.

No production edits, no hardware cache counters, no array attribution. The
Capstone disassembly must match the native function pointer recorded by
original Callgrind in this same process; 8-state RNG/stop must replay.
"""
import argparse
import ctypes
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
import llvmlite
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone.x86 import X86_OP_MEM
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



MAX_JIT_SNAPSHOT=16384
EXISTING_HOT_OFFSETS=(0x36b,0x3a0,0x3a6,0x5ad,0x620,0x7d2)


def snapshot_native_step(dispatcher, output_file):
    """Read executable bytes at the EXACT original same-process LLVM JIT pointer.

    Addresses in JIT code are process-specific; the pointer and raw bytes must
    be collected in the same Valgrind process as the actual 1000-step fit.
    This is ONLY machine disassembly, NOT an array/CPU cache profiler.
    """
    if sys.platform!="linux":
        raise AssertionError("native JIT code snapshot is Linux only")
    if platform.machine().lower() not in ("x86_64","amd64"):
        raise AssertionError("Capstone x86-64 source machine required")
    compiled=list(dispatcher.overloads.items())
    if len(compiled)!=1:
        raise AssertionError("expected one original JIT source signature")
    signature,code=compiled[0]
    ptr=int(code.library.get_pointer_to_function(code.fndesc.llvm_func_name))
    if ptr<=0:
        raise AssertionError("original LLVM native entry pointer is zero")
    segment=None
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields=line.split()
        if len(fields)<2:
            continue
        left,right=map(lambda n:int(n,16),fields[0].split("-",1))
        if left<=ptr<right:
            if "r" not in fields[1] or "x" not in fields[1]:
                raise AssertionError("original native JIT pointer is not readable executable")
            segment=dict(begin=hex(left),end=hex(right),
                         permissions=fields[1])
            break
    if segment is None:
        raise AssertionError("original source LLVM pointer missing from /proc/self/maps")
    extent=min(MAX_JIT_SNAPSHOT,int(segment["end"],16)-ptr)
    if extent<0x1000:
        raise AssertionError("original JIT memory region too small for prior hot offsets")
    raw=ctypes.string_at(ptr,extent)
    if len(raw)!=extent:
        raise AssertionError("short original native executable bytes")
    md=Cs(CS_ARCH_X86,CS_MODE_64)
    md.detail=True
    listing=[]
    for inst in md.disasm(raw,ptr):
        displacement=int(inst.address)-ptr
        if displacement<0 or displacement+inst.size>len(raw):
            raise AssertionError("disassembly outside exact mapped JIT snapshot")
        members=[]
        for op in inst.operands:
            if op.type==X86_OP_MEM:
                members.append(dict(base=inst.reg_name(op.mem.base),
                                    index=inst.reg_name(op.mem.index),
                                    scale=int(op.mem.scale),
                                    displacement=int(op.mem.disp),
                                    operand_size=int(op.size)))
        listing.append(dict(
            offset=hex(displacement),address=hex(inst.address),
            size=int(inst.size),hex_bytes=inst.bytes.hex(),
            mnemonic=inst.mnemonic,operands=inst.op_str,
            memory_operands=members))
    if not listing or listing[0]["offset"]!="0x0":
        raise AssertionError("failed to decode original Numba function entry")
    offsets={int(v["offset"],16):v for v in listing}
    selected={}
    for offset in EXISTING_HOT_OFFSETS:
        inst=offsets.get(offset)
        index=listing.index(inst) if inst is not None else None
        selected[hex(offset)]=dict(
            exact_instruction_start=inst is not None,
            context=[] if index is None else listing[max(0,index-5):index+6])
    result=dict(
        original_true_native_llvm_pointer=hex(ptr),
        signature=str(signature),original_llvm_symbol=code.fndesc.llvm_func_name,
        jit_snapshot_bytes=len(raw),jit_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        original_proc_maps_segment=segment,
        python=sys.version,numba=numba.__version__,llvmlite=llvmlite.__version__,
        capstone_version=__import__("capstone").__version__,
        architecture=platform.machine(),
        raw_snapshot_filename="original_numba_jit_step.bin",
        original_source_code_only=True,
        machine_instruction_listing=listing,prior_237_hot_windows=selected)
    directory=Path(output_file).parent
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"original_numba_jit_step.bin").write_bytes(raw)
    Path(output_file).write_text(json.dumps(result,indent=2)+"\n",
                                 encoding="utf-8")
    return dict(pointer=ptr,raw=raw,len=len(raw),sha=result["jit_snapshot_sha256"],
                selected=selected)


def validate_same_native_bytes(before):
    reread=ctypes.string_at(before["pointer"],before["len"])
    if hashlib.sha256(reread).hexdigest()!=before["sha"] or reread!=before["raw"]:
        raise AssertionError("native JIT bytes changed during original fit")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=("wine","digits"),required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--policy",type=float,choices=POLICIES,required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--reference-only",action="store_true")
    p.add_argument("--asm-out",required=True,
                   help="full same-process original native JIT bytes/decoded instructions")
    args=p.parse_args()

    if args.dataset!="digits" or args.seed!=89021 or args.policy!=0.5:
        raise AssertionError("frozen single original Digits k10 p_grd=.5 only")
    if args.reference_only:
        raise AssertionError("frozen native disassembly requires genuine Linux Valgrind")
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

    native=snapshot_native_step(pcc_numba._pcc_step_numba_sequential,args.asm_out)
    if native["pointer"]!=int(
        native_entries["_pcc_step_numba_sequential"][0]["address"],16):
        raise AssertionError("raw native bytes pointer differs from Callgrind LLVM step")
    model=make_model(arrays)
    if not args.reference_only:
        invoke_command(True)
    try:
        prediction=fit(model,arrays["labels"],args.policy,count=1000)
    finally:
        if not args.reference_only:
            invoke_command(False)
    validate_same_native_bytes(native)
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
        isolated_profile=not args.reference_only,
        genuine_same_process_native_pointer=hex(native["pointer"]),
        genuine_same_process_original_machine_sha256=native["sha"],
        genuine_same_process_original_machine_bytes=native["len"],
        raw_native_machine_disassembly_saved=True)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("LIVE_ORIGINAL_NUMBA_JIT_BYTES_SOURCE_STATE_RNG_PASS",
          args.dataset,args.seed,args.policy,sys.platform,flush=True)


if __name__=="__main__":
    main()
