"""One original native Numba Digits PCC fit with READ-ONLY GDB address sampling.

Only a research-level Python backend wrapper exports actual NumPy allocations.
No original kernel, motion, data, RNG, graph or stopping change. No CPU timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys

import numpy as np
import pcc_numba
from benchmarks.fixtures import Case, REAL_DATASETS, prepare_fixture
from experiments import profile_numba_instruction_case as orig

HOT = {
    "0x36b": {"bytes": "4d8b0436", "allocation": "neib_list", "bytes_read": 8},
    "0x3a0": {"bytes": "c4a17b1004c9", "allocation": "dominance", "bytes_read": 8},
    "0x3a6": {"bytes": "460fb60402", "allocation": "dist_table", "bytes_read": 1},
}
ARG_NAMES = (
    "neib_list","neib_qt","labels","p_grd","delta_v","c","zerovec",
    "part_curnode","part_label","part_strength","dist_table","dominance",
    "owndeg","deltap","dexp","dom_row","reduc","dom_list","dist_list","prob",
    "slices","dist_weights","max_iter","early_stop","es_chk","stop_max"
)


def snapshot_native(args, folder, sha):
    dispatcher = pcc_numba._pcc_step_numba_sequential
    overloads = list(dispatcher.overloads.items())
    if len(overloads) != 1:
        raise AssertionError("not one exact warmed original native step signature")
    signature, compiled = overloads[0]
    pointer = compiled.library.get_pointer_to_function(compiled.fndesc.llvm_func_name)
    if not isinstance(pointer, int) or pointer <= 0:
        raise AssertionError("original step LLVM native pointer unavailable")
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise AssertionError("this frozen hot opcode probe is x86-64 only")
    with open("/proc/self/mem", "rb", buffering=0) as mem:
        found = {}
        for off, item in HOT.items():
            offset = int(off, 16)
            raw = os.pread(mem.fileno(), len(bytes.fromhex(item["bytes"])), pointer + offset)
            if raw.hex() != item["bytes"]:
                raise AssertionError("original live opcode differs at "+off+
                                     ": expected "+item["bytes"]+" got "+raw.hex())
            found[off] = raw.hex()
    if len(args) < len(ARG_NAMES):
        raise AssertionError("source-exact original backend args not present")
    allocations = {}
    for name, value in zip(ARG_NAMES, args):
        if not isinstance(value, np.ndarray):
            continue
        if not (value.flags.c_contiguous or value.flags.f_contiguous):
            raise AssertionError("noncontiguous runtime kernel array: "+name)
        ptr = int(value.__array_interface__["data"][0])
        if ptr <= 0 or value.nbytes <= 0:
            raise AssertionError("invalid native runtime buffer: "+name)
        allocations[name] = {
            "data_ptr": hex(ptr), "end_ptr": hex(ptr+int(value.nbytes)),
            "dtype": str(value.dtype), "shape": list(value.shape),
            "strides": list(value.strides), "itemsize": value.itemsize,
            "nbytes": int(value.nbytes), "C": bool(value.flags.c_contiguous),
            "F": bool(value.flags.f_contiguous),
        }
    for left, a in allocations.items():
        for right, b in allocations.items():
            if left >= right:
                continue
            a0, a1 = int(a["data_ptr"],16), int(a["end_ptr"],16)
            b0, b1 = int(b["data_ptr"],16), int(b["end_ptr"],16)
            if a0 < b1 and b0 < a1:
                raise AssertionError("aliased allocations prevent unique attribution: "+
                                     left+" "+right)
    assert allocations["neib_list"]["dtype"] == "int64"
    assert allocations["dominance"]["dtype"] == "float64"
    assert allocations["dist_table"]["dtype"] == "uint8"
    assert allocations["dominance"]["C"] is True
    assert allocations["dist_table"]["F"] is True
    assert allocations["neib_list"]["C"] is True
    return {
        "schema":1, "original_master":orig.BASE, "source_sha":sha,
        "llvm_entry":hex(pointer), "llvm_symbol":compiled.fndesc.llvm_func_name,
        "numba_step_signature":str(signature), "hot_machine_bytes":found,
        "hot_expectations":HOT, "allocations":allocations,
        "original_case":"digits-k10-labels89021-p0.5", "native_gdb_only":True,
        "not_a_cpu_benchmark":True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    options = parser.parse_args()
    if sys.platform != "linux" or not os.environ.get("PCC_NATIVE_GDB_PROBE"):
        raise AssertionError("requires one native Linux GDB-controlled genuine fit")
    folder = Path(options.out).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if str(folder) != os.environ.get("PCC_NATIVE_GDB_PROBE"):
        raise AssertionError("GDB and native process do not share exact artifact directory")
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS","NUMBA_ENABLE_PROFILING","NUMBA_DEBUGINFO"):
        if os.environ.get(name) != "1":
            raise AssertionError("original native Numba environment was not frozen: "+name)
    guard = orig.guard_production_files()
    n,features,classes,_ = REAL_DATASETS["digits"]
    case=Case("digits",n,features,classes,k_nn=10,labels_per_class=3,
              data_seed=0,label_seed=89021)
    arrays,meta=prepare_fixture(case)
    # Recheck original reciprocal undirected (NOT directed) graph.
    neib=arrays["neighbors"]; degree=arrays["degrees"]
    for i,k in enumerate(degree):
        if any(i not in neib[j,:degree[j]] for j in neib[i,:k]):
            raise AssertionError("not the true original reciprocal undirected graph")
    warm=orig.make_model(arrays)
    orig.fit(warm, arrays["labels"], .5, count=2)
    actual=orig.make_model(arrays)
    original_fn=pcc_numba.pcc_propagate_numba
    entered=[]

    def capture_only(*args,**kwargs):
        if entered:
            raise AssertionError("multiple original propagation calls during single fit")
        entered.append(True)
        metadata=snapshot_native(args,folder,guard)
        (folder/"native_live_array_ranges.json").write_text(
            json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print("GDB_ARRAY_RANGES_READY", metadata["llvm_entry"], flush=True)
        # The parent GDB inspects only native registers at actual opcode breaks.
        os.kill(os.getpid(),signal.SIGTRAP)
        return original_fn(*args,**kwargs)

    # Original pcc.py examines backend __name__ for C-order dominance selection.
    capture_only.__name__="pcc_propagate_numba"
    actual._get_backend_fn=lambda:capture_only
    actual_pred=orig.fit(actual,arrays["labels"],.5,count=1000)
    if len(entered)!=1:
        raise AssertionError("original native full-fit backend interception missing")
    observed=orig.hash_state(actual,actual_pred)
    native_rng=orig.original_next_numba_rng()
    ref=orig.make_model(arrays)
    details={}
    def state_replay(*args,**kwargs):
        kwargs["return_info"]=True
        details.update(original_fn(*args,**kwargs))
        return None
    state_replay.__name__="pcc_propagate_numba"
    ref._get_backend_fn=lambda:state_replay
    ref_pred=orig.fit(ref,arrays["labels"],.5,count=1000)
    second=orig.hash_state(ref,ref_pred)
    second_rng=orig.original_next_numba_rng()
    if observed!=second or native_rng!=second_rng:
        raise AssertionError("GDB native fit changed original eight-state or MT19937")
    if (details["effective_iterations"] != 1000 or
        details["stop_reason"] != "max_iter" or details["checks"] != 0):
        raise AssertionError("original 1000 fixed iterations not preserved")
    output={
       "schema":1, "base":orig.BASE, "source_hashes":guard,
       "dataset":"digits","seed":89021,"policy":.5,"steps":1000,
       "state_hashes":observed, "rng_next":float(native_rng),
       "original_stop":{k:details[k] for k in
                        ("effective_iterations","stop_reason","checks","stalled_checks","stop_max")},
       "graph_sha":[orig.array_hash(neib),orig.array_hash(degree)],
       "labels_sha":orig.array_hash(arrays["labels"]),
       "fixture_sha":meta["fixture_sha256"],
       "native_layout":"C","owndeg_f_layout":True,"distances_uint8":True,
       "original_dense_padded_int64":True,"full_fit_under_native_gdb":True,
       "state_and_rng_replayed_without_breakpoints":True
    }
    (folder/"numba_native_gdb_full_fit_source.json").write_text(
         json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("GDB_NATIVE_ORIGINAL_FIT_EIGHT_STATE_RNG_STOP_PASS",flush=True)


if __name__ == "__main__":
    main()
