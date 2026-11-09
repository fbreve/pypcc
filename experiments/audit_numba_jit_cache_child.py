"""One fresh Python interpreter's original-or-cache Numba first and hot fit.

DO NOT import an experiment-proposed kernel into production, and DO NOT time
fixture construction, imports, JIT compilation unrelated to the PCC call, or
sample RNG after the first fit within the first-fit timer.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from numba import njit
from benchmarks.fixtures import Case, REAL_DATASETS, prepare_fixture, array_hash
from pcc import ParticleCompetitionAndCooperation as PCC


@njit
def next_original_numba_rng_draw():
    return np.random.random()


def load_original_source_module(path):
    path=Path(path)
    spec=importlib.util.spec_from_file_location(path.stem,str(path))
    if spec is None or spec.loader is None:
        raise AssertionError("not a real file backed Python module")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if module.pcc_propagate_numba.__name__!="pcc_propagate_numba":
        raise AssertionError("production Numba source function name altered")
    return module


def native_fit(arrays,backend,classes,cap):
    model=PCC(impl="numba")
    model.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
    actual=backend.pcc_propagate_numba
    metrics={}

    def original_native_info(*args,**kwargs):
        kwargs["return_info"]=True
        observed=actual(*args,**kwargs)
        metrics.update(observed)
        return None

    original_native_info.__name__="pcc_propagate_numba"
    model._get_backend_fn=lambda:original_native_info

    wall_before=time.perf_counter_ns()
    cpu_before=time.process_time_ns()
    predicted=model.fit_predict(
        arrays["labels"],max_iter=cap,early_stop=True,seed=0,
        p_grd=.5,delta_v=.1,deltap=1.0,dexp=2.0,es_chk=2000)
    cpu_ns=time.process_time_ns()-cpu_before
    wall_ns=time.perf_counter_ns()-wall_before

    arrays_to_check=dict(
        predictions=predicted,dominance=model.node.dominance,
        positions=model.part.curnode,strengths=model.part.strength,
        distances=model.part.dist_table,owndeg=model.owndeg)
    hashes={k:array_hash(v) for k,v in arrays_to_check.items()}
    got_layout="C" if model.node.dominance.flags.c_contiguous else "F"
    want_layout="C" if classes>=8 else "F"
    if got_layout!=want_layout:
        raise AssertionError("true native Numba dominance layout changed")
    if metrics.get("effective_iterations")!=cap or metrics.get("stop_reason")!="max_iter":
        raise AssertionError("short native no-stop compilation sample changed")
    if cpu_ns < 0 or wall_ns <= 0:
        raise AssertionError("negative CPU or absent wall time; Windows sub-ms CPU may round to zero")
    rng_next=float(next_original_numba_rng_draw())
    if not 0<=rng_next<1:
        raise AssertionError("invalid Numba next native RNG sample")
    return dict(cpu_ns=cpu_ns,wall_ns=wall_ns,
                state_hashes=hashes,
                native_rng_next=rng_next,
                stop_info=dict(
                    (name,metrics[name]) for name in (
                        "effective_iterations","stop_reason",
                        "checks","stalled_checks","stop_max")),
                layout=got_layout,mode="early_stopping",
                iteration_cap=cap,
                original_backend_name=actual.__name__)


HOT_BATCH_REPETITIONS=512


def high_resolution_already_hot_cpu(arrays,backend,classes,expected,cap=300):
    """Amortize coarse Windows CPU clock ticks across 512 real native fits."""
    model=PCC(impl="numba")
    model.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
    source_fn=backend.pcc_propagate_numba
    observed={}

    def same_native_fn(*args,**kwargs):
        kwargs["return_info"]=True
        observed.update(source_fn(*args,**kwargs))
        return None

    same_native_fn.__name__="pcc_propagate_numba"
    model._get_backend_fn=lambda:same_native_fn

    before_cpu=time.process_time_ns()
    before_wall=time.perf_counter_ns()
    for _ in range(HOT_BATCH_REPETITIONS):
        predicted=model.fit_predict(
            arrays["labels"],max_iter=cap,early_stop=True,seed=0,
            p_grd=.5,delta_v=.1,deltap=1.0,dexp=2.0,es_chk=2000)
    elapsed_wall=time.perf_counter_ns()-before_wall
    elapsed_cpu=time.process_time_ns()-before_cpu
    if elapsed_cpu<=0 or elapsed_wall<=0:
        raise AssertionError("512 original Numba fits still lack measurable CPU time")
    hashes={k:array_hash(v) for k,v in dict(
        predictions=predicted,dominance=model.node.dominance,
        positions=model.part.curnode,strengths=model.part.strength,
        distances=model.part.dist_table,owndeg=model.owndeg).items()}
    actual_rng=float(next_original_numba_rng_draw())
    stop_info=dict(
        (key,observed[key]) for key in (
            "effective_iterations","stop_reason","checks",
            "stalled_checks","stop_max"))
    if (hashes!=expected["state_hashes"] or
            actual_rng!=expected["native_rng_next"] or
            stop_info!=expected["stop_info"]):
        raise AssertionError("reused-model high-resolution hot batch changed original native PCC")
    layout="C" if model.node.dominance.flags.c_contiguous else "F"
    if layout!=expected["layout"]:
        raise AssertionError("hot batch changed native C/F memory layout")
    return dict(count=HOT_BATCH_REPETITIONS,cpu_ns=elapsed_cpu,
                wall_ns=elapsed_wall,last_native_state_hashes=hashes,
                last_numba_rng_next=actual_rng,last_stop_info=stop_info)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--module",required=True)
    p.add_argument("--dataset",choices=("wine","breast_cancer","digits"),
                   required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--variant",choices=("original","candidate"),required=True)
    p.add_argument("--phase",choices=("cold","fresh_reuse"),required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()

    source=Path(a.module).read_bytes()
    source_sha=hashlib.sha256(source).hexdigest()
    backend=load_original_source_module(a.module)
    nodes,features,classes,_=REAL_DATASETS[a.dataset]
    case=Case(a.dataset,nodes,features,classes,
              k_nn=10,labels_per_class=3,data_seed=0,label_seed=a.seed)
    arrays,meta=prepare_fixture(case)
    # First call below must really be the first native PCC compilation/load
    # for this unique module path in this new Python interpreter.
    first=native_fit(arrays,backend,classes,cap=300)
    # Hot call: all native specializations have already been compiled/loaded.
    second=native_fit(arrays,backend,classes,cap=300)
    batch=high_resolution_already_hot_cpu(arrays,backend,classes,first,cap=300)
    for field in ("state_hashes","native_rng_next","stop_info","layout"):
        if first[field]!=second[field]:
            raise AssertionError("same native module repeated seeded fits changed "+
                                 repr(field))
    output=dict(
        schema=1,variant=a.variant,phase=a.phase,
        dataset=a.dataset,seed=a.seed,case_id=case.case_id,
        module_sha256=source_sha,platform=sys.platform,
        host=platform.platform(),python=sys.version,
        numpy=np.__version__,numba=__import__("numba").__version__,
        fixture_sha=meta["fixture_sha256"],
        input_sha=array_hash(arrays["x"]),
        ordered_graph_sha=[
            array_hash(arrays["neighbors"]),array_hash(arrays["degrees"])],
        labels_sha=array_hash(arrays["labels"]),
        real_nodes=nodes,classes=classes,
        first=first,hot=second,hot_batch=batch)
    Path(a.out).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print(f"NATIVE-FIRST-FIT {a.variant} {a.phase} {case.case_id}: "
          f"first process CPU {first['cpu_ns']/1e6:.3f} ms; "
          f"hot CPU (single) {second['cpu_ns']/1e6:.3f} ms; "
          f"hot batch per fit CPU {batch['cpu_ns']/batch['count']/1e6:.3f} ms",
          flush=True)


if __name__=="__main__":
    main()
