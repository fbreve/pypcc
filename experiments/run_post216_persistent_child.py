"""A real persistent interpreter: 20 original PCC fits, one original graph.

No hashing, RNG probing, replay instrumentation, or new graph construction is
allowed BETWEEN the start and final checkpoint of timed fits.
"""
import time
SCRIPT_ENTRY_NS=time.perf_counter_ns()
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_frozen_numba_source(path):
    if "pcc" in sys.modules or "pcc_numba" in sys.modules:
        raise AssertionError("frozen original module must be imported before pcc")
    path=Path(path).resolve()
    if path.name!="pcc_numba.py":
        raise AssertionError("source module not a real pcc_numba.py file")
    spec=importlib.util.spec_from_file_location("pcc_numba",str(path))
    if spec is None or spec.loader is None:
        raise AssertionError("source file not importable")
    module=importlib.util.module_from_spec(spec)
    sys.modules["pcc_numba"]=module
    spec.loader.exec_module(module)
    return module,hashlib.sha256(path.read_bytes()).hexdigest()


def state_hashes(model,pred,array_hash):
    return dict(
        predictions=array_hash(pred),
        dominance=array_hash(model.node.dominance),
        positions=array_hash(model.part.curnode),
        strengths=array_hash(model.part.strength),
        distances=array_hash(model.part.dist_table),
        owndeg=array_hash(model.owndeg),
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--module",required=True)
    p.add_argument("--variant",choices=("original","production"),required=True)
    p.add_argument("--phase",choices=("cold","warm"),required=True)
    p.add_argument("--dataset",choices=("wine","breast_cancer","digits"),required=True)
    p.add_argument("--first-seed",type=int,required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()

    source_module,source_sha=load_frozen_numba_source(a.module)
    from benchmarks.fixtures import (
        REAL_DATASETS,Case,FixturePreparer,prepare_fixture,array_hash)
    from pcc import ParticleCompetitionAndCooperation as PCC
    from numba import njit
    import numpy as np

    @njit
    def next_numba_rng():
        return np.random.random()

    import_done=time.perf_counter_ns()
    nodes,features,classes,_=REAL_DATASETS[a.dataset]
    all_seeds=(70013,70117)
    if a.first_seed not in all_seeds:
        raise AssertionError("not a preregistered balanced first-mask seed")
    order=(a.first_seed,next(s for s in all_seeds if s!=a.first_seed))
    shared=FixturePreparer(prepare_fixture)
    prepared={}
    graph_start=time.perf_counter_ns()
    for seed in order:
        case=Case(a.dataset,nodes,features,classes,k_nn=10,
                  labels_per_class=3,data_seed=0,label_seed=seed)
        prepared[seed]=shared.prepare(case)
    graph_done=time.perf_counter_ns()
    first_arrays,first_meta=prepared[order[0]]
    other_arrays,other_meta=prepared[order[1]]
    if (first_meta["labeled_nodes"]!=3*classes or
            other_meta["labeled_nodes"]!=3*classes or
            not other_meta["preparation"]["reused"] or
            first_arrays["neighbors"] is not other_arrays["neighbors"] or
            first_arrays["degrees"] is not other_arrays["degrees"] or
            np.array_equal(first_arrays["labels"],other_arrays["labels"])):
        raise AssertionError("not two different masks with truly shared original graph")
    graph_hashes=[
        array_hash(first_arrays["neighbors"]),
        array_hash(first_arrays["degrees"])]
    labels_sha={str(seed):array_hash(prepared[seed][0]["labels"])
                for seed in order}
    fixtures={str(seed):prepared[seed][1]["fixture_sha256"] for seed in order}
    model_ready_start=time.perf_counter_ns()
    model=PCC(impl="numba",update_mode="sequential")
    model.set_graph(first_arrays["neighbors"],first_arrays["degrees"],k_nn=10)
    native=model._get_backend_fn()
    if native is not source_module.pcc_propagate_numba:
        raise AssertionError("model did not select frozen actual Numba source")
    original_neighbors_id=id(model.neib_list)
    original_degrees_id=id(model.neib_qt)
    model_ready=time.perf_counter_ns()

    # The whole timed region: NO SHA, replay or native RNG probes during it.
    # This uses the SAME model and the SAME installed graph for all 20 fits.
    checkpoint_ns={}
    checkpoint_cpu_ns={}
    final_pred=None
    first_fit_start=time.perf_counter_ns()
    for i in range(1,21):
        seed=order[(i-1)%2]
        final_pred=model.fit_predict(
            prepared[seed][0]["labels"],seed=0,
            p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,
            max_iter=30000,early_stop=True,es_chk=2000)
        if i in (1,2,5,10,20):
            checkpoint_ns[str(i)]=time.perf_counter_ns()
            checkpoint_cpu_ns[str(i)]=time.process_time_ns()
    fit20_done_ns=time.perf_counter_ns()
    if (id(model.neib_list)!=original_neighbors_id or
            id(model.neib_qt)!=original_degrees_id):
        raise AssertionError("20 public fits unexpectedly changed the installed graph")
    final_state_hash=state_hashes(model,final_pred,array_hash)
    final_mapping_sha=array_hash(model.mapped_labels)
    final_native_rng=float(next_numba_rng())

    # Diagnostic-only independent seeded replay occurs AFTER last measured fit.
    diagnostic=PCC(impl="numba",update_mode="sequential")
    diagnostic.set_graph(first_arrays["neighbors"],first_arrays["degrees"],k_nn=10)
    metrics={}
    def native_replay(*args,**kwargs):
        kwargs["return_info"]=True
        metrics.update(native(*args,**kwargs))
        return None
    native_replay.__name__=native.__name__
    diagnostic._get_backend_fn=lambda:native_replay
    diag_neighbors=id(diagnostic.neib_list)
    diag_degrees=id(diagnostic.neib_qt)
    outcomes=[]
    for i in range(1,21):
        label_seed=order[(i-1)%2]
        replay_pred=diagnostic.fit_predict(
            prepared[label_seed][0]["labels"],seed=0,
            p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,
            max_iter=30000,early_stop=True,es_chk=2000)
        real_states=state_hashes(diagnostic,replay_pred,array_hash)
        real_rng=float(next_numba_rng())
        layout="C" if diagnostic.node.dominance.flags.c_contiguous else "F"
        if layout!=("C" if classes>=8 else "F"):
            raise AssertionError("real Numba C/F dominance layout drift")
        native_stop={k:metrics[k] for k in (
            "effective_iterations","stop_reason","checks",
            "stalled_checks","stop_max")}
        it=native_stop["effective_iterations"]
        if (not 0<it<=30000 or
                ((it<30000)!=(native_stop["stop_reason"]=="early_stop"))):
            raise AssertionError("original native monitored stop report inconsistent")
        outcomes.append(dict(
            index=i,label_seed=label_seed,state_hashes=real_states,
            mapped_sha=array_hash(diagnostic.mapped_labels),
            numba_rng_next=real_rng,stop=native_stop,
            native_layout=layout))
    if (id(diagnostic.neib_list)!=diag_neighbors or
            id(diagnostic.neib_qt)!=diag_degrees or
            final_state_hash!=outcomes[-1]["state_hashes"] or
            final_mapping_sha!=outcomes[-1]["mapped_sha"] or
            final_native_rng!=outcomes[-1]["numba_rng_next"]):
        raise AssertionError("timed 20-fit state/RNG mismatched independent replay")

    if not (SCRIPT_ENTRY_NS<=import_done<=graph_start<graph_done<=
            model_ready_start<=model_ready<=first_fit_start<
            checkpoint_ns["1"]<=checkpoint_ns["2"]<=
            checkpoint_ns["5"]<=checkpoint_ns["10"]<=checkpoint_ns["20"]<=
            fit20_done_ns):
        raise AssertionError("invalid original application monotonic timing checkpoints")
    if any(checkpoint_cpu_ns[str(i)]<=0 for i in (1,2,5,10,20)):
        raise AssertionError("missing cumulative original process CPU")
    result=dict(
        schema=1,platform=sys.platform,variant=a.variant,phase=a.phase,
        dataset=a.dataset,first_seed=a.first_seed,mask_order=list(order),
        source_sha256=source_sha,native_backend_name=native.__name__,
        backend_identity_direct=bool(native is source_module.pcc_propagate_numba),
        real_nodes=nodes,classes=classes,k_nn=10,labels_per_class=3,
        max_iter=30000,early_stop=True,propagation_seed=0,
        threads={k:os.environ.get(k) for k in (
            "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
            "MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
        graph_hashes=graph_hashes,labels_sha=labels_sha,fixture_sha=fixtures,
        input_sha=array_hash(first_arrays["x"]),
        model_graph_stable=True,original_graph_prepared_once=True,
        imported_ns=import_done,script_entry_ns=SCRIPT_ENTRY_NS,
        graph_start_ns=graph_start,graph_done_ns=graph_done,
        model_ready_start_ns=model_ready_start,
        model_ready_ns=model_ready,first_fit_start_ns=first_fit_start,
        fit20_done_ns=fit20_done_ns,
        checkpoints_end_ns=checkpoint_ns,
        checkpoints_process_cpu_ns=checkpoint_cpu_ns,
        final_timed_state_hashes=final_state_hash,
        final_timed_mapped_sha=final_mapping_sha,
        final_timed_rng_next=final_native_rng,
        replay_outcomes=outcomes)
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("PERSISTENT_20_NATIVE_FITS_SUCCESS "
          f"{a.dataset} {a.first_seed} {a.variant} {a.phase} "
          f"20th_fit_wall_span_ms={(fit20_done_ns-first_fit_start)/1e6:.3f}",
          flush=True)


if __name__=="__main__":
    main()
