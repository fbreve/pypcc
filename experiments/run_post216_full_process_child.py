"""Real first full data-to-prediction PCC workflow in a FRESH Python process.

Parent monotonic clock begins before interpreter creation. The first output
timestamp is sampled IMMEDIATELY after unmodified public fit_predict returns.
Any diagnostic hashing/replay/RNG/checking is strictly AFTER that time boundary.
"""
import time
SCRIPT_ENTRY_PERF_NS=time.perf_counter_ns()
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

def initialize_source(path):
    """Load the exact pinned pcc_numba module BEFORE importing production pcc."""
    if "pcc" in sys.modules or "pcc_numba" in sys.modules:
        raise RuntimeError("backend source control must precede all pcc imports")
    source=Path(path).resolve()
    if source.name!="pcc_numba.py" or not source.is_file():
        raise RuntimeError("not an ordinary filesystem-backed pcc_numba module")
    spec=importlib.util.spec_from_file_location("pcc_numba",str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import original Numba backend from real source")
    module=importlib.util.module_from_spec(spec)
    sys.modules["pcc_numba"]=module
    spec.loader.exec_module(module)
    return module,hashlib.sha256(source.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--variant",choices=("original","production"),required=True)
    parser.add_argument("--phase",choices=("cold","warm1","warm2","warm3"),
                        required=True)
    parser.add_argument("--dataset",choices=("wine","breast_cancer","digits"),
                        required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--result",required=True)
    a=parser.parse_args()

    module,source_sha=initialize_source(a.source)
    from benchmarks.fixtures import (
        REAL_DATASETS,Case,prepare_fixture,array_hash)
    from pcc import ParticleCompetitionAndCooperation as PCC
    from numba import njit
    import numpy as np

    @njit
    def peek_next_numba_rng():
        return np.random.random()

    imported_ns=time.perf_counter_ns()
    nodes,features,classes,_=REAL_DATASETS[a.dataset]
    case=Case(a.dataset,nodes,features,classes,k_nn=10,
              labels_per_class=3,data_seed=0,label_seed=a.seed)

    prepared_start=time.perf_counter_ns()
    arrays,metadata=prepare_fixture(case)
    prepared_end=time.perf_counter_ns()
    if metadata["labeled_nodes"]!=3*classes or (
        arrays["neighbors"].dtype!=np.int64 or
        arrays["neighbors"].ndim!=2):
        raise AssertionError("wrong real original dense graph and particle labels")

    model_start=time.perf_counter_ns()
    model=PCC(impl="numba",update_mode="sequential")
    model.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
    native_backend=model._get_backend_fn()
    if native_backend is not module.pcc_propagate_numba:
        raise RuntimeError("not the original high-level production Numba dispatch")
    model_ready=time.perf_counter_ns()

    # This is the FIRST Numba propagation in the interpreter and NO extra
    # instrumentation, model monkeypatch or RNG probe runs beforehand.
    fit_start_cpu=time.process_time_ns()
    fit_start=time.perf_counter_ns()
    pred=model.fit_predict(
        arrays["labels"],max_iter=30000,early_stop=True,seed=0,
        p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)
    # Time-to-first-prediction absolute endpoint shared with parent clock.
    fitted_ns=time.perf_counter_ns()
    fit_cpu_end_ns=time.process_time_ns()

    if not (SCRIPT_ENTRY_PERF_NS<=imported_ns<=prepared_start<=prepared_end
            <=model_start<=model_ready<=fit_start<fitted_ns):
        raise AssertionError("invalid original application phase timestamp order")
    if fit_cpu_end_ns<=fit_start_cpu:
        raise AssertionError("no actual original fit process CPU")
    if fit_cpu_end_ns<=0:
        raise AssertionError("missing total Python-process CPU time")

    # Diagnostics strictly OUTSIDE the production app time-to-result.
    current_state=dict(
        predictions=pred,dominance=model.node.dominance,
        positions=model.part.curnode,strengths=model.part.strength,
        distances=model.part.dist_table,owndeg=model.owndeg)
    states={k:array_hash(v) for k,v in current_state.items()}
    mapped_sha=array_hash(model.mapped_labels)
    native_rng_endpoint=float(peek_next_numba_rng())
    layout="C" if model.node.dominance.flags.c_contiguous else "F"
    if layout!=("C" if classes>=8 else "F"):
        raise AssertionError("Numba genuine C/F class dominance layout changed")

    # Independent diagnostic replay starts AFTER the first-result endpoint
    # to learn the actual stop iteration/checks WITHOUT a wrapper on the
    # timed, unmodified production fit.
    replay=PCC(impl="numba",update_mode="sequential")
    replay.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
    info={}
    def replay_observer(*args,**kwargs):
        kwargs["return_info"]=True
        info.update(native_backend(*args,**kwargs))
        return None
    replay_observer.__name__=native_backend.__name__
    replay._get_backend_fn=lambda:replay_observer
    again=replay.fit_predict(
        arrays["labels"],max_iter=30000,early_stop=True,seed=0,
        p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)
    replay_states={
        "predictions":again,"dominance":replay.node.dominance,
        "positions":replay.part.curnode,"strengths":replay.part.strength,
        "distances":replay.part.dist_table,"owndeg":replay.owndeg,
    }
    replay_hashes={k:array_hash(v) for k,v in replay_states.items()}
    replay_rng=float(peek_next_numba_rng())
    if states!=replay_hashes or mapped_sha!=array_hash(replay.mapped_labels):
        raise AssertionError("unmodified original public fit != diagnostic replay states")
    if native_rng_endpoint!=replay_rng:
        raise AssertionError("Numba original RNG endpoint changed on diagnostic replay")
    if not 0<info["effective_iterations"]<=30000:
        raise AssertionError("invalid actual original early stop")
    if info["stop_reason"] not in ("max_iter","early_stop"):
        raise AssertionError("invalid native stop reason")
    if info["effective_iterations"]<30000 and info["stop_reason"]!="early_stop":
        raise AssertionError("unexpected native truncation")
    if info["effective_iterations"]==30000 and info["stop_reason"]!="max_iter":
        raise AssertionError("invalid full-cap reason")
    observed={k:info[k] for k in (
        "effective_iterations","stop_reason","checks","stalled_checks",
        "stop_max")}
    phase=dict(
        entry_ns=SCRIPT_ENTRY_PERF_NS,imported_ns=imported_ns,
        prep_start_ns=prepared_start,prep_end_ns=prepared_end,
        model_start_ns=model_start,model_ready_ns=model_ready,
        fit_start_ns=fit_start,fit_end_ns=fitted_ns,
        total_cpu_to_first_fit_ns=fit_cpu_end_ns,
        fit_cpu_ns=fit_cpu_end_ns-fit_start_cpu,
        import_within_script_ns=imported_ns-SCRIPT_ENTRY_PERF_NS,
        prep_ns=prepared_end-prepared_start,
        graph_build_ns=int(metadata["timings"]["graph_build_s"]*1e9),
        data_load_ns=int(metadata["timings"]["dataset_load_s"]*1e9),
        preprocessing_ns=int(
            metadata["timings"]["preprocessing_and_labels_s"]*1e9),
        model_and_set_graph_ns=model_ready-model_start,
        fit_wall_ns=fitted_ns-fit_start)
    output=dict(
        schema=1,variant=a.variant,phase=a.phase,platform=sys.platform,
        source_sha256=source_sha,backend_module="pcc_numba",
        true_original_pcc_dispatch=True,
        dataset=a.dataset,seed=a.seed,case_id=case.case_id,
        real_nodes=nodes,classes=classes,
        threads={x:os.environ.get(x) for x in (
            "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
            "MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
        host=platform.platform(),python=sys.version,
        numpy=np.__version__,numba=__import__("numba").__version__,
        fixture_sha=metadata["fixture_sha256"],
        graph_hashes=[
            array_hash(arrays["neighbors"]),array_hash(arrays["degrees"])],
        input_hash=array_hash(arrays["x"]),label_hash=array_hash(arrays["labels"]),
        real_k_nn=10,labels_per_class=3,propagation_seed=0,
        full_fit_max_iter=30000,early_stop=True,
        phase_timing=phase,
        state_hashes=states,mapped_labels_hash=mapped_sha,
        native_rng_endpoint=native_rng_endpoint,native_stop=observed)
    Path(a.result).write_text(
        json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"WHOLE_PROGRAM_NATIVE_RESULT {case.case_id} {a.variant} {a.phase} "
          f"fit={phase['fit_wall_ns']/1e6:.3f}ms "
          f"original_real_stop={observed['effective_iterations']} "
          f"states_rng_pass",flush=True)


if __name__=="__main__":
    main()
