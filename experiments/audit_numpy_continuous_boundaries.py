"""Real NumPy continuous-observer boundary tests: no seed, zero, synchronous.

Separate calls have the same *externally controlled initial* global MT19937
state, but do NOT pass a fit seed. The observer must have zero causal effect.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.audit_numpy_continuous_observer import (
    AuditedNumpyPCC, check_last_snapshot, graph_and_labels,
)
from experiments.numpy_continuous_observer import rng_equal, rng_digest

OSES=("ubuntu-latest","windows-latest")
DATASETS=("iris","digits")
SEEDS=(14003,14107,14209)
POLICIES=("unseeded","zero_iterations","synchronous")
FIELDS=(
    "runner_os","dataset","seed","policy","baseline_first",
    "num_steps","early_stop","es_chk","effective_iterations",
    "stop_reason","stop_max","checks","stalled_checks",
    "exact_six_arrays","exact_native_stop_info","exact_global_rng",
    "exact_next_16","exact_initial_observer_rng",
    "exact_final_snapshot","exact_final_rng","zero_did_not_draw",
    "actual_snapshot_iterations","expected_snapshot_iterations",
    "snapshot_count","observed_bytes","snapshot_digests",
    "baseline_seconds","observed_seconds",
)


def run_one(graph,degrees,labels,external_seed,policy,record):
    if policy=="zero_iterations":
        cap,early,es_chk,mode=0,True,10,"sequential"
    elif policy=="synchronous":
        cap,early,es_chk,mode=111,True,10,"synchronous"
    else:
        cap,early,es_chk,mode=111,True,10,"sequential"
    model=AuditedNumpyPCC(impl="numpy",record=record,update_mode=mode)
    model.set_graph(graph,degrees,k_nn=10)
    # This is the *harness's* controlled RNG state, not a fit_predict seed.
    # Fit_predict receives no seed= kwarg, exercising the default None.
    np.random.seed(external_seed)
    initial=np.random.get_state()
    from time import perf_counter
    start=perf_counter()
    prediction=model.fit_predict(
        labels.copy(),max_iter=cap,early_stop=early,es_chk=es_chk,
        p_grd=.5,delta_v=.1,deltap=1.,dexp=2.)
    elapsed=perf_counter()-start
    output={
        "predictions":np.asarray(prediction).copy(),
        "positions":model.part.curnode.copy(),
        "strengths":model.part.strength.copy(),
        "distances":model.part.dist_table.copy(),
        "dominance":model.node.dominance.copy(),
        "ownership":model.owndeg.copy(),
    }
    rng_after=np.random.get_state()
    draws=np.random.random(16)
    if not isinstance(model.info,dict):
        raise AssertionError("Missing native NumPy propagation metadata")
    if not 0<=model.info["effective_iterations"]<=cap:
        raise AssertionError("Invalid effective iterations")
    if policy=="zero_iterations" and (
            model.info["effective_iterations"]!=0 or
            model.info["checks"]!=0 or
            model.info["stop_reason"]!="max_iter"):
        raise AssertionError("A zero-step original algorithm unexpectedly moved")
    return dict(
        state=output,info=model.info,initial_rng=initial,
        final_rng=rng_after,next_draws=draws,elapsed=elapsed,
        observer=model.observer,model=model,
    )


def evaluate(dataset,osname,output):
    with open(output,"w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader()
        for idx,seed in enumerate(SEEDS):
            graph,degrees,labels,_=graph_and_labels(dataset,seed)
            for j,policy in enumerate(POLICIES):
                baseline_first=(idx+j)%2==0
                order=(False,True) if baseline_first else (True,False)
                runs={}
                for observe in order:
                    runs[observe]=run_one(graph,degrees,labels,seed,policy,observe)
                baseline,observed=runs[False],runs[True]
                arrays_equal=all(
                    np.array_equal(baseline["state"][name],
                                   observed["state"][name])
                    for name in baseline["state"])
                info_equal=baseline["info"]==observed["info"]
                rng_equal_final=rng_equal(
                    baseline["final_rng"],observed["final_rng"])
                next_equal=np.array_equal(
                    baseline["next_draws"],observed["next_draws"])
                initial_equal=rng_equal(
                    observed["observer"].samples[0]["rng"],
                    observed["initial_rng"])
                schedules,final_rng_ok=check_last_snapshot(observed)
                cap=0 if policy=="zero_iterations" else 111
                expected=sorted(set([
                    k for k in (0,1,11,101,501,2001,6001)
                    if k<=observed["info"]["effective_iterations"]
                ]+[observed["info"]["effective_iterations"]]))
                if schedules!=expected:
                    raise AssertionError("Incorrect actual checkpoint schedule")
                zero_no_draws=(policy!="zero_iterations" or (
                    rng_equal(observed["initial_rng"],
                              observed["final_rng"]) and
                    rng_equal(baseline["initial_rng"],
                              baseline["final_rng"]) and
                    schedules==[0] and
                    len(observed["observer"].samples)==1))
                if not all((
                        arrays_equal,info_equal,rng_equal_final,next_equal,
                        initial_equal,final_rng_ok,zero_no_draws)):
                    raise AssertionError(
                        f"Boundary test diverged {dataset}/{seed}/{policy}: "
                        f"states={arrays_equal}, stop={info_equal}, "
                        f"rng={rng_equal_final}, next={next_equal}, "
                        f"initial={initial_equal}, final={final_rng_ok}, "
                        f"zero={zero_no_draws}")
                obs=observed["observer"]
                if len(obs.samples)>9 or obs.used_bytes>48*1024*1024:
                    raise AssertionError("Collector exceeded memory cap")
                writer.writerow(dict(
                    runner_os=osname,dataset=dataset,seed=seed,policy=policy,
                    baseline_first=baseline_first,
                    num_steps=cap,early_stop=True,es_chk=10,
                    effective_iterations=observed["info"]["effective_iterations"],
                    stop_reason=observed["info"]["stop_reason"],
                    stop_max=observed["info"]["stop_max"],
                    checks=observed["info"]["checks"],
                    stalled_checks=observed["info"]["stalled_checks"],
                    exact_six_arrays=arrays_equal,
                    exact_native_stop_info=info_equal,
                    exact_global_rng=rng_equal_final,
                    exact_next_16=next_equal,
                    exact_initial_observer_rng=initial_equal,
                    exact_final_snapshot=True,
                    exact_final_rng=final_rng_ok,
                    zero_did_not_draw=zero_no_draws,
                    actual_snapshot_iterations=json.dumps(schedules),
                    expected_snapshot_iterations=json.dumps(expected),
                    snapshot_count=len(obs.samples),
                    observed_bytes=obs.used_bytes,
                    snapshot_digests=json.dumps([
                        {"iteration":r["iteration"],
                         "rng":r["rng_digest"],
                         "arrays":r["array_digests"]}
                        for r in obs.samples],sort_keys=True),
                    baseline_seconds=baseline["elapsed"],
                    observed_seconds=observed["elapsed"],
                ))
                file.flush()
                print(f"{osname}/{dataset}/{seed}/{policy}: "
                      "exact state/stop/full RNG/next draws/snapshots PASS",
                      flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=DATASETS,required=True)
    parser.add_argument("--runner-os",choices=OSES,required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args(argv)
    evaluate(a.dataset,a.runner_os,a.output)


if __name__=="__main__":
    main()
