"""Phase-4 preregistered all-iteration observation, 0..111, without altered PCC dynamics."""
import argparse
import csv
import json
import time

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numpy import pcc_propagate_numpy
from experiments.audit_numpy_continuous_observer import graph_and_labels
from experiments.numpy_continuous_observer import ContinuousNumpyObserver, MAX_BYTES, rng_equal

SEEDS=(16007,16111)
MILESTONES=tuple(range(112))
FIELDS=("os","dataset","seed","arrays_exact","stop_exact","rng_exact",
        "next16_exact","final_snapshot_exact","dense_schedule_exact",
        "row_count","effective_iterations","used_bytes","max_bytes",
        "baseline_seconds","observed_seconds","overhead_percent")


class DensePCC(PCC):
    def __init__(self,record):
        super().__init__(impl="numpy")
        self.record=record
        self.observer=None
        self.info=None

    def _get_backend_fn(self):
        native=PCC._get_backend_fn(self)
        if native is not pcc_propagate_numpy:
            raise AssertionError("native NumPy backend required")

        def propagate(*args,**kwargs):
            if "observer" in kwargs or "return_info" in kwargs:
                raise AssertionError("Unexpected nonnative fit hook")
            if self.record:
                self.observer=ContinuousNumpyObserver(
                    args[2],milestones=MILESTONES,
                    max_observations=len(MILESTONES),max_bytes=MAX_BYTES)
                kwargs["observer"]=self.observer
            self.info=dict(native(*args,return_info=True,**kwargs))
            return None

        propagate.__name__=native.__name__
        return propagate


def fit(graph,degrees,labels,seed,record):
    model=DensePCC(record)
    model.set_graph(graph,degrees,k_nn=10)
    start=time.perf_counter()
    pred=model.fit_predict(labels.copy(),seed=seed,max_iter=111,
                           early_stop=False,es_chk=10,
                           p_grd=.5,delta_v=.1,deltap=1.,dexp=2.)
    seconds=time.perf_counter()-start
    state=dict(
        predictions=np.asarray(pred).copy(),
        positions=model.part.curnode.copy(),
        strengths=model.part.strength.copy(),
        distances=model.part.dist_table.copy(),
        dominance=model.node.dominance.copy(),
        ownership=model.owndeg.copy())
    rng=np.random.get_state()
    after=np.random.random(16)
    return dict(state=state,info=model.info,rng=rng,after=after,
                model=model,seconds=seconds)


def check_final(result):
    model=result["model"]
    obs=model.observer
    if obs is None:
        raise AssertionError("Missing dense observer")
    rows=obs.samples
    schedule=[row["iteration"] for row in rows]
    if schedule!=list(MILESTONES) or len(rows)!=112:
        raise AssertionError("Missing/duplicated dense milestone")
    if obs.used_bytes<=0 or obs.used_bytes>MAX_BYTES:
        raise AssertionError("Retained budget exceeded")
    if obs.final_info!=result["info"] or rows[-1]["final_info"]!=result["info"]:
        raise AssertionError("Incorrect actual terminal metadata")
    final=rows[-1]["state"]
    actual=result["state"]
    mapped=model.mapped_labels
    predicted=np.argmax(actual["dominance"],axis=1)
    predicted[mapped!=-1]=mapped[mapped!=-1]
    sums=final["raw_ownership"].sum(axis=1,keepdims=True)
    sums[sums==0]=1.
    checks=(
        np.array_equal(final["assignments"],predicted),
        np.array_equal(final["positions"],actual["positions"]),
        np.array_equal(final["strengths"],actual["strengths"]),
        np.array_equal(final["distances"],actual["distances"]),
        np.array_equal(final["dominance"],actual["dominance"]),
        np.array_equal(final["raw_ownership"]/sums,actual["ownership"]),
        rng_equal(rows[-1]["rng"],result["rng"]),
    )
    if not all(checks):
        raise AssertionError(f"Dense final state/RNG incorrect: {checks}")
    if result["info"]["effective_iterations"]!=111:
        raise AssertionError("Unexpected stopping in no-stop dense cohort")
    return obs.used_bytes


def evaluate(dataset,os_name,output):
    with open(output,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        for j,seed in enumerate(SEEDS):
            graph,degrees,labels,_=graph_and_labels(dataset,seed)
            order=(False,True) if j%2==0 else (True,False)
            result={record:fit(graph,degrees,labels,seed,record) for record in order}
            baseline,observed=result[False],result[True]
            arrays=all(np.array_equal(baseline["state"][k],observed["state"][k])
                       for k in baseline["state"])
            stop=baseline["info"]==observed["info"]
            rng=rng_equal(baseline["rng"],observed["rng"])
            next16=np.array_equal(baseline["after"],observed["after"])
            used=check_final(observed)
            if not all((arrays,stop,rng,next16)):
                raise AssertionError("High-frequency observer changed final state/RNG/stop")
            overhead=100*(observed["seconds"]/baseline["seconds"]-1)
            writer.writerow(dict(
                os=os_name,dataset=dataset,seed=seed,arrays_exact=arrays,
                stop_exact=stop,rng_exact=rng,next16_exact=next16,
                final_snapshot_exact=True,dense_schedule_exact=True,
                row_count=112,effective_iterations=111,used_bytes=used,
                max_bytes=MAX_BYTES,baseline_seconds=baseline["seconds"],
                observed_seconds=observed["seconds"],overhead_percent=overhead))
            f.flush()
            print(f"{os_name}/{dataset}/{seed}: 112/112 rows, "
                  f"{used}/{MAX_BYTES} bytes, overhead {overhead:+.2f}%, "
                  "state+RNG+stop exact",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,choices=("iris","digits"))
    p.add_argument("--runner-os",required=True,choices=("ubuntu-latest","windows-latest"))
    p.add_argument("--output",required=True)
    a=p.parse_args()
    evaluate(a.dataset,a.runner_os,a.output)


if __name__=="__main__":
    main()
