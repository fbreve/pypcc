"""Strict real-data NumPy continuous-observer vs unobserved propagation audit."""
import argparse
import csv
import json
import time

import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
)
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph
from pcc_numpy import pcc_propagate_numpy
from experiments.numpy_continuous_observer import (
    ContinuousNumpyObserver, MAX_BYTES, MAX_OBSERVATIONS,
    DEFAULT_MILESTONES, STATE_NAMES, digest, rng_digest, rng_equal,
)

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}
SEEDS = (12007, 12101, 12203)
POLICIES = {
    "no_stop": (6001, False, 2000),
    "real_potential": (6001, True, 2000),
    "aggressive_potential": (6001, True, 10),
}
FIELDS = (
    "runner_os","dataset","seed","policy","n_nodes","n_classes","n_particles",
    "graph_reciprocal","requested_actual_backend","baseline_first",
    "cap","es_chk","early_stop","effective_iterations","stop_reason",
    "stalled_checks","checks","stop_max",
    "all_six_final_arrays_exact","native_full_stop_info_exact",
    "complete_rng_state_exact","next_16_random_draws_exact",
    "native_last_sample_exact","final_snapshot_rng_exact",
    "expected_milestones_exact","initial_snapshot_exact",
    "snapshot_count","snapshot_iterations","snapshot_json",
    "observer_used_bytes","observer_budget_bytes",
    "baseline_fit_seconds","observed_fit_seconds",
    "observer_overhead_percent","accuracy_unlabeled",
)


class AuditedNumpyPCC(PCC):
    """Preserve normal high-level fit API; only the private dispatch is wrapped."""
    def __init__(self, *args, record=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.record = record
        self.observer = None
        self.info = None

    def _get_backend_fn(self):
        native = PCC._get_backend_fn(self)
        if native is not pcc_propagate_numpy:
            raise AssertionError(
                "This experiment requires real NumPy propagation, no fallback")

        def run(*args, **kwargs):
            if "return_info" in kwargs or "observer" in kwargs:
                raise AssertionError("Original model unexpectedly passed observer/info")
            if self.record:
                if self.observer is not None:
                    raise AssertionError("Unexpected reused instrumented model")
                self.observer = ContinuousNumpyObserver(
                    args[2], milestones=DEFAULT_MILESTONES,
                    max_observations=MAX_OBSERVATIONS,max_bytes=MAX_BYTES)
                kwargs["observer"] = self.observer
            self.info = dict(native(*args,return_info=True,**kwargs))
            return None

        # Private graph initializer examines a backend's __name__ to choose
        # layout for C-order Numba matrices. Preserve NumPy name and semantics.
        run.__name__ = native.__name__
        return run


def graph_and_labels(dataset, seed):
    data = DATASETS[dataset]()
    truth = np.asarray(data.target,dtype=np.int64)
    x = StandardScaler().fit_transform(
        np.asarray(data.data,dtype=np.float64))
    graph,degrees=build_knn_graph(x,k_nn=10)
    neighbors=[set(map(int,graph[i,:int(degrees[i])]))
               for i in range(len(truth))]
    for i,adj in enumerate(neighbors):
        if i in adj or len(adj)!=int(degrees[i]):
            raise AssertionError("Production kNN graph has self/duplicate edges")
        if any(i not in neighbors[j] for j in adj):
            raise AssertionError("Production graph is not undirected")
    labels=np.full(len(truth),-1,dtype=np.int64)
    rng=np.random.default_rng(seed)
    for cls in np.unique(truth):
        pool=np.flatnonzero(truth==cls)
        labels[rng.choice(pool,2,replace=False)]=cls
    if int(np.count_nonzero(labels>=0))!=2*len(np.unique(truth)):
        raise AssertionError("Wrong frozen labeled-per-class cohort")
    return graph,degrees,labels,truth


def run_one(graph,degrees,labels,seed,policy,record):
    cap,early,es_chk=POLICIES[policy]
    model=AuditedNumpyPCC(impl="numpy",record=record)
    model.set_graph(graph,degrees,k_nn=10)
    start=time.perf_counter()
    predictions=model.fit_predict(
        labels.copy(),max_iter=cap,early_stop=early,es_chk=es_chk,
        p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,seed=seed,
    )
    elapsed=time.perf_counter()-start
    state={
        "predictions":np.asarray(predictions).copy(),
        "positions":model.part.curnode.copy(),
        "strengths":model.part.strength.copy(),
        "distances":model.part.dist_table.copy(),
        "dominance":model.node.dominance.copy(),
        "ownership":model.owndeg.copy(),
    }
    if not np.array_equal(predictions[labels>=0],labels[labels>=0]):
        raise AssertionError("Labeled vertices changed")
    for name,a in state.items():
        if a.dtype.kind=="f" and not np.isfinite(a).all():
            raise AssertionError(f"Nonfinite post-fit {name}")
    if not isinstance(model.info,dict):
        raise AssertionError("No genuine NumPy propagation return_info")
    if not 0 <= model.info["effective_iterations"] <= cap:
        raise AssertionError("Invalid effective iteration count")
    if not early and (
            model.info["effective_iterations"]!=cap
            or model.info["checks"]!=0
            or model.info["stop_reason"]!="max_iter"):
        raise AssertionError("Unexpected no-stop termination")
    rng_after=np.random.get_state()
    subsequent_draws=np.random.random(16)
    return dict(state=state,info=model.info,observer=model.observer,
                rng_after=rng_after,subsequent=subsequent_draws,
                elapsed=elapsed,model=model)


def check_last_snapshot(result):
    observer=result["observer"]
    if observer is None:
        raise AssertionError("No continuous observer attached")
    snapshots=observer.samples
    info=result["info"]
    if (not snapshots or snapshots[0]["iteration"]!=0
            or snapshots[-1]["iteration"]!=info["effective_iterations"]
            or snapshots[-1]["final_info"]!=info
            or observer.final_info!=info):
        raise AssertionError("Missing correct initial/final trajectory records")
    expected=sorted(set(
        [i for i in DEFAULT_MILESTONES if i<=info["effective_iterations"]]
        +[info["effective_iterations"]]))
    actual=[x["iteration"] for x in snapshots]
    if expected!=actual:
        raise AssertionError(
            f"Wrong observed milestone schedule: {actual} != {expected}")
    if len(snapshots)>MAX_OBSERVATIONS or not 0<observer.used_bytes<=MAX_BYTES:
        raise AssertionError("Observer exceeded frozen byte/row budget")
    final=snapshots[-1]["state"]
    model=result["model"]
    truth_state=result["state"]
    parity={
        "positions":np.array_equal(final["positions"],truth_state["positions"]),
        "strengths":np.array_equal(final["strengths"],truth_state["strengths"]),
        "distances":np.array_equal(final["distances"],truth_state["distances"]),
        "dominance":np.array_equal(final["dominance"],truth_state["dominance"]),
    }
    mapped=np.asarray(model.mapped_labels)
    predicted=np.argmax(truth_state["dominance"],axis=1)
    predicted[mapped!=-1]=mapped[mapped!=-1]
    parity["assignments"]=np.array_equal(final["assignments"],predicted)
    own=final["raw_ownership"]
    sums=own.sum(axis=1,keepdims=True)
    sums[sums==0]=1.0
    parity["raw_ownership_normalized"]=np.array_equal(
        own/sums,truth_state["ownership"])
    if not all(parity.values()):
        raise AssertionError(f"Snapshot differs from actual complete state: {parity}")
    rng_ok=rng_equal(snapshots[-1]["rng"],result["rng_after"])
    if not rng_ok:
        raise AssertionError("Final snapshot RNG state is different from live RNG")
    initial=snapshots[0]["state"]
    home=np.flatnonzero(mapped!=-1)
    if (not np.array_equal(initial["positions"],home)
            or not np.all(initial["strengths"]==1)
            or not np.array_equal(initial["assignments"][home],mapped[home])):
        raise AssertionError("Initial observation was not before the first step")
    return actual,rng_ok


def evaluate(dataset,output,*,seeds=SEEDS,policies=None,runner_os="ubuntu-latest"):
    if policies is None:
        policies=tuple(POLICIES)
    if (not seeds or len(set(seeds))!=len(seeds) or
            not policies or len(set(policies))!=len(policies) or
            set(policies)-set(POLICIES)):
        raise ValueError("invalid frozen observer cohort")
    with open(output,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader()
        f.flush()
        for seed_idx,seed in enumerate(seeds):
            graph,degrees,labels,truth=graph_and_labels(dataset,seed)
            for policy_idx,policy in enumerate(policies):
                baseline_first=(seed_idx+policy_idx)%2==0
                order=(False,True) if baseline_first else (True,False)
                results={}
                for record in order:
                    results[record]=run_one(
                        graph,degrees,labels,seed,policy,record)
                a,b=results[False],results[True]
                array_parity=all(
                    np.array_equal(a["state"][name],b["state"][name])
                    for name in a["state"])
                info_parity=a["info"]==b["info"]
                rng_parity=rng_equal(a["rng_after"],b["rng_after"])
                next_parity=np.array_equal(a["subsequent"],b["subsequent"])
                if not all((array_parity,info_parity,rng_parity,next_parity)):
                    differing=[
                        n for n in a["state"] if not np.array_equal(
                            a["state"][n],b["state"][n])]
                    raise AssertionError(
                        f"Observer modified dynamics/RNG/stop for "
                        f"{dataset}/{seed}/{policy}: arrays={differing},"
                        f"info={a['info'],b['info']},"
                        f"rng={rng_parity},next={next_parity}")
                milestones,last_rng_ok=check_last_snapshot(b)
                observed=b["observer"]
                changed=b["state"]["predictions"][labels==-1]
                accuracy=float(np.mean(changed==truth[labels==-1]))
                overhead=100*(b["elapsed"]/a["elapsed"]-1)
                w.writerow(dict(
                    runner_os=runner_os,dataset=dataset,seed=seed,policy=policy,
                    n_nodes=len(labels),n_classes=len(np.unique(truth)),
                    n_particles=int(np.count_nonzero(labels>=0)),
                    graph_reciprocal=True,
                    requested_actual_backend="pcc_propagate_numpy",
                    baseline_first=baseline_first,
                    cap=POLICIES[policy][0],
                    es_chk=POLICIES[policy][2],
                    early_stop=POLICIES[policy][1],
                    effective_iterations=a["info"]["effective_iterations"],
                    stop_reason=a["info"]["stop_reason"],
                    stalled_checks=a["info"]["stalled_checks"],
                    checks=a["info"]["checks"],
                    stop_max=a["info"]["stop_max"],
                    all_six_final_arrays_exact=array_parity,
                    native_full_stop_info_exact=info_parity,
                    complete_rng_state_exact=rng_parity,
                    next_16_random_draws_exact=next_parity,
                    native_last_sample_exact=True,
                    final_snapshot_rng_exact=last_rng_ok,
                    expected_milestones_exact=True,
                    initial_snapshot_exact=True,
                    snapshot_count=len(milestones),
                    snapshot_iterations=json.dumps(milestones),
                    snapshot_json=observed.summarize_json(),
                    observer_used_bytes=observed.used_bytes,
                    observer_budget_bytes=MAX_BYTES,
                    baseline_fit_seconds=a["elapsed"],
                    observed_fit_seconds=b["elapsed"],
                    observer_overhead_percent=overhead,
                    accuracy_unlabeled=accuracy,
                ))
                f.flush()
                print(f"{dataset}/{seed}/{policy}: "
                      f"{a['info']['effective_iterations']} steps, "
                      f"{len(milestones)} actual snapshots, "
                      f"overhead={overhead:+.2f}%, state+RNG+stop PASS",
                      flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,choices=tuple(DATASETS))
    p.add_argument("--output",required=True)
    p.add_argument("--runner-os",choices=("ubuntu-latest","windows-latest"),default="ubuntu-latest")
    p.add_argument("--seeds",type=int,nargs="+",default=None)
    p.add_argument("--policies",nargs="+",choices=tuple(POLICIES),default=None)
    args=p.parse_args(argv)
    evaluate(args.dataset,args.output,
             seeds=tuple(args.seeds) if args.seeds is not None else SEEDS,
             policies=tuple(args.policies) if args.policies is not None else None,
             runner_os=args.runner_os)


if __name__=="__main__":
    main()
