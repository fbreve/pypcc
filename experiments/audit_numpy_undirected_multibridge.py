"""Frozen nested 0/1/8/32 reciprocal bridge-dose NumPy PCC study.

No production code changed. Observer captures actual native motion on every
physical iteration, preserving exactly the random state and final six arrays.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numpy import pcc_propagate_numpy
from experiments.numpy_continuous_observer import clone_rng, rng_equal
from experiments.audit_numpy_undirected_sparsity_mobility import (
    MobilityObserver, verify_actual_final, PARAMS, CAP, WINDOW,
)
from experiments.audit_numpy_undirected_bridge_disconnect import (
    build_variants, labelled_nodes, validate_graph,
    edges_from_neighbor_arrays, arrays_from_edges,
)

SEEDS=(22103,22229,22307,22441)
DOSES=(0,1,8,32)
MAX_BRIDGES=max(DOSES)
N=1797


def bridge_candidates(x,prior_first):
    """32 lowest cross-distances, *vertex-disjoint* endpoints; stable ties."""
    groups=(np.arange(0,len(x),2,dtype=np.int64),
            np.arange(1,len(x),2,dtype=np.int64))
    distances=cdist(x[groups[0]],x[groups[1]],metric="sqeuclidean")
    used_a=set()
    used_b=set()
    chosen=[]
    for ix in np.argsort(distances,axis=None,kind="stable"):
        a,b=np.unravel_index(int(ix),distances.shape)
        if a in used_a or b in used_b:
            continue
        used_a.add(int(a))
        used_b.add(int(b))
        u,v=int(groups[0][a]),int(groups[1][b])
        edge=(min(u,v),max(u,v))
        chosen.append(edge)
        if len(chosen)==MAX_BRIDGES:
            break
    if (len(chosen)!=MAX_BRIDGES or
            len(set(chosen))!=MAX_BRIDGES or
            set(chosen[0])!=set(prior_first) or
            len({u for u,v in chosen for _ in (0,)})!=MAX_BRIDGES or
            len({v for u,v in chosen for _ in (0,)})!=MAX_BRIDGES):
        raise AssertionError("Non-nested, unstable or repeated bridge endpoints")
    if any(u%2==v%2 for u,v in chosen):
        raise AssertionError("Bridge endpoints share the same partition")
    return chosen


def edge_digest(edges):
    values=np.array(sorted(edges),dtype="<i8").reshape(-1,2)
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_graphs(x,truth):
    prior,prior_first=build_variants(x,truth)
    split,base_meta=prior["split_disconnected"]
    base_edges=edges_from_neighbor_arrays(*split)
    bridges=bridge_candidates(x,prior_first)
    out={}
    for dose in DOSES:
        target=base_edges.union(bridges[:dose])
        graph=split if dose==0 else arrays_from_edges(len(x),target)
        edges,comp,meta=validate_graph(*graph)
        additions=edges-base_edges
        if (edges!=target or additions!=set(bridges[:dose]) or
                base_edges-edges or len(edges)!=len(base_edges)+dose or
                meta["ncomponents"]!=(2 if dose==0 else 1) or
                meta["component_sizes"]!=
                   ([899,898] if dose==0 else [N]) or
                meta["directed_edges"]!=
                    base_meta["directed_edges"]+2*dose or
                sum(u%2!=v%2 for u,v in edges)!=dose):
            raise AssertionError(f"Broken graph nesting or reciprocity, B={dose}")
        out[dose]=(graph,meta,edge_digest(edges))
    old_first_edges=edges_from_neighbor_arrays(
        *prior["single_bridge"][0])
    if edges_from_neighbor_arrays(*out[1][0])!=old_first_edges:
        raise AssertionError("One-edge arm is not exactly #185's fixed graph")
    return out,bridges,edge_digest(base_edges)


class MultiBridgeObserver(MobilityObserver):
    """Exact cross-edge events, by class, particle, 1000-step window and edge."""
    def __init__(self,mapped,graph,degrees,allowed_edges,*,cap=CAP,window=WINDOW):
        super().__init__(mapped,graph,degrees,cap=cap,window=window)
        self.allowed={tuple(sorted((int(u),int(v)))) for u,v in allowed_edges}
        if len(self.allowed)!=len(allowed_edges):
            raise AssertionError("Repeated allowed bridge")
        self.crossing_by_particle=np.zeros(self.nparticles,dtype=np.int64)
        self.crossing_by_class=np.zeros(self.classes,dtype=np.int64)
        self.window_cross=np.zeros(self.classes,dtype=np.int64)
        self.first_crossing_iteration=None
        self.edge_crossings={f"{u}-{v}":0 for u,v in sorted(self.allowed)}
        self.cross_windows=[]

    def capture(self,iteration,dominance,positions,strengths,distances,
                ownership,rng,info):
        if info is None and self.previous is not None:
            pos=np.asarray(positions)
            changed=np.flatnonzero((pos%2)!=(self.previous%2))
            for pid in changed:
                old=int(self.previous[pid])
                new=int(pos[pid])
                e=(min(old,new),max(old,new))
                if e not in self.allowed:
                    raise AssertionError(
                        f"Forbidden cross-partition accepted move: {e}")
                self.edge_crossings[f"{e[0]}-{e[1]}"]+=1
                self.crossing_by_particle[pid]+=1
                cls=int(self.particle_classes[pid])
                self.crossing_by_class[cls]+=1
                self.window_cross[cls]+=1
                if self.first_crossing_iteration is None:
                    self.first_crossing_iteration=iteration

        super().capture(iteration,dominance,positions,strengths,distances,
                        ownership,rng,info)
        if info is None and iteration>0 and iteration%self.window==0:
            self.cross_windows.append(dict(
                start=iteration-self.window,end=iteration,
                per_class=self.window_cross.tolist(),
                total=int(np.sum(self.window_cross))))
            self.window_cross[:]=0

    def summary(self):
        d=super().summary()
        total=int(np.sum(self.crossing_by_particle))
        if (total!=int(np.sum(self.crossing_by_class)) or
                total!=sum(w["total"] for w in self.cross_windows) or
                total!=sum(self.edge_crossings.values()) or
                total>d["total_accepted_moves"] or
                len(self.cross_windows)!=self.cap//self.window or
                (self.first_crossing_iteration is None)!=(total==0)):
            raise AssertionError("Crossing accounting diverged from native movement")
        if len(self.allowed)==0 and total:
            raise AssertionError("Disconnected graph leaked a particle")
        d.update(
            cross_partition_moves=total,
            cross_partition_moves_by_class=self.crossing_by_class.tolist(),
            cross_partition_moves_by_particle=self.crossing_by_particle.tolist(),
            particles_ever_crossed=int(np.count_nonzero(
                self.crossing_by_particle)),
            bridges_ever_traversed=sum(x>0 for x in self.edge_crossings.values()),
            crossing_count_by_bridge=self.edge_crossings,
            first_crossing_iteration=self.first_crossing_iteration,
            cross_partition_windows=self.cross_windows,
        )
        return d


class ObservedPCC(PCC):
    def __init__(self,observe,bridges):
        super().__init__(impl="numpy")
        self.observe=observe
        self.bridges=bridges
        self.audit=None
        self.native_info=None

    def _get_backend_fn(self):
        native=PCC._get_backend_fn(self)
        if native is not pcc_propagate_numpy:
            raise AssertionError("Wrong native backend")
        def wrapper(*args,**kwargs):
            if "observer" in kwargs or "return_info" in kwargs:
                raise AssertionError("Public API unexpectedly forwarded observer")
            if self.observe:
                self.audit=MultiBridgeObserver(
                    args[2],args[0],args[1],self.bridges)
                kwargs["observer"]=self.audit
            self.native_info=dict(native(*args,return_info=True,**kwargs))
            return None
        wrapper.__name__=native.__name__
        return wrapper


def fit_native(graph,degrees,labels,seed,observe,bridges):
    model=ObservedPCC(observe,bridges)
    model.set_graph(graph,degrees,k_nn=10)
    start=time.perf_counter()
    prediction=model.fit_predict(
        labels.copy(),seed=seed,max_iter=CAP,
        early_stop=False,es_chk=2000,**PARAMS)
    elapsed=time.perf_counter()-start
    state=dict(
        predictions=np.asarray(prediction).copy(),
        positions=model.part.curnode.copy(),
        strengths=model.part.strength.copy(),
        distances=model.part.dist_table.copy(),
        dominance=model.node.dominance.copy(),
        ownership=model.owndeg.copy())
    if not np.array_equal(state["predictions"][labels!=-1],
                          labels[labels!=-1]):
        raise AssertionError("Initial labels changed in production")
    rng=clone_rng(np.random.get_state())
    next16=np.random.random(16)
    return dict(model=model,state=state,info=model.native_info,
                rng=rng,next16=next16,elapsed=elapsed)


def evaluate(seed,outfile):
    digits=load_digits()
    x=StandardScaler().fit_transform(np.asarray(digits.data,dtype=np.float64))
    truth=np.asarray(digits.target,dtype=np.int64)
    graphs,bridges,base_digest=make_graphs(x,truth)
    labels=labelled_nodes(truth,seed)
    unknown=(labels==-1)
    masks=[unknown&(np.arange(len(labels))%2==side) for side in (0,1)]
    result=dict(dataset="digits",seed=seed,cap=CAP,window=WINDOW,
                graph_family="official_split_union10_plus_nested_disjoint_bridges",
                doses=list(DOSES),ordered_bridges=[list(e) for e in bridges],
                base_edge_sha256=base_digest,
                label_positions=np.flatnonzero(~unknown).tolist(),
                labeled_homes="one per class per parity component",
                cases=[])
    for i,dose in enumerate(DOSES):
        (graph,degrees),structure,graph_digest=graphs[dose]
        order=(False,True) if (SEEDS.index(seed)+i)%2==0 else (True,False)
        runs={enable:fit_native(graph,degrees,labels,seed,enable,
                                bridges[:dose])
              for enable in order}
        a,b=runs[False],runs[True]
        parity=dict(
            six_full_state_exact=all(
                np.array_equal(a["state"][k],b["state"][k])
                for k in a["state"]),
            native_info_exact=a["info"]==b["info"],
            complete_rng_exact=rng_equal(a["rng"],b["rng"]),
            next16_exact=np.array_equal(a["next16"],b["next16"]),
        )
        if not all(parity.values()):
            raise AssertionError(f"Instrumented/unobserved state divergence: {seed}/{dose} {parity}")
        verify_actual_final(b)
        motion=b["model"].audit.summary()
        acc=float(np.mean(a["state"]["predictions"][unknown]==truth[unknown]))
        acc_sides=[float(np.mean(
            a["state"]["predictions"][m]==truth[m])) for m in masks]
        result["cases"].append(dict(
            bridges=dose,n_labeled=20,n_particles=20,
            structure=structure,full_edge_sha256=graph_digest,
            accuracy_unlabeled=acc,accuracy_unlabeled_by_partition=acc_sides,
            parity={**parity,"true_final_low_level_exact":True},
            elapsed=dict(baseline=a["elapsed"],observed=b["elapsed"]),
            dynamics=motion))
        Path(outfile).write_text(json.dumps(
            result,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        print(f"Digits seed {seed}, B={dose}: crossings="
              f"{motion['cross_partition_moves']}, active_bridges="
              f"{motion['bridges_ever_traversed']}/{dose}, "
              f"class_lockouts={motion['operationally_locked_classes']}, "
              f"max_still={motion['longest_individual_streak']}, "
              f"accuracy={acc:.5f}; exact state/RNG parity PASS",flush=True)
    if [case["bridges"] for case in result["cases"]]!=list(DOSES):
        raise AssertionError("Incomplete fixed bridge-dose set")
    Path(outfile).write_text(json.dumps(
        result,indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    evaluate(args.seed,args.output)


if __name__=="__main__":
    main()
