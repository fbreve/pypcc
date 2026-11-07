"""Strict undirected two-component vs single-bridge Digits PCC experiment.

See the precommitted numpy_undirected_bridge_disconnect_protocol.md.
No new production graph, movement, stopping or random-number code.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph, build_graph_from_edge_index
from pcc_numpy import pcc_propagate_numpy
from experiments.numpy_continuous_observer import clone_rng, rng_equal
from experiments.audit_numpy_undirected_sparsity_mobility import (
    MobilityObserver, verify_actual_final, PARAMS, CAP,
)

SEEDS=(21107,21221,21317,21401)
VARIANTS=("full_union10","split_disconnected","single_bridge")
K=10
WINDOW=1000


def edges_from_neighbor_arrays(neighbors,degrees):
    undirected=set()
    n=len(degrees)
    for u in range(n):
        arr=neighbors[u,:int(degrees[u])]
        for v in arr:
            v=int(v)
            if u==v or not 0<=v<n:
                raise AssertionError("Invalid undirected edge index")
            undirected.add((min(u,v),max(u,v)))
    return undirected


def arrays_from_edges(n,edges):
    arr=np.array(sorted(edges),dtype=np.int64).reshape(-1,2)
    # build_graph_from_edge_index adds the symmetric arcs; no rounding
    # and no change to the production PCC builder itself.
    return build_graph_from_edge_index(n,arr.T.copy())


def validate_graph(neighbors,degrees):
    n=len(degrees)
    if neighbors.shape[0]!=n or np.min(degrees)<=0:
        raise AssertionError("Unexpected graph shape or isolated vertices")
    e=edges_from_neighbor_arrays(neighbors,degrees)
    row=[];col=[]
    for u,v in e:
        row.extend((u,v));col.extend((v,u))
    adj=csr_matrix((np.ones(len(row),dtype=np.int8),(row,col)),
                   shape=(n,n))
    ncomponents,components=connected_components(adj,directed=False)
    if np.sum(degrees)!=2*len(e):
        raise AssertionError("Reciprocity, uniqueness or graph degree violated")
    for u in range(n):
        active=neighbors[u,:int(degrees[u])]
        if len(np.unique(active))!=len(active) or u in active:
            raise AssertionError("Duplicated or looped active edge")
        for v in active:
            if (min(u,int(v)),max(u,int(v))) not in e:
                raise AssertionError("Invalid edge set")
    return e,components,dict(
        n_nodes=n,ncomponents=int(ncomponents),
        component_sizes=sorted((int(x) for x in
                                np.bincount(components,minlength=ncomponents)),
                               reverse=True),
        directed_edges=int(np.sum(degrees)),
        undirected_edges=len(e),
        mean_degree=float(np.mean(degrees)),
        min_degree=int(np.min(degrees)),
        max_degree=int(np.max(degrees)),
        no_isolates=True,reciprocal=True)


def build_variants(x,truth):
    n=len(x)
    parity=np.arange(n,dtype=np.int64)%2
    groups=[np.flatnonzero(parity==s) for s in (0,1)]
    if list(map(len,groups))!=[899,898]:
        raise AssertionError("Unexpected fixed index parity partitions")
    if any(np.any(np.bincount(truth[g],minlength=10)==0)
           for g in groups):
        raise AssertionError("Supervision class unavailable in a partition")

    full=build_knn_graph(x,k_nn=K)
    e_full,c_full,meta_full=validate_graph(*full)
    if meta_full["ncomponents"]!=1:
        raise AssertionError("Full official union10 unexpectedly disconnected")

    separate_edges=set()
    for g in groups:
        local_graph,local_degrees=build_knn_graph(x[g],k_nn=K)
        local_edges=edges_from_neighbor_arrays(local_graph,local_degrees)
        for u,v in local_edges:
            separate_edges.add(tuple(sorted((int(g[u]),int(g[v])))))
    separated=arrays_from_edges(n,separate_edges)
    e_split,c_split,meta_split=validate_graph(*separated)
    if (e_split!=separate_edges or
            meta_split["ncomponents"]!=2 or
            meta_split["component_sizes"]!=[899,898] or
            np.any(c_split[groups[0]]!=c_split[groups[0][0]]) or
            np.any(c_split[groups[1]]!=c_split[groups[1][0]]) or
            c_split[groups[0][0]]==c_split[groups[1][0]] or
            any(u%2!=v%2 for u,v in e_split) or
            meta_split["min_degree"]<K):
        raise AssertionError("Split did not yield exactly two supervised components")

    # Closest cross-partition pair by standardized Euclidean distances.
    # One well-defined edge; tie-breaking is stable row-major argmin.
    dist=cdist(x[groups[0]],x[groups[1]],metric="sqeuclidean")
    a,b=np.unravel_index(int(np.argmin(dist)),dist.shape)
    cross_edge=tuple(sorted((int(groups[0][a]),int(groups[1][b]))))
    if cross_edge in separate_edges or cross_edge[0]%2==cross_edge[1]%2:
        raise AssertionError("Bridge not a novel cross-partition edge")
    bridged_edges=separate_edges|{cross_edge}
    bridged=arrays_from_edges(n,bridged_edges)
    e_bridge,c_bridge,meta_bridge=validate_graph(*bridged)
    if (meta_bridge["ncomponents"]!=1 or
            e_bridge-e_split!={cross_edge} or
            e_split-e_bridge or
            len(e_bridge)!=len(e_split)+1 or
            meta_bridge["directed_edges"]!=meta_split["directed_edges"]+2 or
            sum(u%2!=v%2 for u,v in e_bridge)!=1):
        raise AssertionError("Bridge changed more than one undirected edge")

    return {
        "full_union10":(full,meta_full),
        "split_disconnected":(separated,meta_split),
        "single_bridge":(bridged,meta_bridge),
    },cross_edge


def labelled_nodes(truth,seed):
    labels=np.full(len(truth),-1,dtype=np.int64)
    rng=np.random.default_rng(seed)
    for cls in range(10):
        for side in (0,1):
            eligible=np.flatnonzero((truth==cls)&
                                    (np.arange(len(truth))%2==side))
            if len(eligible)<2:
                raise AssertionError("No robust class coverage across partitions")
            labels[int(rng.choice(eligible))]=cls
    if (np.sum(labels!=-1)!=20 or
            any(np.count_nonzero((labels==cls)&
                                 (np.arange(len(labels))%2==side))!=1
                for cls in range(10) for side in (0,1))):
        raise AssertionError("Missing one labelled home per class per component")
    return labels


class CrossingObserver(MobilityObserver):
    """Observe exact crossings on every native step, without modifying PCC."""
    def __init__(self,mapped,graph,degrees,variant,bridge):
        super().__init__(mapped,graph,degrees,cap=CAP,window=WINDOW)
        self.variant=variant
        self.bridge=frozenset(bridge)
        self.old_positions=None
        self.crossing_counts=np.zeros(self.nparticles,dtype=np.int64)
        self.crossing_counts_by_class=np.zeros(self.classes,dtype=np.int64)

    def capture(self,iteration,dominance,positions,strengths,distances,
                ownership,rng,info):
        if info is None:
            pos=np.asarray(positions)
            if self.old_positions is not None:
                crossing=(pos%2!=self.old_positions%2)
                if self.variant=="split_disconnected" and np.any(crossing):
                    raise AssertionError("Particle crossed disconnected partition")
                if self.variant=="single_bridge":
                    for a,b in zip(self.old_positions[crossing],pos[crossing]):
                        if frozenset((int(a),int(b)))!=self.bridge:
                            raise AssertionError("Particle crossed off the unique bridge")
                self.crossing_counts+=crossing
                self.crossing_counts_by_class+=np.bincount(
                    self.particle_classes[crossing],minlength=self.classes)
            self.old_positions=pos.copy()
        super().capture(iteration,dominance,positions,strengths,distances,
                        ownership,rng,info)

    def summary(self):
        result=super().summary()
        cross=int(np.sum(self.crossing_counts))
        if self.variant=="split_disconnected" and cross!=0:
            raise AssertionError("Disconnected graph had cross-partition moves")
        if cross!=int(np.sum(self.crossing_counts_by_class)):
            raise AssertionError("Cross-partition counters disagree")
        result.update(
            cross_partition_moves=cross,
            cross_partition_moves_by_class=self.crossing_counts_by_class.tolist(),
            cross_partition_moves_by_particle=self.crossing_counts.tolist())
        return result


class NativePCC(PCC):
    def __init__(self,record,variant,bridge):
        super().__init__(impl="numpy")
        self.record=record
        self.variant=variant
        self.bridge=bridge
        self.audit=None
        self.native_info=None

    def _get_backend_fn(self):
        native=PCC._get_backend_fn(self)
        if native is not pcc_propagate_numpy:
            raise AssertionError("Wrong native NumPy backend selected")
        def wrapped(*args,**kwargs):
            if "observer" in kwargs or "return_info" in kwargs:
                raise AssertionError("High-level model unexpectedly passed observer")
            if self.record:
                self.audit=CrossingObserver(args[2],args[0],args[1],
                                            self.variant,self.bridge)
                kwargs["observer"]=self.audit
            self.native_info=dict(native(*args,return_info=True,**kwargs))
            return None
        wrapped.__name__=native.__name__
        return wrapped


def fit(graph,degrees,labels,seed,record,variant,bridge):
    model=NativePCC(record,variant,bridge)
    model.set_graph(graph,degrees,k_nn=K)
    start=time.perf_counter()
    output=model.fit_predict(labels.copy(),seed=seed,max_iter=CAP,
                             early_stop=False,es_chk=2000,**PARAMS)
    elapsed=time.perf_counter()-start
    state=dict(
        predictions=np.asarray(output).copy(),
        positions=model.part.curnode.copy(),
        strengths=model.part.strength.copy(),
        distances=model.part.dist_table.copy(),
        dominance=model.node.dominance.copy(),
        ownership=model.owndeg.copy())
    if not np.array_equal(output[labels!=-1],labels[labels!=-1]):
        raise AssertionError("Labeled vertices changed")
    last_rng=clone_rng(np.random.get_state())
    next16=np.random.random(16)
    return dict(model=model,state=state,info=model.native_info,
                rng=last_rng,next16=next16,elapsed=elapsed)


def experiment(seed,outfile):
    digits=load_digits()
    x=StandardScaler().fit_transform(np.asarray(digits.data,dtype=np.float64))
    truth=np.asarray(digits.target,dtype=np.int64)
    graphs,bridge=build_variants(x,truth)
    labels=labelled_nodes(truth,seed)
    unknown=labels==-1
    result=dict(dataset="digits",seed=seed,cap=CAP,window=WINDOW,
                bridge_edge=list(bridge),
                graph_family="undirected_partition_parity_k10",
                labels_by_class_and_side="one per class per parity side",
                label_positions=np.flatnonzero(~unknown).tolist(),
                cases=[])
    for i,variant in enumerate(VARIANTS):
        ((graph,degrees),structure)=graphs[variant]
        order=(False,True) if (SEEDS.index(seed)+i)%2==0 else (True,False)
        runs={record:fit(graph,degrees,labels,seed,record,
                         variant,bridge)
              for record in order}
        a,b=runs[False],runs[True]
        parity={
            "six_full_state_exact":all(
                np.array_equal(a["state"][name],b["state"][name])
                for name in a["state"]),
            "native_info_exact":a["info"]==b["info"],
            "complete_rng_exact":rng_equal(a["rng"],b["rng"]),
            "next16_exact":np.array_equal(a["next16"],b["next16"]),
        }
        if not all(parity.values()):
            raise AssertionError(f"Observer changed native PCC: {seed}/{variant}/{parity}")
        # Reuses the strict low-level true-final state and full RNG verifier.
        verify_actual_final(b)
        obs=b["model"].audit
        dynamics=obs.summary()
        acc=float(np.mean(a["state"]["predictions"][unknown]==truth[unknown]))
        acc_by_side=[]
        for side in (0,1):
            m=unknown & (np.arange(len(truth))%2==side)
            acc_by_side.append(float(np.mean(
                a["state"]["predictions"][m]==truth[m])))
        result["cases"].append(dict(
            variant=variant,structure=structure,
            n_particles=20,n_labeled=20,
            accuracy_unlabeled=acc,
            accuracy_unlabeled_by_partition=acc_by_side,
            parity={**parity,"true_final_low_level_exact":True},
            elapsed=dict(baseline=a["elapsed"],observed=b["elapsed"]),
            dynamics=dynamics))
        Path(outfile).write_text(json.dumps(
            result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(f"Digits seed {seed}, {variant}: components={structure['ncomponents']}; "
              f"class_lockouts={dynamics['operationally_locked_classes']}; "
              f"zero-acceptance class-windows={dynamics['zero_move_class_windows']}; "
              f"bridge_crossings={dynamics['cross_partition_moves']}; "
              f"max_stationary={dynamics['longest_individual_streak']}; "
              f"unlabeled_accuracy={acc:.4f}; full state/RNG parity PASS",
              flush=True)
    cases={case["variant"]:case for case in result["cases"]}
    result["accuracy_bridge_minus_split_pp"]=100*(
        cases["single_bridge"]["accuracy_unlabeled"]-
        cases["split_disconnected"]["accuracy_unlabeled"])
    result["accuracy_full_minus_split_pp"]=100*(
        cases["full_union10"]["accuracy_unlabeled"]-
        cases["split_disconnected"]["accuracy_unlabeled"])
    Path(outfile).write_text(json.dumps(
        result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    experiment(a.seed,a.output)


if __name__=="__main__":
    main()
