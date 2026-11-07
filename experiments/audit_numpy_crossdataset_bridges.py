"""Out-of-dataset Wine / Breast Cancer native undirected bridge experiment.

All parameters and seeds frozen in numpy_crossdataset_bridge_protocol.md.
Use unchanged original production graph builder and native NumPy dynamics.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler

from pcc_graph import build_knn_graph
from experiments.audit_numpy_undirected_bridge_disconnect import (
    arrays_from_edges,edges_from_neighbor_arrays,validate_graph,
)
from experiments.audit_numpy_undirected_multibridge import (
    DOSES,bridge_candidates,edge_digest,fit_native,
)
from experiments.audit_numpy_undirected_sparsity_mobility import verify_actual_final
from experiments.numpy_continuous_observer import rng_equal

DATASETS=("wine","breast_cancer")
SEEDS=(23111,23209,23321,23417)
ARMS=("B0","B1","B8","B32","full_union10")
NODES={"wine":178,"breast_cancer":569}
NCLASSES={"wine":3,"breast_cancer":2}
PARTITION_SIZES={"wine":[89,89],"breast_cancer":[285,284]}
CAP=100000
WINDOW=1000
K=10


def inputs(dataset):
    if dataset not in DATASETS:
        raise AssertionError("Unsupported non-preregistered dataset")
    raw=load_wine() if dataset=="wine" else load_breast_cancer()
    truth=np.asarray(raw.target,dtype=np.int64)
    x=StandardScaler().fit_transform(
        np.asarray(raw.data,dtype=np.float64))
    n=NODES[dataset]
    nc=NCLASSES[dataset]
    if (len(x)!=n or len(truth)!=n or
            not np.array_equal(np.unique(truth),np.arange(nc))):
        raise AssertionError("Unexpected sklearn dataset layout")
    parity=np.arange(n)%2
    groups=[np.flatnonzero(parity==side) for side in (0,1)]
    if [len(g) for g in groups]!=PARTITION_SIZES[dataset]:
        raise AssertionError("Unexpected parity partition sizes")
    for group in groups:
        if np.any(np.bincount(truth[group],minlength=nc)<2):
            raise AssertionError("Some class cannot be labeled on each side")
    return x,truth,groups


def select_labels(truth,seed):
    n=len(truth)
    nclasses=int(np.max(truth))+1
    rng=np.random.default_rng(seed)
    labels=np.full(n,-1,dtype=np.int64)
    parity=np.arange(n)%2
    for cls in range(nclasses):
        for side in (0,1):
            eligible=np.flatnonzero((truth==cls)&(parity==side))
            if len(eligible)<2:
                raise AssertionError("Class not present on both sides")
            labels[int(rng.choice(eligible))]=cls
    if (np.count_nonzero(labels!=-1)!=2*nclasses or
            any(np.count_nonzero((labels==cls)&(parity==side))!=1
                for cls in range(nclasses) for side in (0,1))):
        raise AssertionError("Incomplete independent component supervision")
    return labels


def build_graphs(x,truth,groups):
    n=len(x)
    split_edges=set()
    for group in groups:
        neighbors,degrees=build_knn_graph(x[group],k_nn=K)
        edges=edges_from_neighbor_arrays(neighbors,degrees)
        for u,v in edges:
            split_edges.add(tuple(sorted(
                (int(group[u]),int(group[v])))))
    split=arrays_from_edges(n,split_edges)
    actual_edges,comps,split_meta=validate_graph(*split)
    if (actual_edges!=split_edges or
            split_meta["ncomponents"]!=2 or
            split_meta["component_sizes"]!=sorted(
                (len(g) for g in groups),reverse=True) or
            split_meta["min_degree"]<K or
            any(u%2!=v%2 for u,v in split_edges) or
            any(np.any(comps[group]!=comps[group[0]])
                for group in groups) or
            comps[groups[0][0]]==comps[groups[1][0]]):
        raise AssertionError("Expected two independently connected kNN components")

    dist=cdist(x[groups[0]],x[groups[1]],metric="sqeuclidean")
    i,j=np.unravel_index(int(np.argmin(dist)),dist.shape)
    prior_first=tuple(sorted(
        (int(groups[0][i]),int(groups[1][j]))))
    bridges=bridge_candidates(x,prior_first)
    if len(bridges)!=32 or bridges[0]!=prior_first:
        raise AssertionError("Bridge ranking changed")

    graph_variants={}
    base_digest=edge_digest(split_edges)
    for b in DOSES:
        proposed=split_edges|set(bridges[:b])
        graph=split if b==0 else arrays_from_edges(n,proposed)
        e,_,meta=validate_graph(*graph)
        if (e!=proposed or
                meta["ncomponents"]!=(2 if b==0 else 1) or
                meta["component_sizes"]!=
                    (sorted((len(g) for g in groups),reverse=True)
                     if b==0 else [n]) or
                meta["min_degree"]<K or
                meta["undirected_edges"]!=len(split_edges)+b or
                meta["directed_edges"]!=
                    split_meta["directed_edges"]+2*b or
                sum(u%2!=v%2 for u,v in e)!=b):
            raise AssertionError(f"Graph nesting/connectedness violated B={b}")
        meta=dict(meta,cross_partition_edges=b,
                  controlled_bridge_dose=b)
        graph_variants[f"B{b}"]=(
            graph,meta,edge_digest(e),
            list(bridges[:b]))
    full=build_knn_graph(x,k_nn=K)
    full_edges,_,full_meta=validate_graph(*full)
    full_cross=sorted(
        e for e in full_edges if e[0]%2!=e[1]%2)
    if (full_meta["ncomponents"]!=1 or
            full_meta["component_sizes"]!=[n] or
            full_meta["min_degree"]<K or
            not full_cross):
        raise AssertionError("Ordinary production kNN full reference invalid")
    full_meta=dict(full_meta,cross_partition_edges=len(full_cross),
                   controlled_bridge_dose=None)
    graph_variants["full_union10"]=(
        full,full_meta,edge_digest(full_edges),full_cross)
    return graph_variants,bridges,base_digest


def evaluate(dataset,seed,outfile):
    x,truth,groups=inputs(dataset)
    nclasses=NCLASSES[dataset]
    graphs,bridges,base_digest=build_graphs(x,truth,groups)
    labels=select_labels(truth,seed)
    unlabelled=labels==-1
    if int(np.sum(unlabelled))!=len(truth)-2*nclasses:
        raise AssertionError("Unexpected held-out data count")
    partitions=[
        unlabelled&(np.arange(len(truth))%2==side)
        for side in (0,1)
    ]
    result=dict(
        dataset=dataset,seed=seed,cap=CAP,window=WINDOW,
        graph_family="crossdataset_official_union10_nested_parity_bridges",
        doses=list(DOSES),arms=list(ARMS),
        ordered_bridges=[list(e) for e in bridges],
        base_edge_sha256=base_digest,
        labeled_homes="one per class per parity component",
        label_positions=np.flatnonzero(~unlabelled).tolist(),
        n_classes=nclasses,n_nodes=len(truth),n_labeled=2*nclasses,
        partitions=PARTITION_SIZES[dataset],
        cases=[])
    for i,arm in enumerate(ARMS):
        (graph,degrees),structure,fingerprint,allowed=graphs[arm]
        alternate=(SEEDS.index(seed)+i+DATASETS.index(dataset))%2
        order=(False,True) if alternate==0 else (True,False)
        runs={
            observed:fit_native(
                graph,degrees,labels,seed,observed,allowed)
            for observed in order}
        baseline,observed=runs[False],runs[True]
        checks=dict(
            six_full_state_exact=all(
                np.array_equal(baseline["state"][name],
                               observed["state"][name])
                for name in baseline["state"]),
            native_info_exact=baseline["info"]==observed["info"],
            complete_rng_exact=rng_equal(
                baseline["rng"],observed["rng"]),
            next16_exact=np.array_equal(
                baseline["next16"],observed["next16"]),
        )
        if not all(checks.values()):
            raise AssertionError(
                f"Native observer changed state or RNG: {dataset}/{seed}/{arm}: {checks}")
        verify_actual_final(observed)
        m=observed["model"].audit.summary()
        acc=float(np.mean(
            baseline["state"]["predictions"][unlabelled]==
            truth[unlabelled]))
        acc_partitions=[
            float(np.mean(
                baseline["state"]["predictions"][part]==truth[part]))
            for part in partitions]
        result["cases"].append(dict(
            arm=arm,bridges=(None if arm=="full_union10" else int(arm[1:])),
            n_labeled=2*nclasses,n_particles=2*nclasses,
            n_unlabelled=int(np.sum(unlabelled)),
            structure=structure,full_edge_sha256=fingerprint,
            accuracy_unlabeled=acc,
            accuracy_unlabeled_by_partition=acc_partitions,
            parity={**checks,"true_final_low_level_exact":True},
            elapsed=dict(baseline=baseline["elapsed"],
                         observed=observed["elapsed"]),
            dynamics=m))
        Path(outfile).write_text(json.dumps(
            result,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        print(
            f"{dataset} seed {seed} {arm}: "
            f"components={structure['ncomponents']}, "
            f"cross_edges={structure['cross_partition_edges']}, "
            f"crossings={m['cross_partition_moves']}, "
            f"used_edges={m['bridges_ever_traversed']}, "
            f"locked_classes={m['operationally_locked_classes']}, "
            f"zero_move_class_windows={m['zero_move_class_windows']}, "
            f"accuracy={acc:.5f}; exact full-state/RNG PASS",
            flush=True)
    if [c["arm"] for c in result["cases"]]!=list(ARMS):
        raise AssertionError("Missing prespecified graph arm")
    c={case["arm"]:case["accuracy_unlabeled"] for case in result["cases"]}
    result["accuracy_deltas_vs_split_pp"]={
        arm:100*(c[arm]-c["B0"]) for arm in ARMS if arm!="B0"}
    Path(outfile).write_text(json.dumps(
        result,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=DATASETS,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    evaluate(args.dataset,args.seed,args.output)


if __name__=="__main__":
    main()
