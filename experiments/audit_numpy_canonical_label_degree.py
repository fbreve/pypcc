"""Canonical *uncut undirected union10* degree-stratified PCC label-home study.

See the protocol precommitted before outcomes. Entire model graph fixed across
label strategies and seeds. Native propagation and observers are unchanged.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler

from pcc_graph import build_knn_graph
from experiments.audit_numpy_undirected_bridge_disconnect import (
    validate_graph, edges_from_neighbor_arrays,
)
from experiments.audit_numpy_undirected_multibridge import edge_digest
from experiments.audit_numpy_undirected_sparsity_mobility import (
    run_fit, verify_actual_final, CAP, WINDOW,
)
from experiments.numpy_continuous_observer import rng_equal

DATASETS=("digits","wine","breast_cancer")
SEEDS=(24103,24223,24317,24439)
STRATEGIES=("random","low_degree_quartile","high_degree_quartile")
ROWS={"digits":1797,"wine":178,"breast_cancer":569}
CLASSES={"digits":10,"wine":3,"breast_cancer":2}
K=10
LABELS_PER_CLASS=2


def dataset_graph(dataset):
    if dataset not in DATASETS:
        raise AssertionError("Dataset not prospectively selected")
    loaders={"digits":load_digits,"wine":load_wine,
             "breast_cancer":load_breast_cancer}
    data=loaders[dataset]()
    truth=np.asarray(data.target,dtype=np.int64)
    x=StandardScaler().fit_transform(
        np.asarray(data.data,dtype=np.float64))
    n,c=ROWS[dataset],CLASSES[dataset]
    if (len(truth)!=n or len(x)!=n or
            not np.array_equal(np.unique(truth),np.arange(c))):
        raise AssertionError("Unexpected dataset/label structure")
    graph,degrees=build_knn_graph(x,k_nn=K)
    edges,components,meta=validate_graph(graph,degrees)
    if (meta["ncomponents"]!=1 or meta["component_sizes"]!=[n]
            or meta["min_degree"]<K or len(edges)<n*K//2):
        raise AssertionError("Unmodified canonical graph is not connected")
    return graph,degrees,truth,meta,edge_digest(edges)


def class_pools(truth,degrees):
    """Per-class complete random pool and bounded low/high degree ranked pools.

    Score ties are deterministically resolved by original integer node ID.
    No labels or predictions learned by PCC enter the construction.
    """
    degree=np.asarray(degrees,dtype=np.int64)
    nclass=int(np.max(truth))+1
    result={}
    for cls in range(nclass):
        eligible=np.flatnonzero(truth==cls)
        count=max(LABELS_PER_CLASS,int(math.ceil(len(eligible)/4)))
        if len(eligible)<2*count:
            raise AssertionError("Class too small for disjoint quartile pools")
        low_order=eligible[np.lexsort((eligible,degree[eligible]))]
        high_order=eligible[np.lexsort((eligible,-degree[eligible]))]
        low=low_order[:count].copy()
        high=high_order[:count].copy()
        if (len(set(map(int,low)))!=count or
                len(set(map(int,high)))!=count or
                not set(map(int,low)).isdisjoint(set(map(int,high)))):
            raise AssertionError("Unexpected overlapping extreme-degree pools")
        if (np.any(np.diff(degree[low])<0) or
                np.any(np.diff(degree[high])>0)):
            raise AssertionError("Degree rank/tie ordering violated")
        result[cls]=dict(random=eligible,
                         low_degree_quartile=low,
                         high_degree_quartile=high)
    return result


def select_label_masks(truth,degrees,seed):
    pools=class_pools(truth,degrees)
    nc=int(np.max(truth))+1
    n=len(truth)
    masks={}
    for strategy in STRATEGIES:
        rng=np.random.default_rng(seed)  # fresh, one independent source per arm
        labels=np.full(n,-1,dtype=np.int64)
        for cls in range(nc):
            options=pools[cls][strategy]
            selected=rng.choice(options,LABELS_PER_CLASS,replace=False)
            if len(np.unique(selected))!=LABELS_PER_CLASS:
                raise AssertionError("Duplicate initial particle within class")
            labels[selected]=cls
        if (np.sum(labels!=-1)!=nc*LABELS_PER_CLASS or
                any(np.count_nonzero(labels==cls)!=LABELS_PER_CLASS
                    for cls in range(nc))):
            raise AssertionError("Wrong frozen particle homes per class")
        for cls in range(nc):
            if not set(np.flatnonzero(labels==cls)).issubset(
                    set(map(int,pools[cls][strategy]))):
                raise AssertionError("Selected node not in frozen degree pool")
        masks[strategy]=labels
    shared=np.logical_and.reduce(
        [labels==-1 for labels in masks.values()])
    if (np.count_nonzero(shared)<=0 or
            np.any([np.any(labels[shared]!=-1)
                    for labels in masks.values()])):
        raise AssertionError("No common actually held-out subset")
    pool_json={
        str(cls):{
            s:[int(i) for i in pools[cls][s]]
            for s in STRATEGIES
        } for cls in range(nc)}
    return masks,shared,pool_json


def home_diagnostics(graph,degrees,truth,labels):
    home=np.flatnonzero(labels!=-1)
    deg=np.asarray(degrees,dtype=np.int64)
    if len(home)!=2*(int(np.max(truth))+1):
        raise AssertionError("Wrong particle home count")
    same_edges=0
    total_edges=0
    entries=[]
    for i in home:
        neighbors=np.asarray(graph[i,:int(deg[i])],dtype=np.int64)
        same=int(np.count_nonzero(truth[neighbors]==truth[i]))
        same_edges+=same
        total_edges+=len(neighbors)
        entries.append(dict(node=int(i),class_id=int(labels[i]),
                            degree=int(deg[i]),same_class_neighbors=same,
                            total_neighbors=int(len(neighbors))))
    if total_edges<=0:
        raise AssertionError("Missing natural graph neighbors")
    return dict(
        home_ids=home.tolist(),
        homes=entries,
        selected_degree_mean=float(np.mean(deg[home])),
        selected_degree_min=int(np.min(deg[home])),
        selected_degree_max=int(np.max(deg[home])),
        selected_neighbor_label_agreement=float(same_edges/total_edges),
        # The above is a retrospectively oracle-computed analysis diagnostic,
        # NOT available to the native PCC and NEVER used for selection.
    )


def evaluate(dataset,seed,outfile):
    graph,degrees,truth,meta,sha=dataset_graph(dataset)
    masks,common,pools=select_label_masks(truth,degrees,seed)
    all_masked=np.logical_or.reduce([labels!=-1 for labels in masks.values()])
    if not np.array_equal(common,~all_masked):
        raise AssertionError("Common held-out set was not exact complement")
    nc=CLASSES[dataset]
    report=dict(
        dataset=dataset,seed=seed,cap=CAP,window=WINDOW,
        graph_family="canonical_original_undirected_full_union10",
        graph_edge_sha256=sha,graph_structure=meta,
        n_nodes=ROWS[dataset],n_classes=nc,
        labels_per_class=LABELS_PER_CLASS,
        k=K,strategies=list(STRATEGIES),
        class_pools=pools,
        common_unlabeled_count=int(np.count_nonzero(common)),
        common_unlabeled_ids=np.flatnonzero(common).tolist(),
        selected_union_ids=np.flatnonzero(all_masked).tolist(),
        cases=[],
    )
    for i,strategy in enumerate(STRATEGIES):
        labels=masks[strategy]
        own_unlabeled=labels==-1
        if np.any(~own_unlabeled & common):
            raise AssertionError("Hidden leakage on common evaluation set")
        order=((False,True) if (SEEDS.index(seed)+i+DATASETS.index(dataset))%2==0
               else (True,False))
        runs={
            record:run_fit(graph,degrees,labels,seed,K,record)
            for record in order}
        a,b=runs[False],runs[True]
        exact=dict(
            six_full_state_exact=all(
                np.array_equal(a["state"][key],b["state"][key])
                for key in a["state"]),
            native_info_exact=a["info"]==b["info"],
            complete_rng_exact=rng_equal(a["rng"],b["rng"]),
            next16_exact=np.array_equal(a["following"],b["following"]),
        )
        if not all(exact.values()):
            raise AssertionError(
                f"Observer violated original trajectory {dataset}/{seed}/{strategy}: {exact}")
        verify_actual_final(b)
        dynamics=b["model"].audit.summary()
        pred=a["state"]["predictions"]
        common_acc=float(np.mean(pred[common]==truth[common]))
        own_acc=float(np.mean(pred[own_unlabeled]==truth[own_unlabeled]))
        diag=home_diagnostics(graph,degrees,truth,labels)
        report["cases"].append(dict(
            strategy=strategy,
            n_particles=nc*LABELS_PER_CLASS,
            n_labeled=int(np.sum(~own_unlabeled)),
            n_own_unlabeled=int(np.sum(own_unlabeled)),
            own_unlabeled_count=int(np.count_nonzero(own_unlabeled)),
            accuracy_common_unlabeled=common_acc,
            accuracy_own_unlabeled=own_acc,
            selected=diag,
            parity={**exact,"true_final_low_level_exact":True},
            times_seconds=dict(
                baseline=a["seconds"],observed=b["seconds"]),
            dynamics=dynamics,
        ))
        Path(outfile).write_text(json.dumps(
            report,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        print(
            f"{dataset} seed {seed} {strategy}: "
            f"selected_mean_degree={diag['selected_degree_mean']:.3f}, "
            f"selected_oracle_homophily={diag['selected_neighbor_label_agreement']:.3f}, "
            f"common_heldout={np.count_nonzero(common)}, "
            f"common_accuracy={common_acc:.5f}, "
            f"own_accuracy={own_acc:.5f}, "
            f"class_lockouts={dynamics['operationally_locked_classes']}, "
            f"zero_class_windows={dynamics['zero_move_class_windows']}; "
            f"all-six/RNG/true-final parity PASS",
            flush=True)
    acc={case["strategy"]:case["accuracy_common_unlabeled"]
         for case in report["cases"]}
    report["primary_accuracy_delta_pp"]={
        "low_minus_random":100*(acc["low_degree_quartile"]-acc["random"]),
        "high_minus_random":100*(acc["high_degree_quartile"]-acc["random"]),
        "high_minus_low":100*(acc["high_degree_quartile"]-
                             acc["low_degree_quartile"]),
    }
    Path(outfile).write_text(json.dumps(
        report,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=DATASETS,required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    evaluate(a.dataset,a.seed,a.output)


if __name__=="__main__":
    main()
