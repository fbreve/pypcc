"""Fail-closed all-36 full-state native label-home placement cohort."""
import argparse
import json
import statistics
from pathlib import Path

DATASETS=("digits","wine","breast_cancer")
SEEDS=(24103,24223,24317,24439)
STRATEGIES=("random","low_degree_quartile","high_degree_quartile")
NODES={"digits":1797,"wine":178,"breast_cancer":569}
CLASSES={"digits":10,"wine":3,"breast_cancer":2}
EXACT=("six_full_state_exact","native_info_exact",
       "complete_rng_exact","next16_exact","true_final_low_level_exact")


def enforce(paths):
    conditions={}
    contexts=set()
    graph_refs={}
    for path in paths:
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        ds=d.get("dataset")
        seed=d.get("seed")
        if ds not in DATASETS or seed not in SEEDS or (ds,seed) in contexts:
            raise AssertionError(f"Missing/duplicate/unplanned dataset×seed {ds}/{seed}")
        contexts.add((ds,seed))
        nc=CLASSES[ds]
        n=NODES[ds]
        structure=d.get("graph_structure",{})
        if (d.get("cap")!=100000 or d.get("window")!=1000 or
                d.get("graph_family")!=
                    "canonical_original_undirected_full_union10" or
                d.get("n_nodes")!=n or d.get("n_classes")!=nc or
                d.get("labels_per_class")!=2 or d.get("k")!=10 or
                d.get("strategies")!=list(STRATEGIES) or
                structure.get("n_nodes")!=n or
                structure.get("ncomponents")!=1 or
                structure.get("component_sizes")!=[n] or
                structure.get("reciprocal") is not True or
                structure.get("no_isolates") is not True or
                structure.get("min_degree",0)<10 or
                structure.get("directed_edges")!=
                    2*structure.get("undirected_edges",0)):
            raise AssertionError(f"Canonical graph/metadata mismatch {ds}/{seed}")
        digest=d.get("graph_edge_sha256")
        if not isinstance(digest,str) or len(digest)!=64:
            raise AssertionError(f"Missing graph SHA {ds}/{seed}")
        graph_ref=(digest,structure)
        if ds not in graph_refs:
            graph_refs[ds]=graph_ref
        elif graph_refs[ds]!=graph_ref:
            raise AssertionError(f"Graph topology changed across seeds for {ds}")
        pools=d.get("class_pools",{})
        if set(pools)!={str(cls) for cls in range(nc)}:
            raise AssertionError(f"Frozen class pool structure changed {ds}/{seed}")
        own_ids_all={strategy:set() for strategy in STRATEGIES}
        for cls in range(nc):
            pp=pools[str(cls)]
            if set(pp)!=set(STRATEGIES):
                raise AssertionError(f"Missing pool strategy {ds}/{seed}/{cls}")
            random_ids=pp["random"]
            if (len(random_ids)!=len(set(random_ids)) or
                    len(random_ids)<8 or
                    any(not isinstance(v,int) or not 0<=v<n
                        for v in random_ids)):
                raise AssertionError(f"Invalid class member set {ds}/{seed}/{cls}")
            required=max(2,(len(random_ids)+3)//4)
            for strategy in STRATEGIES[1:]:
                ids=pp[strategy]
                if (len(ids)!=required or len(set(ids))!=required or
                        not set(ids).issubset(set(random_ids))):
                    raise AssertionError(f"Invalid rank-pool count {ds}/{seed}/{cls}/{strategy}")
            if set(pp[STRATEGIES[1]]) & set(pp[STRATEGIES[2]]):
                raise AssertionError(f"Overlapping high/low quartile pools {ds}/{seed}/{cls}")
            own_ids_all["random"].update(random_ids)

        rows=d.get("cases",[])
        if (len(rows)!=3 or
                set(c.get("strategy") for c in rows)!=set(STRATEGIES)):
            raise AssertionError(f"Partial or repeated label strategies {ds}/{seed}")
        selected_union=set()
        for c in rows:
            strategy=c["strategy"]
            key=(ds,seed,strategy)
            if key in conditions:
                raise AssertionError(f"Duplicate seed/strategy {key}")
            if (c.get("n_particles")!=2*nc or
                    c.get("n_labeled")!=2*nc or
                    c.get("n_own_unlabeled")!=n-2*nc or
                    c.get("own_unlabeled_count")!=n-2*nc or
                    any(c.get("parity",{}).get(k) is not True for k in EXACT)):
                raise AssertionError(f"Wrong particle count or exact state/RNG gate {key}")
            homes=c.get("selected",{})
            ids=homes.get("home_ids",[])
            detail=homes.get("homes",[])
            if (len(ids)!=2*nc or len(set(ids))!=2*nc or
                    len(detail)!=len(ids) or
                    ids!=[p.get("node") for p in detail]):
                raise AssertionError(f"Duplicated/missing selected node {key}")
            ids_set=set(ids)
            selected_union|=ids_set
            if (not 0<=homes.get("selected_neighbor_label_agreement",-1)<=1 or
                    not 10<=homes.get("selected_degree_min",0)<=
                    homes.get("selected_degree_mean",0)<=
                    homes.get("selected_degree_max",0) or
                    homes.get("selected_degree_mean")!=statistics.mean(
                        p["degree"] for p in detail)):
                raise AssertionError(f"Invalid measured home-degree diagnostics {key}")
            for cls in range(nc):
                byclass=[p for p in detail if p.get("class_id")==cls]
                if (len(byclass)!=2 or
                        not all(p["node"] in pools[str(cls)][strategy]
                                for p in byclass)):
                    raise AssertionError(f"Home outside class/degree pool {key}/{cls}")
            if (any(p.get("degree",0)<10 or
                    p.get("total_neighbors")!=p.get("degree") or
                    not 0<=p.get("same_class_neighbors",-1)<=
                        p.get("total_neighbors",0) for p in detail)):
                raise AssertionError(f"Corrupted retrospective home neighbor counts {key}")
            if (not 0<=c.get("accuracy_common_unlabeled",-1)<=1 or
                    not 0<=c.get("accuracy_own_unlabeled",-1)<=1 or
                    set(c.get("times_seconds",{}))!=
                        {"baseline","observed"} or
                    any(x<=0 for x in c["times_seconds"].values())):
                raise AssertionError(f"Invalid accuracy or runtime {key}")
            motion=c.get("dynamics",{})
            if (motion.get("physical_iterations")!=100001 or
                    motion.get("n_windows")!=100 or
                    len(motion.get("windows",[]))!=100*nc or
                    len(motion.get("total_moves_by_class",[]))!=nc or
                    len(motion.get("individual_max_streaks",[]))!=2*nc or
                    len(motion.get("class_max_zero_window_streaks",[]))!=nc or
                    len(motion.get("first_class_lockout",[]))!=nc or
                    len(motion.get("first_recovered_move",[]))!=nc or
                    motion.get("total_accepted_moves")!=
                        sum(motion["total_moves_by_class"]) or
                    motion.get("longest_individual_streak")!=
                        max(motion["individual_max_streaks"]) or
                    motion.get("zero_move_class_windows")!=sum(
                        p["accepted_moves"]==0 for p in motion["windows"]) or
                    motion.get("operationally_locked_classes")!=sum(
                        v is not None for v in motion["first_class_lockout"]) or
                    motion.get("any_operational_lockout")!=
                        (motion["operationally_locked_classes"]>0)):
                raise AssertionError(f"Truncated or contradictory per-step mobility {key}")
            accepted_class=[0]*nc
            for idx,row in enumerate(motion["windows"]):
                w,cls=divmod(idx,nc)
                attempts=2000
                if (row.get("class_id")!=cls or
                        row.get("start")!=w*1000 or
                        row.get("end")!=(w+1)*1000 or
                        row.get("attempts")!=attempts or
                        not 0<=row["accepted_moves"]<=attempts or
                        abs(row["acceptance_fraction"]-
                            row["accepted_moves"]/attempts)>1e-15):
                    raise AssertionError(f"Broken frozen class window {key}#{idx}")
                accepted_class[cls]+=row["accepted_moves"]
            if accepted_class!=motion["total_moves_by_class"]:
                raise AssertionError(f"Class-window accepted movement sums mismatch {key}")
            conditions[key]=c

        common=d.get("common_unlabeled_ids",[])
        selected=d.get("selected_union_ids",[])
        if (len(selected)!=len(selected_union) or
                selected!=sorted(selected_union) or
                len(common)+len(selected_union)!=n or
                common!=sorted(set(range(n))-selected_union) or
                d.get("common_unlabeled_count")!=len(common) or
                not 0<len(common)<=n-2*nc):
            raise AssertionError(f"Common primary evaluation mask corrupted {ds}/{seed}")
        changes=d.get("primary_accuracy_delta_pp",{})
        if set(changes)!={"low_minus_random","high_minus_random",
                          "high_minus_low"}:
            raise AssertionError(f"Missing preregistered common-subset comparisons {ds}/{seed}")
        acc={strategy:conditions[ds,seed,strategy][
            "accuracy_common_unlabeled"] for strategy in STRATEGIES}
        expected={
            "low_minus_random":100*(acc["low_degree_quartile"]-acc["random"]),
            "high_minus_random":100*(acc["high_degree_quartile"]-acc["random"]),
            "high_minus_low":100*(
                acc["high_degree_quartile"]-acc["low_degree_quartile"])}
        if any(abs(changes[k]-expected[k])>1e-12 for k in expected):
            raise AssertionError(f"Wrong paired common-heldout delta {ds}/{seed}")
    needed={(ds,seed,s) for ds in DATASETS
            for seed in SEEDS for s in STRATEGIES}
    if len(paths)!=len(DATASETS)*len(SEEDS) or set(conditions)!=needed:
        raise AssertionError(f"Missing frozen cases: {sorted(needed-set(conditions))}")
    return conditions


def describe(conditions):
    lines=[
        "# Canonical undirected PCC: frozen degree-stratified label-home cohort",
        "",
        "**All 36/36 native observed-vs-unobserved full 100k fits passed",
        "strict final-state/native-metadata/MT19937/next16 and actual raw",
        "low-level observer parity** (72 real fits; 36 single uninterrupted",
        "100001-physical-state streaming observations). The exact production",
        "full union10 graph is unchanged across all strategies and seeds",
        "within each dataset. Two oracle selected home labels per class;",
        "the three arms choose from random/all-class, lower-ranked-degree",
        "quarter, or upper-ranked-degree quarter of each ground-truth class.",
        "**Primary accuracy is ALWAYS on the exact common unlabeled subset**",
        "within a dataset/seed; own-arm unlabeled accuracy is secondary.",
        "",
        "| Dataset | Seed | Strategy | Mean home degree | Retrospective home homophily | Common held-out accuracy | Own held-out accuracy | Locked classes | Zero-accepted class windows |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for seed in SEEDS:
            for strat in STRATEGIES:
                c=conditions[ds,seed,strat]
                diag=c["selected"];m=c["dynamics"]
                lines.append(
                    f"| {ds} | {seed} | {strat} | "
                    f"{diag['selected_degree_mean']:.2f} | "
                    f"{100*diag['selected_neighbor_label_agreement']:.1f}% | "
                    f"{100*c['accuracy_common_unlabeled']:.2f}% | "
                    f"{100*c['accuracy_own_unlabeled']:.2f}% | "
                    f"{m['operationally_locked_classes']} | "
                    f"{m['zero_move_class_windows']} |")
    lines.extend(["",
        "## Paired **same-evaluation-set** changes, percentage points",
        "",
        "| Dataset | Seed | Low−random | High−random | High−low |",
        "|---|---:|---:|---:|---:|",
    ])
    for ds in DATASETS:
        for seed in SEEDS:
            acc={s:conditions[ds,seed,s]["accuracy_common_unlabeled"]
                 for s in STRATEGIES}
            lines.append(
                f"| {ds} | {seed} | "
                f"{100*(acc['low_degree_quartile']-acc['random']):+.3f} | "
                f"{100*(acc['high_degree_quartile']-acc['random']):+.3f} | "
                f"{100*(acc['high_degree_quartile']-acc['low_degree_quartile']):+.3f} |")
    lines.extend(["",
        "## Per-dataset descriptive means over exactly four seeds",
        "",
        "| Dataset | Strategy | Mean home degree | Mean home neighbor agreement | Mean common-heldout accuracy | Mean change vs random (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for ds in DATASETS:
        base=statistics.mean(
            conditions[ds,seed,"random"]["accuracy_common_unlabeled"]
            for seed in SEEDS)
        for s in STRATEGIES:
            members=[conditions[ds,seed,s] for seed in SEEDS]
            lines.append(
                f"| {ds} | {s} | "
                f"{statistics.mean(c['selected']['selected_degree_mean'] for c in members):.2f} | "
                f"{100*statistics.mean(c['selected']['selected_neighbor_label_agreement'] for c in members):.1f}% | "
                f"{100*statistics.mean(c['accuracy_common_unlabeled'] for c in members):.2f}% | "
                f"{100*(statistics.mean(c['accuracy_common_unlabeled'] for c in members)-base):+.3f} |")
    lines.extend(["",
        "**Limits:** the class-balanced low/high-degree strategies require",
        "class labels for candidate eligibility (oracle benchmark, NOT",
        "a deployable unsupervised degree-only strategy). Local graph",
        "degree is NOT equivalent to global centrality or graph periphery.",
        "Class-neighbor homophily is a retrospective diagnostic that",
        "uses true labels and NEVER affects PCC or the selected pool.",
        "Graph edges are unmodified and reciprocal union10 throughout.",
        "Native 100k cap does NOT prove convergence or accuracy stability;",
        "4 seeds/dataset do not justify a production default, policy",
        "change, force reset, causal degree mechanism, or p-value.",
        "Keep stacked experimental PRs DRAFT/UNMERGED.",
    ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=describe(enforce(a.files))
    Path(a.output).write_text(result,encoding="utf-8")
    print(result,flush=True)


if __name__=="__main__":
    main()
