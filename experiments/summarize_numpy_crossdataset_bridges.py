"""Fail closed unless all 40 frozen Wine/Breast Cancer native pairs pass."""
import argparse
import json
import statistics
from pathlib import Path

DATASETS=("wine","breast_cancer")
SEEDS=(23111,23209,23321,23417)
ARMS=("B0","B1","B8","B32","full_union10")
DOSES=(0,1,8,32)
NODES={"wine":178,"breast_cancer":569}
NCLASSES={"wine":3,"breast_cancer":2}
PARTITION_SIZES={"wine":[89,89],"breast_cancer":[285,284]}
EXACT=("six_full_state_exact","native_info_exact","complete_rng_exact",
       "next16_exact","true_final_low_level_exact")


def validate(paths):
    all_cases={}
    seen_documents=set()
    graph_refs={}
    for path in paths:
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        dataset=d.get("dataset")
        seed=d.get("seed")
        if dataset not in DATASETS or seed not in SEEDS or (
            dataset,seed) in seen_documents:
            raise AssertionError(f"Unknown or repeated dataset×seed: {dataset}/{seed}")
        seen_documents.add((dataset,seed))
        nc=NCLASSES[dataset]
        n=NODES[dataset]
        if (d.get("cap")!=100000 or
                d.get("window")!=1000 or
                d.get("n_classes")!=nc or
                d.get("n_nodes")!=n or
                d.get("n_labeled")!=2*nc or
                d.get("partitions")!=PARTITION_SIZES[dataset] or
                d.get("graph_family")!=
                    "crossdataset_official_union10_nested_parity_bridges" or
                d.get("labeled_homes")!=
                    "one per class per parity component" or
                d.get("doses")!=list(DOSES) or
                d.get("arms")!=list(ARMS)):
            raise AssertionError(f"Frozen dataset/horizon metadata changed: {dataset}/{seed}")
        labels=d.get("label_positions",[])
        if (len(labels)!=2*nc or len(set(labels))!=2*nc or
                any(not isinstance(v,int) or not 0<=v<n for v in labels)):
            raise AssertionError(f"Label mask not as preregistered: {dataset}/{seed}")
        bridges=d.get("ordered_bridges",[])
        if (len(bridges)!=32 or
                len({tuple(e) for e in bridges})!=32 or
                len({v for e in bridges for v in e})!=64 or
                any(len(e)!=2 or not all(isinstance(v,int) and 0<=v<n
                    for v in e) or e[0]>=e[1] or e[0]%2==e[1]%2
                    for e in bridges)):
            raise AssertionError(f"Non-disjoint or malformed nested bridges: {dataset}/{seed}")
        base_digest=d.get("base_edge_sha256")
        if not isinstance(base_digest,str) or len(base_digest)!=64:
            raise AssertionError("Missing frozen baseline graph digest")
        meta=(base_digest,tuple(map(tuple,bridges)))
        if ("dataset_base",dataset) not in graph_refs:
            graph_refs["dataset_base",dataset]=meta
        elif graph_refs["dataset_base",dataset]!=meta:
            raise AssertionError(f"Graph/dose ordering changed across seeds: {dataset}")
        arms=d.get("cases",[])
        if len(arms)!=len(ARMS) or set(x.get("arm") for x in arms)!=set(ARMS):
            raise AssertionError(f"Partial or repeated arms for {dataset}/{seed}")
        for c in arms:
            arm=c["arm"]
            key=(dataset,seed,arm)
            if key in all_cases:
                raise AssertionError(f"Duplicated arm {key}")
            if c.get("n_particles")!=2*nc or c.get("n_labeled")!=2*nc or (
                    c.get("n_unlabelled")!=n-2*nc):
                raise AssertionError(f"Incorrect particle count or held-out mask {key}")
            if any(c.get("parity",{}).get(field) is not True
                   for field in EXACT):
                raise AssertionError(f"Native bitwise state/RNG parity gate failed {key}")
            graph=c.get("structure",{})
            dose=(None if arm=="full_union10" else int(arm[1:]))
            if (c.get("bridges")!=dose or
                    graph.get("n_nodes")!=n or
                    graph.get("no_isolates") is not True or
                    graph.get("reciprocal") is not True or
                    graph.get("min_degree",0)<10 or
                    graph.get("undirected_edges",0)*2!=
                        graph.get("directed_edges",None) or
                    graph.get("ncomponents")!=
                        (2 if arm=="B0" else 1) or
                    graph.get("component_sizes")!=
                        (PARTITION_SIZES[dataset] if arm=="B0" else [n]) or
                    graph.get("cross_partition_edges")!=
                        (dose if dose is not None else graph.get("cross_partition_edges")) or
                    not isinstance(c.get("full_edge_sha256"),str) or
                    len(c["full_edge_sha256"])!=64):
                raise AssertionError(f"Undirected graph invalid or incomplete {key}")
            if arm=="full_union10" and graph.get("cross_partition_edges",0)<=0:
                raise AssertionError(f"Full graph has no natural partition-crossing edges: {key}")
            graph_ref=(c["full_edge_sha256"],graph)
            if ("arm",dataset,arm) not in graph_refs:
                graph_refs["arm",dataset,arm]=graph_ref
            elif graph_refs["arm",dataset,arm]!=graph_ref:
                raise AssertionError(f"Graph topology varied across label seeds: {key}")
            if (not 0<=c.get("accuracy_unlabeled",-1)<=1 or
                    len(c.get("accuracy_unlabeled_by_partition",[]))!=2 or
                    any(not 0<=x<=1 for x in
                        c["accuracy_unlabeled_by_partition"]) or
                    set(c.get("elapsed",{}))!={"baseline","observed"} or
                    any(x<=0 for x in c["elapsed"].values())):
                raise AssertionError(f"Accuracy/timing data invalid {key}")
            m=c.get("dynamics",{})
            if (m.get("physical_iterations")!=100001 or
                    m.get("n_windows")!=100 or
                    len(m.get("windows",[]))!=100*nc or
                    len(m.get("cross_partition_windows",[]))!=100 or
                    len(m.get("individual_max_streaks",[]))!=2*nc or
                    len(m.get("total_moves_by_class",[]))!=nc or
                    len(m.get("cross_partition_moves_by_class",[]))!=nc or
                    len(m.get("cross_partition_moves_by_particle",[]))!=2*nc or
                    len(m.get("class_max_zero_window_streaks",[]))!=nc or
                    len(m.get("first_class_lockout",[]))!=nc or
                    len(m.get("first_recovered_move",[]))!=nc):
                raise AssertionError(f"Incomplete physical streaming trajectory: {key}")
            crossing=m["cross_partition_moves"]
            count_by_bridge=m.get("crossing_count_by_bridge",{})
            if (crossing!=sum(m["cross_partition_moves_by_class"]) or
                    crossing!=sum(m["cross_partition_moves_by_particle"]) or
                    crossing!=sum(v["total"]
                                  for v in m["cross_partition_windows"]) or
                    crossing!=sum(count_by_bridge.values()) or
                    len(count_by_bridge)!=graph["cross_partition_edges"] or
                    (dose is not None and set(count_by_bridge)!=
                        {f"{u}-{v}" for u,v in bridges[:dose]}) or
                    crossing>m["total_accepted_moves"] or crossing<0 or
                    m["total_accepted_moves"]!=sum(m["total_moves_by_class"]) or
                    m["bridges_ever_traversed"]!=sum(
                        v>0 for v in count_by_bridge.values()) or
                    not 0<=m["particles_ever_crossed"]<=2*nc or
                    m["particles_ever_crossed"]!=sum(
                        x>0 for x in m["cross_partition_moves_by_particle"]) or
                    (m["first_crossing_iteration"] is None)!=(crossing==0) or
                    (crossing and not 1<=m["first_crossing_iteration"]<=100000) or
                    (arm=="B0" and crossing!=0)):
                raise AssertionError(f"Bridge crossing count corrupted: {key}")
            if (not 0<=m["zero_movement_steps"]<=100000 or
                    not 0<=m["never_moved_particles"]<=2*nc or
                    m["longest_individual_streak"]!=max(
                        m["individual_max_streaks"]) or
                    m["operationally_locked_classes"]!=sum(
                        x is not None for x in m["first_class_lockout"]) or
                    m["any_operational_lockout"]!=
                        (m["operationally_locked_classes"]>0) or
                    m["zero_move_class_windows"]!=sum(
                        row["accepted_moves"]==0 for row in m["windows"])):
                raise AssertionError(f"Native class mobility counts corrupted: {key}")
            accepted_totals=[0]*nc
            for i,row in enumerate(m["windows"]):
                w=i//nc
                cls=i%nc
                if (row.get("start")!=w*1000 or
                        row.get("end")!=(w+1)*1000 or
                        row.get("class_id")!=cls or
                        row.get("attempts")!=2000 or
                        not 0<=row["accepted_moves"]<=2000 or
                        abs(row["acceptance_fraction"]-
                            row["accepted_moves"]/2000)>1e-15):
                    raise AssertionError(f"Malformed class-window entry: {key}/{i}")
                accepted_totals[cls]+=row["accepted_moves"]
            if accepted_totals!=m["total_moves_by_class"]:
                raise AssertionError(f"Per-class motion count differs from windows: {key}")
            crossings_by_class=[0]*nc
            for i,row in enumerate(m["cross_partition_windows"]):
                if (row.get("start")!=i*1000 or
                        row.get("end")!=(i+1)*1000 or
                        len(row.get("per_class",[]))!=nc or
                        row["total"]!=sum(row["per_class"])):
                    raise AssertionError(f"Invalid cross-window entry: {key}/{i}")
                for cls,value in enumerate(row["per_class"]):
                    crossings_by_class[cls]+=value
            if crossings_by_class!=m["cross_partition_moves_by_class"]:
                raise AssertionError(f"Cross-window totals do not sum: {key}")
            all_cases[key]=c
        original=all_cases[dataset,seed,"B0"]["structure"]
        if (all_cases[dataset,seed,"B0"]["full_edge_sha256"]!=base_digest or
                any(all_cases[dataset,seed,f"B{b}"]["structure"]["undirected_edges"]!=
                    original["undirected_edges"]+b for b in DOSES) or
                any(all_cases[dataset,seed,f"B{b}"]["structure"]["directed_edges"]!=
                    original["directed_edges"]+2*b for b in DOSES)):
            raise AssertionError(f"Controlled graph base differs between bridge doses: {dataset}/{seed}")
        deltas=d.get("accuracy_deltas_vs_split_pp",{})
        if set(deltas)!={arm for arm in ARMS if arm!="B0"}:
            raise AssertionError(f"Missing preregistered accuracy deltas: {dataset}/{seed}")
        for arm,value in deltas.items():
            correct=100*(all_cases[dataset,seed,arm]["accuracy_unlabeled"]-
                          all_cases[dataset,seed,"B0"]["accuracy_unlabeled"])
            if abs(value-correct)>1e-12:
                raise AssertionError(f"Incorrect signed paired accuracy delta {dataset}/{seed}/{arm}")
    expected={(ds,seed,arm) for ds in DATASETS
              for seed in SEEDS for arm in ARMS}
    if (set(all_cases)!=expected or len(paths)!=len(DATASETS)*len(SEEDS)
            or len(seen_documents)!=len(DATASETS)*len(SEEDS)):
        raise AssertionError(f"Incomplete frozen 40-pair cohort: {sorted(expected-set(all_cases))}")
    return all_cases


def summarize(cases):
    lines=[
        "# Wine and Breast Cancer: cross-dataset undirected bridge-dose pilot",
        "",
        "**Strict 40/40 native pairs (80 actual 100k fits) completed.** Four",
        "fresh matched seeds per dataset; same internal reciprocal union10",
        "graph and labeled vertices across controlled B=0/1/8/32 arms;",
        "original full union10 is descriptive context, NOT a controlled bridge dose.",
        "Each observed fit captures 100001 real physical iterations and exactly",
        "reproduces original final six full arrays, native metadata, complete",
        "global MT19937 state and next 16 draws.",
        "",
        "| Dataset | Seed | Arm | Components | Installed cross edges | Accepted cross moves | Used cross edges | Locked classes | Zero-move class-windows | Held-out accuracy |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                c=cases[ds,seed,arm]
                g=c["structure"];m=c["dynamics"]
                lines.append(
                    f"| {ds} | {seed} | {arm} | {g['ncomponents']} | "
                    f"{g['cross_partition_edges']} | "
                    f"{m['cross_partition_moves']} | "
                    f"{m['bridges_ever_traversed']} | "
                    f"{m['operationally_locked_classes']} | "
                    f"{m['zero_move_class_windows']} | "
                    f"{100*c['accuracy_unlabeled']:.2f}% |")
    lines.extend(["",
        "## Paired changes on the same held-out nodes (percentage points)",
        "",
        "| Dataset | Seed | B1−B0 | B8−B0 | B32−B0 | Full−B0 (descriptive) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for ds in DATASETS:
        for seed in SEEDS:
            baseline=cases[ds,seed,"B0"]["accuracy_unlabeled"]
            changes=[100*(cases[ds,seed,arm]["accuracy_unlabeled"]-baseline)
                     for arm in ("B1","B8","B32","full_union10")]
            lines.append(
                f"| {ds} | {seed} | " +
                " | ".join(f"{x:+.3f}" for x in changes) + " |")
    lines.extend(["",
        "## Fixed-dose descriptive means per dataset",
        "",
        "| Dataset | Arm | Mean cross-partition accepted moves | Cross range | Mean held-out accuracy | Δ mean accuracy vs B0 (pp) |",
        "|---|---|---:|---|---:|---:|",
    ])
    for ds in DATASETS:
        base=statistics.mean(cases[ds,seed,"B0"]["accuracy_unlabeled"]
                             for seed in SEEDS)
        for arm in ARMS:
            traversals=[cases[ds,seed,arm]["dynamics"]["cross_partition_moves"]
                        for seed in SEEDS]
            acc=[cases[ds,seed,arm]["accuracy_unlabeled"]
                 for seed in SEEDS]
            lines.append(
                f"| {ds} | {arm} | "
                f"{statistics.mean(traversals):.1f} | "
                f"{min(traversals)}–{max(traversals)} | "
                f"{100*statistics.mean(acc):.2f}% | "
                f"{100*(statistics.mean(acc)-base):+.3f} |")
    lines.extend(["",
        "Four label-sampling seeds per dataset and a deliberately artificial",
        "even/odd partition do NOT establish population-level prevalence,",
        "statistical equivalence, a dose-optimum, convergence beyond 100k,",
        "or a causal effect of physical traffic itself on accuracy.",
        "Same numeric seed is NOT a common random tape across graph variants.",
        "All graphs are reciprocal, no self loops, original production",
        "union-kNN builders. No particle resets, forced strength, stopping",
        "changes or production defaults were modified. Keep PR draft.",
    ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    cases=validate(a.files)
    report=summarize(cases)
    Path(a.output).write_text(report,encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":
    main()
