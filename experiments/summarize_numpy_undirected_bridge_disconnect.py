"""Fail closed unless all 12 predeclared native bridge/disconnect pairs pass."""
import argparse
import json
from pathlib import Path

SEEDS=(21107,21221,21317,21401)
VARIANTS=("full_union10","split_disconnected","single_bridge")
EXPECTED={(seed,variant) for seed in SEEDS for variant in VARIANTS}
EXACT=("six_full_state_exact","native_info_exact","complete_rng_exact",
       "next16_exact","true_final_low_level_exact")


def enforce(paths):
    cases={}
    seen=set()
    for path in paths:
        doc=json.loads(Path(path).read_text(encoding="utf-8"))
        seed=doc.get("seed")
        if seed not in SEEDS or seed in seen:
            raise AssertionError(f"Wrong or duplicate seed {seed}")
        seen.add(seed)
        if (doc.get("cap")!=100000 or doc.get("window")!=1000
                or doc.get("dataset")!="digits"
                or doc.get("graph_family")!="undirected_partition_parity_k10"
                or doc.get("labels_by_class_and_side")!=
                    "one per class per parity side"):
            raise AssertionError(f"Protocol metadata altered: {seed}")
        edge=doc.get("bridge_edge")
        if (not isinstance(edge,list) or len(edge)!=2
                or not all(isinstance(x,int) and 0<=x<1797 for x in edge)
                or {edge[0]%2,edge[1]%2}!={0,1} or edge[0]>=edge[1]):
            raise AssertionError(f"Bridge vertex identity corrupted: {seed}")
        labels=doc.get("label_positions",[])
        if (len(labels)!=20 or len(set(labels))!=20
                or any(not isinstance(x,int) or not 0<=x<1797
                       for x in labels)):
            raise AssertionError(f"Seed label mask corrupted: {seed}")
        entries=doc.get("cases",[])
        if (len(entries)!=3 or
                set(row.get("variant") for row in entries)!=set(VARIANTS)):
            raise AssertionError(f"Missing/duplicated arms for {seed}")
        for row in entries:
            variant=row["variant"]
            key=(seed,variant)
            if key in cases:
                raise AssertionError(f"Repeated condition {key}")
            if (row.get("n_particles")!=20 or row.get("n_labeled")!=20
                    or any(row.get("parity",{}).get(x) is not True
                           for x in EXACT)):
                raise AssertionError(f"Instrumented full-state/RNG parity failed {key}")
            g=row.get("structure",{})
            if (g.get("n_nodes")!=1797 or
                    g.get("reciprocal") is not True or
                    g.get("no_isolates") is not True or
                    g.get("min_degree",0)<10 or
                    g.get("undirected_edges",0)*2!=g.get("directed_edges")
                    or g.get("ncomponents") not in (1,2)
                    or sum(g.get("component_sizes",[]))!=1797):
                raise AssertionError(f"Invalid reciprocal graph report {key}")
            if variant=="split_disconnected":
                if (g["ncomponents"]!=2 or
                        g["component_sizes"]!=[899,898]):
                    raise AssertionError(f"Split not two components {key}")
            else:
                if (g["ncomponents"]!=1 or
                        g["component_sizes"]!=[1797]):
                    raise AssertionError(f"Unexpected graph connectivity {key}")
            if (not 0<=row["accuracy_unlabeled"]<=1 or
                    len(row.get("accuracy_unlabeled_by_partition",[]))!=2
                    or any(not 0<=x<=1
                           for x in row["accuracy_unlabeled_by_partition"])):
                raise AssertionError(f"Accuracy/partition issue {key}")
            d=row["dynamics"]
            if (d.get("physical_iterations")!=100001 or
                    d.get("n_windows")!=100 or
                    len(d.get("windows",[]))!=1000 or
                    len(d.get("individual_max_streaks",[]))!=20 or
                    len(d.get("total_moves_by_class",[]))!=10 or
                    len(d.get("cross_partition_moves_by_class",[]))!=10 or
                    len(d.get("cross_partition_moves_by_particle",[]))!=20 or
                    len(d.get("class_max_zero_window_streaks",[]))!=10 or
                    len(d.get("first_class_lockout",[]))!=10 or
                    len(d.get("first_recovered_move",[]))!=10 or
                    sum(d["total_moves_by_class"])!=d["total_accepted_moves"]
                    or sum(d["cross_partition_moves_by_class"])!=
                       d["cross_partition_moves"]
                    or sum(d["cross_partition_moves_by_particle"])!=
                       d["cross_partition_moves"]
                    or not 0<=d["cross_partition_moves"]<=
                       d["total_accepted_moves"] or
                    d["longest_individual_streak"]!=
                       max(d["individual_max_streaks"]) or
                    not 0<=d["zero_movement_steps"]<=100000 or
                    not 0<=d["never_moved_particles"]<=20 or
                    sum(x is not None for x in
                        d["first_class_lockout"])!=
                       d["operationally_locked_classes"] or
                    d["any_operational_lockout"]!=
                       (d["operationally_locked_classes"]>0) or
                    d["zero_move_class_windows"]!=sum(
                       x["accepted_moves"]==0 for x in d["windows"])):
                raise AssertionError(f"Malformed full per-iteration trajectory {key}")
            for ix,w in enumerate(d["windows"]):
                window=ix//10
                cls=ix%10
                if (w["class_id"]!=cls or w["start"]!=window*1000
                        or w["end"]!=(window+1)*1000
                        or w["attempts"]!=2000
                        or not 0<=w["accepted_moves"]<=2000
                        or abs(w["acceptance_fraction"]-
                               w["accepted_moves"]/2000)>1e-15):
                    raise AssertionError(f"Malformed class window {key}#{ix}")
            if variant=="split_disconnected" and d["cross_partition_moves"]!=0:
                raise AssertionError(f"Impossible crossing in disconnected graph: {key}")
            if any(t<=0 for t in row["elapsed"].values()):
                raise AssertionError(f"Invalid timing {key}")
            cases[key]=row

        split=cases[seed,"split_disconnected"]["structure"]
        bridged=cases[seed,"single_bridge"]["structure"]
        if (bridged["undirected_edges"]!=split["undirected_edges"]+1
                or bridged["directed_edges"]!=split["directed_edges"]+2):
            raise AssertionError(f"Single bridge changed other edges {seed}")
        accurate={key:cases[seed,key]["accuracy_unlabeled"]
                  for key in VARIANTS}
        if (abs(doc["accuracy_bridge_minus_split_pp"]-
                100*(accurate["single_bridge"]-
                     accurate["split_disconnected"]))>1e-12 or
            abs(doc["accuracy_full_minus_split_pp"]-
                100*(accurate["full_union10"]-
                     accurate["split_disconnected"]))>1e-12):
            raise AssertionError(f"Signed descriptive delta inconsistent {seed}")
    if set(cases)!=EXPECTED:
        raise AssertionError(f"Missing entire frozen cohort {sorted(EXPECTED-set(cases))}")
    return cases


def summary(cases):
    lines=[
        "# Controlled undirected Digits connectivity: 12/12 exact native pairs",
        "",
        "Four fresh seeds, three reciprocal variants: original full union10,",
        "two independently connected index-parity components (899+898),",
        "and the exact same split graph plus one reciprocal cross edge.",
        "One ground-truth-labeled home for each class on EACH side.",
        "24 real fits, 12 uninterrupted 100001-state live traces;",
        "bitwise final 6-state/native-metadata/MT19937/next16 parity.",
        "",
        "| Seed | Variant | Components | Degree mean | Locked classes | Zero-move class windows / 1000 | Bridge/parity crossings | Max stationary steps | Unlabeled accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for variant in VARIANTS:
            c=cases[seed,variant];g=c["structure"];d=c["dynamics"]
            lines.append(
                f"| {seed} | {variant} | {g['ncomponents']} | "
                f"{g['mean_degree']:.3f} | "
                f"{d['operationally_locked_classes']} | "
                f"{d['zero_move_class_windows']} | "
                f"{d['cross_partition_moves']} | "
                f"{d['longest_individual_streak']} | "
                f"{100*c['accuracy_unlabeled']:.2f}% |")
    lines.extend([
        "",
        "| Seed | Bridge − disconnected accuracy (pp) | Full − disconnected accuracy (pp) |",
        "|---|---:|---:|",
    ])
    for seed in SEEDS:
        a={name:cases[seed,name]["accuracy_unlabeled"]
           for name in VARIANTS}
        lines.append(
            f"| {seed} | {100*(a['single_bridge']-a['split_disconnected']):+.3f} | "
            f"{100*(a['full_union10']-a['split_disconnected']):+.3f} |")
    lines.extend([
        "",
        "The **split-to-bridge** difference is exactly one reciprocal edge.",
        "Full union10 differs in many within-partition connections and is",
        "a descriptive familiar baseline, not a one-edge counterfactual.",
        "No bridge crossing is required for a scientifically valid run.",
        "Operational class lockout requires >=5 consecutive 1000-step",
        "zero-accepted-movement class windows, class max strength <=0.01",
        "and zero current winning neighbor entries. Lockout is a",
        "descriptive state, NOT proof of irreversibility or necessarily",
        "an algorithm bug. Any accuracy change is not automatically",
        "caused by mobility. 100k is not proven convergence, and",
        "4 same-dataset seeds are not population prevalence estimates.",
        "No production PCC or default change, reset or forced strength.",
    ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=summary(enforce(args.files))
    args.output.write_text(report,encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":
    main()
