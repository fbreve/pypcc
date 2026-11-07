"""Strict aggregated report: all 16 preregistered undirected bridge doses."""
import argparse
import json
import statistics
from pathlib import Path

SEEDS=(22103,22229,22307,22441)
DOSES=(0,1,8,32)
N=1797
EXACT=("six_full_state_exact","native_info_exact",
       "complete_rng_exact","next16_exact","true_final_low_level_exact")


def validate(files):
    conditions={}
    seen_seeds=set()
    shared=None
    for path in files:
        report=json.loads(Path(path).read_text(encoding="utf-8"))
        seed=report.get("seed")
        if seed not in SEEDS or seed in seen_seeds:
            raise AssertionError(f"Unexpected or duplicated seed {seed}")
        seen_seeds.add(seed)
        if (report.get("dataset")!="digits" or
                report.get("cap")!=100000 or report.get("window")!=1000 or
                report.get("graph_family")!=
                    "official_split_union10_plus_nested_disjoint_bridges" or
                report.get("labeled_homes")!=
                    "one per class per parity component" or
                report.get("doses")!=list(DOSES)):
            raise AssertionError(f"Preregistered graph/dose metadata changed: {seed}")
        bridges=report.get("ordered_bridges",[])
        if (len(bridges)!=32 or len({tuple(edge) for edge in bridges})!=32
                or len({v for edge in bridges for v in edge})!=64 or
                any(len(edge)!=2 or not all(isinstance(v,int)
                       and 0<=v<N for v in edge) or
                       edge[0]>=edge[1] or
                       edge[0]%2==edge[1]%2 for edge in bridges)):
            raise AssertionError(f"Wrong nested vertex-disjoint bridges: {seed}")
        meta=(report.get("base_edge_sha256"),tuple(map(tuple,bridges)))
        if not isinstance(meta[0],str) or len(meta[0])!=64:
            raise AssertionError("Invalid frozen base adjacency fingerprint")
        if shared is None:
            shared=meta
        elif shared!=meta:
            raise AssertionError("Base graph / ordered bridge set differs across seeds")
        labels=report.get("label_positions",[])
        if (len(labels)!=20 or len(set(labels))!=20 or
                any(not isinstance(v,int) or not 0<=v<N for v in labels)):
            raise AssertionError(f"Invalid or altered seeded labels: {seed}")
        cases=report.get("cases",[])
        if len(cases)!=len(DOSES) or {c.get("bridges") for c in cases}!=set(DOSES):
            raise AssertionError(f"Missing or repeated bridge-dose arms: {seed}")
        for c in cases:
            b=c["bridges"]
            key=(seed,b)
            if key in conditions:
                raise AssertionError(f"Duplicate condition: {key}")
            if c.get("n_labeled")!=20 or c.get("n_particles")!=20:
                raise AssertionError(f"Unexpected class-particle count: {key}")
            if any(c.get("parity",{}).get(field) is not True for field in EXACT):
                raise AssertionError(f"Exact 6-array/RNG/final parity failure: {key}")
            g=c.get("structure",{})
            if (g.get("n_nodes")!=N or
                    g.get("reciprocal") is not True or
                    g.get("no_isolates") is not True or
                    g.get("min_degree",0)<10 or
                    g.get("component_sizes")!=
                       ([899,898] if b==0 else [N]) or
                    g.get("ncomponents")!=(2 if b==0 else 1) or
                    g.get("undirected_edges",0)*2!=g.get("directed_edges") or
                    not isinstance(c.get("full_edge_sha256"),str) or
                    len(c["full_edge_sha256"])!=64):
                raise AssertionError(f"Invalid undirected graph or component count: {key}")
            if (c.get("accuracy_unlabeled") is None or
                    not 0<=c["accuracy_unlabeled"]<=1 or
                    len(c.get("accuracy_unlabeled_by_partition",[]))!=2 or
                    any(not 0<=x<=1 for x in
                        c["accuracy_unlabeled_by_partition"]) or
                    any(x<=0 for x in c.get("elapsed",{}).values()) or
                    set(c.get("elapsed",{}))!={"baseline","observed"}):
                raise AssertionError(f"Invalid full accuracy/elapsed results: {key}")
            d=c.get("dynamics",{})
            if (d.get("physical_iterations")!=100001 or
                    d.get("n_windows")!=100 or
                    len(d.get("windows",[]))!=1000 or
                    len(d.get("cross_partition_windows",[]))!=100 or
                    len(d.get("individual_max_streaks",[]))!=20 or
                    len(d.get("total_moves_by_class",[]))!=10 or
                    len(d.get("cross_partition_moves_by_class",[]))!=10 or
                    len(d.get("cross_partition_moves_by_particle",[]))!=20 or
                    len(d.get("class_max_zero_window_streaks",[]))!=10 or
                    len(d.get("first_class_lockout",[]))!=10 or
                    len(d.get("first_recovered_move",[]))!=10):
                raise AssertionError(f"Incomplete native continuous trace: {key}")
            cross=d["cross_partition_moves"]
            if (cross!=sum(d["cross_partition_moves_by_class"]) or
                    cross!=sum(d["cross_partition_moves_by_particle"]) or
                    cross!=sum(w["total"]
                               for w in d["cross_partition_windows"]) or
                    cross!=sum(d["crossing_count_by_bridge"].values()) or
                    not 0<=cross<=d["total_accepted_moves"] or
                    sum(d["total_moves_by_class"])!=d["total_accepted_moves"] or
                    not 0<=d["particles_ever_crossed"]<=20 or
                    d["particles_ever_crossed"]!=sum(
                        x>0 for x in d["cross_partition_moves_by_particle"]) or
                    d["bridges_ever_traversed"]!=sum(
                        x>0 for x in d["crossing_count_by_bridge"].values()) or
                    not 0<=d["bridges_ever_traversed"]<=b or
                    (d["first_crossing_iteration"] is None)!=(cross==0) or
                    (cross>0 and not 1<=d["first_crossing_iteration"]<=100000) or
                    len(d["crossing_count_by_bridge"])!=b or
                    set(d["crossing_count_by_bridge"])!=
                       {f"{u}-{v}" for u,v in bridges[:b]} or
                    (b==0 and cross!=0)):
                raise AssertionError(f"Crossing count / graph support mismatch: {key}")
            if (d["longest_individual_streak"]!=
                    max(d["individual_max_streaks"]) or
                    not 0<=d["longest_individual_streak"]<=100000 or
                    not 0<=d["zero_movement_steps"]<=100000 or
                    not 0<=d["never_moved_particles"]<=20 or
                    d["operationally_locked_classes"]!=sum(
                        x is not None for x in d["first_class_lockout"]) or
                    d["any_operational_lockout"]!=
                       (d["operationally_locked_classes"]>0) or
                    d["zero_move_class_windows"]!=sum(
                        row["accepted_moves"]==0 for row in d["windows"])):
                raise AssertionError(f"Movement/streak totals mismatch: {key}")
            perclass=[0]*10
            for i,row in enumerate(d["windows"]):
                w=i//10
                cls=i%10
                if (row.get("start")!=w*1000 or
                        row.get("end")!=(w+1)*1000 or
                        row.get("class_id")!=cls or
                        row.get("attempts")!=2000 or
                        not 0<=row["accepted_moves"]<=2000 or
                        abs(row["acceptance_fraction"]-
                            row["accepted_moves"]/2000)>1e-15):
                    raise AssertionError(f"Malformed 1000-step native class window {key}/{i}")
                perclass[cls]+=row["accepted_moves"]
            if perclass!=d["total_moves_by_class"]:
                raise AssertionError(f"Per-class accepted movement sum mismatch: {key}")
            perclasscross=[0]*10
            for i,w in enumerate(d["cross_partition_windows"]):
                if (w.get("start")!=i*1000 or
                        w.get("end")!=(i+1)*1000 or
                        len(w.get("per_class",[]))!=10 or
                        not all(isinstance(v,int) and v>=0
                                for v in w["per_class"]) or
                        w["total"]!=sum(w["per_class"])):
                    raise AssertionError(f"Malformed bridge crossing window {key}/{i}")
                for cls,value in enumerate(w["per_class"]):
                    perclasscross[cls]+=value
            if perclasscross!=d["cross_partition_moves_by_class"]:
                raise AssertionError(f"Native per-class crossing totals do not add up: {key}")
            conditions[key]=c

        for b in DOSES[1:]:
            g1=conditions[seed,b]["structure"]
            g0=conditions[seed,0]["structure"]
            if (g1["undirected_edges"]!=g0["undirected_edges"]+b or
                    g1["directed_edges"]!=g0["directed_edges"]+2*b):
                raise AssertionError(f"Unexpected within-component graph change: {seed}/{b}")
        if conditions[seed,0]["full_edge_sha256"]!=report["base_edge_sha256"]:
            raise AssertionError(f"Original disconnected edge set altered: {seed}")

    expected={(seed,b) for seed in SEEDS for b in DOSES}
    if set(conditions)!=expected or len(files)!=len(SEEDS):
        raise AssertionError(f"Incomplete 16-case pilot: {sorted(expected-set(conditions))}")
    return conditions


def describe(conditions):
    lines=[
        "# Frozen undirected nested bridge-dose experiment: 16/16 exact pairs",
        "",
        "Standardized Digits; two 899/898 components, same internal official",
        "union10 edge set; B=0/1/8/32 nearest vertex-disjoint nested",
        "undirected bridge edges; fresh 4 seeds; one labeled home/class/side.",
        "32 independent real NumPy no-stop 100k fits; 16 true uninterrupted",
        "100001-step observed traces, exact six-array/native-stop/full MT19937",
        "and next16 parity; 100 disjoint 1000-step windows per class.",
        "",
        "| Seed | Bridges | Components | Cross-partition accepted moves | Used bridges | Crossing particles / 20 | Class lockouts | Zero-movement class windows /1000 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for b in DOSES:
            c=conditions[seed,b]
            d=c["dynamics"]
            lines.append(
                f"| {seed} | {b} | {c['structure']['ncomponents']} | "
                f"{d['cross_partition_moves']} | {d['bridges_ever_traversed']} | "
                f"{d['particles_ever_crossed']} | "
                f"{d['operationally_locked_classes']} | "
                f"{d['zero_move_class_windows']} | "
                f"{100*c['accuracy_unlabeled']:.2f}% |")
    lines.extend(["",
        "| Seed | Δ accuracy B1−B0 (pp) | B8−B0 | B32−B0 | B32−B1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for seed in SEEDS:
        v={b:conditions[seed,b]["accuracy_unlabeled"] for b in DOSES}
        lines.append(
            f"| {seed} | {100*(v[1]-v[0]):+.3f} | "
            f"{100*(v[8]-v[0]):+.3f} | {100*(v[32]-v[0]):+.3f} | "
            f"{100*(v[32]-v[1]):+.3f} |")
    for b in DOSES:
        x=[conditions[seed,b]["dynamics"]["cross_partition_moves"]
           for seed in SEEDS]
        acc=[conditions[seed,b]["accuracy_unlabeled"] for seed in SEEDS]
        lines.append(
            f"- **B={b}**: accepted crossing count range "
            f"{min(x)}–{max(x)}, mean {statistics.mean(x):.1f}; "
            f"held-out accuracy range {100*min(acc):.2f}–"
            f"{100*max(acc):.2f}%, mean {100*statistics.mean(acc):.2f}%")
    lines.extend(["",
        "No monotonicity, convergence, statistical equivalence, or causal",
        "mediation claim is justified by four seeds on one artificially",
        "partitioned dataset. Crossings are physical accepted particle moves",
        "not independent trials. Adding edges changes transition probabilities",
        "and future RNG consumption, even under the same numeric seed.",
        "Whole-class operational lockout still requires five 1000-step",
        "zero-movement class windows, max strength <=.01 and zero",
        "winning outgoing neighbors; no result-based tuning or reset.",
        "The ordinary full union10 from PR #185 has many additional",
        "within- and cross-partition edges and is not a controlled B dose.",
        "Production master and defaults remain untouched.",
    ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    conditions=validate(a.files)
    text=describe(conditions)
    Path(a.output).write_text(text,encoding="utf-8")
    print(text,flush=True)


if __name__=="__main__":
    main()
