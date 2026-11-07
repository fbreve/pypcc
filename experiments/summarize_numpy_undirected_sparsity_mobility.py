"""Strict complete-cohort report for canonical Digits k=3 vs k=10 mobility."""
import argparse
import json
from pathlib import Path

SEEDS=(20107,20201,20309,20411)
KS=(3,10)
CHECKS=(
    "full_six_state_exact","full_native_stop_exact",
    "full_mt19937_rng_exact","next_16_draws_exact",
    "final_low_level_state_exact",
)


def enforce(paths):
    cases={}
    masks={}
    for path in paths:
        document=json.loads(Path(path).read_text(encoding="utf-8"))
        seed=document.get("seed")
        if (seed not in SEEDS or
                document.get("dataset")!="digits" or
                document.get("cap")!=100000 or
                document.get("window")!=1000 or
                document.get("classes")!=10 or
                document.get("graph_type")!="canonical_undirected_union_knn"):
            raise AssertionError(f"Unexpected seed/cohort metadata: {path}")
        if len(document.get("cases",[]))!=2:
            raise AssertionError(f"Missing k-arm in {path}")
        if set(c["k"] for c in document["cases"])!=set(KS):
            raise AssertionError(f"Wrong/missing k in {path}")
        cases_in_order={c["k"]:c for c in document["cases"]}
        if document["paired_accuracy_sparse_minus_dense_pp"]!=100*(
            cases_in_order[3]["accuracy_unlabeled"]-
            cases_in_order[10]["accuracy_unlabeled"]):
            raise AssertionError(f"Inconsistent paired accuracy delta seed {seed}")
        for c in document["cases"]:
            key=(seed,c["k"])
            if key in cases:
                raise AssertionError(f"Duplicate seed×graph {key}")
            if not all(c.get(x) is True for x in CHECKS):
                raise AssertionError(f"Exact parity failed {key}")
            g=c["structure"]
            if (not g.get("reciprocal") or not g.get("no_self_edges")
                    or g.get("requested_k")!=c["k"]
                    or g.get("n_nodes")!=1797
                    or g.get("min_degree")<c["k"]
                    or sum(g.get("component_sizes",[]))!=1797
                    or g.get("ncomponents")!=len(g["component_sizes"])
                    or g["ncomponents"]<1):
                raise AssertionError(f"Graph invariants violated for {key}")
            if not (c["labels_count"]==20
                    and len(c["label_positions"])==20
                    and len(set(c["label_positions"]))==20
                    and 0<=c["accuracy_unlabeled"]<=1):
                raise AssertionError(f"Label/accuracy metadata wrong: {key}")
            d=c["dynamics"]
            if (d["physical_iterations"]!=100001 or
                    d["n_windows"]!=100
                    or len(d["windows"])!=1000
                    or len(d["individual_max_streaks"])!=20
                    or len(d["total_moves_by_class"])!=10
                    or len(d["class_max_zero_window_streaks"])!=10
                    or len(d["first_class_lockout"])!=10
                    or len(d["first_recovered_move"])!=10
                    or sum(d["total_moves_by_class"])!=d["total_accepted_moves"]
                    or not 0<=d["zero_movement_steps"]<=100000
                    or not 0<=d["never_moved_particles"]<=20
                    or not 0<=d["longest_individual_streak"]<=100000
                    or d["longest_individual_streak"]!=max(d["individual_max_streaks"])
                    or sum(x is not None for x in d["first_class_lockout"])!=
                       d["operationally_locked_classes"]
                    or d["any_operational_lockout"]!=
                       (d["operationally_locked_classes"]>0)
                    or d["zero_move_class_windows"]!=sum(
                       row["accepted_moves"]==0 for row in d["windows"])):
                raise AssertionError(f"Invalid streaming totals: {key}")
            for i,row in enumerate(d["windows"]):
                w=i//10
                cls=i%10
                if (row["start"]!=w*1000
                        or row["end"]!=(w+1)*1000
                        or row["class_id"]!=cls
                        or row["attempts"]!=2000
                        or not 0<=row["accepted_moves"]<=2000
                        or abs(row["acceptance_fraction"]-
                               row["accepted_moves"]/2000)>1e-15):
                    raise AssertionError(f"Window/class trace corrupt: {key} row {i}")
            if any(v<=0 for v in c["times_seconds"].values()):
                raise AssertionError(f"Nonpositive measured runtime: {key}")
            cases[key]=c
        masks[seed]=set(tuple(c["label_positions"]) for c in document["cases"])
        if len(masks[seed])!=1:
            raise AssertionError(f"Graphs for seed {seed} have different labels")
    expected={(seed,k) for seed in SEEDS for k in KS}
    if set(cases)!=expected:
        raise AssertionError(f"Missing predeclared cases: {sorted(expected-set(cases))}")
    return cases


def format_report(cases):
    lines=[
        "# Canonical undirected Digits kNN sparsity mobility — 8/8 exact pairs",
        "",
        "Four prospectively chosen seeds; k=3 and k=10 production undirected union",
        "graphs on exactly the same standardized nodes and initially labeled",
        "vertices per seed, native sequential NumPy, 100k no-stop iterations.",
        "8 original + 8 single-uninterrupted observed fits, exact final six-state,",
        "complete NumPy MT19937/16-next-draw and real stopping-info parity.",
        "",
        "| Seed | k | Degree mean | Components | Locked classes | Zero-move class-windows (of 1000) | Longest particle stationary streak | Zero-system-motion steps (of 100k) | Accuracy unlabeled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key,c in sorted(cases.items()):
        g=c["structure"]
        d=c["dynamics"]
        lines.append(
            f"| {key[0]} | {key[1]} | {g['mean_degree']:.2f} | "
            f"{g['ncomponents']} | {d['operationally_locked_classes']} | "
            f"{d['zero_move_class_windows']} | "
            f"{d['longest_individual_streak']} | {d['zero_movement_steps']} | "
            f"{100*c['accuracy_unlabeled']:.2f}% |")
    lines.extend(["",
        "| Seed | Sparse k=3 minus canonical k=10 accuracy (pp) |",
        "|---|---:|",
    ])
    for seed in SEEDS:
        d=100*(cases[seed,3]["accuracy_unlabeled"]-
               cases[seed,10]["accuracy_unlabeled"])
        lines.append(f"| {seed} | {d:+.3f} |")
    lines.extend(["",
        "Operational class lockout is five consecutive **1000-step** windows",
        "with zero changed-position accepted movements for all class particles,",
        "max class strength <=0.01 and zero currently winning neighbor entries.",
        "Position changes are exact accepted moves here because active graphs",
        "have no self-loops and every particle executes one valid visit per step.",
        "The presence or absence of lockout is **not a pass/fail gate** and",
        "does not prove irreversibility or causal effects on classification.",
        "The 100k cap does not establish convergence, and k=3 was chosen as",
        "a deliberately sparse topology, not a production default candidate.",
        "Host timing is exploratory and full-step observation is expensive.",
        "No changes to production graphs, movement, RNG or stopping policies.",
    ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cases=enforce(args.files)
    report=format_report(cases)
    args.output.write_text(report,encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":
    main()
