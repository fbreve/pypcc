"""Verify all Windows/Linux cache replication and stress artifacts; no exclusions."""
import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import mean, median

from experiments.audit_cython_rowmax_cache_cross_platform import (
    DATASETS, SEEDS, POLICIES,
)
from experiments.audit_cython_rowmax_cache_numeric_edges import (
    CLASSES, GREEDY, DELTA_V, DELTAP, STOP_MAX, LAYOUTS,
    SEEDS as EDGE_SEEDS, CAPS, EXPECTED_CASES,
)

OSES=("ubuntu-latest","windows-latest")
VARIANTS=("legacy","cache")


def read_replication(paths):
    if len(paths)!=len(DATASETS)*len(OSES):
        raise AssertionError("Missing OS×dataset replication artifacts")
    rows={}
    for path in paths:
        with open(path,newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                osname,ds,seed,policy,rep,variant=(
                    row["runner_os"],row["dataset"],int(row["seed"]),
                    row["policy"],int(row["repeat"]),row["variant"])
                if (osname not in OSES or ds not in DATASETS or
                        seed not in SEEDS or variant not in VARIANTS or
                        policy not in [x[0] for x in POLICIES]):
                    raise AssertionError(f"Unexpected replication entry {row}")
                key=(osname,ds,seed,policy,rep,variant)
                if key in rows:
                    raise AssertionError("Duplicate OS/seed/policy/repetition")
                if (row["all_six_arrays_exact"]!="True" or
                        row["complete_stop_info_exact"]!="True" or
                        row["graph_reciprocal"]!="True"):
                    raise AssertionError(f"Hard state/stop/graph gate FAIL {key}")
                rows[key]=row
    expected=set()
    for osname,ds,seed,(policy,_,_,n),variant in product(
            OSES,DATASETS,SEEDS,POLICIES,VARIANTS):
        for rep in range(n):
            expected.add((osname,ds,seed,policy,rep,variant))
    if set(rows)!=expected:
        raise AssertionError(
            f"Missing or unexpected replication entries "
            f"{len(expected-set(rows))} / {len(set(rows)-expected)}")
    if len(rows)!=192:
        raise AssertionError(f"Expected exactly 192 repeated-fit rows, got {len(rows)}")
    pairs={}
    for osname,ds,seed in product(OSES,DATASETS,SEEDS):
        for policy,cap,es_chk,count in POLICIES:
            for rep in range(count):
                old=rows[(osname,ds,seed,policy,rep,"legacy")]
                new=rows[(osname,ds,seed,policy,rep,"cache")]
                if (int(old["cap"]) != cap or int(new["cap"]) != cap or
                        int(old["es_chk"]) != es_chk or
                        int(new["es_chk"]) != es_chk):
                    raise AssertionError("Wrong frozen cap/es_chk")
                for field in ("effective_iterations","checks","stalled_checks",
                              "stop_max","stop_reason","accuracy"):
                    if old[field]!=new[field]:
                        raise AssertionError(
                            f"Different stop metadata/accuracy {field}")
                if int(old["order_index"])==int(new["order_index"]):
                    raise AssertionError("Order balance not recorded")
                if policy=="forced_no_stop" and (
                        int(old["effective_iterations"]) != cap):
                    raise AssertionError("Forced-no-stop model stopped early")
                oldt=float(old["elapsed_seconds"])
                newt=float(new["elapsed_seconds"])
                if not (oldt>0 and newt>0):
                    raise AssertionError("Invalid wall time")
                saved=100*(1-newt/oldt)
                if any(abs(float(z["cached_time_saved_percent"])-saved)>1e-8
                       for z in (old,new)):
                    raise AssertionError("Pair timing fields inconsistent")
                pairs[(osname,ds,seed,policy,rep)] = saved
    if len(pairs)!=96:
        raise AssertionError("Expected 96 parity/timing pairs")
    return pairs


def read_edge_cases(paths):
    if len(paths)!=len(OSES):
        raise AssertionError("Must have both OS edge-test CSV artifacts")
    all_cases={}
    for p in paths:
        with open(p,newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                osname=row["runner_os"]
                key=(
                    int(row["classes"]),float(row["p_grd"]),
                    float(row["delta_v"]),float(row["deltap"]),
                    int(row["stop_max"]),row["layout"],
                    int(row["seed"]),int(row["cap"]))
                if osname not in OSES or (osname,key) in all_cases:
                    raise AssertionError("Unexpected/duplicate edge-case row")
                if (row["exact_state"]!="True" or
                        row["exact_full_stop_metadata"]!="True" or
                        row["verified_cache_at_all_checkpoints"]!="True" or
                        row["isolated_labeled_node"]!="True" or
                        int(row["checks"])<1):
                    raise AssertionError(f"Direct Cython numeric parity FAIL {key}")
                all_cases[(osname,key)]=row
    expected=set(product(
        CLASSES,GREEDY,DELTA_V,DELTAP,STOP_MAX,LAYOUTS,EDGE_SEEDS,CAPS))
    if len(expected)!=EXPECTED_CASES:
        raise AssertionError("Frozen direct grid count mismatch")
    for osname in OSES:
        actual={case for os_,case in all_cases if os_==osname}
        if actual!=expected:
            raise AssertionError(
                f"Missing {len(expected-actual)} edge cases on {osname}")
    if len(all_cases)!=2*EXPECTED_CASES:
        raise AssertionError("Incomplete numerical edge coverage")
    return len(all_cases)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replication-files",nargs="+",required=True)
    p.add_argument("--numeric-files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    paired=read_replication(args.replication_files)
    numeric_count=read_edge_cases(args.numeric_files)
    lines=[
        "# Cython row-max cache: independent Windows/Linux replication",
        "",
        "**Data integrity: 192/192 full-fit observations, 96/96 exact state + "
        "stop-metadata parity comparisons, 2 operating systems, "
        "fresh seeds 7307/7403/7507/7603, no exclusions.**",
        f"Both OS direct-kernel numerical stress grids passed: "
        f"**{numeric_count}/{numeric_count} scenarios**, including "
        "a degree-zero labeled node, C/F dominance layouts, "
        "zero and tiny delta_v, varied p_grd, deltap, stop_max and "
        "stop-check boundary caps. Every cached stress run "
        "independently compared the original global potential scalar "
        "at every actual checkpoint.",
        "",
        "## Whole fit-time savings, positive = faster",
        "",
        "| Runner OS | Dataset | Policy | Four-seed median of seed medians | "
        "Mean over raw pairs | Range over raw pairs |",
        "|---|---|---|---:|---:|---:|",
    ]
    grouped={}
    for osname,ds,(policy,_,_,reps) in product(OSES,DATASETS,POLICIES):
        values=[paired[(osname,ds,seed,policy,rep)]
                for seed in SEEDS for rep in range(reps)]
        seeds=[median(paired[(osname,ds,seed,policy,rep)]
                      for rep in range(reps)) for seed in SEEDS]
        grouped[(osname,ds,policy)] = median(seeds)
        lines.append(
            f"| {osname} | {ds} | {policy} | "
            f"{median(seeds):+.2f}% | {mean(values):+.2f}% | "
            f"[{min(values):+.2f}, {max(values):+.2f}]% |")
    digits=[grouped[(osname,"digits","forced_no_stop")] for osname in OSES]
    breast=[grouped[(osname,"breast_cancer","forced_no_stop")] for osname in OSES]
    passes=all(x>=15 for x in digits) and all(x>=-5 for x in breast)
    lines.extend([
        "",
        "## Frozen replication decision gate",
        "",
        f"- Digits whole-fit speedup medians on Linux/Windows: "
        f"**{digits[0]:+.2f}% / {digits[1]:+.2f}%**. Both must exceed +15%.",
        f"- Breast Cancer speedup medians on Linux/Windows: "
        f"**{breast[0]:+.2f}% / {breast[1]:+.2f}%**. Neither may be below -5%.",
        f"- **{'PASS: independent replication supports further review' if passes else 'FAIL: hold/reject candidate pending investigation'}**.",
        "",
        "## Caveats and next work",
        "",
        "There was one CI runner for each OS×dataset and five alternating "
        "paired blocks within that runner for each of four seeds. "
        "Repeated pairs reduce temporal noise but are not independent "
        "machines; Windows and Linux also use different native C "
        "compilers. The original and cached Cython paths are compiled "
        "together: the default-disabled branch needs an independent "
        "baseline-to-untouched-master comparison before production merge. "
        "The candidate changes only the global potential check, which "
        "is still summed in original node order; no change to the "
        "particle dynamics or stopping schedule is intended. "
        "Even if all replication gates pass, `master` is not "
        "changed automatically.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:18]),flush=True)


if __name__=="__main__":
    main()
