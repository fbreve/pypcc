"""Validate complete opt-in Cython metric A/B and apply frozen decision gate."""
import argparse
import csv
from pathlib import Path
from statistics import mean, median

from experiments.audit_cython_columnwise_potential import (
    DATASETS, SEEDS, POLICIES,
)

VARIANTS = ("legacy", "columnwise")


def collect(paths):
    if len(paths) != 4:
        raise AssertionError("Need all four dataset artifacts")
    rows = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as file:
            for r in csv.DictReader(file):
                dataset, seed = r["dataset"], int(r["seed"])
                policy, variant = r["policy"], r["variant"]
                if (dataset not in DATASETS or seed not in SEEDS
                        or policy not in [p[0] for p in POLICIES]
                        or variant not in VARIANTS):
                    raise AssertionError(f"Unexpected cohort row: {r}")
                key = (dataset, seed, policy, variant)
                if key in rows:
                    raise AssertionError(f"Duplicate cohort row {key}")
                if (r["all_state_arrays_bitwise_equal"] != "True"
                        or r["all_propagation_info_equal"] != "True"
                        or r["dominance_fortran_contiguous"] != "True"
                        or abs(float(r["reciprocal_arc_fraction"]) - 1.0) > 1e-12):
                    raise AssertionError(f"Hard equivalence or graph gate failed: {key}")
                rows[key] = r
    expected = {
        (d,s,p[0],v) for d in DATASETS for s in SEEDS
        for p in POLICIES for v in VARIANTS
    }
    if set(rows) != expected:
        raise AssertionError(
            f"Incomplete cohort, missing={expected-set(rows)}")
    if len(rows)!=64:
        raise AssertionError("Unexpected number of timed fits")
    pairs={}
    for dataset in DATASETS:
        for seed in SEEDS:
            for name,cap,es_chk in POLICIES:
                a=rows[(dataset,seed,name,"legacy")]
                b=rows[(dataset,seed,name,"columnwise")]
                if (int(a["cap"]) != cap or int(b["cap"]) != cap or
                        int(a["es_chk"]) != es_chk or int(b["es_chk"]) != es_chk):
                    raise AssertionError("Wrong fixed cohort parameters")
                for f in ("effective_iterations", "stalled_checks",
                          "stop_reason", "stop_max", "checks"):
                    if a[f] != b[f]:
                        raise AssertionError(f"Stop metadata differed: {f}")
                if int(a["order_index"]) == int(b["order_index"]):
                    raise AssertionError("Missing order balance")
                t0,t1=float(a["elapsed_seconds"]),float(b["elapsed_seconds"])
                if not (t0 > 0 and t1 > 0):
                    raise AssertionError("Nonpositive fit time")
                if name == "forced_no_stop" and int(a["effective_iterations"]) != cap:
                    raise AssertionError("Forced case stopped")
                saving = 100*(1-t1/t0)
                if any(abs(float(r["time_saved_percent"])-saving) > 1e-8
                       for r in (a,b)):
                    raise AssertionError("Inconsistent saved-time report")
                pairs[(dataset,seed,name)] = saving
    if len(pairs)!=32:
        raise AssertionError("Expected 32 exact-parity comparisons")
    return pairs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    paired=collect(args.files)
    lines=[
        "# Opt-in Cython columnwise potential scan: exact-parity A/B",
        "",
        "All **64/64** timed fits and **32/32** exact six-array **and** "
        "full stopping-info parity gates PASS, across four canonical "
        "undirected datasets, four seeds and two monitoring regimes. "
        "All cases and measurements are retained; zero exclusions.",
        "",
        "## Total fit-time savings from experimental columnwise metric",
        "",
        "| Dataset | Policy | Mean savings | Median | Minimum | Maximum |",
        "|---|---|---:|---:|---:|---:|",
    ]
    agg={}
    for dataset in DATASETS:
        for policy,_,_ in POLICIES:
            values=[paired[(dataset,seed,policy)] for seed in SEEDS]
            agg[(dataset,policy)] = dict(
                mean=mean(values),median=median(values),
                min=min(values),max=max(values))
            x=agg[(dataset,policy)]
            lines.append(
                f"| {dataset} | {policy} | {x['mean']:+.2f}% | "
                f"{x['median']:+.2f}% | {x['min']:+.2f}% | "
                f"{x['max']:+.2f}% |"
            )
    target=agg[("digits","forced_no_stop")]["median"]
    worst_regression=min(agg[(ds,"forced_no_stop")]["median"]
                         for ds in DATASETS)
    passes=(target >= 10.0 and worst_regression >= -5.0)
    lines.extend([
        "",
        "## Preregistered exploratory speed gate",
        "",
        f"- Digits forced-workload median improvement: "
        f"**{target:+.2f}%** (must be >= +10%).",
        f"- Worst forced-workload median improvement across datasets: "
        f"**{worst_regression:+.2f}%** (must be >= -5%).",
        f"- Exploratory performance gate: "
        f"**{'PASS — investigate more broadly, do not merge' if passes else 'FAIL — discard this candidate as currently implemented'}**.",
        "",
        "## Interpretation and safeguards",
        "",
        "Savings use `100*(1-T_columnwise/T_legacy)`, i.e. positive "
        "means **less total runtime**. The baseline code path is the "
        "same original calculator inside the same Cython extension, "
        "with only a branch selecting the opt-in alternative; thus "
        "this is a fair paired compilation but not a comparison with "
        "an untouched historical binary.",
        "",
        "This is **a single CI runner observation per pair** with "
        "balanced variant order and just four seeds per dataset. "
        "No claim of statistically established speedup or all-platform "
        "performance follows. Passing the exploratory threshold does "
        "**not** approve a production merge. A separate independent "
        "replication, broader memory-layout and rare-state tests are "
        "needed. `master` is unchanged, and the original Cython "
        "potential calculator remains the default.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:22]),flush=True)


if __name__=="__main__":
    main()
