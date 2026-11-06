"""Audit all direct step calls, strict binary isolation and timing gate."""
import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import median

from experiments.benchmark_finite_single_step import (
    OSES,DATASETS,SEEDS,VARIANTS,STEPS,
)

BASE_SHA="c449444d99eac16d89f4a2492d44795e490d874b"


def validate(paths):
    if len(paths)!=4:
        raise AssertionError("Need four OS×dataset artifacts")
    rows={}
    for path in paths:
        with open(path,encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                key=(row["runner_os"],row["dataset"],
                     int(row["seed"]),row["variant"])
                if (key[0] not in OSES or key[1] not in DATASETS
                        or key[2] not in SEEDS or key[3] not in VARIANTS
                        or key in rows):
                    raise AssertionError(f"Unexpected or duplicate case: {key}")
                if (row["exact_all_seven_state_arrays"]!="True"
                        or row["canonical_graph"]!="True"
                        or int(row["steps"])!=STEPS
                        or len(row["extension_sha256"])!=64):
                    raise AssertionError(f"Invalid state/work/binary: {key}")
                rows[key]=row
    expected=set(product(OSES,DATASETS,SEEDS,VARIANTS))
    if set(rows)!=expected or len(rows)!=24:
        raise AssertionError(f"Expected all 24 native runs, missing={expected-set(rows)}")
    pairs={}
    shas=set()
    for osname,ds,seed in product(OSES,DATASETS,SEEDS):
        a=rows[(osname,ds,seed,"master")]
        b=rows[(osname,ds,seed,"preflight")]
        if (a["source_commit"]!=BASE_SHA or
                b["source_commit"]==BASE_SHA or
                a["extension_sha256"]==b["extension_sha256"] or
                {int(a["order"]),int(b["order"])}!={0,1}):
            raise AssertionError("Wrong baseline, binary identity or order")
        shas.add(b["source_commit"])
        start=float(a["elapsed_seconds"])
        new=float(b["elapsed_seconds"])
        if min(start,new)<=0:
            raise AssertionError("Nonpositive timing")
        overhead=100*(new/start-1)
        for x in (a,b):
            for key,value in (
                ("baseline_seconds",start),
                ("preflight_seconds",new),
                ("slowdown_percent",overhead)):
                if abs(float(x[key])-value)>1e-8:
                    raise AssertionError(f"Corrupted timing pair: {key}")
        pairs[(osname,ds,seed)]=(overhead,start,new)
    if len(pairs)!=12 or len(shas)!=1:
        raise AssertionError("Need 12 exact paired comparisons, one candidate head")
    return pairs,next(iter(shas))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    pairs,sha=validate(args.files)
    lines=[
        "# Cython public single-step nonfinite preflight cost",
        "",
        "**24/24 independently compiled native `pcc_step` subprocesses and "
        "12/12 exact all-state pairs PASS; 5,000 sequential calls per process.** "
        "No excluded samples. The original master and candidate load "
        "different compiled extension hashes and separate Git revisions.",
        f"Original master `{BASE_SHA}`; candidate `{sha}`.",
        "",
        "## Percentage slowdown for the public single-step boundary",
        "",
        "| OS | Dataset | Median slowdown | Raw range | "
        "Master 5,000-step duration (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    groups={}
    for osname,ds in product(OSES,DATASETS):
        values=[pairs[(osname,ds,seed)] for seed in SEEDS]
        slowdown=median(v[0] for v in values)
        groups[(osname,ds)]=slowdown
        lines.append(
            f"| {osname} | {ds} | {slowdown:+.2f}% | "
            f"[{min(v[0] for v in values):+.2f}, "
            f"{max(v[0] for v in values):+.2f}]% | "
            f"{1000*median(v[1] for v in values):.3f} |")
    worst=max(groups.values())
    passes=worst<=20.0
    lines.extend([
        "",
        "## Frozen direct-step performance gate",
        "",
        f"- Maximum OS×dataset median slowdown: **{worst:+.2f}%**.",
        "- Gate: **<= +20%** in every stratum.",
        f"- **{'PASS — safe to review jointly with propagation benchmark' if passes else 'FAIL — redesign direct-step validation before merging'}**.",
        "",
        "This is the public single-step Cython API called repeatedly from "
        "Python, not the main `fit_predict` propagation path. "
        "Invalid NaN/Inf active-state rejection is separately tested. "
        "Even a PASS is not a universal performance guarantee; CI-hosted "
        "runners, Python call dispatch and clock scaling introduce noise. "
        "The current production master is not modified by this study.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)
    if not passes:
        raise AssertionError(
            "Direct-step preflight overhead exceeded frozen +20% median; "
            "do not merge the candidate as implemented")


if __name__=="__main__":
    main()
