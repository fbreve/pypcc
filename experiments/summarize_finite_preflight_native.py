"""Validate all 96 native fits and all 48 exact master-vs-preflight pairs."""
import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import mean,median

from experiments.audit_finite_preflight_native import (
    OSES,DATASETS,SEEDS,MODES,VARIANTS,REPEATS,CAP,
)

BASE_SHA="c449444d99eac16d89f4a2492d44795e490d874b"


def audit(paths):
    if len(paths)!=len(OSES)*len(DATASETS):
        raise AssertionError("Require every OS×dataset experiment CSV")
    rows={}
    for path in paths:
        with open(path,newline="",encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key=(
                    row["runner_os"],row["dataset"],int(row["seed"]),
                    row["mode"],int(row["repeat"]),row["variant"])
                if (key[0] not in OSES or key[1] not in DATASETS or
                        key[2] not in SEEDS or
                        key[3] not in [x[0] for x in MODES] or
                        key[4] not in range(REPEATS) or
                        key[5] not in VARIANTS):
                    raise AssertionError(f"Unexpected condition {key}")
                if key in rows:
                    raise AssertionError(f"Duplicate condition {key}")
                if any(row[f]!="True" for f in (
                    "all_state_arrays_exact","all_stop_metadata_exact",
                    "canonical_reciprocal_graph")):
                    raise AssertionError(f"State or graph gate FAILED {key}")
                if int(row["effective_iterations"])!=CAP:
                    raise AssertionError(f"Propagation terminated early {key}")
                if (len(row["extension_sha256"])!=64 or
                        len(row["source_commit"])!=40):
                    raise AssertionError(f"Missing binary/source identity {key}")
                rows[key]=row
    expected=set(product(
        OSES,DATASETS,SEEDS,
        (x[0] for x in MODES),range(REPEATS),VARIANTS))
    if len(rows)!=96 or set(rows)!=expected:
        raise AssertionError(
            f"Missing or duplicate data: {len(rows)} expected 96; "
            f"missing={len(expected-set(rows))}")
    pairs={}
    candidate_shas=set()
    for osname,ds,seed,mode,repeat in product(
            OSES,DATASETS,SEEDS,
            [x[0] for x in MODES],range(REPEATS)):
        a=rows[(osname,ds,seed,mode,repeat,"master")]
        b=rows[(osname,ds,seed,mode,repeat,"preflight")]
        if {int(a["order"]),int(b["order"])}!={0,1}:
            raise AssertionError("Required counterbalanced order missing")
        if (a["source_commit"]!=BASE_SHA or
                b["source_commit"]==a["source_commit"] or
                b["extension_sha256"]==a["extension_sha256"]):
            raise AssertionError("Not independent compiled source revisions")
        candidate_shas.add(b["source_commit"])
        for key in ("effective_iterations","stop_reason","checks",
                    "stop_max","stalled_checks"):
            if a[key]!=b[key]:
                raise AssertionError(f"Stop metadata differ: {key}")
        baseline=float(a["elapsed_seconds"])
        experimental=float(b["elapsed_seconds"])
        if min(baseline,experimental)<=0:
            raise AssertionError("Nonpositive wall time")
        saved=100*(1-experimental/baseline)
        for item in (a,b):
            for field,value in (
                ("baseline_seconds",baseline),
                ("preflight_seconds",experimental),
                ("preflight_saving_percent",saved)
            ):
                if abs(float(item[field])-value)>1e-8:
                    raise AssertionError(f"Incorrect timing pair: {field}")
        pairs[(osname,ds,seed,mode,repeat)]=(
            saved,baseline,experimental)
    if len(pairs)!=48 or len(candidate_shas)!=1:
        raise AssertionError("Missing pairs or inconsistent candidate revision")
    return pairs,next(iter(candidate_shas))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",required=True,nargs="+")
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    pairs,candidate=audit(args.files)
    lines=[
        "# Clean Cython active-input finite preflight: OS/native parity and overhead",
        "",
        "**Exactly 96/96 native compiled subprocess fits and 48/48 "
        "strict master/candidate state-and-stop metadata comparisons PASS**, "
        "no excluded conditions; 200,000 actual native steps per fit, "
        "same canonical reciprocal graphs/seeds. ",
        f"Original master revision: `{BASE_SHA}`; candidate: `{candidate}`.",
        "",
        "## Timing of added finite preflight (positive = candidate faster)",
        "",
        "| OS | Dataset | Regime | Per-seed median saved | "
        "Mean raw saving | Range of paired savings | Master median (ms) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    strata={}
    for osname,ds,mode in product(
            OSES,DATASETS,[x[0] for x in MODES]):
        values=[pairs[(osname,ds,seed,mode,rep)]
                for seed in SEEDS for rep in range(REPEATS)]
        median_by_seed=[
            median(pairs[(osname,ds,seed,mode,rep)][0]
                   for rep in range(REPEATS))
            for seed in SEEDS
        ]
        s=median(median_by_seed)
        strata[(osname,ds,mode)]=s
        lines.append(
            f"| {osname} | {ds} | {mode} | {s:+.2f}% | "
            f"{mean(v[0] for v in values):+.2f}% | "
            f"[{min(v[0] for v in values):+.2f}, "
            f"{max(v[0] for v in values):+.2f}]% | "
            f"{median(v[1] for v in values)*1000:.2f} |"
        )
    worst=min(strata.values())
    passes=worst>=-8.0
    lines.extend([
        "",
        "## Frozen exploratory overhead gate",
        "",
        f"All eight OS×dataset×regime median savings must be >=−8%. "
        f"Worst observed is **{worst:+.2f}%**. "
        f"**{'PASS' if passes else 'FAIL'}**.",
        "",
        "This benchmark times only native `pcc_propagate` (not graph build/"
        "initialization). It does not estimate the overhead of repeated "
        "public single-step calls. Per-seed pairs on GitHub-hosted "
        "runners are still noisy; no universal performance guarantee "
        "is implied. The candidate does **not** change the already "
        "merged Cython cache or its default activation. Inputs with "
        "nonfinite active probability values must fail before mutation "
        "in both public Cython entry points; direct regression tests "
        "are required independently of the performance gate.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:22]),flush=True)
    if not passes:
        raise AssertionError(
            "Preregistered overhead gate failed; review raw timings and "
            "retain all samples before any integration")


if __name__=="__main__":
    main()
