"""Aggregate strict independent-original-master NumPy default-off comparisons."""
import argparse
import csv
import math
from itertools import product
from pathlib import Path
from statistics import median

from experiments.audit_numpy_master_default_off import (
    BASE_SHA,DATASETS,OSES,SEEDS,REPEATS,
)


def validate(files):
    if len(files)!=len(OSES)*len(DATASETS):
        raise AssertionError("Expected exactly six OS×dataset CSV artifacts")
    rows={}
    for file in files:
        with open(file,newline="",encoding="utf-8") as source:
            for row in csv.DictReader(source):
                osname,ds,seed,rep=(
                    row["runner_os"],row["dataset"],
                    int(row["seed"]),int(row["repeat"]))
                key=(osname,ds,seed,rep)
                if (osname not in OSES or ds not in DATASETS
                        or seed not in SEEDS or not 0<=rep<REPEATS
                        or key in rows):
                    raise AssertionError(f"Invalid or duplicate independent-fit pair {key}")
                if (row["all_state_and_rng_exact"]!="True"
                        or row["both_normal_public_api"]!="True"
                        or row["actual_graph_reciprocal"]!="True"
                        or row["both_distinct_checkouts"]!="True"
                        or row["baseline_commit"]!=BASE_SHA
                        or row["candidate_commit"]==BASE_SHA
                        or len(row["baseline_source_sha256"])!=64
                        or len(row["candidate_source_sha256"])!=64
                        or row["baseline_source_sha256"]==
                           row["candidate_source_sha256"]):
                    raise AssertionError(f"Source isolation/state/RNG gate failed: {key}")
                expected_order=(
                    "master/candidate_off" if (SEEDS.index(seed)+rep)%2==0
                    else "candidate_off/master")
                if row["variant_order"]!=expected_order:
                    raise AssertionError(f"Incorrect balanced variant order: {key}")
                base=float(row["original_seconds"])
                candidate=float(row["candidate_seconds"])
                saving=float(row["saving_percent"])
                if (not all(math.isfinite(x) and x>0 for x in (base,candidate))
                        or not math.isfinite(saving)
                        or abs(saving-100*(1-candidate/base))>1e-8):
                    raise AssertionError(f"Incorrect timing: {key}")
                rows[key]=row
    expected=set(product(OSES,DATASETS,SEEDS,range(REPEATS)))
    if set(rows)!=expected or len(rows)!=24:
        raise AssertionError(
            f"Incomplete 24 independent revision pairs: missing={expected-set(rows)}")
    for osname,ds in product(OSES,DATASETS):
        cases=[rows[(osname,ds,seed,rep)] for seed in SEEDS
               for rep in range(REPEATS)]
        base_ids={r["baseline_source_sha256"] for r in cases}
        cand_ids={r["candidate_source_sha256"] for r in cases}
        cand_commits={r["candidate_commit"] for r in cases}
        if len(base_ids)!=1 or len(cand_ids)!=1 or len(cand_commits)!=1:
            raise AssertionError(f"Independent source identities drifted: {osname}/{ds}")
    if len({r["candidate_commit"] for r in rows.values()})!=1:
        raise AssertionError("Candidate commit moved between matrix jobs")
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",required=True,nargs="+")
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    rows=validate(args.files)
    sha=next(iter(rows.values()))["candidate_commit"]
    lines=[
        "# NumPy default-off: original master vs candidate source checkout",
        "",
        "**24/24 full state+full MT19937 RNG+next 16 draws exact parity pairs PASS "
        "across 48 independently launched real `fit_predict` calls**. "
        "Two Git source trees, independent Python subprocesses, imported "
        "actual `pcc.py` and `pcc_numpy.py` from their respective "
        "checkouts, independently recorded revision IDs and source SHA-256. "
        "Observer never enabled in either source. "
        f"Untouched baseline `{BASE_SHA}`, candidate `{sha}`. "
        "Normal undirected union-10NN graph, 6001 no-stop iterations, "
        "two new seeds, two balanced order blocks per OS/dataset. "
        "All pairs and outliers retained.",
        "",
        "## Full high-level fit_predict runtime difference",
        "",
        "Positive saving means candidate with observer disabled is faster.",
        "",
        "| OS | Dataset | Median of seed-wise paired savings | "
        "Lowest individual pair | Highest individual pair | "
        "Original master median (ms) | Candidate median (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    medians={}
    for osname,ds in product(OSES,DATASETS):
        seedmed=[]
        raw=[]
        bases=[]
        candidates=[]
        for seed in SEEDS:
            values=[
                float(rows[(osname,ds,seed,rep)]["saving_percent"])
                for rep in range(REPEATS)
            ]
            seedmed.append(median(values))
            raw.extend(values)
            bases.extend(float(rows[(osname,ds,seed,rep)]["original_seconds"])
                         for rep in range(REPEATS))
            candidates.extend(float(rows[(osname,ds,seed,rep)]["candidate_seconds"])
                              for rep in range(REPEATS))
        medians[(osname,ds)]=median(seedmed)
        lines.append(
            f"| {osname} | {ds} | {median(seedmed):+.2f}% | "
            f"{min(raw):+.2f}% | {max(raw):+.2f}% | "
            f"{1000*median(bases):.2f} | {1000*median(candidates):.2f} |")
    worst=min(medians.values())
    pass_gate=worst>=-8.0
    lines.extend([
        "",
        "## Frozen default-off decision",
        "",
        f"- Worst OS/dataset median: **{worst:+.2f}%**; "
        "the preregistered lower bound is **-8.00%**.",
        f"- **{'PASS: default-off source overhead acceptable for further review' if pass_gate else 'FAIL: independent master-vs-candidate performance gate'}**.",
        "",
        "Each full fit includes model's normal propagation and per-fit setup, "
        "but excludes Python process startup, dependency import, graph building "
        "and serialization. Tests isolate **separate source checkouts** rather "
        "than merely toggling a runtime flag in one candidate binary. "
        "Hosted Windows/Linux runners may still have noisy CPU scheduling, "
        "so a positive 2-seed median is not proof of a universal speedup.",
        "",
        "Even if performance and parity pass, this is an opt-in research "
        "diagnostic, not a reason to merge automatically. "
        "The production master remains untouched.",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:24]),flush=True)
    if not pass_gate:
        raise AssertionError(
            f"Pre-registered default-off slowdown gate failed: worst {worst:+.2f}%")

if __name__=="__main__":
    main()
