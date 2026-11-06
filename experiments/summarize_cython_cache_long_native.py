"""Audit full long-duration native Cython timing cohort and frozen gates."""
import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import mean, median

from experiments.audit_cython_cache_long_native import (
    OSES, DATASETS, SEEDS, VARIANTS, CAP, BLOCKS, INNER_REPEATS,
)

BASE_SHA = "2902009c80b8edd3fc5f4961769444563fcf5b8a"


def validate(paths):
    if len(paths) != len(OSES)*len(DATASETS):
        raise AssertionError("Need all four OS×dataset long native artifacts")
    rows = {}
    for path in paths:
        with open(path,encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                key = (
                    row["runner_os"],row["dataset"],int(row["seed"]),
                    int(row["block"]),int(row["repeat"]),row["variant"])
                if (key[0] not in OSES or key[1] not in DATASETS
                        or key[2] not in SEEDS
                        or not 0 <= key[3] < BLOCKS
                        or not 0 <= key[4] < INNER_REPEATS
                        or key[5] not in VARIANTS):
                    raise AssertionError(f"Unexpected case {key}")
                if key in rows:
                    raise AssertionError(f"Duplicate case {key}")
                if any(row[k] != "True" for k in (
                        "all_three_state_digests_equal",
                        "all_three_stop_info_equal",
                        "real_native_extension","reciprocal_graph")):
                    raise AssertionError(f"Failed native correctness {key}")
                if (len(row["state_sha256"]) != 64 or
                        len(row["extension_sha256"]) != 64
                        or len(row["source_commit"]) != 40
                        or int(row["iterations"]) != CAP
                        or not float(row["elapsed_seconds"]) > 0):
                    raise AssertionError(f"Invalid digest, revision, cap or time {key}")
                rows[key] = row
    expected = set(product(
        OSES,DATASETS,SEEDS,range(BLOCKS),
        range(INNER_REPEATS),VARIANTS))
    if set(rows) != expected or len(rows) != 384:
        raise AssertionError(
            f"Incomplete long benchmark: seen={len(rows)}, "
            f"missing={len(expected-set(rows))}")
    block_savings = {}
    native_sha = set()
    for osname,ds,seed,block in product(
            OSES,DATASETS,SEEDS,range(BLOCKS)):
        triples = {}
        for variant in VARIANTS:
            group = [rows[(osname,ds,seed,block,i,variant)]
                     for i in range(INNER_REPEATS)]
            for f in ("source_commit","extension_sha256","extension_file",
                      "iterations","checks","stop_reason","stop_max",
                      "stalled_checks","state_sha256","order_index",
                      "master_block_median_seconds","off_block_median_seconds",
                      "on_block_median_seconds","off_saving_percent",
                      "on_saving_percent"):
                if len({r[f] for r in group}) != 1:
                    raise AssertionError(f"Inconsistent repeat metadata {f}")
            group_median = median(float(row["elapsed_seconds"]) for row in group)
            triples[variant] = (group,group_median)
        commits = [triples[v][0][0]["source_commit"] for v in VARIANTS]
        if commits[0] != BASE_SHA or commits[1] != commits[2] or commits[0] == commits[1]:
            raise AssertionError("Unexpected baseline/candidate source revision")
        hashes = [triples[v][0][0]["extension_sha256"] for v in VARIANTS]
        if hashes[0] == hashes[1] or hashes[1] != hashes[2]:
            raise AssertionError("Not independently compiled distinct variants")
        native_sha.add(commits[1])
        if {triples[v][0][0]["order_index"] for v in VARIANTS} != {"0","1","2"}:
            raise AssertionError("Variant order not cyclically balanced")
        state_digests = {triples[v][0][0]["state_sha256"] for v in VARIANTS}
        if len(state_digests) != 1:
            raise AssertionError("Native final-state digest mismatch across variants")
        for f in ("iterations","checks","stop_reason","stop_max","stalled_checks"):
            if len({triples[v][0][0][f] for v in VARIANTS}) != 1:
                raise AssertionError(f"Stopping info diverged: {f}")
        base = triples["master"][1]
        off = triples["candidate_off"][1]
        on = triples["candidate_on"][1]
        saving_off = 100 * (1 - off / base)
        saving_on = 100 * (1 - on / base)
        for v in VARIANTS:
            ref = triples[v][0][0]
            for column, calculated in (
                ("master_block_median_seconds",base),
                ("off_block_median_seconds",off),
                ("on_block_median_seconds",on),
                ("off_saving_percent",saving_off),
                ("on_saving_percent",saving_on),
            ):
                if abs(float(ref[column])-calculated)>1e-8:
                    raise AssertionError(f"Incorrect paired median {column}")
        block_savings[(osname,ds,seed,block)] = dict(
            off=saving_off,on=saving_on,
            master_seconds=base,off_seconds=off,on_seconds=on)
    if len(native_sha)!=1 or len(block_savings)!=32:
        raise AssertionError("Unexpected candidate revisions or paired group count")
    return block_savings, next(iter(native_sha))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    paired,sha=validate(args.files)
    lines=[
        "# Long-duration native Cython benchmark: original master vs cache OFF/ON",
        "",
        "**Complete cohort: 384/384 timed 300k-step fits, 32/32 exact "
        "three-way native-state+stop triplets, four seed conditions per "
        "OS×dataset, two independently launched blocks with four native "
        "repeats/variant. No exclusions.**",
        f"Original master baseline `{BASE_SHA}`; activated candidate `{sha}`.",
        "",
        "## Stable-workload median time savings versus untouched master",
        "",
        "| OS | Dataset | Cache explicitly OFF | Cache default ON | "
        "Master baseline median (ms) | Candidate OFF median (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    stats={}
    for osname,ds in product(OSES,DATASETS):
        per_seed={}
        master_times=[]
        off_times=[]
        for seed in SEEDS:
            blocks=[paired[(osname,ds,seed,b)] for b in range(BLOCKS)]
            per_seed[seed] = dict(
                off=median(x["off"] for x in blocks),
                on=median(x["on"] for x in blocks))
            master_times.extend(x["master_seconds"] for x in blocks)
            off_times.extend(x["off_seconds"] for x in blocks)
        off_med=median(v["off"] for v in per_seed.values())
        on_med=median(v["on"] for v in per_seed.values())
        stats[(osname,ds)] = (off_med,on_med,per_seed)
        lines.append(
            f"| {osname} | {ds} | {off_med:+.2f}% | {on_med:+.2f}% | "
            f"{1000*median(master_times):.3f} | "
            f"{1000*median(off_times):.3f} |")
    digits=[stats[(o,"digits")][1] for o in OSES]
    breast=[stats[(o,"breast_cancer")][1] for o in OSES]
    off=[stats[(o,d)][0] for o,d in product(OSES,DATASETS)]
    passed=all(x>=15 for x in digits) and all(x>=-5 for x in breast) and all(x>=-8 for x in off)
    lines.extend([
        "",
        "## Pre-registered independent longer-workload speed gates",
        "",
        f"- Digits cache ON must save >=15% per OS: "
        f"Linux/Windows **{digits[0]:+.2f}% / {digits[1]:+.2f}%**.",
        f"- Breast Cancer cache ON may not regress beyond -5% per OS: "
        f"Linux/Windows **{breast[0]:+.2f}% / {breast[1]:+.2f}%**.",
        f"- Cache OFF may not regress beyond -8% on any stratum: "
        f"worst measured **{min(off):+.2f}%**.",
        f"- Decision: **{'PASS longer-run diagnostic' if passed else 'FAIL longer-run diagnostic'}**.",
        "",
        "## Methodological limitations",
        "",
        "The 300k workload avoids relying on 3–5 ms propagation windows "
        "and measures the same native Cython potential-check path without "
        "early termination. It is **not equivalent to actual default "
        "stopping runtime**, which was evaluated separately under shorter "
        "conditions. The historical 20k run with a -17.41% Linux Breast "
        "Cancer cache-OFF regression is retained as a real observed "
        "result rather than erased. Only a cross-platform scoped "
        "performance conclusion is supported; GitHub runner CPU "
        "frequencies, OS scheduler load and compiler optimization may "
        "change between runs. The production master remains untouched.",
        "",
        "## Seed-by-seed observations",
        "",
        "| OS | Dataset | Seed | Cache OFF | Cache ON |",
        "|---|---|---:|---:|---:|",
    ])
    for osname,ds in product(OSES,DATASETS):
        for seed,v in stats[(osname,ds)][2].items():
            lines.append(
                f"| {osname} | {ds} | {seed} | "
                f"{v['off']:+.2f}% | {v['on']:+.2f}% |")
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:25]),flush=True)


if __name__=="__main__":
    main()
