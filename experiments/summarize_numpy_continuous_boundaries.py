"""Reject omitted/duplicated Windows/Linux boundary parity observations."""
import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from statistics import median
from experiments.audit_numpy_continuous_boundaries import (
    OSES,DATASETS,SEEDS,POLICIES,FIELDS,
)


def validate(files):
    if len(files)!=len(OSES)*len(DATASETS):
        raise AssertionError("Expected exactly 4 OS×dataset artifacts")
    rows={}
    for file in files:
        with open(file,newline="",encoding="utf-8") as source:
            for row in csv.DictReader(source):
                osname,ds,seed,policy=(
                    row["runner_os"],row["dataset"],int(row["seed"]),row["policy"])
                key=(osname,ds,seed,policy)
                if (osname not in OSES or ds not in DATASETS
                        or seed not in SEEDS or policy not in POLICIES
                        or key in rows):
                    raise AssertionError(f"Invalid or duplicate boundary case {key}")
                must_true=(
                    "exact_six_arrays","exact_native_stop_info",
                    "exact_global_rng","exact_next_16",
                    "exact_initial_observer_rng","exact_final_snapshot",
                    "exact_final_rng","zero_did_not_draw",
                )
                if any(row[name]!="True" for name in must_true):
                    raise AssertionError(f"Failed strict boundary gate {key}")
                effective=int(row["effective_iterations"])
                expected=json.loads(row["expected_snapshot_iterations"])
                actual=json.loads(row["actual_snapshot_iterations"])
                if (not 0<=effective<=int(row["num_steps"])
                        or not actual or actual!=expected
                        or actual[0]!=0 or actual[-1]!=effective
                        or len(actual)!=int(row["snapshot_count"])
                        or not 0<int(row["observed_bytes"])<=48*1024*1024):
                    raise AssertionError(f"Invalid or incomplete snapshot sequence {key}")
                snapshots=json.loads(row["snapshot_digests"])
                if len(snapshots)!=len(actual) or any(
                        item["iteration"]!=actual[i] or
                        len(item["rng"])!=64 or
                        len(item["arrays"])!=6 or
                        any(len(d)!=64 for d in item["arrays"].values())
                        for i,item in enumerate(snapshots)):
                    raise AssertionError(f"Missing full snapshot fingerprints {key}")
                if policy=="zero_iterations" and (
                        int(row["num_steps"])!=0 or effective!=0
                        or int(row["checks"])!=0
                        or row["stop_reason"]!="max_iter" or actual!=[0]):
                    raise AssertionError(f"Zero-step behavior diverged {key}")
                if policy!="zero_iterations" and int(row["num_steps"])!=111:
                    raise AssertionError("Wrong iteration cap")
                if any(not math.isfinite(float(row[name])) or float(row[name])<=0
                       for name in ("baseline_seconds","observed_seconds")):
                    raise AssertionError(f"Invalid durations {key}")
                rows[key]=row
    expected=set(product(OSES,DATASETS,SEEDS,POLICIES))
    if set(rows)!=expected or len(rows)!=36:
        raise AssertionError(
            f"Expected all 36 OS×dataset×seed×policy pairs; missing={expected-set(rows)}")
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",required=True,nargs="+")
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    rows=validate(a.files)
    lines=[
        "# NumPy single-run observer phase-3: boundary parity",
        "",
        "**36/36 exact pairs (72 fresh fits) PASS**, across Windows/Linux, "
        "canonical undirected Iris and Digits, 3 independent harness RNG seeds, "
        "and the distinct unseeded, zero-iterations and synchronous policies. "
        "Every tested pair exactly matches six final state arrays, complete "
        "native stop metadata, MT19937 RNG state and next 16 draws. "
        "The initial observer RNG fingerprint matches the externally installed "
        "state. Each actual final snapshot matches the live final state. "
        "The zero-step mode creates exactly one physical snapshot at iteration 0 "
        "and consumes zero random draws. No cases omitted.",
        "",
        "## Descriptive observed full-fit overhead",
        "",
        "| OS | Dataset | Unseeded median | Zero-step median | Synchronous median |",
        "|---|---|---:|---:|---:|",
    ]
    for osname,ds in product(OSES,DATASETS):
        values={
            policy:median(
                100*(float(rows[(osname,ds,seed,policy)]["observed_seconds"]) /
                float(rows[(osname,ds,seed,policy)]["baseline_seconds"])-1)
                for seed in SEEDS)
            for policy in POLICIES
        }
        lines.append(
            f"| {osname} | {ds} | "
            f"{values['unseeded']:+.2f}% | "
            f"{values['zero_iterations']:+.2f}% | "
            f"{values['synchronous']:+.2f}% |")
    lines.extend([
        "",
        "The near-zero iteration timed results are dominated by Python/setup "
        "overhead; do not treat those percentages as an algorithm benchmark. "
        "The experiment pins the global NumPy RNG state **outside** each "
        "unseeded fit to make paired inputs reproducible; this is not the "
        "same as claiming independently launched unseeded runs match.",
        "",
        "An exception raised at the initial observer callback is covered by "
        "the separate synthetic safety unit test, and does not modify "
        "particle buffers before the first iteration. Arbitrary callback "
        "exceptions after updates remain outside the guarantees of this "
        "experimental optional observer.",
        "",
        "No change to normal NumPy behavior, its Cython or Numba counterparts, "
        "the graph, or the existing stopping criterion. PR #182 stays draft.",
    ])
    Path(a.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:22]),flush=True)


if __name__=="__main__":
    main()
