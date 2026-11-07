"""Require all phase-two Windows/Linux live NumPy observer comparisons."""
import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from statistics import median

from experiments.numpy_continuous_observer import (
    DEFAULT_MILESTONES,MAX_BYTES,MAX_OBSERVATIONS,STATE_NAMES,
)
from experiments.audit_numpy_continuous_observer import (
    DATASETS,POLICIES,
)

OSES=("ubuntu-latest","windows-latest")
SEEDS=(13001,13109)
POLICY_NAMES=("no_stop","aggressive_potential")
TRUE_FLAGS=(
    "graph_reciprocal","all_six_final_arrays_exact",
    "native_full_stop_info_exact","complete_rng_state_exact",
    "next_16_random_draws_exact","native_last_sample_exact",
    "final_snapshot_rng_exact","expected_milestones_exact",
    "initial_snapshot_exact",
)


def validate(files):
    if len(files)!=8:
        raise AssertionError("Require all eight OS/dataset raw artifacts")
    rows={}
    for path in files:
        with open(path,newline="",encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key=(row["runner_os"],row["dataset"],int(row["seed"]),row["policy"])
                osname,ds,seed,policy=key
                if (osname not in OSES or ds not in DATASETS
                        or seed not in SEEDS or policy not in POLICY_NAMES
                        or key in rows):
                    raise AssertionError(f"Unknown or repeated OS/dataset/seed/policy: {key}")
                cap,early,es_chk=POLICIES[policy]
                if (row["requested_actual_backend"]!="pcc_propagate_numpy"
                        or int(row["cap"])!=cap or
                        row["early_stop"]!=str(early)
                        or int(row["es_chk"])!=es_chk
                        or int(row["n_particles"])!=2*int(row["n_classes"])):
                    raise AssertionError(f"Invalid frozen workload: {key}")
                if any(row[flag]!="True" for flag in TRUE_FLAGS):
                    raise AssertionError(f"State, stop, RNG or graph gate failed: {key}")
                steps=int(row["effective_iterations"])
                if not 0<=steps<=cap:
                    raise AssertionError(f"Invalid number of steps: {key}")
                if policy=="no_stop" and (
                        steps!=cap or row["stop_reason"]!="max_iter"
                        or int(row["checks"])!=0):
                    raise AssertionError(f"No-stop workload stopped early: {key}")
                expected=sorted(set(
                    [m for m in DEFAULT_MILESTONES if m<=steps]+[steps]))
                snapshots=json.loads(row["snapshot_json"])
                actual=json.loads(row["snapshot_iterations"])
                if (actual!=expected or int(row["snapshot_count"])!=len(actual)
                        or len(actual)>MAX_OBSERVATIONS
                        or int(row["observer_budget_bytes"])!=MAX_BYTES
                        or not 0<int(row["observer_used_bytes"])<=MAX_BYTES
                        or len(snapshots)!=len(actual)):
                    raise AssertionError(f"Invalid actual trace schedule or memory: {key}")
                for i,ss in enumerate(snapshots):
                    if (ss["iteration"]!=actual[i] or
                            bool(ss["final"])!=(i==len(actual)-1)
                            or set(ss["array_digests"])!=set(STATE_NAMES)
                            or any(len(v)!=64 for v in ss["array_digests"].values())
                            or len(ss["rng_digest"])!=64):
                        raise AssertionError(f"Corrupt trace record: {key}/{i}")
                measured=float(row["observer_overhead_percent"])
                base=float(row["baseline_fit_seconds"])
                new=float(row["observed_fit_seconds"])
                if (not all(math.isfinite(v) for v in (measured,base,new))
                        or min(base,new)<=0
                        or abs(measured-100*(new/base-1))>1e-8):
                    raise AssertionError(f"Inconsistent runtime: {key}")
                rows[key]=row
    expected=set(product(OSES,DATASETS,SEEDS,POLICY_NAMES))
    if set(rows)!=expected or len(rows)!=32:
        raise AssertionError(
            f"Incomplete 32-pair cross-OS cohort: missing={expected-set(rows)}")
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files",required=True,nargs="+")
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    rows=validate(args.files)
    lines=[
        "# NumPy continuous observer: independent Windows/Linux parity replication",
        "",
        "**8/8 OS×dataset raw cohort artifacts, 32/32 exact observed/unobserved "
        "six-array + full stopping-metadata + full NumPy RNG state + "
        "next-16-RNG-draw equivalences PASS**: 64 seeded full fits, "
        "four reciprocal union-10NN datasets, two independent new seeds "
        "13001/13109, real continuous single-run state samples on "
        "the scheduled milestones and actual stop. Every sample and "
        "byte budget is accounted for.",
        "",
        "## Median whole-fit observer-on overhead (two new seeds per row)",
        "",
        "| OS | Dataset | No-stop 6001 | Aggressive potential | "
        "Aggressive early stops / 2 |",
        "|---|---|---:|---:|---:|",
    ]
    worst=-float("inf")
    for osname,ds in product(OSES,DATASETS):
        no=median(float(rows[(osname,ds,seed,"no_stop")][
            "observer_overhead_percent"]) for seed in SEEDS)
        aggressive=median(float(rows[(osname,ds,seed,"aggressive_potential")][
            "observer_overhead_percent"]) for seed in SEEDS)
        early=sum(rows[(osname,ds,seed,"aggressive_potential")][
            "stop_reason"]=="early_stop" for seed in SEEDS)
        worst=max(worst,no)
        lines.append(f"| {osname} | {ds} | {no:+.2f}% | "
                     f"{aggressive:+.2f}% | {early}/2 |")
    passes=worst<=50
    lines.extend([
        "",
        "## Frozen phase-two decision",
        "",
        f"- Worst per-OS/dataset no-stop observer median overhead: "
        f"**{worst:+.2f}%** (maximum allowed +50%).",
        f"- **{'PASS: cross-OS observational parity and overhead gates' if passes else 'FAIL: overhead exceeded frozen gate'}**.",
        "",
        "## Interpretation",
        "",
        "Every OS/backend pair compares its observed and unobserved "
        "NumPy fits with matched initial graph/labels/seed. It does not "
        "assert bitwise identity across Windows and Linux. "
        "The collector does not enumerate per-step flips between "
        "scheduled snapshots, and an unchanged assignment between "
        "samples does not guarantee convergence.",
        "",
        "This phase compared the default-unobserved path and opt-in path "
        "within one candidate Python source version. It did not measure "
        "the old master Python source compiled separately; further "
        "native baseline work and callback failure semantics review "
        "are still necessary before a production merge. "
        "No NumPy default, stopping rule, Numba or Cython source "
        "was changed by the replication protocol.",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:24]),flush=True)


if __name__=="__main__":
    main()
