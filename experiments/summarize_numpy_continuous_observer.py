"""Validate frozen continuous NumPy-only trajectory cohort, without exclusions."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean,median

from experiments.audit_numpy_continuous_observer import (
    DATASETS,SEEDS,POLICIES,
)
from experiments.numpy_continuous_observer import (
    DEFAULT_MILESTONES,MAX_BYTES,MAX_OBSERVATIONS,STATE_NAMES,
)


def validate(paths):
    if len(paths)!=len(DATASETS):
        raise AssertionError("Require exactly four dataset artifacts")
    rows={}
    for path in paths:
        with open(path,encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                ds,seed,policy=row["dataset"],int(row["seed"]),row["policy"]
                key=(ds,seed,policy)
                if (ds not in DATASETS or seed not in SEEDS or
                        policy not in POLICIES or key in rows):
                    raise AssertionError(f"Unexpected or duplicate key {key}")
                expected_cap,expected_stop,expected_chk=POLICIES[policy]
                if (int(row["cap"])!=expected_cap or
                        row["early_stop"]!=str(expected_stop) or
                        int(row["es_chk"])!=expected_chk or
                        int(row["n_particles"])!=2*int(row["n_classes"]) or
                        row["requested_actual_backend"]!="pcc_propagate_numpy"):
                    raise AssertionError(f"Unexpected frozen policy/backend {key}")
                if any(row[field]!="True" for field in (
                        "graph_reciprocal","all_six_final_arrays_exact",
                        "native_full_stop_info_exact","complete_rng_state_exact",
                        "next_16_random_draws_exact","native_last_sample_exact",
                        "final_snapshot_rng_exact","expected_milestones_exact",
                        "initial_snapshot_exact")):
                    raise AssertionError(f"Strict parity, graph or RNG failure {key}")
                effective=int(row["effective_iterations"])
                if not 0<=effective<=expected_cap:
                    raise AssertionError(f"Invalid effective iteration {key}")
                if policy=="no_stop" and (
                        effective!=expected_cap or
                        int(row["checks"])!=0 or
                        row["stop_reason"]!="max_iter"):
                    raise AssertionError(f"Unexpected no-stop result {key}")
                observed=json.loads(row["snapshot_iterations"])
                expected=sorted(set(
                    [m for m in DEFAULT_MILESTONES if m<=effective]
                    +[effective]))
                if (observed!=expected or
                        int(row["snapshot_count"])!=len(observed) or
                        int(row["snapshot_count"])>MAX_OBSERVATIONS or
                        observed[0]!=0 or observed[-1]!=effective or
                        int(row["observer_used_bytes"])>MAX_BYTES or
                        int(row["observer_used_bytes"])<=0 or
                        int(row["observer_budget_bytes"])!=MAX_BYTES):
                    raise AssertionError(f"Missing real snapshots or unsafe memory {key}")
                snapshots=json.loads(row["snapshot_json"])
                if len(snapshots)!=len(observed):
                    raise AssertionError("Incomplete snapshot payload")
                for i,snapshot in enumerate(snapshots):
                    if (int(snapshot["iteration"])!=observed[i] or
                            bool(snapshot["final"])!=(i==len(snapshots)-1) or
                            set(snapshot["array_digests"])!=set(STATE_NAMES) or
                            len(snapshot["rng_digest"])!=64 or
                            any(len(d)!=64 for d in snapshot["array_digests"].values())):
                        raise AssertionError(f"Corrupt snapshot digests {key}/{i}")
                    if (i==0 and snapshot["net_assignment_changes"] is not None or
                            i>0 and (
                                not isinstance(snapshot["net_assignment_changes"],int)
                                or snapshot["net_assignment_changes"]<0
                                or snapshot["net_assignment_changes"]>int(row["n_nodes"]))):
                        raise AssertionError(f"Invalid observed interval changes {key}")
                    if not math.isfinite(snapshot["mean_max_dominance"]):
                        raise AssertionError("Invalid mean max dominance")
                for field in ("baseline_fit_seconds","observed_fit_seconds"):
                    if not math.isfinite(float(row[field])) or float(row[field])<=0:
                        raise AssertionError(f"Invalid runtime {key}")
                if abs(
                    float(row["observer_overhead_percent"])-
                    100*(float(row["observed_fit_seconds"])/float(
                        row["baseline_fit_seconds"])-1)
                )>1e-8:
                    raise AssertionError(f"Misreported timing {key}")
                if not 0<=float(row["accuracy_unlabeled"])<=1:
                    raise AssertionError(f"Invalid accuracy {key}")
                rows[key]=row
    expected={(ds,s,p) for ds in DATASETS for s in SEEDS for p in POLICIES}
    if set(rows)!=expected or len(rows)!=36:
        raise AssertionError(
            f"Frozen 36-case coverage failure, missing={expected-set(rows)}")
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    rows=validate(args.files)
    lines=[
        "# Continuous single-run NumPy observer — complete preregistered audit",
        "",
        "**All 36/36 true observed/unobserved propagation pairs PASS exact "
        "six-array state, complete native stop metadata, full NumPy "
        "MT19937 RNG state and next 16 random outputs.** "
        "72 fresh fits, no exclusions, 4 production canonical undirected "
        "union-10NN datasets × three seeds × three policies. All actual "
        "observed checkpoint iteration arrays match the original "
        "propagation cap/real stop. The final low-level snapshot and "
        "RNG state match the observed high-level run before owner "
        "normalization. This is an **uninterrupted single-run trace**, "
        "not the independent prefix replays in PR #181.",
        "",
        "## Full-model runtime overhead (exploratory; positive = slower)",
        "",
        "| Dataset | No stop median overhead | Real-potential median overhead "
        "| Aggressive-potential median overhead | No-stop avg snapshots |",
        "|---|---:|---:|---:|---:|",
    ]
    stopping={}
    for ds in DATASETS:
        values={}
        for policy in POLICIES:
            records=[rows[(ds,seed,policy)] for seed in SEEDS]
            values[policy]=median(float(r["observer_overhead_percent"])
                                  for r in records)
        count=mean(int(rows[(ds,s,"no_stop")]["snapshot_count"]) for s in SEEDS)
        lines.append(
            f"| {ds} | {values['no_stop']:+.2f}% | "
            f"{values['real_potential']:+.2f}% | "
            f"{values['aggressive_potential']:+.2f}% | {count:.1f} |")
        stopping[ds]={
            "real":sum(rows[(ds,s,"real_potential")]["stop_reason"]=="early_stop"
                       for s in SEEDS),
            "aggressive":sum(
                rows[(ds,s,"aggressive_potential")]["stop_reason"]=="early_stop"
                for s in SEEDS),
        }
    no_stop_medians=[
        median(float(rows[(ds,seed,"no_stop")]["observer_overhead_percent"])
               for seed in SEEDS)
        for ds in DATASETS
    ]
    merit=all(x<=50 for x in no_stop_medians)
    lines.extend([
        "",
        "## Stopping observations",
        "",
        "| Dataset | Real potential: early stops / 3 | "
        "Aggressive potential: early stops / 3 |",
        "|---|---:|---:|",
    ])
    for ds,x in stopping.items():
        lines.append(
            f"| {ds} | {x['real']}/3 | {x['aggressive']}/3 |")
    lines.extend([
        "",
        "## Frozen exploratory gate",
        "",
        f"- Worst no-stop median observer overhead: "
        f"**{max(no_stop_medians):+.2f}%**, allowed <= +50%.",
        f"- Decision: **{'PASS, retain isolated observer for review' if merit else 'FAIL, reduce sampling/abandon scope'}**.",
        "",
        "## Scope and limitations",
        "",
        "- Samples are taken from one true uninterrupted NumPy execution "
        "at iteration 0, selected milestones, and the **actual final "
        "iteration**, including an early stop where present. "
        "The normal NumPy kernel is called unchanged for the ordinary model; "
        "the opt-in observer adds read-only callbacks. No snapshots are "
        "fabricated beyond early stopping.",
        "- Full RNG state and next-16-draw parity rigorously demonstrate "
        "that observation did not advance NumPy's random sequence in the "
        "tested cases. Comparison is within NumPy only and does not extend "
        "to Numba or Cython.",
        "- Differences between observed checkpoints are net changes, "
        "NOT counts of every intermediate class flip. Neither a stable "
        "sample interval nor unchanged mean dominance proves convergence.",
        "- The performance data are single shared-runner Linux timing "
        "comparisons with small cohorts and not independent throughput "
        "replication. A production merge would still require OS-native "
        "baseline-vs-candidate performance, expanded edge tests, budget "
        "safety and review of callback failure semantics.",
        "- The old continuous tracer PR #18 remains unmerged and stale "
        "against today's Cython cache and seeded kernels. No request "
        "to merge it is implied by a NumPy-only tracer success.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[:24]),flush=True)


if __name__=="__main__":
    main()
