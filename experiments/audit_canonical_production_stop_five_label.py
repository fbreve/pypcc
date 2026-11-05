"""Predeclared production potential-stop baseline at five labels/class.

Graph and seed protocol: canonical_production_stop_five_label_protocol.md
"""
import argparse
import csv
import numpy as np

from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.audit_canonical_production_stop_baseline import (
    production_run, FIELDS)
from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import DATASETS


def compare(dataset, seed, cap):
    graph, degrees, labels, truth, graph_stats = canonical_prepare(
        dataset, seed, k=10, labeled_per_class=5)
    unknown = labels == -1
    counts = np.bincount(labels[~unknown])
    if not np.all(counts == 5):
        raise AssertionError("Missing five labeled points per class")
    prod_ref, ref_seconds, ref_info = production_run(
        graph, degrees, labels, seed, cap, early_stop=False)
    if int(ref_info["effective_iterations"]) != cap:
        raise AssertionError("Production reference failed full cap")
    candidate_ref, _, candidate_ref_info = run(
        graph, degrees, labels, seed, cap, 0, 1000)
    parity = bool(np.array_equal(prod_ref, candidate_ref))
    if (int(candidate_ref_info["effective_iterations"]) != cap
            or not parity):
        raise AssertionError(
            f"Candidate vs production full-cap mismatch "
            f"for {dataset} seed {seed}")
    prod_stop, prod_seconds, prod_info = production_run(
        graph, degrees, labels, seed, cap, early_stop=True)
    candidate_stop, candidate_seconds, candidate_info = run(
        graph, degrees, labels, seed, cap, 10, 1000,
        criterion="label_change", tolerance=0.0)

    policies = [
        ("production_no_stop", prod_ref, ref_seconds, ref_info),
        ("production_potential_default", prod_stop,
         prod_seconds, prod_info),
        ("experimental_label_patience_10", candidate_stop,
         candidate_seconds, candidate_info),
    ]
    ref_acc = float(np.mean(prod_ref[unknown] == truth[unknown]))
    rows = []
    for policy, pred, seconds, info in policies:
        steps = int(info["effective_iterations"])
        if not 0 < steps <= cap:
            raise AssertionError("Invalid iteration count")
        if policy == "production_potential_default":
            other = candidate_stop
        elif policy == "experimental_label_patience_10":
            other = prod_stop
        else:
            other = prod_ref
        acc = float(np.mean(pred[unknown] == truth[unknown]))
        other_acc = float(np.mean(other[unknown] == truth[unknown]))
        rows.append(dict(
            dataset=dataset, seed=seed, policy=policy, cap=cap,
            iterations=steps, early_stop=(steps < cap),
            saved_iterations_fraction=(cap - steps) / cap,
            accuracy=acc, reference_accuracy=ref_acc,
            accuracy_delta_pp=100.0 * (acc - ref_acc),
            agreement_with_reference=float(np.mean(
                pred[unknown] == prod_ref[unknown])),
            elapsed_seconds=float(seconds),
            reference_seconds=float(ref_seconds),
            saved_wall_time_fraction=1.0 - seconds / ref_seconds,
            agreement_with_other_stopping_policy=float(np.mean(
                pred[unknown] == other[unknown])),
            accuracy_delta_vs_other_stopping_policy_pp=100.0 * (
                acc - other_acc),
            unlabeled_count=int(np.count_nonzero(unknown)),
            mean_degree=graph_stats["mean_degree"],
            reciprocal_arc_fraction=graph_stats[
                "reciprocal_arc_fraction"],
            candidate_production_no_stop_parity=parity,
        ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[3701, 3803, 3911])
    parser.add_argument("--cap", type=int, default=100000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.cap < 1):
        parser.error("Invalid fixed five-label protocol")
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            rows = compare(args.dataset, seed, args.cap)
            writer.writerows(rows)
            print(f"{args.dataset} seed={seed}: " +
                  ", ".join(
                      f"{r['policy']}={r['iterations']}"
                      for r in rows), flush=True)


if __name__ == "__main__":
    main()
