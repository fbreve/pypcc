"""Predeclared in-loop stop criteria on *canonical undirected* PCC graphs.

See canonical_undirected_stopping_protocol.md. Unlike former pilots, this
imports graph construction from a production-graph-validated preparer.
"""
import argparse
import csv
import numpy as np

from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import DATASETS


FIELDS = (
    "dataset", "seed", "policy", "criterion", "tolerance",
    "interval", "patience", "reference_cap",
    "iterations", "early_stop", "saved_iterations",
    "saved_iteration_fraction", "stop_reason", "accuracy",
    "reference_accuracy", "accuracy_delta_pp", "agreement_with_reference",
    "elapsed_seconds", "reference_seconds", "elapsed_ratio",
    "unlabeled_count", "mean_degree", "reciprocal_arc_fraction",
)


def compare(name, seed, cap, interval, patience, hybrid_tolerance):
    graph, degree, labels, truth, graph_stats = canonical_prepare(name, seed)
    unknown = labels == -1
    ref, ref_elapsed, info = run(
        graph, degree, labels, seed, cap, 0, interval)
    if info["effective_iterations"] != cap:
        raise AssertionError("No-stop reference did not reach its cap")
    ref_acc = float(np.mean(ref[unknown] == truth[unknown]))
    variants = [
        ("no_stop", None, 0.0, 0),
        ("label_stability", "label_change", 0.0, patience),
        ("dominance_label_hybrid", "dominance_label_hybrid",
         hybrid_tolerance, patience),
    ]
    records = []
    for policy, criterion, tol, requested_patience in variants:
        if criterion is None:
            pred, elapsed, details = ref, ref_elapsed, info
            criterion = "none"
        else:
            pred, elapsed, details = run(
                graph, degree, labels, seed, cap,
                requested_patience, interval, criterion, tol)
        iterations = int(details["effective_iterations"])
        if not 0 < iterations <= cap:
            raise AssertionError("Invalid stopping iteration count")
        acc = float(np.mean(pred[unknown] == truth[unknown]))
        records.append(dict(
            dataset=name, seed=seed, policy=policy,
            criterion=criterion, tolerance=tol,
            interval=interval, patience=requested_patience,
            reference_cap=cap, iterations=iterations,
            early_stop=(iterations < cap),
            saved_iterations=cap - iterations,
            saved_iteration_fraction=(cap - iterations) / cap,
            stop_reason=details.get("stop_reason", ""),
            accuracy=acc, reference_accuracy=ref_acc,
            accuracy_delta_pp=(acc - ref_acc) * 100.0,
            agreement_with_reference=float(np.mean(
                pred[unknown] == ref[unknown])),
            elapsed_seconds=float(elapsed),
            reference_seconds=float(ref_elapsed),
            elapsed_ratio=float(elapsed / ref_elapsed)
            if ref_elapsed else float("nan"),
            unlabeled_count=int(np.count_nonzero(unknown)),
            mean_degree=graph_stats["mean_degree"],
            reciprocal_arc_fraction=graph_stats[
                "reciprocal_arc_fraction"],
        ))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2203, 2309, 2411])
    parser.add_argument("--output", required=True)
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--hybrid-tolerance", type=float, default=0.001)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.cap <= 0 or args.interval < 1 or
            args.cap % args.interval or args.patience < 1
            or not np.isfinite(args.hybrid_tolerance)
            or args.hybrid_tolerance < 0):
        parser.error("Invalid canonical stopping protocol")
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            records = compare(args.dataset, seed, args.cap, args.interval,
                              args.patience, args.hybrid_tolerance)
            writer.writerows(records)
            print(
                f"{args.dataset} seed={seed}: " +
                ", ".join(
                    f"{r['policy']}={r['iterations']}"
                    for r in records),
                flush=True)


if __name__ == "__main__":
    main()
