"""Predeclared long-horizon label-stability stopping on canonical graphs."""
import argparse
import csv
import numpy as np

from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import DATASETS

FIELDS = (
    "dataset", "seed", "policy", "criterion", "patience", "interval",
    "reference_cap", "iterations", "early_stop",
    "iterations_saved", "fraction_iterations_saved",
    "stop_reason", "accuracy", "reference_accuracy",
    "accuracy_delta_pp", "agreement_with_reference",
    "elapsed_seconds", "reference_seconds", "time_saved_fraction",
    "reference_50k_accuracy", "reference_50k_to_100k_accuracy_delta_pp",
    "reference_50k_to_100k_flips", "reference_50k_to_100k_agreement",
    "initially_unlabeled_nodes", "mean_degree",
    "reciprocal_arc_fraction", "reference_matches_no_callback",
)


def audit_case(dataset, seed, cap, interval, patience_values):
    graph, degrees, labels, truth, graph_info = canonical_prepare(
        dataset, seed)
    mask = labels == -1
    states = {}
    def observe(it, dominance, positions, classes, strengths):
        if it in (cap // 2, cap):
            if not np.isfinite(dominance).all():
                raise AssertionError("Nonfinite checkpoint dominances")
            states[it] = np.argmax(dominance[mask], axis=1).copy()

    ref, ref_time, ref_info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        state_checkpoint_callback=observe, checkpoint_interval=cap // 2)
    if ref_info["effective_iterations"] != cap or set(states) != {
            cap // 2, cap}:
        raise AssertionError("Missing canonical full-reference checkpoints")
    if not np.array_equal(states[cap], ref[mask]):
        raise AssertionError("Reference last checkpoint mismatch")
    uninstrumented, _, control_info = run(
        graph, degrees, labels, seed, cap, 0, interval)
    if (control_info["effective_iterations"] != cap or
            not np.array_equal(ref, uninstrumented)):
        raise AssertionError("Reference observer altered predictions")
    previous = states[cap // 2]
    final = states[cap]
    ref_acc = float(np.mean(final == truth[mask]))
    halfway_acc = float(np.mean(previous == truth[mask]))
    drift_flips = int(np.count_nonzero(previous != final))
    drift_agreement = float(np.mean(previous == final))
    rows = []
    for p in [0] + list(patience_values):
        if p == 0:
            pred, elapsed, details = ref, ref_time, ref_info
            policy, criterion = "no_stop", "none"
        else:
            pred, elapsed, details = run(
                graph, degrees, labels, seed, cap, p,
                interval, "label_change", 0.0)
            policy, criterion = f"label_patience_{p}", "label_change"
        steps = int(details["effective_iterations"])
        if not 0 < steps <= cap:
            raise AssertionError("Invalid stopping iteration count")
        acc = float(np.mean(pred[mask] == truth[mask]))
        rows.append(dict(
            dataset=dataset, seed=seed, policy=policy,
            criterion=criterion, patience=p, interval=interval,
            reference_cap=cap, iterations=steps,
            early_stop=(steps < cap), iterations_saved=cap - steps,
            fraction_iterations_saved=(cap - steps) / cap,
            stop_reason=details.get("stop_reason", ""),
            accuracy=acc, reference_accuracy=ref_acc,
            accuracy_delta_pp=100.0 * (acc - ref_acc),
            agreement_with_reference=float(np.mean(
                pred[mask] == final)),
            elapsed_seconds=float(elapsed),
            reference_seconds=float(ref_time),
            time_saved_fraction=1.0 - elapsed / ref_time,
            reference_50k_accuracy=halfway_acc,
            reference_50k_to_100k_accuracy_delta_pp=100.0 * (
                ref_acc - halfway_acc),
            reference_50k_to_100k_flips=drift_flips,
            reference_50k_to_100k_agreement=drift_agreement,
            initially_unlabeled_nodes=int(np.count_nonzero(mask)),
            mean_degree=graph_info["mean_degree"],
            reciprocal_arc_fraction=graph_info["reciprocal_arc_fraction"],
            reference_matches_no_callback=True,
        ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2609, 2707, 2801, 2903])
    parser.add_argument("--patience", nargs="+", type=int,
                        default=[3, 6, 10])
    parser.add_argument("--cap", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or
            len(args.patience) != 3 or sorted(args.patience) != [3, 6, 10]
            or args.cap <= 0 or args.interval <= 0 or
            args.cap % (2 * args.interval)):
        parser.error("Invalid predeclared long-horizon protocol")
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            rows = audit_case(
                args.dataset, seed, args.cap, args.interval, args.patience)
            writer.writerows(rows)
            print(
                f"{args.dataset} seed={seed}: " +
                ", ".join(
                    f"{row['policy']}={row['iterations']}"
                    for row in rows),
                flush=True)


if __name__ == "__main__":
    main()
