"""Compare stability patience on identical decision checkpoints and future horizons.

Retrospective node-level diagnostic, not a global PCC stopping rule.
All patience values are evaluated at the same checkpoint indices.
"""
import argparse
import csv
import numpy as np

from experiments.compare_dominance_stopping import DATASETS, prepare
from experiments.analyze_late_prediction_changes import snapshot

FIELDS = ("dataset", "seed", "window_checks", "checkpoint_step",
          "decision_checkpoints", "eligible_node_windows",
          "apparently_stable_windows", "stable_then_changed",
          "stable_then_changed_fraction", "nodes_with_false_stability")


def summarize_fixed(predictions, window_checks, max_window, step,
                    future_checks=1):
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[0] < max_window + future_checks + 1:
        raise ValueError("insufficient checkpoints")
    if not 1 <= window_checks <= max_window or future_checks < 1:
        raise ValueError("invalid window or future horizon")
    n_times, n_nodes = predictions.shape
    false_nodes = np.zeros(n_nodes, dtype=bool)
    stable_count = false_count = 0
    decision_points = range(max_window, n_times - future_checks)
    for end in decision_points:
        stable = np.all(predictions[end - window_checks:end + 1] ==
                        predictions[end], axis=0)
        future_changed = np.any(
            predictions[end + 1:end + future_checks + 1] !=
            predictions[end], axis=0)
        failures = stable & future_changed
        stable_count += int(stable.sum())
        false_count += int(failures.sum())
        false_nodes |= failures
    return dict(window_checks=window_checks, checkpoint_step=step,
                decision_checkpoints=len(decision_points),
                eligible_node_windows=len(decision_points) * n_nodes,
                apparently_stable_windows=stable_count,
                stable_then_changed=false_count,
                stable_then_changed_fraction=(false_count / stable_count
                                              if stable_count else 0.0),
                nodes_with_false_stability=int(false_nodes.sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=["breast_cancer"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 23, 37, 41, 53])
    parser.add_argument("--caps", nargs="+", type=int,
                        default=list(range(40000, 50001, 1000)))
    parser.add_argument("--windows", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--future-checks", type=int, default=1)
    args = parser.parse_args(argv)
    caps = sorted(set(args.caps))
    steps = np.diff(caps)
    if len(caps) < 3 or caps[0] < 1 or not np.all(steps == steps[0]):
        parser.error("at least three uniformly spaced positive caps required")
    if (not args.windows or min(args.windows) < 1 or
            max(args.windows) + args.future_checks >= len(caps) or
            args.future_checks < 1):
        parser.error("invalid windows/future horizon")
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                graph, degrees, labels, _ = prepare(name, seed)
                mask = labels == -1
                predictions = np.stack([
                    snapshot(graph, degrees, labels, seed, cap)[0][mask]
                    for cap in caps])
                for w in args.windows:
                    writer.writerow(dict(dataset=name, seed=seed,
                        **summarize_fixed(predictions, w, max(args.windows),
                                          int(steps[0]), args.future_checks)))


if __name__ == "__main__":
    main()
