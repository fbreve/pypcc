"""Audit sampled label-stability windows followed by a later class change.

This is a retrospective per-node diagnostic, NOT a validated stopping rule.
Checkpoints are independently replayed from identical seeds.
"""
import argparse
import csv
import numpy as np
from experiments.compare_dominance_stopping import DATASETS, prepare
from experiments.analyze_late_prediction_changes import snapshot

FIELDS = ("dataset", "seed", "window_checks", "checkpoint_step",
          "eligible_node_windows", "apparently_stable_windows",
          "stable_then_changed", "stable_then_changed_fraction",
          "nodes_with_false_stability", "max_stable_checks_before_change")


def summarize(predictions, window_checks, step):
    """Rows are ordered checkpoints, columns are initially unlabeled nodes."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[0] < 3:
        raise ValueError("predictions must have >=3 checkpoints and 2 dimensions")
    if not 1 <= window_checks < predictions.shape[0] - 1:
        raise ValueError("window_checks must leave a future checkpoint")
    n_times, n_nodes = predictions.shape
    stable = 0
    changed_later = 0
    false_nodes = np.zeros(n_nodes, dtype=bool)
    max_streak = 0
    eligible = 0
    for end in range(window_checks, n_times - 1):
        eligible += n_nodes
        start = end - window_checks
        is_stable = np.all(predictions[start:end + 1] ==
                           predictions[end], axis=0)
        future_change = np.any(predictions[end + 1:] !=
                               predictions[end], axis=0)
        stable += int(is_stable.sum())
        failed = is_stable & future_change
        changed_later += int(failed.sum())
        false_nodes |= failed
    for node in range(n_nodes):
        streak = 0
        for t in range(1, n_times):
            if predictions[t, node] == predictions[t - 1, node]:
                streak += 1
            else:
                max_streak = max(max_streak, streak)
                streak = 0
    return dict(window_checks=window_checks, checkpoint_step=step,
                eligible_node_windows=eligible,
                apparently_stable_windows=stable,
                stable_then_changed=changed_later,
                stable_then_changed_fraction=(changed_later / stable
                                              if stable else 0.0),
                nodes_with_false_stability=int(false_nodes.sum()),
                max_stable_checks_before_change=max_streak)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=["breast_cancer"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 23, 37, 41, 53])
    parser.add_argument("--caps", nargs="+", type=int,
                        default=list(range(40000, 50001, 1000)))
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args(argv)
    caps = sorted(set(args.caps))
    if len(caps) < 3 or caps[0] <= 0:
        parser.error("at least three positive checkpoints are required")
    steps = np.diff(caps)
    if not np.all(steps == steps[0]):
        parser.error("checkpoint spacing must be uniform")
    if any(w < 1 or w >= len(caps) - 1 for w in args.windows):
        parser.error("windows must be >=1 and leave a future checkpoint")
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
                                         **summarize(predictions, w, int(steps[0]))))


if __name__ == "__main__":
    main()
