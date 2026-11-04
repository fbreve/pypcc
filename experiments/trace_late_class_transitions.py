"""Targeted 1k-resolution diagnostics of late Wine class transitions.

Records class populations and pre/post dominant-class margins of *flipped*
nodes, not just a global mean margin. Observation never changes propagation.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


TRAJECTORY_FIELDS = (
    "dataset", "seed", "iteration", "accuracy", "class_0_count",
    "class_1_count", "class_2_count", "changed_since_previous",
    "pre_flip_margin_median", "post_flip_margin_median",
    "pre_flip_margin_at_least_half", "p95_node_dominance_change",
    "max_node_dominance_change", "median_flip_node_dominance_change",
    "max_flip_node_dominance_change", "class_0_dominance_mass",
    "class_1_dominance_mass", "class_2_dominance_mass",
    "max_simplex_deviation", "nonfinite_dominance_entries",
)
TRANSITION_FIELDS = (
    "dataset", "seed", "from_iteration", "to_iteration",
    "from_class", "to_class", "count",
)


def margin(dominance):
    top_two = np.partition(dominance, -2, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


def trace(seed, start, end, interval):
    graph, degrees, labels, truth = prepare("wine", seed)
    unknown = labels == -1
    snapshots = {}

    def observe(it, dominance):
        if it >= start:
            snapshots[it] = dominance[unknown].copy()

    final, _, info = run(
        graph, degrees, labels, seed, end, 0, interval,
        checkpoint_callback=observe, checkpoint_interval=interval)
    if info["effective_iterations"] != end:
        raise AssertionError("unexpected early stop")
    if sorted(snapshots) != list(range(start, end + 1, interval)):
        raise AssertionError("unexpected checkpoint schedule")
    if not np.array_equal(
            np.argmax(snapshots[end], axis=1), final[unknown]):
        raise AssertionError("checkpoint differs from final prediction")

    rows, transitions = [], []
    previous = None
    for iteration in sorted(snapshots):
        dom = snapshots[iteration]
        predicted = np.argmax(dom, axis=1)
        if dom.shape[1] != 3:
            raise AssertionError("Wine class-count changed")
        counts = np.bincount(predicted, minlength=3)
        class_mass = dom.sum(axis=0)
        simplex_error = np.abs(dom.sum(axis=1) - 1.0)
        finite = np.isfinite(dom)
        row = dict(
            dataset="wine", seed=seed, iteration=iteration,
            accuracy=float(np.mean(predicted == truth[unknown])),
            class_0_count=int(counts[0]), class_1_count=int(counts[1]),
            class_2_count=int(counts[2]),
            changed_since_previous=None,
            pre_flip_margin_median=None, post_flip_margin_median=None,
            pre_flip_margin_at_least_half=None,
            p95_node_dominance_change=None,
            max_node_dominance_change=None,
            median_flip_node_dominance_change=None,
            max_flip_node_dominance_change=None,
            class_0_dominance_mass=float(class_mass[0]),
            class_1_dominance_mass=float(class_mass[1]),
            class_2_dominance_mass=float(class_mass[2]),
            max_simplex_deviation=float(np.max(simplex_error)),
            nonfinite_dominance_entries=int(np.size(finite) - finite.sum()),
        )
        if previous is not None:
            prev_iteration, prev_dom = previous
            prev_labels = np.argmax(prev_dom, axis=1)
            changed = predicted != prev_labels
            row["changed_since_previous"] = int(np.count_nonzero(changed))
            per_node_change = np.max(np.abs(dom - prev_dom), axis=1)
            row["p95_node_dominance_change"] = float(
                np.percentile(per_node_change, 95))
            row["max_node_dominance_change"] = float(
                np.max(per_node_change))
            if np.any(changed):
                row["median_flip_node_dominance_change"] = float(
                    np.median(per_node_change[changed]))
                row["max_flip_node_dominance_change"] = float(
                    np.max(per_node_change[changed]))
                pre_margin = margin(prev_dom)[changed]
                post_margin = margin(dom)[changed]
                row["pre_flip_margin_median"] = float(
                    np.median(pre_margin))
                row["post_flip_margin_median"] = float(
                    np.median(post_margin))
                row["pre_flip_margin_at_least_half"] = int(
                    np.count_nonzero(pre_margin >= 0.5))
                for source in range(3):
                    for target in range(3):
                        if source == target:
                            continue
                        count = int(np.count_nonzero(
                            changed & (prev_labels == source) &
                            (predicted == target)))
                        if count:
                            transitions.append(dict(
                                dataset="wine", seed=seed,
                                from_iteration=prev_iteration,
                                to_iteration=iteration,
                                from_class=source, to_class=target,
                                count=count))
        rows.append(row)
        previous = (iteration, dom)
    return rows, transitions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-output", required=True)
    parser.add_argument("--transitions-output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 83])
    parser.add_argument("--start", type=int, default=50000)
    parser.add_argument("--end", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args(argv)
    if (args.start <= 0 or args.end <= args.start or args.interval <= 0
            or args.start % args.interval or args.end % args.interval
            or not args.seeds):
        parser.error("invalid trajectory interval or seeds")
    with (open(args.trajectory_output, "w", newline="",
               encoding="utf-8") as first,
          open(args.transitions_output, "w", newline="",
               encoding="utf-8") as second):
        trajectory_writer = csv.DictWriter(
            first, fieldnames=TRAJECTORY_FIELDS)
        transition_writer = csv.DictWriter(
            second, fieldnames=TRANSITION_FIELDS)
        trajectory_writer.writeheader()
        transition_writer.writeheader()
        for seed in args.seeds:
            rows, transitions = trace(seed, args.start, args.end,
                                      args.interval)
            trajectory_writer.writerows(rows)
            transition_writer.writerows(transitions)


if __name__ == "__main__":
    main()
