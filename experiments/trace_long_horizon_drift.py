"""Trace label drift in one uninterrupted PCC run; no early stopping.

Designed for targeted cases where 50k and 100k capped predictions differ.
A separate 50k replay verifies that checkpoint observation preserves results.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


CHECKPOINT_FIELDS = (
    "dataset", "seed", "iteration", "unlabeled", "accuracy",
    "changed_since_prior", "changed_vs_50k", "changed_vs_100k",
    "p95_dominance_change_since_prior", "mean_top_two_margin",
)
SUMMARY_FIELDS = (
    "dataset", "seed", "unlabeled", "checkpoints", "accuracy_50k",
    "accuracy_100k", "agreement_50k_100k", "nodes_changed_after_50k",
    "nodes_returned_to_50k_by_100k", "total_adjacent_flips_after_50k",
)


def trace_case(name, seed, interval, long_cap, short_cap):
    graph, degrees, labels, truth = prepare(name, seed)
    unknown = labels == -1
    snapshots = {}

    def observe(iteration, dominance):
        if iteration in snapshots:
            raise AssertionError("duplicate checkpoint")
        # The callback receives a defensive copy of the current state.
        snapshots[iteration] = dominance[unknown].copy()

    reference_50, _, _ = run(
        graph, degrees, labels, seed, short_cap, 0, interval)
    reference_100, _, info = run(
        graph, degrees, labels, seed, long_cap, 0, interval,
        checkpoint_callback=observe, checkpoint_interval=interval)
    if info["effective_iterations"] != long_cap:
        raise AssertionError("uninterrupted run stopped early")
    for cap, reference in [(short_cap, reference_50),
                           (long_cap, reference_100)]:
        if cap not in snapshots:
            raise AssertionError("reference checkpoint missing")
        observed = np.argmax(snapshots[cap], axis=1)
        if not np.array_equal(observed, reference[unknown]):
            raise AssertionError(
                f"checkpoint changed the trajectory: {name} {seed} at {cap}")

    fixed_50 = np.argmax(snapshots[short_cap], axis=1)
    fixed_100 = np.argmax(snapshots[long_cap], axis=1)
    rows = []
    previous_dom = None
    previous_labels = None
    changed_after_50 = np.zeros(fixed_50.size, dtype=bool)
    adjacent_flips_after_50 = 0
    for iteration, dom in sorted(snapshots.items()):
        predicted = np.argmax(dom, axis=1)
        top_two = np.partition(dom, -2, axis=1)[:, -2:]
        margins = top_two[:, 1] - top_two[:, 0]
        changed_prior = (int(np.count_nonzero(predicted != previous_labels))
                         if previous_labels is not None else None)
        p95_dom = (float(np.percentile(
            np.max(np.abs(dom - previous_dom), axis=1), 95))
            if previous_dom is not None else None)
        if iteration > short_cap:
            changed_after_50 |= predicted != fixed_50
            adjacent_flips_after_50 += changed_prior
        rows.append(dict(
            dataset=name, seed=seed, iteration=iteration,
            unlabeled=int(unknown.sum()),
            accuracy=float(np.mean(predicted == truth[unknown])),
            changed_since_prior=changed_prior,
            changed_vs_50k=int(np.count_nonzero(predicted != fixed_50)),
            changed_vs_100k=int(np.count_nonzero(predicted != fixed_100)),
            p95_dominance_change_since_prior=p95_dom,
            mean_top_two_margin=float(np.mean(margins)),
        ))
        previous_dom = dom
        previous_labels = predicted

    summary = dict(
        dataset=name, seed=seed, unlabeled=int(unknown.sum()),
        checkpoints=len(rows),
        accuracy_50k=float(np.mean(fixed_50 == truth[unknown])),
        accuracy_100k=float(np.mean(fixed_100 == truth[unknown])),
        agreement_50k_100k=float(np.mean(fixed_50 == fixed_100)),
        nodes_changed_after_50k=int(changed_after_50.sum()),
        nodes_returned_to_50k_by_100k=int(np.count_nonzero(
            changed_after_50 & (fixed_100 == fixed_50))),
        total_adjacent_flips_after_50k=adjacent_flips_after_50,
    )
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--cases", nargs="+", default=[
        "wine:37", "wine:67", "wine:83", "iris:11", "iris:37"])
    parser.add_argument("--interval", type=int, default=5000)
    parser.add_argument("--short-cap", type=int, default=50000)
    parser.add_argument("--long-cap", type=int, default=100000)
    args = parser.parse_args(argv)
    if (args.interval < 1 or not 0 < args.short_cap < args.long_cap
            or args.short_cap % args.interval
            or args.long_cap % args.interval):
        parser.error("invalid caps or checkpoint interval")

    with (open(args.checkpoints_output, "w", newline="",
               encoding="utf-8") as checkpoint_file,
          open(args.summary_output, "w", newline="",
               encoding="utf-8") as summary_file):
        writer = csv.DictWriter(checkpoint_file, fieldnames=CHECKPOINT_FIELDS)
        summary_writer = csv.DictWriter(summary_file,
                                        fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        summary_writer.writeheader()
        for case in args.cases:
            try:
                name, value = case.split(":")
                seed = int(value)
            except ValueError:
                parser.error(f"invalid case: {case}; expected dataset:seed")
            checkpoints, summary = trace_case(
                name, seed, args.interval, args.long_cap, args.short_cap)
            writer.writerows(checkpoints)
            summary_writer.writerow(summary)


if __name__ == "__main__":
    main()
