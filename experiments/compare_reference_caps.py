"""Compare capped reference trajectories; no stopping heuristic is enabled.

Run: python -m experiments.compare_reference_caps --output reference-caps.csv
The longest cap is a comparison baseline, not a convergence guarantee.
"""
import argparse
import csv

import numpy as np

from experiments.compare_dominance_stopping import DATASETS, prepare, run_case


FIELDS = ("dataset", "seed", "cap", "baseline_cap", "neighbors",
          "labeled_per_class", "accuracy", "agreement_with_longest_cap",
          "elapsed_seconds", "iterations", "previous_cap",
          "agreement_with_previous_cap", "changed_since_previous_cap")


def compare_caps(name, seed, caps=(1000, 3000, 10000), *,
                 neighbors=10, labeled_per_class=2):
    caps = sorted(set(caps))
    if not caps or caps[0] < 1:
        raise ValueError("caps must be positive")
    graph, degrees, labels, truth = prepare(
        name, seed, neighbors=neighbors, labeled_per_class=labeled_per_class)
    unknown = labels == -1
    results = {}
    for cap in caps:
        predicted, accuracy, elapsed, info = run_case(
            graph, degrees, labels, truth, seed, "reference", cap, 0.0)
        results[cap] = (predicted, accuracy, elapsed, info)
    baseline = results[caps[-1]][0]
    rows = []
    for index, cap in enumerate(caps):
        predicted, accuracy, elapsed, info = results[cap]
        previous_cap = caps[index - 1] if index else None
        previous = results[previous_cap][0] if previous_cap is not None else None
        agreement_previous = (float(np.mean(predicted[unknown] == previous[unknown]))
                              if previous is not None else None)
        rows.append(dict(
            dataset=name, seed=seed, cap=cap, baseline_cap=caps[-1],
            neighbors=neighbors, labeled_per_class=labeled_per_class,
            accuracy=accuracy,
            agreement_with_longest_cap=float(np.mean(
                predicted[unknown] == baseline[unknown])),
            elapsed_seconds=elapsed,
            iterations=info["effective_iterations"],
            previous_cap=previous_cap,
            agreement_with_previous_cap=agreement_previous,
            changed_since_previous_cap=(int(np.count_nonzero(
                predicted[unknown] != previous[unknown]))
                if previous is not None else None)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=sorted(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--caps", nargs="+", type=int,
                        default=[1000, 3000, 10000])
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--labeled-per-class", type=int, default=2)
    args = parser.parse_args(argv)
    if any(cap < 1 for cap in args.caps):
        parser.error("caps must be positive")
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                writer.writerows(compare_caps(
                    name, seed, args.caps, neighbors=args.neighbors,
                    labeled_per_class=args.labeled_per_class))


if __name__ == "__main__":
    main()
