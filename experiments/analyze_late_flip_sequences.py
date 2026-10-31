"""Count repeated prediction changes across fine-grained late checkpoints.

Checkpoints are independently replayed with identical initial labels and seed.
An A-B-A pattern is evidence of sampled reversion, not a complete trajectory.
"""
import argparse
import csv

import numpy as np

from experiments.compare_dominance_stopping import DATASETS, prepare
from experiments.analyze_late_prediction_changes import snapshot


FIELDS = ("dataset", "seed", "first_cap", "last_cap", "checkpoint_count",
          "unlabeled", "nodes_changed_at_least_once",
          "nodes_changed_more_than_once", "total_adjacent_changes",
          "nodes_reverted_to_initial", "nodes_changed_at_endpoints",
          "nodes_changed_internally_but_same_endpoints")


def analyze(name, seed, caps, labeled_per_class=2):
    graph, degrees, labels, _ = prepare(
        name, seed, labeled_per_class=labeled_per_class)
    mask = labels == -1
    predictions = np.stack([
        snapshot(graph, degrees, labels, seed, cap)[0][mask]
        for cap in caps
    ])
    counts = np.count_nonzero(predictions[1:] != predictions[:-1], axis=0)
    same_endpoints = predictions[0] == predictions[-1]
    reverted = np.any(
        (predictions[1:-1] != predictions[0]) &
        (predictions[1:-1] != predictions[-1]), axis=0
    ) & same_endpoints
    return dict(
        dataset=name, seed=seed, first_cap=caps[0], last_cap=caps[-1],
        checkpoint_count=len(caps), unlabeled=int(mask.sum()),
        nodes_changed_at_least_once=int(np.count_nonzero(counts)),
        nodes_changed_more_than_once=int(np.count_nonzero(counts > 1)),
        total_adjacent_changes=int(counts.sum()),
        nodes_reverted_to_initial=int(np.count_nonzero(reverted)),
        nodes_changed_at_endpoints=int(np.count_nonzero(~same_endpoints)),
        nodes_changed_internally_but_same_endpoints=int(np.count_nonzero(
            (counts > 0) & same_endpoints)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=sorted(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 23, 37, 41, 53])
    parser.add_argument("--caps", nargs="+", type=int,
                        default=[40000, 42000, 44000, 46000, 48000, 50000])
    parser.add_argument("--labeled-per-class", type=int, default=2)
    args = parser.parse_args(argv)
    caps = sorted(set(args.caps))
    if len(caps) < 3 or caps[0] < 1:
        parser.error("at least three positive distinct caps required")
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                writer.writerow(analyze(name, seed, caps, args.labeled_per_class))


if __name__ == "__main__":
    main()
