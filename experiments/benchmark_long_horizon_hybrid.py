"""Audit prospective hybrid stopping against two independently capped references.

The 50k and 100k no-stop runs start from the same graph, labels and RNG seed.
The longer endpoint is an *additional* benchmark, not proof of convergence.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


FIELDS = (
    "dataset", "seed", "short_cap", "long_cap", "interval", "patience",
    "tolerance", "unlabeled", "reference_short_seconds",
    "reference_long_seconds", "hybrid_seconds", "hybrid_iterations",
    "hybrid_iterations_saved", "reference_short_accuracy",
    "reference_long_accuracy", "hybrid_accuracy",
    "reference_short_long_agreement", "hybrid_short_agreement",
    "hybrid_long_agreement", "hybrid_long_accuracy_delta",
)


def evaluate(graph, degrees, labels, truth, seed, short_cap, long_cap,
             interval, patience, tolerance):
    unknown = labels == -1
    short_pred, short_seconds, short_info = run(
        graph, degrees, labels, seed, short_cap, 0, 1)
    long_pred, long_seconds, long_info = run(
        graph, degrees, labels, seed, long_cap, 0, 1)
    hybrid_pred, hybrid_seconds, hybrid_info = run(
        graph, degrees, labels, seed, long_cap, patience, interval,
        criterion="dominance_label_hybrid", tolerance=tolerance)

    if short_info["effective_iterations"] != short_cap:
        raise AssertionError("short reference stopped unexpectedly")
    if long_info["effective_iterations"] != long_cap:
        raise AssertionError("long reference stopped unexpectedly")

    def agreement(a, b):
        return float(np.mean(a[unknown] == b[unknown]))

    def accuracy(pred):
        return float(np.mean(pred[unknown] == truth[unknown]))

    hybrid_iterations = int(hybrid_info["effective_iterations"])
    return dict(
        seed=seed, short_cap=short_cap, long_cap=long_cap,
        interval=interval, patience=patience, tolerance=tolerance,
        unlabeled=int(unknown.sum()),
        reference_short_seconds=short_seconds,
        reference_long_seconds=long_seconds,
        hybrid_seconds=hybrid_seconds,
        hybrid_iterations=hybrid_iterations,
        hybrid_iterations_saved=long_cap - hybrid_iterations,
        reference_short_accuracy=accuracy(short_pred),
        reference_long_accuracy=accuracy(long_pred),
        hybrid_accuracy=accuracy(hybrid_pred),
        reference_short_long_agreement=agreement(short_pred, long_pred),
        hybrid_short_agreement=agreement(hybrid_pred, short_pred),
        hybrid_long_agreement=agreement(hybrid_pred, long_pred),
        hybrid_long_accuracy_delta=accuracy(hybrid_pred) - accuracy(long_pred),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", default=["iris", "wine"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 23, 37, 67, 83])
    parser.add_argument("--short-cap", type=int, default=50000)
    parser.add_argument("--long-cap", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    if (not 0 < args.short_cap < args.long_cap or args.interval < 1
            or args.patience < 1 or not np.isfinite(args.tolerance)
            or args.tolerance < 0 or not args.seeds):
        parser.error("invalid experiment parameters")
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                graph, degrees, labels, truth = prepare(name, seed)
                writer.writerow(dict(dataset=name, **evaluate(
                    graph, degrees, labels, truth, seed, args.short_cap,
                    args.long_cap, args.interval, args.patience,
                    args.tolerance)))


if __name__ == "__main__":
    main()
