"""Counterfactual global label-stability stop on independently replayed checkpoints.

No early stop is executed: this audits the first sampled checkpoint at which
ALL initially unlabeled nodes retain their predicted class for a given patience.
The last cap is only a capped reference, not proof of convergence.
"""
import argparse
import csv
import numpy as np
from experiments.compare_dominance_stopping import DATASETS, prepare
from experiments.analyze_late_prediction_changes import snapshot

FIELDS = ("dataset", "seed", "patience", "first_cap", "reference_cap",
          "stop_cap", "triggered", "iterations_saved",
          "reference_accuracy", "stop_accuracy", "accuracy_delta",
          "reference_agreement", "unlabeled")


def evaluate(predictions, truth, patience, caps):
    predictions = np.asarray(predictions)
    caps = np.asarray(caps)
    if predictions.ndim != 2 or len(predictions) != len(caps):
        raise ValueError("predictions/caps mismatch")
    if not 1 <= patience < len(caps):
        raise ValueError("patience out of range")
    # Each check means no change across one consecutive sampled interval.
    stop_index = len(caps) - 1
    for end in range(patience, len(caps) - 1):
        if np.all(predictions[end - patience:end + 1] ==
                  predictions[end]):
            stop_index = end
            break
    reference = predictions[-1]
    stopped = predictions[stop_index]
    return dict(stop_cap=int(caps[stop_index]),
                triggered=bool(stop_index < len(caps) - 1),
                iterations_saved=int(caps[-1] - caps[stop_index]),
                reference_accuracy=float(np.mean(reference == truth)),
                stop_accuracy=float(np.mean(stopped == truth)),
                accuracy_delta=float(np.mean(stopped == truth) -
                                     np.mean(reference == truth)),
                reference_agreement=float(np.mean(stopped == reference)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=sorted(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 23, 37, 41, 53])
    parser.add_argument("--caps", nargs="+", type=int,
                        default=[10000, 20000, 30000, 40000, 50000])
    parser.add_argument("--patience", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args(argv)
    caps = sorted(set(args.caps))
    if len(caps) < 3 or caps[0] <= 0 or not args.patience or (
            min(args.patience) < 1 or max(args.patience) >= len(caps)):
        parser.error("invalid caps or patience")
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                graph, degrees, labels, truth = prepare(name, seed)
                mask = labels == -1
                predictions = np.stack([
                    snapshot(graph, degrees, labels, seed, cap)[0][mask]
                    for cap in caps])
                for patience in args.patience:
                    writer.writerow(dict(dataset=name, seed=seed,
                        patience=patience, first_cap=caps[0],
                        reference_cap=caps[-1], unlabeled=int(mask.sum()),
                        **evaluate(predictions, truth[mask], patience, caps)))


if __name__ == "__main__":
    main()
