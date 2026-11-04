"""Measure actual in-loop global label-stability stopping on small real datasets.

Candidate-only NumPy backend; never modifies production PCC.
"""
import argparse
import csv
import inspect
import time
import numpy as np

from experiments.compare_dominance_stopping import ExperimentalPCC, prepare
from experiments.pcc_numpy_dominance_candidate import pcc_propagate_numpy


class InLoopPCC(ExperimentalPCC):
    def __init__(self, patience, interval, criterion='label_change', tolerance=0.0):
        super().__init__("label_change", 0.0)
        self.patience = patience
        self.interval = interval
        self.criterion = criterion
        self.tolerance = tolerance

    def _get_backend_fn(self):
        def propagate(*args, **kwargs):
            bound = inspect.signature(pcc_propagate_numpy).bind(*args, **kwargs)
            bound.arguments.update(return_info=True, stop_criterion=self.criterion,
                                   dominance_tol=self.tolerance, check_interval=self.interval,
                                   # The tracker stops after stalled_checks > stop_max.
                                   stop_max=max(0, self.patience - 1))
            self.propagation_info = pcc_propagate_numpy(*bound.args, **bound.kwargs)
            return self.propagation_info
        return propagate


FIELDS = ("dataset", "seed", "patience", "interval", "cap", "iterations",
          "elapsed_seconds", "reference_seconds", "iterations_saved",
          "accuracy", "reference_accuracy", "accuracy_delta",
          "reference_agreement", "criterion", "tolerance")


def run(graph, degrees, labels, seed, cap, patience, interval,
        criterion="label_change", tolerance=0.0):
    model = InLoopPCC(patience, interval, criterion, tolerance)
    model.set_graph(graph, degrees)
    started = time.perf_counter()
    prediction = model.fit_predict(labels.copy(), max_iter=cap, seed=seed,
                                   early_stop=(patience > 0))
    return prediction, time.perf_counter() - started, model.propagation_info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", default=["iris", "breast_cancer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--intervals", nargs="+", type=int, default=[1000, 5000])
    parser.add_argument("--patience", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--criteria", nargs="+", choices=["label_change", "dominance_label_hybrid"],
                        default=["label_change", "dominance_label_hybrid"])
    parser.add_argument("--tolerances", nargs="+", type=float, default=[0.001, 0.01])
    args = parser.parse_args(argv)
    if args.cap < 1 or not args.intervals or min(args.intervals) < 1 or (
            not args.patience or min(args.patience) < 1):
        parser.error("invalid cap, interval or patience")
    if not args.criteria or not args.tolerances or any(
            not np.isfinite(t) or t < 0 for t in args.tolerances):
        parser.error("invalid criteria or tolerances")
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                graph, degrees, labels, truth = prepare(name, seed)
                unknown = labels == -1
                ref, ref_time, ref_info = run(graph, degrees, labels, seed,
                                              args.cap, 0, 1)
                ref_accuracy = float(np.mean(ref[unknown] == truth[unknown]))
                for criterion in args.criteria:
                    tolerances = ([0.0] if criterion == "label_change"
                                  else args.tolerances)
                    for tolerance in tolerances:
                        for interval in args.intervals:
                            for patience in args.patience:
                                pred, elapsed, info = run(
                                    graph, degrees, labels, seed, args.cap,
                                    patience, interval, criterion, tolerance)
                                accuracy = float(np.mean(
                                    pred[unknown] == truth[unknown]))
                                writer.writerow(dict(
                                    dataset=name, seed=seed,
                                    patience=patience, interval=interval,
                                    criterion=criterion, tolerance=tolerance,
                                    cap=args.cap,
                                    iterations=info["effective_iterations"],
                                    elapsed_seconds=elapsed,
                                    reference_seconds=ref_time,
                                    iterations_saved=args.cap-info["effective_iterations"],
                                    accuracy=accuracy,
                                    reference_accuracy=ref_accuracy,
                                    accuracy_delta=accuracy-ref_accuracy,
                                    reference_agreement=float(np.mean(
                                        pred[unknown] == ref[unknown]))))



if __name__ == "__main__":
    main()
