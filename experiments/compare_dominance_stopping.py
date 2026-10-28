"""Reproducible real-data comparison of PCC stopping criteria (NumPy only).

Run: python -m experiments.compare_dominance_stopping --output results.csv
The reference is a capped trajectory, NOT proof of convergence.
"""
import argparse
import csv
import time

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.neighbors import NearestNeighbors

from pcc import ParticleCompetitionAndCooperation


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}
FIELDS = ("dataset", "seed", "criterion", "tolerance", "iterations",
          "stop_reason", "cap_hit", "accuracy", "reference_agreement",
          "elapsed_seconds", "labeled_count", "unlabeled_count")


def prepare(name, seed, *, neighbors=10, labeled_per_class=2):
    from sklearn.preprocessing import StandardScaler

    data = DATASETS[name]()
    x = StandardScaler().fit_transform(data.data.astype(np.float64))
    truth = np.asarray(data.target, dtype=np.int64)
    if neighbors < 1 or neighbors >= len(truth):
        raise ValueError("neighbors must be between 1 and n_samples - 1")
    rng = np.random.default_rng(seed)
    labels = np.full(len(truth), -1, dtype=np.int64)
    for cls in np.unique(truth):
        candidates = np.flatnonzero(truth == cls)
        if len(candidates) < labeled_per_class:
            raise ValueError("not enough samples per class")
        labels[rng.choice(candidates, labeled_per_class, replace=False)] = cls
    nn = NearestNeighbors(n_neighbors=neighbors + 1).fit(x)
    raw = nn.kneighbors(x, return_distance=False)
    # Exclude self explicitly; ties may make self appear anywhere.
    edges = np.empty((len(truth), neighbors), dtype=np.int64)
    for i, row in enumerate(raw):
        nonself = row[row != i]
        if len(nonself) < neighbors:
            raise ValueError("nearest-neighbor query omitted a nonself neighbor")
        edges[i] = nonself[:neighbors]
    return edges, np.full(len(truth), neighbors, dtype=np.int64), labels, truth


def run_case(graph, degrees, labels, truth, seed, criterion, cap, tolerance):
    model = ParticleCompetitionAndCooperation(impl="numpy")
    model.set_graph(graph, degrees)
    kwargs = dict(max_iter=cap, seed=seed, early_stop=(criterion != "reference"))
    if criterion == "dominance_change":
        kwargs.update(stop_criterion="dominance_change", dominance_tol=tolerance)
    start = time.perf_counter()
    predicted = model.fit_predict(labels.copy(), **kwargs)
    elapsed = time.perf_counter() - start
    unknown = labels == -1
    return predicted, float(np.mean(predicted[unknown] == truth[unknown])), elapsed


def compare(name, seed, *, cap=1000, neighbors=10, labeled_per_class=2,
            tolerance=1e-6):
    if cap < 1:
        raise ValueError("cap must be positive")
    graph, degrees, labels, truth = prepare(
        name, seed, neighbors=neighbors, labeled_per_class=labeled_per_class)
    # Public API does not expose iteration counts. Record actual stopping
    # metadata in a subsequent instrumented protocol, not guessed values.
    reference, _, _ = run_case(graph, degrees, labels, truth, seed,
                                "reference", cap, tolerance)
    unknown = labels == -1
    rows = []
    for criterion in ("reference", "potential", "dominance_change"):
        predicted, accuracy, elapsed = run_case(
            graph, degrees, labels, truth, seed, criterion, cap, tolerance)
        rows.append(dict(dataset=name, seed=seed, criterion=criterion,
                         tolerance=tolerance if criterion == "dominance_change" else "",
                         iterations="", stop_reason="not_exposed",
                         cap_hit="unknown", accuracy=accuracy,
                         reference_agreement=float(np.mean(
                             predicted[unknown] == reference[unknown])),
                         elapsed_seconds=elapsed,
                         labeled_count=int((~unknown).sum()),
                         unlabeled_count=int(unknown.sum())))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=["iris", "wine", "breast_cancer", "digits"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--labeled-per-class", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                for row in compare(name, seed, cap=args.cap,
                                   neighbors=args.neighbors,
                                   labeled_per_class=args.labeled_per_class,
                                   tolerance=args.tolerance):
                    writer.writerow(row)


if __name__ == "__main__":
    main()
