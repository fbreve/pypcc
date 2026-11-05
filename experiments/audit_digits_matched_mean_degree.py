"""Test directed 14-NN versus symmetric union of directed 10-NN on Digits.

The cohort and endpoints were fixed in digits_matched_mean_degree_protocol.md
before analyzing any output. Experimental sequential NumPy backend only.
"""
import argparse

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from experiments.compare_dominance_stopping import DATASETS, prepare
from experiments.audit_digits_symmetric_graph_lockout import (
    GRAPH_FIELDS, WINDOW_FIELDS, CLASS_FIELDS, CASE_FIELDS,
    make_symmetric_union, graph_stats, observe_variant, save_csv,
)


VARIANTS = (
    "directed_10nn", "directed_14nn", "symmetric_union_10nn")
EXT_CASE_FIELDS = CASE_FIELDS + ("agreement_with_symmetric_union_10nn",)


def build_directed_four_extras(directed, degrees):
    """Append 4 nearest *new* points, retaining original 10-NN prefix."""
    sample = DATASETS["digits"]()
    features = StandardScaler().fit_transform(
        sample.data.astype(np.float64))
    if directed.shape[0] != len(features):
        raise AssertionError("Digits dataset unexpectedly changed")
    nn = NearestNeighbors(n_neighbors=20).fit(features)
    raw_neighbors = nn.kneighbors(features, return_distance=False)
    nodes = len(features)
    if not np.all(degrees == 10) or directed.shape != (nodes, 10):
        raise AssertionError("Expected original fixed 10-NN graph")
    out = np.empty((nodes, 14), dtype=np.int64)
    out[:, :10] = directed
    for node, neighbors in enumerate(raw_neighbors):
        existing = {int(x) for x in directed[node]}
        if len(existing) != 10 or node in existing:
            raise AssertionError("Invalid original outgoing adjacency")
        extras = []
        for target in neighbors:
            target = int(target)
            if (target != node and target not in existing
                    and target not in extras):
                extras.append(target)
                if len(extras) == 4:
                    break
        if len(extras) != 4:
            raise AssertionError(
                f"Could not find 4 unique extra neighbors for node {node}")
        out[node, 10:] = extras
        if node in out[node] or len(set(out[node].tolist())) != 14:
            raise AssertionError("New directed14 adjacency is not unique")
    degree14 = np.full(nodes, 14, dtype=np.int64)
    return out, degree14


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--graph-output", required=True)
    parser.add_argument("--windows-output", required=True)
    parser.add_argument("--classes-output", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--zero-windows", type=int, default=5)
    parser.add_argument("--strength-cutoff", type=float, default=0.01)
    args = parser.parse_args(argv)
    if (args.seed < 0 or args.cap < 1 or args.interval < 1
            or args.cap % args.interval or args.zero_windows < 1
            or not 0 < args.strength_cutoff <= 1):
        parser.error("Invalid protocol parameters")

    directed, d10, labels, truth = prepare(
        "digits", args.seed, neighbors=10, labeled_per_class=2)
    directed14, d14 = build_directed_four_extras(directed, d10)
    symmetric, dsym = make_symmetric_union(directed, d10)
    graphs = {
        "directed_10nn": (directed, d10),
        "directed_14nn": (directed14, d14),
        "symmetric_union_10nn": (symmetric, dsym),
    }

    graph_rows, window_rows, class_rows, case_rows = [], [], [], []
    predictions = {}
    for variant in VARIANTS:
        graph, degrees = graphs[variant]
        if not np.array_equal(graph[:, :10], directed):
            raise AssertionError("Original 10-NN ordering was not preserved")
        graph_rows.append(graph_stats(
            args.seed, variant, graph, degrees, d10))
        prediction, windows, classes, case = observe_variant(
            args.seed, variant, graph, degrees, labels, truth,
            args.cap, args.interval, args.zero_windows,
            args.strength_cutoff)
        predictions[variant] = prediction
        window_rows.extend(windows)
        class_rows.extend(classes)
        case_rows.append(case)
    unlabeled = labels == -1
    for case in case_rows:
        variant = case["variant"]
        case["agreement_with_directed_10nn"] = float(np.mean(
            predictions[variant][unlabeled] ==
            predictions["directed_10nn"][unlabeled]))
        case["agreement_with_symmetric_union_10nn"] = float(np.mean(
            predictions[variant][unlabeled] ==
            predictions["symmetric_union_10nn"][unlabeled]))
    save_csv(args.graph_output, GRAPH_FIELDS, graph_rows)
    save_csv(args.windows_output, WINDOW_FIELDS, window_rows)
    save_csv(args.classes_output, CLASS_FIELDS, class_rows)
    save_csv(args.cases_output, EXT_CASE_FIELDS, case_rows)
    for case in case_rows:
        print(f"seed={args.seed} graph={case['variant']} "
              f"lockouts={case['lockout_count']} "
              f"accuracy={case['accuracy']:.6f} "
              f"seconds={case['uninstrumented_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
