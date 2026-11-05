"""Static graph diagnostic for the predeclared nested Digits label study.

No PCC propagation runs, no algorithm changes. These neighborhood statistics
help determine whether the selected labeled 'home' nodes have obviously
cross-class local neighborhoods. They do not diagnose dynamic lockout by
themselves.
"""
import argparse
import csv

import numpy as np

from experiments.compare_dominance_stopping import prepare


FIELDS = (
    "dataset", "seed", "labeled_per_class", "class_id",
    "labeled_home_nodes", "home_nodes", "home_outgoing_neighbor_entries",
    "home_outgoing_true_same_class", "home_outgoing_true_other_class",
    "home_true_same_class_fraction",
    "home_min_true_same_class_fraction",
    "incoming_edges_to_home_nodes",
    "incoming_edges_from_true_same_class",
    "home_outgoing_labeled_neighbor_entries",
    "home_outgoing_labeled_neighbor_other_class",
)


def build_nested(seed):
    graph, degrees, full_labels, truth = prepare(
        "digits", seed, neighbors=10, labeled_per_class=5)
    rng = np.random.default_rng(seed)
    choices = {}
    for cls in np.unique(truth):
        choices[int(cls)] = rng.choice(
            np.flatnonzero(truth == cls), size=5, replace=False)
    labels = np.full(len(truth), -1, dtype=np.int64)
    for cls, nodes in choices.items():
        labels[nodes] = cls
    if not np.array_equal(labels, full_labels):
        raise AssertionError("Reconstructed five-label choices differ from prepare")
    return graph, degrees, truth, choices


def audit(seed):
    graph, degrees, truth, choices = build_nested(seed)
    n = len(truth)
    if not np.all(degrees == 10):
        raise AssertionError("Unexpected graph degree")
    if graph.shape != (n, 10):
        raise AssertionError("Unexpected graph shape")
    if not np.all((graph >= 0) & (graph < n)):
        raise AssertionError("Invalid graph target")
    if np.any(graph == np.arange(n)[:, None]):
        raise AssertionError("Graph contains a self-edge")

    rows = []
    for n_labeled in [2, 5]:
        label_map = np.full(n, -1, dtype=np.int64)
        for cls, nodes in choices.items():
            label_map[nodes[:n_labeled]] = cls
        for cls, nodes in choices.items():
            home = np.asarray(nodes[:n_labeled])
            outgoing = graph[home]
            neighbors = outgoing.ravel()
            same = truth[neighbors] == cls
            neighbor_labeled = label_map[neighbors] != -1
            incoming_entries = np.isin(graph, home)
            incoming_same = int(np.count_nonzero(
                incoming_entries & (truth[:, None] == cls)))
            rows.append(dict(
                dataset="digits", seed=seed, labeled_per_class=n_labeled,
                class_id=cls, labeled_home_nodes=len(home),
                home_nodes=";".join(map(str, home.tolist())),
                home_outgoing_neighbor_entries=int(neighbors.size),
                home_outgoing_true_same_class=int(same.sum()),
                home_outgoing_true_other_class=int((~same).sum()),
                home_true_same_class_fraction=float(np.mean(same)),
                home_min_true_same_class_fraction=float(np.min(
                    np.mean(truth[outgoing] == cls, axis=1))),
                incoming_edges_to_home_nodes=int(incoming_entries.sum()),
                incoming_edges_from_true_same_class=incoming_same,
                home_outgoing_labeled_neighbor_entries=int(
                    neighbor_labeled.sum()),
                home_outgoing_labeled_neighbor_other_class=int(
                    np.count_nonzero(
                        neighbor_labeled &
                        (label_map[neighbors] != cls))),
            ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[601, 701, 809])
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide unique seeds")
    with open(args.output, "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            writer.writerows(audit(seed))


if __name__ == "__main__":
    main()
