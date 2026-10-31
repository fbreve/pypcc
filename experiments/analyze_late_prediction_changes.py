"""Inspect late label flips and dominance margins without modifying production PCC.

Run: python -m experiments.analyze_late_prediction_changes --output late-flips.csv
Each cap restarts from the same seed and initial labels; this is not a
single continuously sampled trajectory. Margins use the two largest
dominance values at the earlier cap.
"""
import argparse
import csv
import numpy as np

from experiments.compare_dominance_stopping import DATASETS, ExperimentalPCC, prepare


FIELDS = ("dataset", "seed", "from_cap", "to_cap", "unlabeled",
          "changed", "change_fraction", "mean_margin_changed",
          "median_margin_changed", "mean_margin_unchanged",
          "changed_margin_below_001", "unchanged_margin_below_001")


def snapshot(graph, degrees, labels, seed, cap):
    model = ExperimentalPCC("reference", 0.0)
    model.set_graph(graph, degrees)
    model.fit_predict(labels.copy(), max_iter=cap, seed=seed, early_stop=False)
    dominance = model.node.dominance.copy()
    return np.argmax(dominance, axis=1), dominance


def compare(name, seed, caps=(10000, 40000, 50000), labeled_per_class=2):
    graph, degrees, labels, _ = prepare(
        name, seed, labeled_per_class=labeled_per_class)
    mask = labels == -1
    observations = {cap: snapshot(graph, degrees, labels, seed, cap)
                    for cap in caps}
    rows = []
    for earlier, later in zip(caps[:-1], caps[1:]):
        old_predictions, old_dominance = observations[earlier]
        new_predictions, _ = observations[later]
        changed = old_predictions[mask] != new_predictions[mask]
        margins = np.sort(old_dominance[mask], axis=1)
        margins = margins[:, -1] - margins[:, -2]
        unchanged = ~changed
        rows.append(dict(
            dataset=name, seed=seed, from_cap=earlier, to_cap=later,
            unlabeled=int(mask.sum()), changed=int(changed.sum()),
            change_fraction=float(changed.mean()),
            mean_margin_changed=float(margins[changed].mean()) if changed.any() else "",
            median_margin_changed=float(np.median(margins[changed])) if changed.any() else "",
            mean_margin_unchanged=float(margins[unchanged].mean()) if unchanged.any() else "",
            changed_margin_below_001=float(np.mean(margins[changed] < .01))
            if changed.any() else "",
            unchanged_margin_below_001=float(np.mean(margins[unchanged] < .01))
            if unchanged.any() else ""))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=sorted(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 41, 53])
    parser.add_argument("--caps", nargs="+", type=int, default=[10000, 40000, 50000])
    parser.add_argument("--labeled-per-class", type=int, default=2)
    args = parser.parse_args(argv)
    caps = sorted(set(args.caps))
    if len(caps) < 2 or caps[0] < 1:
        parser.error("at least two positive caps required")
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                writer.writerows(compare(name, seed, caps, args.labeled_per_class))


if __name__ == "__main__":
    main()
