"""Summarize experimental dominance traces without changing PCC behavior.

Usage:
    python -m experiments.analyze_dominance_traces dominance-stopping-*.csv

Reports distribution of per-check maximum dominance changes, and the longest
consecutive run of checks below each absolute threshold. These are diagnostics,
not evidence of predictive convergence.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from statistics import median


def longest_streak(values, threshold):
    current = longest = 0
    for value in values:
        current = current + 1 if value <= threshold else 0
        longest = max(longest, current)
    return longest


def summarize(paths, thresholds):
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["criterion"] not in ("dominance_change", "dominance_relative"):
                    continue
                trace = json.loads(row["dominance_change_trace"])
                values = [float(change) for _, change in trace]
                if not values or not all(math.isfinite(x) for x in values):
                    raise ValueError("missing or nonfinite dominance trace in " + path)
                groups[(row["dataset"], row["criterion"], row["tolerance"])].append(
                    (values, float(row["relative_baseline_change"])
                     if row["relative_baseline_change"] else None))
    for (dataset, criterion, tolerance), trials in sorted(groups.items()):
        all_changes = [v for values, _ in trials for v in values]
        print(f"{dataset} {criterion} tolerance={tolerance} runs={len(trials)} "
              f"checks={len(all_changes)} median={median(all_changes):.6g} "
              f"min={min(all_changes):.6g} max={max(all_changes):.6g}")
        for threshold in thresholds:
            streaks = [longest_streak(values, threshold) for values, _ in trials]
            print(f"  absolute <= {threshold:g}: "
                  f"median longest streak={median(streaks):g}, max={max(streaks)}")
        if criterion == "dominance_relative":
            streaks = [longest_streak(values, float(tolerance) * baseline)
                       for values, baseline in trials if baseline is not None]
            if streaks:
                print(f"  configured relative threshold: "
                      f"median longest streak={median(streaks):g}, max={max(streaks)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="Pilot CSV output files")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.001, 0.01, 0.05, 0.1])
    args = parser.parse_args(argv)
    if any(not math.isfinite(x) or x < 0 for x in args.thresholds):
        parser.error("thresholds must be finite and nonnegative")
    summarize(args.csv, args.thresholds)


if __name__ == "__main__":
    main()
