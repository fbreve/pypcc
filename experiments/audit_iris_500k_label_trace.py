"""Observe one full 500k NumPy PCC trajectory without modifying its state.

See iris_500k_label_trace_protocol.md. Strict final-output parity gates
ensure that observer callbacks cannot silently change the experiment.
"""
import argparse
import csv

import numpy as np

from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.audit_iris_long_horizon_stopping import (
    LONG_CAP, production_fit, candidate_fit,
)
from experiments.benchmark_inloop_global_stopping import run as candidate_run

SEEDS = (5003, 5303)
INTERVAL = 1000
CHECKPOINT_FIELDS = (
    "dataset", "seed", "iteration", "accuracy", "flips_since_previous_sample",
    "agreement_with_final_500k", "flips_vs_final_500k",
    "mean_max_dominance", "predicted_class_0", "predicted_class_1",
    "predicted_class_2",
)
SUMMARY_FIELDS = (
    "dataset", "seed", "cap", "interval", "sample_count", "unlabeled",
    "callback_uninstrumented_production_500k_parity",
    "accuracy_100k", "accuracy_250k", "accuracy_500k",
    "flips_100k_vs_250k", "flips_100k_vs_500k",
    "flips_250k_vs_500k",
    "sampled_flips_after_100k",
    "unique_nodes_different_from_100k_after_100k",
    "nodes_changed_and_returned_to_100k_at_500k",
    "longest_zero_flip_sample_streak_before_later_flip",
    "label_p10_iterations", "label_p10_accuracy",
    "label_p10_accuracy_delta_pp_vs_500k",
    "label_p10_agreement_vs_500k", "label_p10_flips_vs_500k",
    "sampled_flips_after_label_p10_stop",
    "nodes_ever_differing_from_label_p10_after_stop",
    "production_500k_seconds", "candidate_unobserved_500k_seconds",
    "candidate_observed_500k_seconds",
)


def audit(seed):
    graph, degrees, labels, truth, stats = canonical_prepare(
        "iris", seed, k=10, labeled_per_class=2)
    if stats["reciprocal_arc_fraction"] != 1.0:
        raise AssertionError("Directed graph in canonical experiment")
    mask = labels == -1
    truth_unknown = truth[mask]
    observed = {}

    def observe(iteration, dominance, positions, pclasses, strengths):
        # Callback receives copies from the preexisting candidate kernel;
        # never change any of the arrays.
        if iteration % INTERVAL or iteration in observed:
            raise AssertionError("Invalid/duplicated checkpoint")
        pred = np.argmax(dominance[mask], axis=1).astype(np.int64)
        observed[iteration] = dict(
            pred=pred.copy(),
            accuracy=float(np.mean(pred == truth_unknown)),
            mean_max_dominance=float(np.mean(np.max(dominance, axis=1))),
        )

    observed_pred, obs_seconds, observed_info = candidate_run(
        graph, degrees, labels, seed, LONG_CAP, 0, INTERVAL,
        criterion="label_change", tolerance=0.0,
        state_checkpoint_callback=observe,
        checkpoint_interval=INTERVAL,
    )
    if (int(observed_info["effective_iterations"]) != LONG_CAP
            or len(observed) != LONG_CAP // INTERVAL
            or list(observed) != list(range(INTERVAL, LONG_CAP + 1, INTERVAL))):
        raise AssertionError("Incomplete read-only 500k checkpoint trace")

    unobserved_pred, unobs_seconds, unobs_steps = candidate_fit(
        graph, degrees, labels, seed, LONG_CAP, patience=0)
    production_pred, prod_seconds, prod_steps = production_fit(
        graph, degrees, labels, "numpy", seed, LONG_CAP, stop=False)
    parity = bool(
        unobs_steps == LONG_CAP and prod_steps == LONG_CAP and
        np.array_equal(observed_pred, unobserved_pred) and
        np.array_equal(unobserved_pred, production_pred) and
        np.array_equal(observed[LONG_CAP]["pred"], production_pred[mask])
    )
    if not parity:
        raise AssertionError(
            f"Observer/candidate/production 500k parity failed for {seed}")

    p10_pred, _, p10_steps = candidate_fit(
        graph, degrees, labels, seed, LONG_CAP, patience=10)
    if not 0 < p10_steps <= LONG_CAP:
        raise AssertionError("Invalid P10 stop count")
    p10_unknown = p10_pred[mask]
    final = observed[LONG_CAP]["pred"]
    base100 = observed[100000]["pred"]
    base250 = observed[250000]["pred"]

    rows = []
    before = None
    zero_streak = 0
    longest_zero_before_change = 0
    sampled_changes_after_100k = 0
    sampled_changes_after_p10 = 0
    ever_100k = np.zeros(len(final), dtype=bool)
    ever_p10 = np.zeros(len(final), dtype=bool)
    for iteration, checkpoint in observed.items():
        pred = checkpoint["pred"]
        flips = None if before is None else int(
            np.count_nonzero(pred != before))
        if flips is not None:
            if flips == 0:
                zero_streak += 1
            else:
                longest_zero_before_change = max(
                    longest_zero_before_change, zero_streak)
                zero_streak = 0
            if iteration > 100000:
                sampled_changes_after_100k += flips
            if iteration > p10_steps:
                sampled_changes_after_p10 += flips
        if iteration > 100000:
            ever_100k |= pred != base100
        if iteration > p10_steps:
            ever_p10 |= pred != p10_unknown
        counts = np.bincount(pred, minlength=3)
        rows.append(dict(
            dataset="iris", seed=seed, iteration=iteration,
            accuracy=checkpoint["accuracy"],
            flips_since_previous_sample=("" if flips is None else flips),
            agreement_with_final_500k=float(np.mean(pred == final)),
            flips_vs_final_500k=int(np.count_nonzero(pred != final)),
            mean_max_dominance=checkpoint["mean_max_dominance"],
            predicted_class_0=int(counts[0]),
            predicted_class_1=int(counts[1]),
            predicted_class_2=int(counts[2]),
        ))
        before = pred

    summary = dict(
        dataset="iris", seed=seed, cap=LONG_CAP, interval=INTERVAL,
        sample_count=len(observed), unlabeled=len(final),
        callback_uninstrumented_production_500k_parity=parity,
        accuracy_100k=observed[100000]["accuracy"],
        accuracy_250k=observed[250000]["accuracy"],
        accuracy_500k=observed[500000]["accuracy"],
        flips_100k_vs_250k=int(np.count_nonzero(base100 != base250)),
        flips_100k_vs_500k=int(np.count_nonzero(base100 != final)),
        flips_250k_vs_500k=int(np.count_nonzero(base250 != final)),
        sampled_flips_after_100k=sampled_changes_after_100k,
        unique_nodes_different_from_100k_after_100k=int(
            np.count_nonzero(ever_100k)),
        nodes_changed_and_returned_to_100k_at_500k=int(
            np.count_nonzero(ever_100k & (base100 == final))),
        longest_zero_flip_sample_streak_before_later_flip=(
            longest_zero_before_change),
        label_p10_iterations=p10_steps,
        label_p10_accuracy=float(np.mean(p10_unknown == truth_unknown)),
        label_p10_accuracy_delta_pp_vs_500k=100.0 * (
            float(np.mean(p10_unknown == truth_unknown)) -
            observed[500000]["accuracy"]),
        label_p10_agreement_vs_500k=float(np.mean(p10_unknown == final)),
        label_p10_flips_vs_500k=int(np.count_nonzero(p10_unknown != final)),
        sampled_flips_after_label_p10_stop=sampled_changes_after_p10,
        nodes_ever_differing_from_label_p10_after_stop=int(
            np.count_nonzero(ever_p10)),
        production_500k_seconds=prod_seconds,
        candidate_unobserved_500k_seconds=unobs_seconds,
        candidate_observed_500k_seconds=obs_seconds,
    )
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    rows, summary = audit(args.seed)
    with open(args.checkpoints, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CHECKPOINT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(args.summary, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary)
    print(f"Iris {args.seed}: 500 read-only checkpoints; "
          f"sampled flips after 100k={summary['sampled_flips_after_100k']}, "
          f"returning nodes={summary['nodes_changed_and_returned_to_100k_at_500k']}, "
          f"P10 stop={summary['label_p10_iterations']}; "
          f"3-way 500k parity passed", flush=True)


if __name__ == "__main__":
    main()
