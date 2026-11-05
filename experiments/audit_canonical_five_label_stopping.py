"""Five-label/class sensitivity of canonical PCC early stopping.

All details predeclared in canonical_five_label_stopping_protocol.md.
"""
import argparse
import csv

import numpy as np

from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import DATASETS


FIELDS = (
    "dataset", "seed", "labeled_per_class", "policy", "patience",
    "interval", "cap", "iterations", "early_stop",
    "saved_iterations_fraction", "accuracy", "reference_accuracy",
    "accuracy_delta_pp", "agreement_with_reference",
    "elapsed_seconds", "reference_seconds", "saved_wall_time_fraction",
    "reference_50k_to_100k_flips", "reference_50k_to_100k_accuracy_delta_pp",
    "reference_50k_to_100k_agreement", "unlabeled_count",
    "mean_degree", "reciprocal_arc_fraction",
    "reference_matches_uninstrumented",
)


def one_case(dataset, seed, cap, interval):
    graph, degrees, labels, truth, stats = canonical_prepare(
        dataset, seed, k=10, labeled_per_class=5)
    unknown = labels == -1
    if not np.all(np.bincount(labels[~unknown]) == 5):
        raise AssertionError("Expected five labeled nodes per class")
    samples = {}

    def observer(it, dominance, positions, classes, strengths):
        if it in (cap // 2, cap):
            if not np.isfinite(dominance).all():
                raise AssertionError("Nonfinite dominance")
            if not np.allclose(dominance.sum(axis=1), 1.0,
                               atol=1e-8, rtol=0.0):
                raise AssertionError("Dominance simplex violation")
            samples[it] = np.argmax(dominance[unknown], axis=1).copy()

    reference, reference_seconds, ref_info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        state_checkpoint_callback=observer, checkpoint_interval=cap // 2)
    if ref_info["effective_iterations"] != cap or set(samples) != {
            cap // 2, cap} or not np.array_equal(
                samples[cap], reference[unknown]):
        raise AssertionError("Incomplete 100k reference")
    plain, _, plain_info = run(graph, degrees, labels, seed,
                              cap, 0, interval)
    if (plain_info["effective_iterations"] != cap
            or not np.array_equal(plain, reference)):
        raise AssertionError("Reference observer changed predictions")

    acc_ref = float(np.mean(reference[unknown] == truth[unknown]))
    halfway = samples[cap // 2]
    late = samples[cap]
    late_delta_pp = 100.0 * (
        acc_ref - float(np.mean(halfway == truth[unknown])))
    late_flips = int(np.count_nonzero(halfway != late))
    rows = []
    for patience in (0, 3, 10):
        if patience == 0:
            prediction, elapsed, info = (
                reference, reference_seconds, ref_info)
            policy = "no_stop"
        else:
            prediction, elapsed, info = run(
                graph, degrees, labels, seed, cap, patience,
                interval, "label_change", 0.0)
            policy = f"label_patience_{patience}"
        steps = int(info["effective_iterations"])
        if not 0 < steps <= cap:
            raise AssertionError("Stopping iteration out of range")
        acc = float(np.mean(prediction[unknown] == truth[unknown]))
        rows.append(dict(
            dataset=dataset, seed=seed, labeled_per_class=5,
            policy=policy, patience=patience, interval=interval, cap=cap,
            iterations=steps, early_stop=steps < cap,
            saved_iterations_fraction=(cap - steps) / cap,
            accuracy=acc, reference_accuracy=acc_ref,
            accuracy_delta_pp=100.0 * (acc - acc_ref),
            agreement_with_reference=float(np.mean(
                prediction[unknown] == reference[unknown])),
            elapsed_seconds=float(elapsed),
            reference_seconds=float(reference_seconds),
            saved_wall_time_fraction=1.0 - elapsed / reference_seconds,
            reference_50k_to_100k_flips=late_flips,
            reference_50k_to_100k_accuracy_delta_pp=late_delta_pp,
            reference_50k_to_100k_agreement=float(np.mean(
                halfway == late)),
            unlabeled_count=int(np.count_nonzero(unknown)),
            mean_degree=stats["mean_degree"],
            reciprocal_arc_fraction=stats["reciprocal_arc_fraction"],
            reference_matches_uninstrumented=True,
        ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[3109, 3203, 3301])
    parser.add_argument("--cap", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.cap <= 0 or args.cap % (2 * args.interval)
            or args.interval <= 0):
        parser.error("Invalid fixed five-label protocol")
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            rows = one_case(args.dataset, seed, args.cap, args.interval)
            writer.writerows(rows)
            print(
                f"{args.dataset} seed={seed}: " +
                ", ".join(
                    f"{r['policy']}={r['iterations']}"
                    for r in rows),
                flush=True)


if __name__ == "__main__":
    main()
