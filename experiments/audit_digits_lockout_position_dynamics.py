"""Trace particle locality and loss of winning neighbors in selected Digits cases.

Diagnostic only; never changes particles, labels, graph, or stopping. Uses
the same nested supervised selections as the prior 2-vs-5 label comparison.
Known challenging seeds 701/809 are for explaining mechanism, not estimating
generalization. All 10 classes are recorded for within-run comparisons.
"""
import argparse
import csv

import numpy as np

from experiments.audit_nested_label_density_lockout import nested_case
from experiments.benchmark_inloop_global_stopping import run


FIELDS = (
    "seed", "labeled_per_class", "iteration", "class_id",
    "particles", "accepted_moves_in_window", "zero_move_streak",
    "particle_strength_min", "particle_strength_max",
    "particle_positions", "distinct_positions", "particles_at_home",
    "particles_on_original_true_class", "true_same_class_neighbor_fraction",
    "winning_neighbor_entries", "neighbor_entries",
    "own_dominance_mean_at_position", "own_dominance_min_at_position",
    "class_dominance_mass_on_unlabeled",
    "class_predicted_unlabeled_count", "unlabeled_accuracy",
)


def trace(seed, labeled_per_class, cap, interval):
    graph, degrees, labels, truth = nested_case(seed, labeled_per_class)
    unknown = labels == -1
    nclasses = int(np.max(truth)) + 1
    home = {cls: set(np.flatnonzero(labels == cls).tolist())
            for cls in range(nclasses)}
    n_windows = cap // interval
    accepted = np.zeros((n_windows, nclasses), dtype=np.int64)
    visits = np.zeros_like(accepted)
    streak = np.zeros(nclasses, dtype=np.int64)
    rows = []

    def event(iteration, cls, is_unlabeled, greedy, moved, donor_loss):
        w = (iteration - 1) // interval
        visits[w, cls] += 1
        accepted[w, cls] += int(moved)

    def state(iteration, dominance, positions, particle_classes, strengths):
        w = iteration // interval - 1
        if not np.isfinite(dominance).all():
            raise AssertionError("Nonfinite dominance")
        if not np.allclose(dominance.sum(axis=1), 1.0, atol=1e-8, rtol=0):
            raise AssertionError("Dominance simplex violation")
        predicted = np.argmax(dominance[unknown], axis=1)
        accuracy = float(np.mean(predicted == truth[unknown]))
        class_counts = np.bincount(predicted, minlength=nclasses)
        masses = dominance[unknown].sum(axis=0)
        for cls in range(nclasses):
            indices = np.flatnonzero(particle_classes == cls)
            if len(indices) != labeled_per_class:
                raise AssertionError("Unexpected class-particle count")
            if visits[w, cls] != interval * len(indices):
                raise AssertionError("Expected one valid visit per particle-step")
            current = positions[indices].astype(np.int64)
            if np.any((current < 0) | (current >= len(truth))):
                raise AssertionError("Invalid particle location")
            class_strength = strengths[indices]
            if not np.isfinite(class_strength).all():
                raise AssertionError("Nonfinite strength")
            own = dominance[current, cls]
            winning = 0
            target_count = 0
            true_same = 0
            for node in current:
                targets = graph[node, :int(degrees[node])]
                if np.any((targets < 0) | (targets >= len(truth))):
                    raise AssertionError("Invalid outgoing graph edge")
                target_count += len(targets)
                true_same += int(np.count_nonzero(truth[targets] == cls))
                winning += int(np.count_nonzero(
                    dominance[targets, cls] ==
                    np.max(dominance[targets], axis=1)))
            streak[cls] = streak[cls] + 1 if accepted[w, cls] == 0 else 0
            rows.append(dict(
                seed=seed, labeled_per_class=labeled_per_class,
                iteration=iteration, class_id=cls, particles=len(indices),
                accepted_moves_in_window=int(accepted[w, cls]),
                zero_move_streak=int(streak[cls]),
                particle_strength_min=float(np.min(class_strength)),
                particle_strength_max=float(np.max(class_strength)),
                particle_positions=";".join(map(str, current.tolist())),
                distinct_positions=int(len(np.unique(current))),
                particles_at_home=sum(int(node in home[cls])
                                      for node in current),
                particles_on_original_true_class=int(np.count_nonzero(
                    truth[current] == cls)),
                true_same_class_neighbor_fraction=float(
                    true_same / target_count),
                winning_neighbor_entries=winning,
                neighbor_entries=target_count,
                own_dominance_mean_at_position=float(np.mean(own)),
                own_dominance_min_at_position=float(np.min(own)),
                class_dominance_mass_on_unlabeled=float(masses[cls]),
                class_predicted_unlabeled_count=int(class_counts[cls]),
                unlabeled_accuracy=accuracy,
            ))

    instrumented, _, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        particle_event_callback=event,
        state_checkpoint_callback=state, checkpoint_interval=interval)
    if info["effective_iterations"] != cap:
        raise AssertionError("Unexpected early stop")
    control, _, _ = run(graph, degrees, labels, seed, cap, 0, interval)
    if not np.array_equal(instrumented, control):
        raise AssertionError("Observer changed reference predictions")
    if len(rows) != n_windows * nclasses:
        raise AssertionError("Missing checkpoint rows")
    if not np.isclose(rows[-1]["unlabeled_accuracy"],
                      np.mean(instrumented[unknown] == truth[unknown])):
        raise AssertionError("Final checkpoint differs from predictions")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labeled-per-class", type=int,
                        choices=[2, 5], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 809])
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.interval < 1 or args.cap < 1
            or args.cap % args.interval):
        parser.error("Invalid run settings")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            writer.writerows(trace(
                seed, args.labeled_per_class, args.cap, args.interval))
            print(f"seed={seed}, labels/class={args.labeled_per_class} done",
                  flush=True)


if __name__ == "__main__":
    main()
