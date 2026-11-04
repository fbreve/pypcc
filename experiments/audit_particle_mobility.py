"""Trace the onset of particle immobility and strength collapse in Wine.

Observes a single, non-stopping NumPy PCC trajectory (seed 83 by default).
Checkpoints are read-only defensive copies; event callbacks only count visits.
A separate uninstrumented run checks the final labels were not changed.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


FIELDS = (
    "dataset", "seed", "iteration", "class_id", "particles",
    "attempted_moves", "accepted_moves", "unlabeled_visits",
    "greedy_visits", "transfer_received", "transfer_donated",
    "min_strength", "mean_strength", "max_strength",
    "unique_particle_positions", "positions_on_labeled_nodes",
    "accessible_winning_neighbors", "accessible_neighbors",
    "mass", "predicted_nodes", "accuracy",
)


def audit(seed, cap, interval):
    name = "wine"
    graph, degrees, labels, truth = prepare(name, seed)
    unknown = labels == -1
    n_classes = int(np.max(truth)) + 1
    windows = (cap + interval - 1) // interval
    attempted = np.zeros((windows, n_classes), dtype=np.int64)
    accepted = np.zeros_like(attempted)
    unlabeled = np.zeros_like(attempted)
    greedy = np.zeros_like(attempted)
    received = np.zeros((windows, n_classes), dtype=np.float64)
    donated = np.zeros_like(received)
    checkpoints = {}

    def event(iteration, cls, is_unlabeled, is_greedy, is_accepted,
              donor_loss):
        wi = (iteration - 1) // interval
        attempted[wi, cls] += 1
        accepted[wi, cls] += int(is_accepted)
        unlabeled[wi, cls] += int(is_unlabeled)
        greedy[wi, cls] += int(is_greedy)
        if is_unlabeled:
            received[wi, cls] += float(np.sum(donor_loss))
            donated[wi] += donor_loss

    def state(iteration, dominance, current, particle_labels, strengths):
        checkpoints[iteration] = (
            dominance, current, particle_labels, strengths)

    pred, _, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        particle_event_callback=event, event_start_iteration=0,
        state_checkpoint_callback=state, checkpoint_interval=interval)
    if info["effective_iterations"] != cap:
        raise AssertionError("unexpected stop")
    reference, _, _ = run(graph, degrees, labels, seed, cap, 0, interval)
    if not np.array_equal(pred[unknown], reference[unknown]):
        raise AssertionError("instrumentation changed final predictions")
    if not np.array_equal(
            np.argmax(checkpoints[cap][0][unknown], axis=1),
            reference[unknown]):
        raise AssertionError("final checkpoint does not match final predictions")
    if len(checkpoints) != cap // interval:
        raise AssertionError("missing or duplicate checkpoints")

    rows = []
    for iteration in sorted(checkpoints):
        dom, current, particle_labels, strengths = checkpoints[iteration]
        wi = (iteration - 1) // interval
        predicted = np.argmax(dom[unknown], axis=1)
        for cls in range(n_classes):
            ids = np.flatnonzero(particle_labels == cls)
            pos = current[ids]
            valid = (pos >= 0) & (pos < len(labels))
            if not np.all(valid):
                raise AssertionError("unexpected particle position")
            neighbors_count = 0
            winning_count = 0
            for node in pos:
                targets = graph[node, :int(degrees[node])]
                # A movement is accepted if the target class is tied for
                # maximum dominance; do not use argmax-only comparisons.
                valid_target = (targets >= 0) & (targets < len(labels))
                targets = targets[valid_target]
                neighbors_count += int(targets.size)
                winning_count += int(np.count_nonzero(
                    dom[targets, cls] == np.max(dom[targets], axis=1)))
            class_strength = strengths[ids]
            rows.append(dict(
                dataset=name, seed=seed, iteration=iteration,
                class_id=cls, particles=len(ids),
                attempted_moves=int(attempted[wi, cls]),
                accepted_moves=int(accepted[wi, cls]),
                unlabeled_visits=int(unlabeled[wi, cls]),
                greedy_visits=int(greedy[wi, cls]),
                transfer_received=float(received[wi, cls]),
                transfer_donated=float(donated[wi, cls]),
                min_strength=float(np.min(class_strength)),
                mean_strength=float(np.mean(class_strength)),
                max_strength=float(np.max(class_strength)),
                unique_particle_positions=int(np.unique(pos).size),
                positions_on_labeled_nodes=int(np.count_nonzero(
                    labels[pos] != -1)),
                accessible_winning_neighbors=winning_count,
                accessible_neighbors=neighbors_count,
                mass=float(dom[unknown, cls].sum()),
                predicted_nodes=int(np.count_nonzero(predicted == cls)),
                accuracy=float(np.mean(predicted == truth[unknown])),
            ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[83])
    parser.add_argument("--cap", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args(argv)
    if (args.cap < 1 or args.interval < 1 or
            args.cap % args.interval != 0 or not args.seeds):
        parser.error("invalid experiment parameters")
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            writer.writerows(audit(seed, args.cap, args.interval))


if __name__ == "__main__":
    main()
