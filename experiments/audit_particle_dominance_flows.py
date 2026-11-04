"""Reconcile class-mass drift with actual particle transfers in one PCC run.

Experimental sequential NumPy backend only. Particle observations and dominance
snapshots are passive: no changed RNG calls or modified propagation arrays.
Each donor class loses dominance to the visiting particle's class.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


CLASS_FIELDS = (
    "dataset", "seed", "from_iteration", "to_iteration", "class_id",
    "visits", "unlabeled_visits", "greedy_visits", "accepted_moves",
    "mass_start", "mass_end", "observed_mass_delta", "transfer_received",
    "transfer_donated", "net_transfer", "mass_balance_residual",
    "class_prediction_count_start", "class_prediction_count_end",
)
FLOW_FIELDS = (
    "dataset", "seed", "from_iteration", "to_iteration",
    "donor_class", "recipient_class", "transferred_mass",
)


def audit_case(seed, start, end, interval):
    name = "wine"
    graph, degrees, labels, truth = prepare(name, seed)
    mask = labels == -1
    classes = int(np.max(truth)) + 1
    if classes != 3:
        raise AssertionError("Expected three Wine classes")
    n_windows = (end - start) // interval
    flow = np.zeros((n_windows, classes, classes), dtype=float)
    visits = np.zeros((n_windows, classes), dtype=int)
    unlabeled_visits = np.zeros_like(visits)
    greedy_visits = np.zeros_like(visits)
    accepted_moves = np.zeros_like(visits)
    snapshots = {}

    def checkpoint(iteration, dominance):
        if iteration >= start:
            snapshots[iteration] = dominance[mask].copy()

    def event(iteration, attack_class, is_unlabeled, greedy, accepted,
              donor_loss):
        window = (iteration - start - 1) // interval
        if not 0 <= window < n_windows:
            raise AssertionError("Particle event outside audited window")
        visits[window, attack_class] += 1
        greedy_visits[window, attack_class] += int(greedy)
        accepted_moves[window, attack_class] += int(accepted)
        if is_unlabeled:
            unlabeled_visits[window, attack_class] += 1
            if donor_loss is None:
                raise AssertionError("Missing donor losses on unlabeled visit")
            # The donor's own class entry is zero by construction.
            flow[window, :, attack_class] += donor_loss
        elif donor_loss is not None:
            raise AssertionError("Labeled visit unexpectedly transferred mass")

    final, _, info = run(
        graph, degrees, labels, seed, end, 0, interval,
        checkpoint_callback=checkpoint, checkpoint_interval=interval,
        particle_event_callback=event, event_start_iteration=start)
    if info["effective_iterations"] != end:
        raise AssertionError("Reference unexpectedly stopped")
    if sorted(snapshots) != list(range(start, end + 1, interval)):
        raise AssertionError("Missing audited checkpoints")
    if not np.array_equal(np.argmax(snapshots[end], axis=1), final[mask]):
        raise AssertionError("Instrumented final labels disagree with snapshot")

    # Verify the instrumentation did not alter the stochastic trajectory.
    control, _, _ = run(graph, degrees, labels, seed, end, 0, interval)
    if not np.array_equal(final[mask], control[mask]):
        raise AssertionError("Particle-flow instrumentation changed predictions")

    rows, pairwise = [], []
    for window in range(n_windows):
        left = start + window * interval
        right = left + interval
        before = snapshots[left]
        after = snapshots[right]
        mass_before = before.sum(axis=0)
        mass_after = after.sum(axis=0)
        received = flow[window].sum(axis=0)
        donated = flow[window].sum(axis=1)
        net = received - donated
        observed = mass_after - mass_before
        residual = observed - net
        # Each update conserves per-node simplex mass; allow rounding.
        if not np.allclose(residual, 0.0, atol=1e-7, rtol=0):
            raise AssertionError(
                f"Flow fails mass balance for seed {seed} at {right}: {residual}")
        before_count = np.bincount(np.argmax(before, axis=1),
                                   minlength=classes)
        after_count = np.bincount(np.argmax(after, axis=1),
                                  minlength=classes)
        for cls in range(classes):
            rows.append(dict(
                dataset=name, seed=seed, from_iteration=left,
                to_iteration=right, class_id=cls,
                visits=int(visits[window, cls]),
                unlabeled_visits=int(unlabeled_visits[window, cls]),
                greedy_visits=int(greedy_visits[window, cls]),
                accepted_moves=int(accepted_moves[window, cls]),
                mass_start=float(mass_before[cls]),
                mass_end=float(mass_after[cls]),
                observed_mass_delta=float(observed[cls]),
                transfer_received=float(received[cls]),
                transfer_donated=float(donated[cls]),
                net_transfer=float(net[cls]),
                mass_balance_residual=float(residual[cls]),
                class_prediction_count_start=int(before_count[cls]),
                class_prediction_count_end=int(after_count[cls]),
            ))
            for other in range(classes):
                if other != cls:
                    pairwise.append(dict(
                        dataset=name, seed=seed,
                        from_iteration=left, to_iteration=right,
                        donor_class=cls, recipient_class=other,
                        transferred_mass=float(flow[window, cls, other]),
                    ))
    return rows, pairwise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-output", required=True)
    parser.add_argument("--flow-output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 83])
    parser.add_argument("--start", type=int, default=50000)
    parser.add_argument("--end", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=5000)
    args = parser.parse_args(argv)
    if (args.start <= 0 or args.end <= args.start or args.interval <= 0
            or args.start % args.interval or args.end % args.interval
            or not args.seeds):
        parser.error("invalid audit window")
    with (open(args.class_output, "w", newline="",
               encoding="utf-8") as class_file,
          open(args.flow_output, "w", newline="",
               encoding="utf-8") as flow_file):
        class_writer = csv.DictWriter(class_file, fieldnames=CLASS_FIELDS)
        flow_writer = csv.DictWriter(flow_file, fieldnames=FLOW_FIELDS)
        class_writer.writeheader()
        flow_writer.writeheader()
        for seed in args.seeds:
            rows, pairwise = audit_case(
                seed, args.start, args.end, args.interval)
            class_writer.writerows(rows)
            flow_writer.writerows(pairwise)


if __name__ == "__main__":
    main()
