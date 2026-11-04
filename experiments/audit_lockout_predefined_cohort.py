"""Predefined cross-dataset cohort for sustained PCC class-particle lockout.

Prospective *observation* only: no early stop, no interventions, no algorithm
changes. Use previously unused seeds for this stopping-study series to reduce
selection bias from examining Wine seed 83. The definition below is fixed
before observing the cohort results.

Operational lockout:
  >= 5 consecutive nonoverlapping 1k-iteration windows with zero accepted
  movements for an entire class, with valid visits, max class-particle
  strength <= 0.01, and no currently winning outgoing neighbors.
This is an empirical event, NOT proof of mathematical irreversibility.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import DATASETS, prepare


WINDOW_FIELDS = (
    "dataset", "seed", "iteration", "class_id", "particles",
    "valid_visits", "accepted_moves", "acceptance_fraction",
    "zero_move_streak", "winning_neighbors", "neighbor_entries",
    "max_strength", "mean_strength", "strength_bounds_excess",
    "dominance_mass",
    "predicted_nodes", "unlabeled_accuracy", "operational_lockout",
)
SUMMARY_FIELDS = (
    "dataset", "seed", "class_id", "particles", "cap", "interval",
    "lockout_windows", "strength_cutoff", "first_lockout_iteration",
    "first_recovery_iteration", "max_zero_move_streak",
    "zero_acceptance_windows", "accepted_moves_total",
    "first_mass", "last_mass", "mass_change",
    "first_unlabeled_accuracy", "last_unlabeled_accuracy",
    "max_strength_bounds_excess",
    "last_max_strength", "last_winning_neighbors",
    "still_zero_at_cap", "matches_uninstrumented",
)
CASE_FIELDS = (
    "dataset", "seed", "n_classes", "cap", "class_lockout_count",
    "unlabeled_accuracy_final", "matches_uninstrumented",
)


def observe_case(dataset, seed, cap, interval, patience, strength_cutoff):
    graph, degrees, labels, truth = prepare(dataset, seed)
    unknown = labels == -1
    nclasses = int(np.max(truth)) + 1
    expected_particles = np.bincount(labels[~unknown], minlength=nclasses)
    if not np.all(expected_particles == 2):
        raise AssertionError("Expected two labeled examples per class")
    windows = cap // interval
    visits = np.zeros((windows, nclasses), dtype=np.int64)
    accepted = np.zeros_like(visits)
    rows = []
    streak = np.zeros(nclasses, dtype=np.int64)

    def event(iteration, cls, is_unlabeled, is_greedy, is_accepted, loss):
        w = (iteration - 1) // interval
        visits[w, cls] += 1
        accepted[w, cls] += int(is_accepted)

    def checkpoint(iteration, dominance, positions, particle_classes,
                   strengths):
        w = iteration // interval - 1
        if not np.isfinite(dominance).all():
            raise AssertionError("Nonfinite dominance at checkpoint")
        if not np.allclose(dominance.sum(axis=1), 1.0,
                           atol=1e-8, rtol=0):
            raise AssertionError("Dominance simplex invariant violated")
        predicted = np.argmax(dominance[unknown], axis=1)
        accuracy = float(np.mean(predicted == truth[unknown]))
        mass = dominance[unknown].sum(axis=0)
        counts = np.bincount(predicted, minlength=nclasses)

        for cls in range(nclasses):
            ids = np.flatnonzero(particle_classes == cls)
            if len(ids) != expected_particles[cls]:
                raise AssertionError("Class particle count changed")
            if visits[w, cls] != interval * len(ids):
                raise AssertionError(
                    f"Missing valid visits: {dataset}, seed={seed}, "
                    f"class={cls}, window={w}")
            class_strength = strengths[ids]
            # Floating-point dominance conservation may differ from 1
            # by roundoff. Do not clip or mutate the particle strength.
            if not np.all(np.isfinite(class_strength)):
                raise AssertionError("Nonfinite particle strength")
            strength_excess = float(max(
                0.0, -float(np.min(class_strength)),
                float(np.max(class_strength)) - 1.0))
            if strength_excess > 1e-8:
                raise AssertionError(
                    f"Material particle strength bound violation: "
                    f"{dataset}, seed={seed}, class={cls}, "
                    f"iteration={iteration}, excess={strength_excess:.17g}")
            current = positions[ids]
            if np.any((current < 0) | (current >= len(labels))):
                raise AssertionError("Particle moved outside graph")
            neighbors_count = 0
            winning = 0
            for node in current:
                neighbors = graph[node, :int(degrees[node])]
                valid = (neighbors >= 0) & (neighbors < len(labels))
                neighbors = neighbors[valid]
                neighbors_count += len(neighbors)
                if len(neighbors):
                    winning += int(np.count_nonzero(
                        dominance[neighbors, cls] ==
                        np.max(dominance[neighbors], axis=1)))
            streak[cls] = (
                streak[cls] + 1 if accepted[w, cls] == 0 else 0)
            max_strength = float(np.max(class_strength))
            operational = bool(
                streak[cls] >= patience
                and max_strength <= strength_cutoff
                and winning == 0)
            rows.append(dict(
                dataset=dataset, seed=seed, iteration=iteration,
                class_id=cls, particles=len(ids),
                valid_visits=int(visits[w, cls]),
                accepted_moves=int(accepted[w, cls]),
                acceptance_fraction=float(
                    accepted[w, cls] / visits[w, cls]),
                zero_move_streak=int(streak[cls]),
                winning_neighbors=winning,
                neighbor_entries=neighbors_count,
                max_strength=max_strength,
                mean_strength=float(np.mean(class_strength)),
                strength_bounds_excess=strength_excess,
                dominance_mass=float(mass[cls]),
                predicted_nodes=int(counts[cls]),
                unlabeled_accuracy=accuracy,
                operational_lockout=operational))

    predicted, _, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        particle_event_callback=event,
        state_checkpoint_callback=checkpoint,
        checkpoint_interval=interval)
    if info["effective_iterations"] != cap:
        raise AssertionError("Unexpected early stop")
    if len(rows) != windows * nclasses:
        raise AssertionError("Missing observation windows")
    # Matched control: assert instrumentation did not alter predictions.
    control, _, _ = run(graph, degrees, labels, seed, cap, 0, interval)
    matching = bool(np.array_equal(predicted, control))
    if not matching:
        raise AssertionError("Instrumented results diverged from control")
    if not np.isclose(rows[-1]["unlabeled_accuracy"],
                      np.mean(predicted[unknown] == truth[unknown])):
        raise AssertionError("Last checkpoint differs from final labels")

    summaries = []
    for cls in range(nclasses):
        history = [row for row in rows if row["class_id"] == cls]
        first_event = next(
            (i for i, row in enumerate(history)
             if row["operational_lockout"]), None)
        first_recovery = (
            next((row["iteration"] for row in history[first_event + 1:]
                  if row["accepted_moves"] > 0), None)
            if first_event is not None else None)
        summaries.append(dict(
            dataset=dataset, seed=seed, class_id=cls,
            particles=int(expected_particles[cls]), cap=cap,
            interval=interval, lockout_windows=patience,
            strength_cutoff=strength_cutoff,
            first_lockout_iteration=(
                history[first_event]["iteration"]
                if first_event is not None else ""),
            first_recovery_iteration=(
                first_recovery if first_recovery is not None else ""),
            max_zero_move_streak=int(max(
                row["zero_move_streak"] for row in history)),
            zero_acceptance_windows=sum(
                row["accepted_moves"] == 0 for row in history),
            accepted_moves_total=sum(
                row["accepted_moves"] for row in history),
            first_mass=history[0]["dominance_mass"],
            last_mass=history[-1]["dominance_mass"],
            mass_change=(
                history[-1]["dominance_mass"] -
                history[0]["dominance_mass"]),
            first_unlabeled_accuracy=history[0]["unlabeled_accuracy"],
            last_unlabeled_accuracy=history[-1]["unlabeled_accuracy"],
            max_strength_bounds_excess=max(
                row["strength_bounds_excess"] for row in history),
            last_max_strength=history[-1]["max_strength"],
            last_winning_neighbors=history[-1]["winning_neighbors"],
            still_zero_at_cap=(
                bool(history[-1]["operational_lockout"])
                if first_event is not None else False),
            matches_uninstrumented=matching))
    case = dict(
        dataset=dataset, seed=seed, n_classes=nclasses, cap=cap,
        class_lockout_count=sum(
            bool(x["first_lockout_iteration"]) for x in summaries),
        unlabeled_accuracy_final=float(np.mean(
            predicted[unknown] == truth[unknown])),
        matches_uninstrumented=matching)
    return rows, summaries, case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--datasets", nargs="+", default=[
        "iris", "wine", "breast_cancer", "digits"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[101, 211, 307, 401, 503])
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--lockout-windows", type=int, default=5)
    parser.add_argument("--strength-cutoff", type=float, default=0.01)
    args = parser.parse_args(argv)
    if (not args.datasets or any(name not in DATASETS
                                  for name in args.datasets)
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.interval < 1 or args.cap < 1
            or args.cap % args.interval
            or args.lockout_windows < 1
            or not 0 < args.strength_cutoff <= 1):
        parser.error("Invalid cohort protocol")
    with (open(args.windows_output, "w", newline="",
               encoding="utf-8") as windows_file,
          open(args.summary_output, "w", newline="",
               encoding="utf-8") as summary_file,
          open(args.cases_output, "w", newline="",
               encoding="utf-8") as cases_file):
        window_writer = csv.DictWriter(
            windows_file, fieldnames=WINDOW_FIELDS)
        summary_writer = csv.DictWriter(
            summary_file, fieldnames=SUMMARY_FIELDS)
        case_writer = csv.DictWriter(cases_file, fieldnames=CASE_FIELDS)
        window_writer.writeheader()
        summary_writer.writeheader()
        case_writer.writeheader()
        for name in args.datasets:
            for seed in args.seeds:
                rows, summaries, case = observe_case(
                    name, seed, args.cap, args.interval,
                    args.lockout_windows, args.strength_cutoff)
                window_writer.writerows(rows)
                summary_writer.writerows(summaries)
                case_writer.writerow(case)
                print(
                    f"{name} seed={seed}: "
                    f"{case['class_lockout_count']} locked classes",
                    flush=True)


if __name__ == "__main__":
    main()
