"""Prespecified Digits lockout audit: nested 2 vs 5 labeled nodes per class.

All comparisons use the same seed, same 10-NN graph, same PCC parameters and
the same maximum five randomly drawn labeled examples in each class.
The two-label set is a subset of the corresponding five-label set.
These are observational trajectories, not stopping or rescue proposals.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


WINDOW_FIELDS = (
    "seed", "labeled_per_class", "iteration", "class_id", "particle_count",
    "accepted_moves", "valid_visits", "zero_move_streak",
    "max_strength", "winning_neighbors", "operational_lockout",
)
CLASS_FIELDS = (
    "seed", "labeled_per_class", "class_id",
    "first_lockout_iteration", "first_recovery_iteration",
    "max_zero_move_streak", "zero_move_windows",
    "last_mass", "last_max_strength", "last_winning_neighbors",
)
CASE_FIELDS = (
    "seed", "labeled_per_class", "cap", "interval",
    "classes_locked", "classes_with_zero_move_windows",
    "accuracy_on_unlabeled", "matches_control", "max_strength_bounds_excess",
)


def nested_case(seed, labeled_per_class):
    graph, degrees, full_labels, truth = prepare(
        "digits", seed, neighbors=10, labeled_per_class=5)
    rng = np.random.default_rng(seed)
    labels = np.full(truth.size, -1, dtype=np.int64)
    for cls in np.unique(truth):
        choices = rng.choice(
            np.flatnonzero(truth == cls), size=5, replace=False)
        labels[choices[:labeled_per_class]] = cls
    if labeled_per_class == 5 and not np.array_equal(labels, full_labels):
        raise AssertionError("Nested label construction does not match prepare")
    return graph, degrees, labels, truth


def observe(seed, labeled_per_class, cap, interval,
            zero_windows, strength_cutoff):
    graph, degrees, labels, truth = nested_case(seed, labeled_per_class)
    unknown = labels == -1
    nclasses = len(np.unique(truth))
    n_windows = cap // interval
    particle_counts = np.bincount(
        labels[~unknown], minlength=nclasses)
    assert np.all(particle_counts == labeled_per_class)
    accepted = np.zeros((n_windows, nclasses), dtype=np.int64)
    visits = np.zeros_like(accepted)
    streak = np.zeros(nclasses, dtype=np.int64)
    windows = []
    max_bounds_excess = 0.0

    def event(iteration, cls, is_unlabeled, greedy, moved, donor_loss):
        wi = (iteration - 1) // interval
        visits[wi, cls] += 1
        accepted[wi, cls] += int(moved)

    def state(iteration, dominance, positions, particle_classes, strengths):
        nonlocal max_bounds_excess
        w = iteration // interval - 1
        if not np.isfinite(dominance).all():
            raise AssertionError("Nonfinite dominance")
        if not np.allclose(dominance.sum(axis=1), 1.0,
                           rtol=0.0, atol=1e-8):
            raise AssertionError("Dominance simplex failure")
        for cls in range(nclasses):
            ids = np.flatnonzero(particle_classes == cls)
            if ids.size != particle_counts[cls]:
                raise AssertionError("Unexpected particle count")
            if visits[w, cls] != interval * ids.size:
                raise AssertionError("Expected one valid visit per particle per iteration")
            cs = strengths[ids]
            if not np.all(np.isfinite(cs)):
                raise AssertionError("Nonfinite particle strength")
            excess = max(0.0, -float(cs.min()), float(cs.max()) - 1.0)
            max_bounds_excess = max(max_bounds_excess, excess)
            if excess > 1e-8:
                raise AssertionError(f"Material strength error: {excess:g}")
            neighbors_won = 0
            for position in positions[ids]:
                targets = graph[position, :int(degrees[position])]
                if np.any((targets < 0) | (targets >= len(truth))):
                    raise AssertionError("Invalid adjacent node")
                neighbors_won += int(np.count_nonzero(
                    dominance[targets, cls] ==
                    np.max(dominance[targets], axis=1)))
            streak[cls] = (streak[cls] + 1
                           if accepted[w, cls] == 0 else 0)
            lockout = bool(
                streak[cls] >= zero_windows
                and float(cs.max()) <= strength_cutoff
                and neighbors_won == 0)
            windows.append(dict(
                seed=seed, labeled_per_class=labeled_per_class,
                iteration=iteration, class_id=cls,
                particle_count=int(ids.size),
                accepted_moves=int(accepted[w, cls]),
                valid_visits=int(visits[w, cls]),
                zero_move_streak=int(streak[cls]),
                max_strength=float(cs.max()),
                winning_neighbors=neighbors_won,
                operational_lockout=lockout))

    result, _, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        particle_event_callback=event,
        state_checkpoint_callback=state, checkpoint_interval=interval)
    if info["effective_iterations"] != cap:
        raise AssertionError("Unexpected stopping")
    control, _, _ = run(
        graph, degrees, labels, seed, cap, 0, interval)
    matching = bool(np.array_equal(result, control))
    if not matching:
        raise AssertionError("Instrumentation changed predictions")
    if len(windows) != nclasses * n_windows:
        raise AssertionError("Missing checkpoint windows")

    rows = []
    for cls in range(nclasses):
        history = [x for x in windows if x["class_id"] == cls]
        event_idx = next((i for i, x in enumerate(history)
                          if x["operational_lockout"]), None)
        recovery = (
            next((x["iteration"] for x in history[event_idx + 1:]
                  if x["accepted_moves"] > 0), None)
            if event_idx is not None else None)
        rows.append(dict(
            seed=seed, labeled_per_class=labeled_per_class,
            class_id=cls,
            first_lockout_iteration=(
                history[event_idx]["iteration"]
                if event_idx is not None else ""),
            first_recovery_iteration=(
                recovery if recovery is not None else ""),
            max_zero_move_streak=max(
                x["zero_move_streak"] for x in history),
            zero_move_windows=sum(x["accepted_moves"] == 0
                                  for x in history),
            last_mass="",  # Class dominance mass is not needed to define lockout.
            last_max_strength=history[-1]["max_strength"],
            last_winning_neighbors=history[-1]["winning_neighbors"],
        ))

    outcome = dict(
        seed=seed, labeled_per_class=labeled_per_class,
        cap=cap, interval=interval,
        classes_locked=sum(bool(x["first_lockout_iteration"]) for x in rows),
        classes_with_zero_move_windows=sum(
            x["zero_move_windows"] > 0 for x in rows),
        accuracy_on_unlabeled=float(np.mean(
            result[unknown] == truth[unknown])),
        matches_control=matching,
        max_strength_bounds_excess=max_bounds_excess)
    return windows, rows, outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-output", required=True)
    parser.add_argument("--class-output", required=True)
    parser.add_argument("--case-output", required=True)
    parser.add_argument("--labeled-per-class", type=int, required=True,
                        choices=(2, 5))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[601, 701, 809])
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--zero-windows", type=int, default=5)
    parser.add_argument("--strength-cutoff", type=float, default=0.01)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.cap <= 0 or args.interval <= 0
            or args.cap % args.interval or args.zero_windows < 1
            or not 0 < args.strength_cutoff <= 1):
        parser.error("Invalid protocol")
    with (open(args.windows_output, "w", newline="",
               encoding="utf-8") as wf,
          open(args.class_output, "w", newline="",
               encoding="utf-8") as cf,
          open(args.case_output, "w", newline="",
               encoding="utf-8") as of):
        w = csv.DictWriter(wf, fieldnames=WINDOW_FIELDS)
        c = csv.DictWriter(cf, fieldnames=CLASS_FIELDS)
        o = csv.DictWriter(of, fieldnames=CASE_FIELDS)
        w.writeheader()
        c.writeheader()
        o.writeheader()
        for seed in args.seeds:
            windows, classes, outcome = observe(
                seed, args.labeled_per_class, args.cap, args.interval,
                args.zero_windows, args.strength_cutoff)
            w.writerows(windows)
            c.writerows(classes)
            o.writerow(outcome)
            print(f"seed {seed}, labeled/class {args.labeled_per_class}: "
                  f"{outcome['classes_locked']} locked classes", flush=True)


if __name__ == "__main__":
    main()
