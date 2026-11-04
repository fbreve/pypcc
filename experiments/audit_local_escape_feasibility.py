"""Test one-visit escape feasibility without changing PCC propagation.

Wine seed 83: compare the temporary class-1 trap at 39k to the persistent trap
at 48k, using actual particle strengths and hypothetical fixed strengths.
For each neighbor, replay only the single PCC dominance update in a *copy*
of the checkpoint state. No counterfactual propagation is executed.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


FIELDS = (
    "dataset", "seed", "iteration", "particle_id", "current_node",
    "actual_strength", "tested_strength", "strength_scenario",
    "neighbors", "already_winning_neighbors",
    "one_visit_accepted_neighbors", "one_visit_conquerable_neighbors",
    "largest_post_visit_own_minus_competitor",
)


def one_visit(row, cls, strength, delta_v, unlabeled):
    updated = row.copy()
    if unlabeled:
        step = strength * delta_v / (updated.size - 1)
        reduction = np.minimum(updated, step)
        updated -= reduction
        updated[cls] += float(np.sum(reduction))
    return updated


def audit(seed, cap, checkpoint_interval, checkpoints,
          strengths, delta_v):
    graph, degrees, labels, truth = prepare("wine", seed)
    states = {}
    needed = set(checkpoints)
    def checkpoint(iteration, dominance, positions, classes, strength):
        if iteration in needed:
            states[iteration] = (dominance, positions, classes, strength)

    prediction, _, info = run(
        graph, degrees, labels, seed, cap, 0, checkpoint_interval,
        state_checkpoint_callback=checkpoint,
        checkpoint_interval=checkpoint_interval)
    control, _, _ = run(graph, degrees, labels, seed, cap, 0,
                        checkpoint_interval)
    if info["effective_iterations"] != cap or not np.array_equal(
            prediction, control):
        raise AssertionError("Instrumentation altered reference predictions")
    if set(states) != needed:
        raise AssertionError("Missing selected checkpoints")

    rows = []
    for iteration in sorted(states):
        dominance, positions, particle_labels, particle_strengths = states[
            iteration]
        for p_i in np.flatnonzero(particle_labels == 1):
            current = int(positions[p_i])
            neighbors = graph[current, :int(degrees[current])]
            neighbors = neighbors[
                (neighbors >= 0) & (neighbors < len(labels))]
            actual = float(particle_strengths[p_i])
            for scenario, strength in [("actual", actual)] + [
                    (f"fixed_{value:g}", value) for value in strengths]:
                already = accepted = conquered = 0
                best_gap = float("-inf")
                for node in neighbors:
                    before = dominance[node]
                    winning_before = (
                        before[1] == np.max(before))
                    updated = one_visit(
                        before, 1, strength, delta_v,
                        labels[node] == -1)
                    winning_after = (
                        updated[1] == np.max(updated))
                    already += int(winning_before)
                    accepted += int(winning_after)
                    conquered += int(winning_after and not winning_before)
                    best_gap = max(
                        best_gap,
                        float(updated[1] - np.max(
                            updated[np.arange(updated.size) != 1])))
                rows.append(dict(
                    dataset="wine", seed=seed, iteration=iteration,
                    particle_id=int(p_i), current_node=current,
                    actual_strength=actual, tested_strength=strength,
                    strength_scenario=scenario, neighbors=len(neighbors),
                    already_winning_neighbors=already,
                    one_visit_accepted_neighbors=accepted,
                    one_visit_conquerable_neighbors=conquered,
                    largest_post_visit_own_minus_competitor=best_gap,
                ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoints", nargs="+", type=int,
                        default=[39000, 40000, 47000, 47500, 47600,
                                 47800, 47900, 48000, 49000, 50000])
    parser.add_argument("--strengths", nargs="+", type=float,
                        default=[0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0])
    parser.add_argument("--delta-v", type=float, default=0.1)
    args = parser.parse_args(argv)
    if (args.cap < 1 or args.checkpoint_interval < 1
            or not args.checkpoints or any(
                t <= 0 or t > args.cap or
                t % args.checkpoint_interval != 0
                for t in args.checkpoints)
            or not args.strengths or any(
                not np.isfinite(s) or not 0 <= s <= 1
                for s in args.strengths)
            or not np.isfinite(args.delta_v)
            or not 0 < args.delta_v <= 1):
        parser.error("invalid experiment parameters")
    rows = audit(args.seed, args.cap, args.checkpoint_interval,
                 args.checkpoints, args.strengths, args.delta_v)
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
