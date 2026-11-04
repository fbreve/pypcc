"""Locate the class-1 mobility failure at particle-level resolution.

Compare a transient co-location/recovery around 39k with the permanent lockout
around 48k in Wine seed 83. Event and state callbacks are observational only.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


FIELDS = (
    "dataset", "seed", "window_start", "window_end", "particle_id",
    "attempts", "accepted_moves", "greedy_visits",
    "winning_target_before_update", "winning_target_after_update",
    "accepted_after_conquest", "last_accepted_iteration",
    "strength_start", "strength_end", "min_strength_after_visit",
    "start_node", "end_node", "winning_neighbors_at_end",
    "valid_neighbors_at_end", "max_own_neighbor_dominance_at_end",
    "mean_own_neighbor_dominance_at_end",
)


def trace(seed, cap, interval, start, end):
    graph, degrees, labels, truth = prepare("wine", seed)
    stats = {}
    snapshots = {}
    label_one = np.flatnonzero(labels == 1)
    if len(label_one) != 2:
        raise AssertionError("Expected two class-1 particles")

    def state(iteration, dominance, current, particle_labels, strengths):
        if start <= iteration <= end:
            snapshots[iteration] = (
                dominance, current, particle_labels, strengths)

    def event(iteration, particle_id, source, target, cls,
              greedy, accepted, strength_before, strength_after,
              own_before, best_before, own_after, best_after):
        if cls != 1:
            return
        window_end = ((iteration - 1) // interval + 1) * interval
        key = (window_end, particle_id)
        if key not in stats:
            stats[key] = {
                "attempts": 0, "accepted_moves": 0, "greedy_visits": 0,
                "winning_target_before_update": 0,
                "winning_target_after_update": 0,
                "accepted_after_conquest": 0,
                "last_accepted_iteration": None,
                "min_strength_after_visit": 1.0,
            }
        row = stats[key]
        row["attempts"] += 1
        row["accepted_moves"] += int(accepted)
        row["greedy_visits"] += int(greedy)
        before_winning = own_before == best_before
        after_winning = own_after == best_after
        row["winning_target_before_update"] += int(before_winning)
        row["winning_target_after_update"] += int(after_winning)
        row["accepted_after_conquest"] += int(
            accepted and not before_winning and after_winning)
        row["min_strength_after_visit"] = min(
            row["min_strength_after_visit"], strength_after)
        if accepted:
            row["last_accepted_iteration"] = iteration

    instrumented, _, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        state_checkpoint_callback=state, checkpoint_interval=interval,
        detailed_particle_callback=event,
        detail_start_iteration=start, detail_end_iteration=end)
    control, _, _ = run(graph, degrees, labels, seed, cap, 0, interval)
    if info["effective_iterations"] != cap:
        raise AssertionError("Unexpected stopping")
    if not np.array_equal(instrumented, control):
        raise AssertionError("Tracing changed predicted labels")
    if len(snapshots) != (end - start) // interval + 1:
        raise AssertionError("Missing state checkpoints")
    if not np.array_equal(
            np.argmax(snapshots[end][0][labels == -1], axis=1),
            instrumented[labels == -1]):
        if end == cap:
            raise AssertionError("Last state checkpoint differs from output")

    rows = []
    for window_end in range(start + interval, end + 1, interval):
        start_dom, start_nodes, part_labels, start_strengths = snapshots[
            window_end - interval]
        end_dom, end_nodes, end_labels, end_strengths = snapshots[window_end]
        if not np.array_equal(part_labels, end_labels):
            raise AssertionError("Particle classes changed during propagation")
        for p_i in np.flatnonzero(end_labels == 1):
            row = stats.get((window_end, int(p_i)))
            if row is None or row["attempts"] != interval:
                raise AssertionError(
                    f"Expected one valid target per particle per iteration: "
                    f"{seed}, {p_i}, {window_end}")
            current_node = int(end_nodes[p_i])
            neighbors = graph[current_node, :int(degrees[current_node])]
            neighbors = neighbors[
                (neighbors >= 0) & (neighbors < end_dom.shape[0])]
            winning = int(np.count_nonzero(
                end_dom[neighbors, 1] == np.max(
                    end_dom[neighbors], axis=1)))
            rows.append(dict(
                dataset="wine", seed=seed,
                window_start=window_end - interval,
                window_end=window_end,
                particle_id=int(p_i), **row,
                strength_start=float(start_strengths[p_i]),
                strength_end=float(end_strengths[p_i]),
                start_node=int(start_nodes[p_i]),
                end_node=current_node,
                winning_neighbors_at_end=winning,
                valid_neighbors_at_end=len(neighbors),
                max_own_neighbor_dominance_at_end=float(
                    np.max(end_dom[neighbors, 1])),
                mean_own_neighbor_dominance_at_end=float(
                    np.mean(end_dom[neighbors, 1])),
            ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--start", type=int, default=38000)
    parser.add_argument("--end", type=int, default=50000)
    args = parser.parse_args(argv)
    if (args.interval < 1 or not 0 <= args.start < args.end <= args.cap
            or args.start % args.interval or args.end % args.interval):
        parser.error("Invalid audit window")
    rows = trace(args.seed, args.cap, args.interval,
                 args.start, args.end)
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
