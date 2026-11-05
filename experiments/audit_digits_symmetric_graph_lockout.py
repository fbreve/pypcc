"""Compare original directed Digits 10-NN with its symmetric union.

All cohorts, thresholds and endpoints were fixed in
digits_symmetric_graph_lockout_protocol.md before any new cohort result.
Experimental sequential NumPy PCC only; no changes to production behavior.
"""
import argparse
import csv

import numpy as np

from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import prepare


VARIANTS = ("directed_10nn", "symmetric_union")
GRAPH_FIELDS = (
    "seed", "variant", "nodes", "total_directed_edges",
    "min_degree", "median_degree", "mean_degree", "p95_degree",
    "max_degree", "reciprocal_directed_edge_fraction",
    "degree_increase_vs_directed_mean",
)
WINDOW_FIELDS = (
    "seed", "variant", "iteration", "class_id", "particles",
    "valid_visits", "accepted_moves", "zero_move_streak",
    "max_strength", "winning_neighbors", "neighbor_entries",
    "class_dominance_mass", "operational_lockout",
)
CLASS_FIELDS = (
    "seed", "variant", "class_id", "first_lockout_iteration",
    "first_recovery_iteration", "zero_movement_windows",
    "max_zero_move_streak", "accepted_moves_total",
    "class_dominance_mass_last", "max_strength_last",
    "winning_neighbor_entries_last", "final_particle_positions",
)
CASE_FIELDS = (
    "seed", "variant", "cap", "interval", "labels_per_class",
    "lockout_count", "has_lockout", "initially_unlabeled_count",
    "accuracy", "agreement_with_directed_10nn", "elapsed_seconds",
    "uninstrumented_seconds", "matches_uninstrumented",
    "max_strength_bound_excess",
)


def make_symmetric_union(directed, degrees):
    """Preserve original ordered neighbors; append sorted reverse-only edges."""
    nodes = directed.shape[0]
    if not np.all(degrees == 10) or directed.shape != (nodes, 10):
        raise AssertionError("Expected original 10-NN directed graph")
    incoming = [set() for _ in range(nodes)]
    originals = []
    for source in range(nodes):
        row = [int(x) for x in directed[source, :degrees[source]]]
        if (len(row) != len(set(row)) or source in row
                or any(not 0 <= x < nodes for x in row)):
            raise AssertionError("Bad directed KNN row")
        originals.append(row)
        for target in row:
            incoming[target].add(source)

    rows = []
    for node in range(nodes):
        original = originals[node]
        additions = sorted(incoming[node].difference(original))
        rows.append(original + additions)

    degrees_union = np.asarray([len(row) for row in rows], dtype=np.int64)
    symmetric = np.full(
        (nodes, int(degrees_union.max())), -1, dtype=np.int64)
    neighbors = [set(row) for row in rows]
    for source, row in enumerate(rows):
        symmetric[source, :len(row)] = row
        if row[:len(originals[source])] != originals[source]:
            raise AssertionError("Original directed prefix not preserved")
        if len(row) != len(neighbors[source]) or source in neighbors[source]:
            raise AssertionError("Self edge or duplicate after symmetrization")
        for target in row:
            if source not in neighbors[target]:
                raise AssertionError("Union graph contains an asymmetric edge")
    if np.any(degrees_union < degrees):
        raise AssertionError("Symmetrization reduced node degree")
    return symmetric, degrees_union


def graph_stats(seed, variant, graph, degrees, directed_degrees):
    nodes = graph.shape[0]
    nbrsets = [
        set(int(x) for x in graph[node, :degrees[node]])
        for node in range(nodes)
    ]
    edge_total = int(degrees.sum())
    reciprocal = sum(
        int(node in nbrsets[target])
        for node in range(nodes)
        for target in nbrsets[node])
    return dict(
        seed=seed, variant=variant, nodes=nodes,
        total_directed_edges=edge_total,
        min_degree=int(degrees.min()),
        median_degree=float(np.median(degrees)),
        mean_degree=float(np.mean(degrees)),
        p95_degree=float(np.percentile(degrees, 95)),
        max_degree=int(degrees.max()),
        reciprocal_directed_edge_fraction=reciprocal / edge_total,
        degree_increase_vs_directed_mean=float(
            np.mean(degrees - directed_degrees)),
    )


def observe_variant(seed, variant, graph, degrees, labels, truth,
                    cap, interval, zero_windows, strength_cutoff):
    nclasses = 10
    unlabeled = labels == -1
    num_windows = cap // interval
    visits = np.zeros((num_windows, nclasses), dtype=np.int64)
    accepted = np.zeros_like(visits)
    streak = np.zeros(nclasses, dtype=np.int64)
    observed = []
    last_positions = {}
    max_excess = 0.0

    def particle_event(iteration, cls, is_unlabeled, is_greedy,
                       is_accepted, donor_loss):
        wi = (iteration - 1) // interval
        visits[wi, cls] += 1
        accepted[wi, cls] += int(is_accepted)

    def state(iteration, dominance, positions, particle_labels, strengths):
        nonlocal max_excess
        wi = iteration // interval - 1
        if not np.isfinite(dominance).all():
            raise AssertionError("Nonfinite node dominance")
        if not np.allclose(
                dominance.sum(axis=1), 1.0, rtol=0, atol=1e-8):
            raise AssertionError("Node dominance simplex violated")
        masses = dominance[unlabeled].sum(axis=0)
        for cls in range(nclasses):
            ids = np.flatnonzero(particle_labels == cls)
            if len(ids) != 2:
                raise AssertionError("Expected 2 particles per class")
            if visits[wi, cls] != interval * len(ids):
                raise AssertionError(
                    f"Invalid visit count: seed={seed} variant={variant} "
                    f"class={cls} iteration={iteration}")
            current = positions[ids]
            if np.any(current < 0) or np.any(current >= graph.shape[0]):
                raise AssertionError("Particle outside valid graph")
            class_strength = strengths[ids]
            if not np.isfinite(class_strength).all():
                raise AssertionError("Nonfinite particle strength")
            excess = max(
                0.0, -float(class_strength.min()),
                float(class_strength.max()) - 1.0)
            max_excess = max(max_excess, excess)
            if excess > 1e-8:
                raise AssertionError("Material particle strength bound violation")
            winning = 0
            total_neighbors = 0
            for node in current:
                targets = graph[node, :int(degrees[node])]
                if not np.all((targets >= 0) & (targets < graph.shape[0])):
                    raise AssertionError("Invalid active adjacency entry")
                total_neighbors += len(targets)
                winning += int(np.count_nonzero(
                    dominance[targets, cls] ==
                    np.max(dominance[targets], axis=1)))
            streak[cls] = streak[cls] + 1 if accepted[wi, cls] == 0 else 0
            lockout = bool(streak[cls] >= zero_windows
                           and float(class_strength.max()) <= strength_cutoff
                           and winning == 0)
            observed.append(dict(
                seed=seed, variant=variant, iteration=iteration,
                class_id=cls, particles=len(ids),
                valid_visits=int(visits[wi, cls]),
                accepted_moves=int(accepted[wi, cls]),
                zero_move_streak=int(streak[cls]),
                max_strength=float(class_strength.max()),
                winning_neighbors=winning,
                neighbor_entries=total_neighbors,
                class_dominance_mass=float(masses[cls]),
                operational_lockout=lockout,
            ))
            if iteration == cap:
                last_positions[cls] = ";".join(map(str, current.tolist()))

    prediction, elapsed, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        particle_event_callback=particle_event,
        state_checkpoint_callback=state, checkpoint_interval=interval)
    if info["effective_iterations"] != cap:
        raise AssertionError("Early stop on no-stop comparison")
    reference, reference_seconds, _ = run(
        graph, degrees, labels, seed, cap, 0, interval)
    match = bool(np.array_equal(prediction, reference))
    if not match:
        raise AssertionError("Instrumented prediction differs from control")
    if len(observed) != num_windows * nclasses:
        raise AssertionError("Missing observation windows")
    classes = []
    for cls in range(nclasses):
        history = [r for r in observed if r["class_id"] == cls]
        first = next((i for i, r in enumerate(history)
                      if r["operational_lockout"]), None)
        later_move = (next((
            r["iteration"] for r in history[first + 1:]
            if r["accepted_moves"] > 0), None)
            if first is not None else None)
        classes.append(dict(
            seed=seed, variant=variant, class_id=cls,
            first_lockout_iteration=(
                history[first]["iteration"] if first is not None else ""),
            first_recovery_iteration=(
                later_move if later_move is not None else ""),
            zero_movement_windows=sum(
                r["accepted_moves"] == 0 for r in history),
            max_zero_move_streak=max(
                r["zero_move_streak"] for r in history),
            accepted_moves_total=int(accepted[:, cls].sum()),
            class_dominance_mass_last=history[-1]["class_dominance_mass"],
            max_strength_last=history[-1]["max_strength"],
            winning_neighbor_entries_last=history[-1]["winning_neighbors"],
            final_particle_positions=last_positions[cls],
        ))
    outcome = dict(
        seed=seed, variant=variant, cap=cap, interval=interval,
        labels_per_class=2,
        lockout_count=sum(bool(c["first_lockout_iteration"])
                          for c in classes),
        has_lockout=int(any(c["first_lockout_iteration"]
                            for c in classes)),
        initially_unlabeled_count=int(np.count_nonzero(unlabeled)),
        accuracy=float(np.mean(prediction[unlabeled] == truth[unlabeled])),
        agreement_with_directed_10nn=None,
        elapsed_seconds=float(elapsed),
        uninstrumented_seconds=float(reference_seconds),
        matches_uninstrumented=match,
        max_strength_bound_excess=max_excess,
    )
    return prediction, observed, classes, outcome


def run_seed(seed, cap, interval, zero_windows, strength_cutoff):
    directed, directed_degrees, labels, truth = prepare(
        "digits", seed, neighbors=10, labeled_per_class=2)
    symmetric, symmetric_degrees = make_symmetric_union(
        directed, directed_degrees)
    graphs = {
        "directed_10nn": (directed, directed_degrees),
        "symmetric_union": (symmetric, symmetric_degrees),
    }
    graph_rows = []
    all_windows, all_classes, cases = [], [], []
    predictions = {}
    for variant in VARIANTS:
        graph, degrees = graphs[variant]
        graph_rows.append(graph_stats(
            seed, variant, graph, degrees, directed_degrees))
        pred, windows, classes, case = observe_variant(
            seed, variant, graph, degrees, labels, truth,
            cap, interval, zero_windows, strength_cutoff)
        predictions[variant] = pred
        all_windows.extend(windows)
        all_classes.extend(classes)
        cases.append(case)
    mask = labels == -1
    agreement = float(np.mean(
        predictions["directed_10nn"][mask] ==
        predictions["symmetric_union"][mask]))
    for case in cases:
        case["agreement_with_directed_10nn"] = (
            1.0 if case["variant"] == "directed_10nn" else agreement)
    return graph_rows, all_windows, all_classes, cases


def save_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--graph-output", required=True)
    parser.add_argument("--windows-output", required=True)
    parser.add_argument("--classes-output", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--zero-windows", type=int, default=5)
    parser.add_argument("--strength-cutoff", type=float, default=0.01)
    args = parser.parse_args(argv)
    if (args.seed < 0 or args.cap < 1 or args.interval < 1
            or args.cap % args.interval or args.zero_windows < 1
            or not 0 < args.strength_cutoff <= 1):
        parser.error("Invalid protocol")
    graph, windows, classes, cases = run_seed(
        args.seed, args.cap, args.interval, args.zero_windows,
        args.strength_cutoff)
    save_csv(args.graph_output, GRAPH_FIELDS, graph)
    save_csv(args.windows_output, WINDOW_FIELDS, windows)
    save_csv(args.classes_output, CLASS_FIELDS, classes)
    save_csv(args.cases_output, CASE_FIELDS, cases)
    for case in cases:
        print(
            f"seed={case['seed']} variant={case['variant']} "
            f"locked={case['lockout_count']} "
            f"accuracy={case['accuracy']:.6f} "
            f"elapsed={case['elapsed_seconds']:.2f}",
            flush=True)


if __name__ == "__main__":
    main()
