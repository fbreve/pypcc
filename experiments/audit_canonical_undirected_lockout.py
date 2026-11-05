"""Canonical, undirected-PCC lockout prevalence and predictive drift audit.

The seeds and endpoints were frozen in canonical_undirected_lockout_protocol.md
before results. Uses production pcc_graph.build_knn_graph; no directed
experimental prepare() path, intervention, or early stopping.
"""
import argparse
import csv

import numpy as np
from sklearn.preprocessing import StandardScaler

from pcc_graph import build_knn_graph
from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import DATASETS

WINDOW_FIELDS = (
    "dataset", "seed", "iteration", "class_id", "particles",
    "accepted_moves", "valid_visits", "zero_move_streak",
    "winning_neighbors", "neighbor_entries",
    "max_strength", "class_mass", "class_predictions",
    "class_recall", "accuracy", "operational_lockout",
)
CLASS_FIELDS = (
    "dataset", "seed", "class_id", "first_lockout_iteration",
    "first_recovery_window_end", "max_zero_move_streak",
    "zero_move_windows", "accepted_moves_total",
    "class_mass_25000", "class_mass_50000",
    "recall_25000", "recall_50000",
    "predicted_count_25000", "predicted_count_50000",
    "last_max_strength", "last_winning_neighbors",
)
CASE_FIELDS = (
    "dataset", "seed", "graph_mode", "n_nodes", "n_classes",
    "labeled_per_class", "mean_degree", "min_degree", "max_degree",
    "reciprocal_arc_fraction", "cap", "interval",
    "lockout_classes", "any_lockout", "max_zero_move_streak",
    "accuracy_25000", "accuracy_50000", "accuracy_delta",
    "prediction_changes_25000_to_50000", "unlabeled_nodes",
    "max_strength_bounds_excess", "matches_uninstrumented",
)


def canonical_prepare(name, seed, k=10, labeled_per_class=2):
    dataset = DATASETS[name]()
    features = StandardScaler().fit_transform(
        dataset.data.astype(np.float64))
    truth = np.asarray(dataset.target, dtype=np.int64)
    rng = np.random.default_rng(seed)
    labels = np.full(len(truth), -1, dtype=np.int64)
    for cls in np.unique(truth):
        candidates = np.flatnonzero(truth == cls)
        labels[rng.choice(
            candidates, labeled_per_class, replace=False)] = int(cls)
    graph, degrees = build_knn_graph(features, k_nn=k)
    n = len(truth)
    if graph.shape[0] != n or len(degrees) != n or np.any(degrees <= 0):
        raise AssertionError("Unexpected graph node count or degree")
    neighset = []
    for src in range(n):
        neighbors = graph[src, :int(degrees[src])]
        row = [int(x) for x in neighbors]
        if (len(row) != len(set(row)) or src in row or
                any(x < 0 or x >= n for x in row)):
            raise AssertionError(f"Invalid production graph row {src}")
        neighset.append(set(row))
    edges = sum(map(len, neighset))
    reciprocal = sum(
        int(src in neighset[dst])
        for src in range(n) for dst in neighset[src])
    if reciprocal != edges:
        raise AssertionError("Production kNN graph is not reciprocal")
    graph_stats = dict(
        n_nodes=n, mean_degree=float(np.mean(degrees)),
        min_degree=int(np.min(degrees)),
        max_degree=int(np.max(degrees)),
        reciprocal_arc_fraction=reciprocal / edges)
    return graph, degrees, labels, truth, graph_stats


def audit_case(name, seed, cap, interval, streak_limit, strength_cutoff):
    if cap % 2:
        raise ValueError("Cap must permit midpoint comparison")
    graph, degrees, labels, truth, graph_stats = canonical_prepare(
        name, seed)
    unknown = labels == -1
    nclasses = int(np.max(truth)) + 1
    n_windows = cap // interval
    visits = np.zeros((n_windows, nclasses), dtype=np.int64)
    accepted = np.zeros_like(visits)
    streak = np.zeros(nclasses, dtype=np.int64)
    window_rows = []
    midpoint_predictions = None
    last_predictions = None
    bound_excess = 0.0

    def event(iteration, cls, is_unlabeled, greedy, moved, loss):
        w = (iteration - 1) // interval
        visits[w, cls] += 1
        accepted[w, cls] += int(moved)

    def checkpoint(it, dominance, positions, pclasses, strengths):
        nonlocal midpoint_predictions, last_predictions, bound_excess
        w = it // interval - 1
        if (not np.isfinite(dominance).all() or
                not np.allclose(dominance.sum(axis=1), 1.0,
                                rtol=0, atol=1e-8)):
            raise AssertionError("Dominance simplex/finite failure")
        predictions = np.argmax(dominance[unknown], axis=1)
        if it == cap // 2:
            midpoint_predictions = predictions.copy()
        if it == cap:
            last_predictions = predictions.copy()
        accuracy = float(np.mean(predictions == truth[unknown]))
        totals = dominance[unknown].sum(axis=0)
        pred_counts = np.bincount(predictions, minlength=nclasses)
        for cls in range(nclasses):
            ids = np.flatnonzero(pclasses == cls)
            if len(ids) != 2:
                raise AssertionError("Expected 2 particles per class")
            if visits[w, cls] != interval * 2:
                raise AssertionError(
                    f"Missing valid visits for {name}/{seed}/{cls}/{it}")
            own_strengths = strengths[ids]
            if not np.isfinite(own_strengths).all():
                raise AssertionError("Nonfinite particle strength")
            excess = max(0.0, -float(own_strengths.min()),
                         float(own_strengths.max()) - 1.0)
            bound_excess = max(bound_excess, excess)
            if excess > 1e-8:
                raise AssertionError("Material particle strength excursion")
            neighbors_winning = 0
            neighbors_total = 0
            for pos in positions[ids]:
                if not 0 <= int(pos) < len(labels):
                    raise AssertionError("Invalid particle position")
                target = graph[pos, :int(degrees[pos])]
                neighbors_total += len(target)
                neighbors_winning += int(np.count_nonzero(
                    dominance[target, cls] ==
                    np.max(dominance[target], axis=1)))
            streak[cls] = streak[cls] + 1 if accepted[w, cls] == 0 else 0
            operational = bool(
                streak[cls] >= streak_limit
                and float(own_strengths.max()) <= strength_cutoff
                and neighbors_winning == 0)
            class_mask = truth[unknown] == cls
            window_rows.append(dict(
                dataset=name, seed=seed, iteration=it, class_id=cls,
                particles=len(ids), accepted_moves=int(accepted[w, cls]),
                valid_visits=int(visits[w, cls]),
                zero_move_streak=int(streak[cls]),
                winning_neighbors=neighbors_winning,
                neighbor_entries=neighbors_total,
                max_strength=float(own_strengths.max()),
                class_mass=float(totals[cls]),
                class_predictions=int(pred_counts[cls]),
                class_recall=float(np.mean(predictions[class_mask] == cls)),
                accuracy=accuracy, operational_lockout=operational))

    pred, _, info = run(
        graph, degrees, labels, seed, cap, 0, interval,
        particle_event_callback=event,
        state_checkpoint_callback=checkpoint,
        checkpoint_interval=interval)
    if (info["effective_iterations"] != cap or
            len(window_rows) != nclasses * n_windows or
            midpoint_predictions is None or last_predictions is None):
        raise AssertionError("Incomplete no-stop trajectory")
    control, _, _ = run(graph, degrees, labels, seed, cap, 0, interval)
    if not np.array_equal(pred, control):
        raise AssertionError("Observer altered final labels")
    if not np.array_equal(last_predictions, pred[unknown]):
        raise AssertionError("Last checkpoint differs from final labels")

    class_rows = []
    for cls in range(nclasses):
        history = [r for r in window_rows if r["class_id"] == cls]
        onset_idx = next((i for i, r in enumerate(history)
                          if r["operational_lockout"]), None)
        recovered = (
            next((r["iteration"] for r in history[onset_idx + 1:]
                  if r["accepted_moves"] > 0), None)
            if onset_idx is not None else None)
        before = history[cap // (2 * interval) - 1]
        after = history[-1]
        class_rows.append(dict(
            dataset=name, seed=seed, class_id=cls,
            first_lockout_iteration=(
                history[onset_idx]["iteration"]
                if onset_idx is not None else ""),
            first_recovery_window_end=(
                recovered if recovered is not None else ""),
            max_zero_move_streak=max(
                r["zero_move_streak"] for r in history),
            zero_move_windows=sum(
                r["accepted_moves"] == 0 for r in history),
            accepted_moves_total=int(accepted[:, cls].sum()),
            class_mass_25000=before["class_mass"],
            class_mass_50000=after["class_mass"],
            recall_25000=before["class_recall"],
            recall_50000=after["class_recall"],
            predicted_count_25000=before["class_predictions"],
            predicted_count_50000=after["class_predictions"],
            last_max_strength=after["max_strength"],
            last_winning_neighbors=after["winning_neighbors"]))
    accuracy_mid = float(
        np.mean(midpoint_predictions == truth[unknown]))
    accuracy_last = float(
        np.mean(last_predictions == truth[unknown]))
    case = dict(
        dataset=name, seed=seed, graph_mode="production_undirected_union_knn",
        n_classes=nclasses, labeled_per_class=2,
        cap=cap, interval=interval,
        lockout_classes=sum(bool(r["first_lockout_iteration"])
                            for r in class_rows),
        any_lockout=any(bool(r["first_lockout_iteration"])
                        for r in class_rows),
        max_zero_move_streak=max(
            r["max_zero_move_streak"] for r in class_rows),
        accuracy_25000=accuracy_mid, accuracy_50000=accuracy_last,
        accuracy_delta=accuracy_last - accuracy_mid,
        prediction_changes_25000_to_50000=int(np.count_nonzero(
            midpoint_predictions != last_predictions)),
        unlabeled_nodes=int(unknown.sum()),
        max_strength_bounds_excess=bound_excess,
        matches_uninstrumented=True,
        **graph_stats)
    return window_rows, class_rows, case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1709, 1801, 1901, 2003, 2111])
    parser.add_argument("--windows-output", required=True)
    parser.add_argument("--classes-output", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--cap", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--lockout-windows", type=int, default=5)
    parser.add_argument("--strength-cutoff", type=float, default=0.01)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or
            args.cap < 2 or args.interval < 1 or
            args.cap % (2 * args.interval) or args.lockout_windows < 1 or
            not 0 < args.strength_cutoff <= 1):
        parser.error("Invalid canonical lockout protocol parameters")
    with (open(args.windows_output, "w", newline="", encoding="utf-8") as wf,
          open(args.classes_output, "w", newline="", encoding="utf-8") as cf,
          open(args.cases_output, "w", newline="", encoding="utf-8") as of):
        w = csv.DictWriter(wf, fieldnames=WINDOW_FIELDS)
        c = csv.DictWriter(cf, fieldnames=CLASS_FIELDS)
        o = csv.DictWriter(of, fieldnames=CASE_FIELDS)
        w.writeheader()
        c.writeheader()
        o.writeheader()
        for seed in args.seeds:
            windows, classes, case = audit_case(
                args.dataset, seed, args.cap, args.interval,
                args.lockout_windows, args.strength_cutoff)
            w.writerows(windows)
            c.writerows(classes)
            o.writerow(case)
            print(
                f"{args.dataset} seed {seed}: "
                f"lockouts={case['lockout_classes']}, "
                f"accuracy_25k={case['accuracy_25000']:.5f}, "
                f"accuracy_50k={case['accuracy_50000']:.5f}",
                flush=True)


if __name__ == "__main__":
    main()
