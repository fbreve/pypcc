"""Experimental full-state parity and overhead audit for production potential checks.

See production_potential_monitoring_overhead_protocol.md. Never edits kernels.
"""
import argparse
import csv
import time

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.audit_canonical_production_backend_stop import (
    BACKENDS, BACKEND_NAMES,
)

DATASETS = ("iris", "wine", "breast_cancer", "digits")
SEEDS = (6101, 6203, 6301, 6407)
CAP = 20000
MONITOR_ES_CHK = 2000000
BASELINE_ES_CHK = 2000
ARRAY_FIELDS = (
    "prediction", "part_position", "part_strength",
    "part_dist_table", "node_dominance", "node_ownership",
)
FIELDS = (
    "dataset", "seed", "backend", "mode", "order_index",
    "actual_backend_function", "cap", "effective_iterations",
    "n_nodes", "n_classes", "n_particles", "mean_degree",
    "reciprocal_arc_fraction", "es_chk", "calculated_stop_max",
    "can_not_stop_early", "elapsed_seconds",
    "paired_baseline_seconds", "paired_monitor_seconds",
    "overhead_percent", "monitor_cost_share_percent",
    "all_six_state_arrays_bitwise_equal",
)


def expected_stop_max(n_nodes, n_particles, k_nn, es_chk, degrees):
    if not np.any(degrees):
        return 1
    return max(1, round((n_nodes / (n_particles * max(k_nn, 1))) *
                        round(es_chk * 0.1)))


def fit(graph, degrees, labels, seed, backend, cap, monitoring):
    model = ParticleCompetitionAndCooperation(impl=backend)
    actual = model._get_backend_fn().__name__
    if actual != BACKEND_NAMES[backend]:
        raise AssertionError(f"Backend fallback: {backend}->{actual}")
    model.set_graph(graph, degrees, k_nn=10)
    es_chk = MONITOR_ES_CHK if monitoring else BASELINE_ES_CHK
    stop_max = expected_stop_max(
        len(labels), int(np.count_nonzero(labels >= 0)), 10,
        es_chk, degrees,
    )
    if monitoring and stop_max <= cap // 10 + 1:
        raise AssertionError(
            f"Not structurally guaranteed no-stop: {stop_max} cap {cap}")

    begin = time.perf_counter()
    prediction, info = model.fit_predict(
        labels.copy(), max_iter=cap, seed=seed,
        p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
        early_stop=monitoring, es_chk=es_chk,
        stop_criterion="potential", return_info=True,
    )
    elapsed = time.perf_counter() - begin
    iterations = int(info["effective_iterations"])
    if iterations != cap:
        raise AssertionError(
            f"Unexpected stopping: {backend} monitor={monitoring} "
            f"iterations={iterations}, cap={cap}")
    if np.any(np.asarray(prediction)[labels >= 0] != labels[labels >= 0]):
        raise AssertionError("A labeled home changed label")

    state = {
        "prediction": np.asarray(prediction).copy(),
        "part_position": model.part.curnode.copy(),
        "part_strength": model.part.strength.copy(),
        "part_dist_table": model.part.dist_table.copy(),
        "node_dominance": model.node.dominance.copy(),
        "node_ownership": model.owndeg.copy(),
    }
    for field in ARRAY_FIELDS:
        if state[field].dtype.kind == "f" and not np.all(
                np.isfinite(state[field])):
            raise AssertionError(f"Nonfinite state {field}")
    return state, elapsed, stop_max, actual


def run_dataset(dataset, output):
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()

        # Warm BOTH Numba specializations for this dataset's layout and
        # early_stop mode. Never include compilation in timing.
        warm_graph, warm_degrees, warm_labels, _, _ = canonical_prepare(
            dataset, 29, k=10, labeled_per_class=2)
        for flag in (False, True):
            fit(warm_graph, warm_degrees, warm_labels, 29,
                "numba", 25, flag)

        for i, seed in enumerate(SEEDS):
            graph, degrees, labels, _truth, stats = canonical_prepare(
                dataset, seed, k=10, labeled_per_class=2)
            if stats["reciprocal_arc_fraction"] != 1.0:
                raise AssertionError("Directed adjacency is not canonical")
            nclasses = len(np.unique(labels[labels >= 0]))
            n_particles = int(np.count_nonzero(labels >= 0))
            flags = (False, True) if i % 2 == 0 else (True, False)

            for backend in BACKENDS:
                results = {}
                for order, flag in enumerate(flags):
                    state, secs, stop_max, actual = fit(
                        graph, degrees, labels, seed, backend, CAP, flag)
                    results[flag] = (state, secs, stop_max, actual, order)
                off_state, off_time, _, _, _ = results[False]
                on_state, on_time, _, _, _ = results[True]
                exact = all(
                    np.array_equal(off_state[field], on_state[field])
                    for field in ARRAY_FIELDS
                )
                if not exact:
                    broken = [
                        field for field in ARRAY_FIELDS
                        if not np.array_equal(
                            off_state[field], on_state[field])
                    ]
                    raise AssertionError(
                        f"Potential monitor changes production state: "
                        f"{dataset}/{seed}/{backend} fields={broken}"
                    )
                pct = 100.0 * (on_time / off_time - 1.0)
                cost_share = 100.0 * (on_time - off_time) / on_time
                for flag in (False, True):
                    _, secs, stop_max, actual, order = results[flag]
                    writer.writerow(dict(
                        dataset=dataset, seed=seed, backend=backend,
                        mode=("metric_on_forced_no_stop"
                              if flag else "metric_off"),
                        order_index=order,
                        actual_backend_function=actual,
                        cap=CAP, effective_iterations=CAP,
                        n_nodes=len(labels), n_classes=nclasses,
                        n_particles=n_particles,
                        mean_degree=stats["mean_degree"],
                        reciprocal_arc_fraction=stats[
                            "reciprocal_arc_fraction"],
                        es_chk=(MONITOR_ES_CHK if flag else BASELINE_ES_CHK),
                        calculated_stop_max=stop_max,
                        can_not_stop_early=(
                            not flag or stop_max > CAP // 10 + 1),
                        elapsed_seconds=secs,
                        paired_baseline_seconds=off_time,
                        paired_monitor_seconds=on_time,
                        overhead_percent=pct,
                        monitor_cost_share_percent=cost_share,
                        all_six_state_arrays_bitwise_equal=exact,
                    ))
                handle.flush()
                print(
                    f"{dataset} seed={seed} backend={backend}: "
                    f"metric off/on={off_time:.4f}/{on_time:.4f}s, "
                    f"overhead={pct:+.2f}%, exact state={exact}",
                    flush=True,
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
