"""Independent Windows/Linux A/B replication of cached global potential.

Frozen in cython_rowmax_cache_replication_protocol.md.
Retain all repeats and compare full state and stop info for every pair.
"""
import argparse
import csv

import numpy as np

from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.audit_production_potential_monitoring_overhead import (
    expected_stop_max,
)
from experiments.audit_cython_incremental_rowmax_cache import (
    fit, STATE_NAMES,
)

DATASETS = ("digits", "breast_cancer")
SEEDS = (7307, 7403, 7507, 7603)
POLICIES = (
    ("forced_no_stop", 20000, 2000000, 5),
    ("production_potential", 100000, 2000, 1),
)
FIELDS = (
    "runner_os", "dataset", "seed", "policy", "repeat", "variant",
    "order_index", "n_nodes", "n_classes", "n_particles",
    "cap", "es_chk", "effective_iterations", "checks",
    "stalled_checks", "stop_max", "stop_reason",
    "legacy_seconds", "cached_seconds", "elapsed_seconds",
    "cached_time_saved_percent", "accuracy",
    "all_six_arrays_exact", "complete_stop_info_exact",
    "graph_reciprocal",
)


def run_case(dataset, runner_os, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        for seed_index, seed in enumerate(SEEDS):
            graph, degrees, labels, truth, stats = canonical_prepare(
                dataset, seed, k=10, labeled_per_class=2)
            if stats["reciprocal_arc_fraction"] != 1.0:
                raise AssertionError("Production kNN graph not undirected")
            n_particles = int(np.count_nonzero(labels >= 0))
            if expected_stop_max(
                    len(labels), n_particles, 10, 2000000, degrees) <= 20001 // 10:
                raise AssertionError("The fixed-work monitor might stop early")
            for policy, cap, es_chk, repeats in POLICIES:
                for repeat in range(repeats):
                    order_list = ("legacy", "cache") if (
                        seed_index + repeat) % 2 == 0 else ("cache", "legacy")
                    outputs = {}
                    for order, variant in enumerate(order_list):
                        outputs[variant] = (
                            *fit(graph, degrees, labels, truth, seed,
                                 cap, es_chk, variant=variant), order)
                    old_state, old_info, old_time, old_acc, _ = outputs["legacy"]
                    new_state, new_info, new_time, new_acc, _ = outputs["cache"]
                    exact_arrays = all(np.array_equal(
                        old_state[name], new_state[name])
                        for name in STATE_NAMES)
                    exact_stop = old_info == new_info
                    if not (exact_arrays and exact_stop and old_acc == new_acc):
                        different = [n for n in STATE_NAMES
                                     if not np.array_equal(
                                         old_state[n],new_state[n])]
                        raise AssertionError(
                            f"{runner_os}/{dataset}/{seed}/{policy}/{repeat} "
                            f"cached metric parity FAIL: {different}; "
                            f"info={(old_info,new_info)}")
                    if (policy == "forced_no_stop"
                            and int(old_info["effective_iterations"]) != cap):
                        raise AssertionError("Fixed workload stopped prematurely")
                    saved = 100*(1-new_time/old_time)
                    for variant in ("legacy", "cache"):
                        _, info, elapsed, accuracy, order = outputs[variant]
                        writer.writerow(dict(
                            runner_os=runner_os, dataset=dataset, seed=seed,
                            policy=policy, repeat=repeat, variant=variant,
                            order_index=order, n_nodes=len(labels),
                            n_classes=len(np.unique(truth)),
                            n_particles=n_particles,
                            cap=cap, es_chk=es_chk,
                            effective_iterations=info["effective_iterations"],
                            checks=info["checks"],
                            stalled_checks=info["stalled_checks"],
                            stop_max=info["stop_max"],
                            stop_reason=info["stop_reason"],
                            legacy_seconds=old_time, cached_seconds=new_time,
                            elapsed_seconds=elapsed,
                            cached_time_saved_percent=saved,
                            accuracy=accuracy,
                            all_six_arrays_exact=exact_arrays,
                            complete_stop_info_exact=exact_stop,
                            graph_reciprocal=True,
                        ))
                    handle.flush()
                    print(f"{runner_os}/{dataset}/{seed}/{policy}/{repeat}: "
                          f"time saved {saved:+.2f}% "
                          f"({old_time:.4f}s -> {new_time:.4f}s); "
                          f"exact state+stop PASS", flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--runner-os", choices=("ubuntu-latest","windows-latest"),
                   required=True)
    p.add_argument("--output", required=True)
    args=p.parse_args(argv)
    run_case(args.dataset,args.runner_os,args.output)


if __name__ == "__main__":
    main()
