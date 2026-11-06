"""Original vs per-node-cache Cython global potential: strict identical-state A/B.

The cache is refreshed on node visits, but global maxima are summed in the
original node order at every 10-step potential checkpoint.
See cython_incremental_rowmax_cache_protocol.md.
"""
import argparse
import csv
import time

import numpy as np

import pcc_step
from pcc import ParticleCompetitionAndCooperation
from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.audit_production_potential_monitoring_overhead import (
    expected_stop_max,
)

DATASETS = ("iris", "wine", "breast_cancer", "digits")
SEEDS = (6907, 7001, 7103, 7207)
POLICIES = (
    ("forced_no_stop", 20000, 2000000),
    ("production_potential", 100000, 2000),
)
VERIFY_CAP = 2000
FIELDS = (
    "dataset", "seed", "n_nodes", "n_classes", "n_particles",
    "reciprocal_arc_fraction", "policy", "variant", "order_index",
    "cap", "es_chk", "effective_iterations", "stop_reason",
    "stalled_checks", "stop_max", "checks",
    "dominance_fortran_contiguous", "elapsed_seconds",
    "legacy_seconds", "cached_seconds",
    "time_saved_percent", "all_state_arrays_bitwise_equal",
    "all_propagation_info_equal", "verified_metric_every_check",
    "verified_metric_checks", "n_unlabeled", "accuracy",
)
STATE_NAMES = (
    "prediction", "particle_position", "particle_strength",
    "particle_distance", "node_dominance", "node_ownership",
)


def pcc_propagate_rowmax_cached(*args, **kwargs):
    return pcc_step.pcc_propagate(
        *args, **kwargs, experimental_rowmax_cache=True)


def pcc_propagate_rowmax_cache_verified(*args, **kwargs):
    return pcc_step.pcc_propagate(
        *args, **kwargs, experimental_rowmax_cache=True,
        experimental_verify_rowmax_cache=True)


class CachedCythonPCC(ParticleCompetitionAndCooperation):
    def _get_backend_fn(self):
        return pcc_propagate_rowmax_cached


class VerifiedCachedCythonPCC(ParticleCompetitionAndCooperation):
    def _get_backend_fn(self):
        return pcc_propagate_rowmax_cache_verified


def fit(graph, degrees, labels, truth, seed, cap, es_chk,
        variant="legacy", timed=True):
    cls = {
        "legacy": ParticleCompetitionAndCooperation,
        "cache": CachedCythonPCC,
        "verified_cache": VerifiedCachedCythonPCC,
    }[variant]
    model = cls(impl="cython")
    actual = model._get_backend_fn().__name__
    expected = {
        "legacy": "pcc_propagate",
        "cache": "pcc_propagate_rowmax_cached",
        "verified_cache": "pcc_propagate_rowmax_cache_verified",
    }[variant]
    if actual != expected:
        raise AssertionError(f"Unexpected backend {actual} != {expected}")
    model.set_graph(graph, degrees, k_nn=10)
    start = time.perf_counter()
    predictions, info = model.fit_predict(
        labels.copy(), p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
        max_iter=cap, seed=seed, early_stop=True, es_chk=es_chk,
        stop_criterion="potential", return_info=True,
    )
    elapsed = time.perf_counter() - start
    if timed and elapsed <= 0:
        raise AssertionError("Unmeasurable run duration")
    steps = int(info["effective_iterations"])
    if not (0 < steps <= cap):
        raise AssertionError(f"Bad count for {variant}: {steps}")
    if np.any(predictions[labels >= 0] != labels[labels >= 0]):
        raise AssertionError(f"Labeled homes changed: {variant}")
    if not model.node.dominance.flags.f_contiguous:
        raise AssertionError("Cython dominance is not in the expected layout")
    state = {
        "prediction": np.asarray(predictions).copy(),
        "particle_position": model.part.curnode.copy(),
        "particle_strength": model.part.strength.copy(),
        "particle_distance": model.part.dist_table.copy(),
        "node_dominance": model.node.dominance.copy(),
        "node_ownership": model.owndeg.copy(),
    }
    for name, array in state.items():
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise AssertionError(f"Nonfinite {name}: {variant}")
    unknown = labels == -1
    accuracy = float(np.mean(predictions[unknown] == truth[unknown]))
    return state, dict(info), elapsed, accuracy


def check_dataset(dataset, path):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        for index, seed in enumerate(SEEDS):
            graph, degrees, labels, truth, graph_stats = canonical_prepare(
                dataset, seed, k=10, labeled_per_class=2)
            if graph_stats["reciprocal_arc_fraction"] != 1.0:
                raise AssertionError("Canonical graph not undirected")
            particles = int(np.count_nonzero(labels >= 0))
            if expected_stop_max(
                    len(labels), particles, 10, 2000000, degrees
            ) <= 20000 // 10 + 1:
                raise AssertionError("Forced metric might stop early")
            # This is NOT a timed variant: check every observed potential
            # checkpoint against an independent full O(N*C) scan.
            _, verified_info, _, _ = fit(
                graph, degrees, labels, truth, seed, VERIFY_CAP,
                2000000, variant="verified_cache", timed=False)
            expected_checks = (VERIFY_CAP + 9) // 10
            if (int(verified_info["effective_iterations"]) != VERIFY_CAP
                    or int(verified_info["checks"]) != expected_checks):
                raise AssertionError("Incomplete per-check metric audit")

            variants = ("legacy", "cache") if index % 2 == 0 else (
                "cache", "legacy")
            for policy, cap, es_chk in POLICIES:
                results = {}
                for order, variant in enumerate(variants):
                    state, info, elapsed, accuracy = fit(
                        graph, degrees, labels, truth, seed, cap,
                        es_chk, variant=variant)
                    results[variant] = (state, info, elapsed, accuracy, order)
                original_state, original_info, original_seconds, original_acc, _ = results["legacy"]
                cache_state, cache_info, cache_seconds, cache_acc, _ = results["cache"]
                equal_state = all(
                    np.array_equal(original_state[k], cache_state[k])
                    for k in STATE_NAMES)
                equal_info = original_info == cache_info
                if not (equal_state and equal_info and original_acc == cache_acc):
                    broken = [n for n in STATE_NAMES if not np.array_equal(
                        original_state[n], cache_state[n])]
                    raise AssertionError(
                        f"Cached metric changed dynamics: {dataset} "
                        f"seed={seed}, {policy}; arrays={broken}, "
                        f"legacy={original_info}, cached={cache_info}"
                    )
                if (policy == "forced_no_stop"
                        and int(original_info["effective_iterations"]) != cap):
                    raise AssertionError("Fixed-work case stopped early")
                saved = 100.0 * (1.0 - cache_seconds / original_seconds)
                for variant in ("legacy", "cache"):
                    _, info, secs, acc, order = results[variant]
                    writer.writerow(dict(
                        dataset=dataset, seed=seed, n_nodes=len(labels),
                        n_classes=len(np.unique(truth)),
                        n_particles=particles,
                        reciprocal_arc_fraction=graph_stats[
                            "reciprocal_arc_fraction"],
                        policy=policy, variant=variant, order_index=order,
                        cap=cap, es_chk=es_chk,
                        effective_iterations=int(info["effective_iterations"]),
                        stop_reason=info["stop_reason"],
                        stalled_checks=info["stalled_checks"],
                        stop_max=info["stop_max"], checks=info["checks"],
                        dominance_fortran_contiguous=True,
                        elapsed_seconds=secs,
                        legacy_seconds=original_seconds,
                        cached_seconds=cache_seconds,
                        time_saved_percent=saved,
                        all_state_arrays_bitwise_equal=equal_state,
                        all_propagation_info_equal=equal_info,
                        verified_metric_every_check=True,
                        verified_metric_checks=expected_checks,
                        n_unlabeled=int(np.count_nonzero(labels == -1)),
                        accuracy=acc,
                    ))
                handle.flush()
                print(
                    f"{dataset} seed={seed} {policy}: "
                    f"orig/cache={original_seconds:.5f}/{cache_seconds:.5f}s "
                    f"saved={saved:+.2f}%, stop={original_info['effective_iterations']} "
                    f"state+metadata parity PASS; 200 scalar checks PASS",
                    flush=True,
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    check_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
