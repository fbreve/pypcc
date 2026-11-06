"""Strict A/B of legacy vs opt-in columnwise Cython potential monitoring.

Protocol: cython_columnwise_potential_protocol.md. No backend default change.
"""
import argparse
import csv
import time

import numpy as np
import pcc_step

from pcc import ParticleCompetitionAndCooperation
from experiments.audit_canonical_undirected_lockout import canonical_prepare

DATASETS = ("iris", "wine", "breast_cancer", "digits")
SEEDS = (6503, 6607, 6701, 6803)
POLICIES = (
    ("forced_no_stop", 20000, 2000000),
    ("production_potential", 100000, 2000),
)
FIELDS = (
    "dataset", "seed", "n_nodes", "n_classes", "n_particles",
    "reciprocal_arc_fraction", "policy", "variant", "order_index",
    "cap", "es_chk", "effective_iterations", "stop_reason",
    "stalled_checks", "stop_max", "checks",
    "dominance_fortran_contiguous", "elapsed_seconds",
    "legacy_seconds", "columnwise_seconds",
    "time_saved_percent", "all_state_arrays_bitwise_equal",
    "all_propagation_info_equal",
)
STATE_NAMES = (
    "prediction", "particle_position", "particle_strength",
    "particle_distance", "node_dominance", "node_ownership",
)


def pcc_propagate_columnwise(*args, **kwargs):
    """Opt-in adapter: no edits to pcc.py or its normal backend dispatch."""
    return pcc_step.pcc_propagate(
        *args, **kwargs, experimental_columnwise_mmpot=True)


class ColumnwiseCythonPCC(ParticleCompetitionAndCooperation):
    def _get_backend_fn(self):
        return pcc_propagate_columnwise


def expected_stop_max(n_nodes, n_particles, es_chk):
    return max(1, round(
        (n_nodes / (n_particles * 10)) * round(es_chk * 0.1)))


def fit(graph, degrees, labels, seed, cap, es_chk, optimized):
    cls = ColumnwiseCythonPCC if optimized else ParticleCompetitionAndCooperation
    model = cls(impl="cython")
    actual = model._get_backend_fn().__name__
    expected = "pcc_propagate_columnwise" if optimized else "pcc_propagate"
    if actual != expected:
        raise AssertionError(f"Wrong Cython implementation: {actual}")
    model.set_graph(graph, degrees, k_nn=10)
    start = time.perf_counter()
    prediction, info = model.fit_predict(
        labels.copy(), p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
        max_iter=cap, seed=seed, early_stop=True, es_chk=es_chk,
        stop_criterion="potential", return_info=True,
    )
    elapsed = time.perf_counter() - start
    steps = int(info["effective_iterations"])
    if not 0 < steps <= cap:
        raise AssertionError("Out-of-range propagation iterations")
    if np.any(prediction[labels >= 0] != labels[labels >= 0]):
        raise AssertionError("Labeled home prediction changed")
    state = dict(
        prediction=np.asarray(prediction).copy(),
        particle_position=model.part.curnode.copy(),
        particle_strength=model.part.strength.copy(),
        particle_distance=model.part.dist_table.copy(),
        node_dominance=model.node.dominance.copy(),
        node_ownership=model.owndeg.copy(),
    )
    if not model.node.dominance.flags.f_contiguous:
        raise AssertionError("Columnwise proposal assumes Fortran dominance")
    for name, arr in state.items():
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise AssertionError(f"Nonfinite {name}")
    return state, dict(info), elapsed


def run_dataset(dataset, output):
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        for index, seed in enumerate(SEEDS):
            graph, degrees, labels, _truth, graph_stats = canonical_prepare(
                dataset, seed, k=10, labeled_per_class=2)
            if graph_stats["reciprocal_arc_fraction"] != 1.0:
                raise AssertionError("Graph not reciprocal")
            n_particles = int(np.count_nonzero(labels >= 0))
            variants = (False, True) if index % 2 == 0 else (True, False)
            for policy, cap, es_chk in POLICIES:
                if policy == "forced_no_stop" and (
                        expected_stop_max(len(labels), n_particles, es_chk)
                        <= cap // 10 + 1):
                    raise AssertionError("Forced monitor could stop early")
                results = {}
                for order, optimized in enumerate(variants):
                    state, info, secs = fit(
                        graph, degrees, labels, seed, cap, es_chk, optimized)
                    results[optimized] = (state, info, secs, order)
                legacy_state, legacy_info, legacy_seconds, _ = results[False]
                opt_state, opt_info, opt_seconds, _ = results[True]
                equal_state = all(np.array_equal(
                    legacy_state[name], opt_state[name]) for name in STATE_NAMES)
                equal_info = legacy_info == opt_info
                if not equal_state or not equal_info:
                    broken = [n for n in STATE_NAMES if not np.array_equal(
                        legacy_state[n], opt_state[n])]
                    raise AssertionError(
                        f"Columnwise Cython not equivalent: "
                        f"{dataset}/{seed}/{policy} "
                        f"arrays={broken}, metadata="
                        f"{(legacy_info, opt_info)}")
                if policy == "forced_no_stop" and (
                        int(legacy_info["effective_iterations"]) != cap):
                    raise AssertionError("Fixed-work monitor unexpectedly stopped")

                saved = 100.0 * (1.0 - opt_seconds / legacy_seconds)
                for optimized in (False, True):
                    _, info, secs, order = results[optimized]
                    writer.writerow(dict(
                        dataset=dataset, seed=seed,
                        n_nodes=len(labels), n_classes=len(
                            np.unique(labels[labels >= 0])),
                        n_particles=n_particles,
                        reciprocal_arc_fraction=graph_stats[
                            "reciprocal_arc_fraction"],
                        policy=policy,
                        variant="columnwise" if optimized else "legacy",
                        order_index=order,
                        cap=cap, es_chk=es_chk,
                        effective_iterations=info["effective_iterations"],
                        stop_reason=info["stop_reason"],
                        stalled_checks=info["stalled_checks"],
                        stop_max=info["stop_max"], checks=info["checks"],
                        dominance_fortran_contiguous=True,
                        elapsed_seconds=secs, legacy_seconds=legacy_seconds,
                        columnwise_seconds=opt_seconds,
                        time_saved_percent=saved,
                        all_state_arrays_bitwise_equal=equal_state,
                        all_propagation_info_equal=equal_info,
                    ))
                handle.flush()
                print(
                    f"{dataset} seed={seed} {policy} "
                    f"steps={legacy_info['effective_iterations']}: "
                    f"legacy/columnwise={legacy_seconds:.4f}/{opt_seconds:.4f}s, "
                    f"saved={saved:+.2f}% (all state/info identical)",
                    flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
