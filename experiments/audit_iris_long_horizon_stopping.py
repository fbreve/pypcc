"""Frozen Iris long-horizon experiment against actual production backends.

Read iris_long_horizon_stopping_protocol.md before interpreting outputs.
This file does not change the PCC implementation or any production defaults.
"""
import argparse
import csv
import time

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.audit_canonical_production_backend_stop import (
    BACKENDS, BACKEND_NAMES, warm_numba,
)
from experiments.benchmark_inloop_global_stopping import run as candidate_run

SEEDS = (5003, 5209, 5303)
SHORT_CAP = 100000
MID_CAP = 250000
LONG_CAP = 500000
FIELDS = (
    "dataset", "seed", "backend", "policy", "cap",
    "effective_iterations", "stopped_early",
    "unlabeled", "accuracy", "accuracy_100k_no_stop",
    "accuracy_500k_no_stop", "accuracy_delta_pp_vs_100k",
    "accuracy_delta_pp_vs_500k",
    "agreement_with_100k", "agreement_with_500k",
    "flips_vs_100k", "flips_vs_500k",
    "ref_100k_to_500k_flips", "ref_100k_to_500k_accuracy_delta_pp",
    "ref_250k_to_500k_flips_numpy",
    "ref_250k_to_500k_accuracy_delta_pp_numpy",
    "seconds", "own_500k_no_stop_seconds",
    "wall_saving_vs_own_500k",
    "iteration_saving_vs_500k",
    "candidate_production_100k_parity",
    "mean_degree", "reciprocal_arc_fraction",
)


def production_fit(graph, degrees, labels, backend, seed, cap, stop):
    model = ParticleCompetitionAndCooperation(impl=backend)
    backend_name = model._get_backend_fn().__name__
    if backend_name != BACKEND_NAMES[backend]:
        raise AssertionError(f"Backend fallback: {backend} -> {backend_name}")
    model.set_graph(graph, degrees, k_nn=10)
    started = time.perf_counter()
    prediction, info = model.fit_predict(
        labels.copy(), p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
        max_iter=cap, early_stop=stop, es_chk=2000,
        stop_criterion="potential", seed=seed, return_info=True,
    )
    elapsed = time.perf_counter() - started
    count = int(info["effective_iterations"])
    if not 0 < count <= cap or (not stop and count != cap):
        raise AssertionError("Production iteration cap mismatch")
    if np.any(prediction[labels >= 0] != labels[labels >= 0]):
        raise AssertionError("Home labels changed")
    return np.asarray(prediction).copy(), elapsed, count


def candidate_fit(graph, degrees, labels, seed, cap, patience):
    started = time.perf_counter()
    # Existing experimental candidate; no changes to its particle dynamics.
    prediction, _, info = candidate_run(
        graph, degrees, labels, seed, cap, patience, 1000,
        criterion="label_change", tolerance=0.0,
    )
    elapsed = time.perf_counter() - started
    count = int(info["effective_iterations"])
    if not 0 < count <= cap or (patience == 0 and count != cap):
        raise AssertionError("Candidate iteration cap mismatch")
    if np.any(prediction[labels >= 0] != labels[labels >= 0]):
        raise AssertionError("Candidate home labels changed")
    return np.asarray(prediction).copy(), elapsed, count


def compare(seed):
    graph, degrees, labels, truth, stats = canonical_prepare(
        "iris", seed, k=10, labeled_per_class=2)
    unknown = labels == -1
    predictions = {}
    times = {}
    iterations = {}
    policies = {}

    for backend in BACKENDS:
        for cap in (SHORT_CAP, LONG_CAP):
            key = (backend, f"no_stop_{cap}")
            predictions[key], times[key], iterations[key] = production_fit(
                graph, degrees, labels, backend, seed, cap, stop=False)
            policies[key] = (cap, False)
        if backend == "numpy":
            key = (backend, f"no_stop_{MID_CAP}")
            predictions[key], times[key], iterations[key] = production_fit(
                graph, degrees, labels, backend, seed, MID_CAP, stop=False)
            policies[key] = (MID_CAP, False)
        key = (backend, "production_potential")
        predictions[key], times[key], iterations[key] = production_fit(
            graph, degrees, labels, backend, seed, LONG_CAP, stop=True)
        policies[key] = (LONG_CAP, True)

    candidate_ref, _, candidate_ref_iters = candidate_fit(
        graph, degrees, labels, seed, SHORT_CAP, 0)
    numpy_short = predictions[("numpy", f"no_stop_{SHORT_CAP}")]
    parity = bool(candidate_ref_iters == SHORT_CAP and
                  np.array_equal(candidate_ref, numpy_short))
    if not parity:
        raise AssertionError(
            f"Candidate vs production 100k no-stop parity FAILED: Iris {seed}")

    key = ("numpy", "experimental_label_p10")
    predictions[key], times[key], iterations[key] = candidate_fit(
        graph, degrees, labels, seed, LONG_CAP, 10)
    policies[key] = (LONG_CAP, True)

    rows = []
    for (backend, policy), pred in predictions.items():
        ref_short = predictions[(backend, f"no_stop_{SHORT_CAP}")]
        ref_long = predictions[(backend, f"no_stop_{LONG_CAP}")]
        ref_time = times[(backend, f"no_stop_{LONG_CAP}")]
        cap, stopping_policy = policies[(backend, policy)]
        short_accuracy = float(np.mean(ref_short[unknown] == truth[unknown]))
        long_accuracy = float(np.mean(ref_long[unknown] == truth[unknown]))
        acc = float(np.mean(pred[unknown] == truth[unknown]))
        mid_flips = ""
        mid_accuracy_delta = ""
        if backend == "numpy":
            ref_mid = predictions[(backend, f"no_stop_{MID_CAP}")]
            mid_flips = int(np.count_nonzero(
                ref_mid[unknown] != ref_long[unknown]))
            mid_accuracy_delta = 100.0 * (
                long_accuracy -
                float(np.mean(ref_mid[unknown] == truth[unknown]))
            )
        steps = iterations[(backend, policy)]
        rows.append(dict(
            dataset="iris", seed=seed, backend=backend, policy=policy,
            cap=cap, effective_iterations=steps,
            stopped_early=bool(stopping_policy and steps < cap),
            unlabeled=int(np.count_nonzero(unknown)),
            accuracy=acc,
            accuracy_100k_no_stop=short_accuracy,
            accuracy_500k_no_stop=long_accuracy,
            accuracy_delta_pp_vs_100k=100.0 * (acc - short_accuracy),
            accuracy_delta_pp_vs_500k=100.0 * (acc - long_accuracy),
            agreement_with_100k=float(np.mean(pred[unknown] == ref_short[unknown])),
            agreement_with_500k=float(np.mean(pred[unknown] == ref_long[unknown])),
            flips_vs_100k=int(np.count_nonzero(
                pred[unknown] != ref_short[unknown])),
            flips_vs_500k=int(np.count_nonzero(
                pred[unknown] != ref_long[unknown])),
            ref_100k_to_500k_flips=int(np.count_nonzero(
                ref_short[unknown] != ref_long[unknown])),
            ref_100k_to_500k_accuracy_delta_pp=100.0 * (
                long_accuracy - short_accuracy),
            ref_250k_to_500k_flips_numpy=mid_flips,
            ref_250k_to_500k_accuracy_delta_pp_numpy=mid_accuracy_delta,
            seconds=times[(backend, policy)],
            own_500k_no_stop_seconds=ref_time,
            wall_saving_vs_own_500k=1.0 - times[(backend, policy)] / ref_time,
            iteration_saving_vs_500k=(LONG_CAP - steps) / LONG_CAP,
            candidate_production_100k_parity=parity,
            mean_degree=stats["mean_degree"],
            reciprocal_arc_fraction=stats["reciprocal_arc_fraction"],
        ))

    for backend in BACKENDS:
        short = predictions[(backend, f"no_stop_{SHORT_CAP}")]
        long = predictions[(backend, f"no_stop_{LONG_CAP}")]
        potential = predictions[(backend, "production_potential")]
        print(f"Iris seed={seed} {backend}: 100k/500k flips="
              f"{np.count_nonzero(short[unknown] != long[unknown])}, "
              f"acc={np.mean(short[unknown] == truth[unknown]):.6f}/"
              f"{np.mean(long[unknown] == truth[unknown]):.6f}, "
              f"potential steps={iterations[(backend, 'production_potential')]}, "
              f"potential 500k agreement="
              f"{np.mean(potential[unknown] == long[unknown]):.6f}", flush=True)
    print(f"Iris seed={seed} NumPy: P10 steps="
          f"{iterations[('numpy', 'experimental_label_p10')]}, "
          f"production 100k parity={parity}", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    # Outside all measured sections. A separate seed cannot affect
    # explicitly seeded production and candidate test trajectories.
    warm_numba("iris")
    rows = compare(args.seed)
    if len(rows) != 11:
        raise AssertionError(f"Unexpected policy count: {len(rows)}")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Iris seed={args.seed}: saved {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
