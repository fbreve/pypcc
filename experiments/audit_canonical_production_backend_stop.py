"""Compare *production* NumPy, Numba and Cython potential stopping.

Protocol predeclared in canonical_production_backend_stop_protocol.md.
Inputs are verified canonical undirected kNN, never raw directed kNN.
"""
import argparse
import csv
import time

import numpy as np

import pcc_step
import pcc_numba
from pcc import ParticleCompetitionAndCooperation

from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.compare_dominance_stopping import DATASETS

BACKENDS = ("numpy", "numba", "cython")
BACKEND_NAMES = {
    "numpy": "pcc_propagate_numpy",
    "numba": "pcc_propagate_numba",
    "cython": "pcc_propagate",
}
FIELDS = (
    "dataset", "seed", "backend", "policy", "requested_impl",
    "actual_backend_function", "iterations", "cap", "stopped_early",
    "unlabeled_nodes", "accuracy", "backend_no_stop_accuracy",
    "accuracy_delta_pp_vs_own_no_stop", "agreement_with_own_no_stop",
    "agreement_with_numpy_no_stop", "all_three_no_stop_identical",
    "pairwise_no_stop_agreement_min", "seconds", "own_no_stop_seconds",
    "saved_wall_time_fraction_vs_own_no_stop",
    "saved_iterations_fraction", "mean_degree",
    "reciprocal_arc_fraction",
)


def fit(graph, degrees, labels, seed, cap, backend, early_stop):
    model = ParticleCompetitionAndCooperation(impl=backend)
    fn_name = model._get_backend_fn().__name__
    if fn_name != BACKEND_NAMES[backend]:
        raise AssertionError(
            f"Requested backend {backend}, got function {fn_name}")
    model.set_graph(graph, degrees, k_nn=10)
    start = time.perf_counter()
    output = model.fit_predict(
        labels.copy(), max_iter=cap, seed=seed,
        early_stop=early_stop, es_chk=2000,
        stop_criterion="potential", return_info=True)
    seconds = time.perf_counter() - start
    if not isinstance(output, tuple) or len(output) != 2:
        raise AssertionError("Missing production propagation metadata")
    prediction, info = output
    steps = int(info["effective_iterations"])
    if not 0 < steps <= cap:
        raise AssertionError("Invalid number of PCC iterations")
    if not early_stop and steps != cap:
        raise AssertionError("No-stop production run stopped early")
    if not np.all(np.isfinite(prediction)):
        raise AssertionError("Nonfinite prediction")
    if np.any(prediction[labels >= 0] != labels[labels >= 0]):
        raise AssertionError("A labeled home changed class")
    return prediction, seconds, steps, fn_name


def warm_numba(dataset):
    """Compile Numba kernels for the current dataset's class/layout signature.

    Numba can specialize on dominance-array contiguity: Digits (10 classes)
    can use a different layout from Iris (3 classes). The original Iris-only
    warmup was insufficient for unbiased per-dataset time comparisons.
    """
    graph, degrees, labels, _, _ = canonical_prepare(
        dataset, 29, k=10, labeled_per_class=2)
    fit(graph, degrees, labels, 29, 25, "numba", False)


def compare(dataset, seed, cap):
    graph, degrees, labels, truth, stats = canonical_prepare(
        dataset, seed, k=10, labeled_per_class=2)
    unknown = labels == -1
    no_stop = {}
    results = {}
    for be in BACKENDS:
        prediction, seconds, steps, fn_name = fit(
            graph, degrees, labels, seed, cap, be, False)
        no_stop[be] = prediction
        results[(be, "production_no_stop")] = (
            prediction, seconds, steps, fn_name)
        results[(be, "production_potential_default")] = fit(
            graph, degrees, labels, seed, cap, be, True)
    pair_agreements = [
        float(np.mean(no_stop[a][unknown] == no_stop[b][unknown]))
        for i, a in enumerate(BACKENDS)
        for b in BACKENDS[i + 1:]
    ]
    all_identical = all(np.array_equal(no_stop["numpy"], no_stop[be])
                        for be in BACKENDS[1:])
    records = []
    for be in BACKENDS:
        own_ref, own_ref_seconds, _, _ = results[(be, "production_no_stop")]
        own_acc = float(np.mean(own_ref[unknown] == truth[unknown]))
        for policy in ("production_no_stop",
                       "production_potential_default"):
            pred, seconds, steps, fn_name = results[(be, policy)]
            acc = float(np.mean(pred[unknown] == truth[unknown]))
            records.append(dict(
                dataset=dataset, seed=seed, backend=be, policy=policy,
                requested_impl=be, actual_backend_function=fn_name,
                iterations=steps, cap=cap, stopped_early=steps < cap,
                unlabeled_nodes=int(np.count_nonzero(unknown)),
                accuracy=acc, backend_no_stop_accuracy=own_acc,
                accuracy_delta_pp_vs_own_no_stop=100*(acc-own_acc),
                agreement_with_own_no_stop=float(np.mean(
                    pred[unknown] == own_ref[unknown])),
                agreement_with_numpy_no_stop=float(np.mean(
                    pred[unknown] == no_stop["numpy"][unknown])),
                all_three_no_stop_identical=all_identical,
                pairwise_no_stop_agreement_min=min(pair_agreements),
                seconds=float(seconds),
                own_no_stop_seconds=float(own_ref_seconds),
                saved_wall_time_fraction_vs_own_no_stop=(
                    1.0 - seconds / own_ref_seconds),
                saved_iterations_fraction=(cap - steps) / cap,
                mean_degree=stats["mean_degree"],
                reciprocal_arc_fraction=stats["reciprocal_arc_fraction"],
            ))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4001, 4093])
    parser.add_argument("--cap", type=int, default=100000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.cap < 1):
        parser.error("Invalid preregistered backend comparison")
    warm_numba(args.dataset)
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            rows = compare(args.dataset, seed, args.cap)
            writer.writerows(rows)
            print(f"{args.dataset} seed={seed}: " +
                  ", ".join(
                      f"{r['backend']}/{r['policy']}={r['iterations']}"
                      for r in rows),
                  flush=True)


if __name__ == "__main__":
    main()
