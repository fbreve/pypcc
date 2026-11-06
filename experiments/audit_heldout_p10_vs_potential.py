"""48-case held-out canonical PCC comparison: production potential vs label P10.

Frozen protocol: experiments/heldout_p10_vs_potential_protocol.md.
No changes to production kernels, graph construction, RNG or stopping APIs.
"""
import argparse
import csv
import time

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.benchmark_inloop_global_stopping import run as candidate_run

DATASETS = ("iris", "wine", "breast_cancer", "digits")
SEEDS = (5407, 5501, 5609, 5701, 5801, 5903)
SUPERVISION = (2, 5)
SHORT_CAP = 100000
LONG_CAP = 200000
PARITY_CAP = 10000
POLICIES = (
    "no_stop_100k",
    "no_stop_200k",
    "potential_default",
    "experimental_label_p10",
)
FIELDS = (
    "dataset", "labeled_per_class", "seed", "policy", "cap",
    "effective_iterations", "stopped_early", "n_unlabeled", "accuracy",
    "accuracy_100k_ref", "accuracy_200k_ref",
    "accuracy_delta_vs_100k_pp", "accuracy_delta_vs_200k_pp",
    "agreement_with_200k", "disagreements_with_200k",
    "accuracy_p10_minus_potential_pp", "agreement_p10_vs_potential",
    "disagreements_p10_vs_potential",
    "reference_100k_200k_accuracy_drift_pp",
    "reference_100k_200k_prediction_flips",
    "elapsed_seconds", "reference_200k_seconds",
    "saved_iterations_fraction_vs_200k",
    "saved_wall_time_fraction_vs_200k",
    "candidate_production_10k_no_stop_parity",
    "mean_degree", "reciprocal_arc_fraction",
)


def production_fit(graph, degrees, labels, seed, cap, stop):
    model = ParticleCompetitionAndCooperation(impl="numpy")
    if model._get_backend_fn().__name__ != "pcc_propagate_numpy":
        raise AssertionError("Unexpected production backend fallback")
    model.set_graph(graph, degrees, k_nn=10)
    start = time.perf_counter()
    predictions, info = model.fit_predict(
        labels.copy(), max_iter=cap, seed=seed,
        p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
        early_stop=stop, es_chk=2000,
        stop_criterion="potential", return_info=True,
    )
    seconds = time.perf_counter() - start
    steps = int(info["effective_iterations"])
    if not 0 < steps <= cap or (not stop and steps != cap):
        raise AssertionError("Invalid production propagation iterations")
    return np.asarray(predictions).copy(), seconds, steps


def experimental_fit(graph, degrees, labels, seed, cap, patience):
    predictions, seconds, info = candidate_run(
        graph, degrees, labels, seed, cap, patience, 1000,
        criterion="label_change", tolerance=0.0,
    )
    steps = int(info["effective_iterations"])
    if not 0 < steps <= cap or (patience == 0 and steps != cap):
        raise AssertionError("Invalid candidate propagation iterations")
    return np.asarray(predictions).copy(), seconds, steps


def evaluate(dataset, labeled_per_class, seed):
    graph, degrees, labels, truth, stats = canonical_prepare(
        dataset, seed, k=10, labeled_per_class=labeled_per_class
    )
    if stats["reciprocal_arc_fraction"] != 1.0:
        raise AssertionError("The production graph must be undirected")
    classes, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(classes) != len(np.unique(truth)) or not np.all(
            counts == labeled_per_class):
        raise AssertionError("Unexpected labeled class counts")

    unknown = labels == -1
    if not np.any(unknown):
        raise AssertionError("Missing initially unlabeled nodes")
    reference_gate, _, _ = production_fit(
        graph, degrees, labels, seed, PARITY_CAP, stop=False)
    candidate_gate, _, _ = experimental_fit(
        graph, degrees, labels, seed, PARITY_CAP, patience=0)
    parity = bool(np.array_equal(reference_gate, candidate_gate))
    if not parity:
        raise AssertionError(
            f"Candidate/production 10k no-stop gate failed: "
            f"{dataset} labels={labeled_per_class} seed={seed}")

    results = {
        "no_stop_100k": production_fit(
            graph, degrees, labels, seed, SHORT_CAP, stop=False),
        "no_stop_200k": production_fit(
            graph, degrees, labels, seed, LONG_CAP, stop=False),
        "potential_default": production_fit(
            graph, degrees, labels, seed, LONG_CAP, stop=True),
        "experimental_label_p10": experimental_fit(
            graph, degrees, labels, seed, LONG_CAP, patience=10),
    }
    for policy, (prediction, _, steps) in results.items():
        if prediction.shape != truth.shape or not np.all(np.isfinite(prediction)):
            raise AssertionError(f"Bad prediction shape/finite: {policy}")
        if np.any(prediction[~unknown] != labels[~unknown]):
            raise AssertionError(f"Home label changed: {policy}")
        expected_cap = SHORT_CAP if policy == "no_stop_100k" else LONG_CAP
        if not 0 < steps <= expected_cap or (
                policy.startswith("no_stop") and steps != expected_cap):
            raise AssertionError(f"Invalid steps for {policy}")

    ref100 = results["no_stop_100k"][0][unknown]
    ref200 = results["no_stop_200k"][0][unknown]
    potential = results["potential_default"][0][unknown]
    p10 = results["experimental_label_p10"][0][unknown]
    target = truth[unknown]

    accuracy = lambda p: float(np.mean(p == target))
    accuracy100 = accuracy(ref100)
    accuracy200 = accuracy(ref200)
    paired_delta = 100.0 * (accuracy(p10) - accuracy(potential))
    paired_agreement = float(np.mean(p10 == potential))
    paired_flips = int(np.count_nonzero(p10 != potential))
    reference_flips = int(np.count_nonzero(ref100 != ref200))
    reference_drift = 100.0 * (accuracy200 - accuracy100)
    ref200_time = results["no_stop_200k"][1]

    rows = []
    for policy in POLICIES:
        prediction, seconds, steps = results[policy]
        pred = prediction[unknown]
        cap = SHORT_CAP if policy == "no_stop_100k" else LONG_CAP
        rows.append(dict(
            dataset=dataset, labeled_per_class=labeled_per_class,
            seed=seed, policy=policy, cap=cap,
            effective_iterations=steps,
            stopped_early=bool(
                policy not in ("no_stop_100k", "no_stop_200k") and steps < cap),
            n_unlabeled=int(np.count_nonzero(unknown)),
            accuracy=accuracy(pred),
            accuracy_100k_ref=accuracy100,
            accuracy_200k_ref=accuracy200,
            accuracy_delta_vs_100k_pp=100.0 * (accuracy(pred) - accuracy100),
            accuracy_delta_vs_200k_pp=100.0 * (accuracy(pred) - accuracy200),
            agreement_with_200k=float(np.mean(pred == ref200)),
            disagreements_with_200k=int(np.count_nonzero(pred != ref200)),
            accuracy_p10_minus_potential_pp=paired_delta,
            agreement_p10_vs_potential=paired_agreement,
            disagreements_p10_vs_potential=paired_flips,
            reference_100k_200k_accuracy_drift_pp=reference_drift,
            reference_100k_200k_prediction_flips=reference_flips,
            elapsed_seconds=seconds,
            reference_200k_seconds=ref200_time,
            saved_iterations_fraction_vs_200k=(LONG_CAP - steps)/LONG_CAP,
            saved_wall_time_fraction_vs_200k=1.0 - seconds / ref200_time,
            candidate_production_10k_no_stop_parity=parity,
            mean_degree=stats["mean_degree"],
            reciprocal_arc_fraction=stats["reciprocal_arc_fraction"],
        ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--labeled-per-class", type=int, required=True,
                        choices=SUPERVISION)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        for seed in SEEDS:
            rows = evaluate(args.dataset, args.labeled_per_class, seed)
            writer.writerows(rows)
            handle.flush()
            p10 = rows[3]
            potential = rows[2]
            print(
                f"{args.dataset} labels={args.labeled_per_class} seed={seed}: "
                f"potential={potential['effective_iterations']}, "
                f"P10={p10['effective_iterations']}, "
                f"P10-minus-potential={p10['accuracy_p10_minus_potential_pp']:+.3f} pp, "
                f"100k->200k ref flips={p10['reference_100k_200k_prediction_flips']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
