"""Compare PyPCC's *existing* potential stop against label stability.

Cohort/settings frozen in canonical_default_stop_comparison_protocol.md.
Run on canonical production undirected graphs only.
"""
import argparse
import csv
import time

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.audit_canonical_undirected_lockout import canonical_prepare
from experiments.benchmark_inloop_global_stopping import run
from experiments.compare_dominance_stopping import DATASETS


FIELDS = (
    "dataset", "seed", "policy", "cap", "iterations",
    "early_stop", "saved_iterations_fraction", "accuracy",
    "reference_accuracy", "accuracy_delta_pp",
    "agreement_with_reference", "elapsed_seconds",
    "reference_seconds", "saved_wall_time_fraction",
    "agreement_with_other_stopping_policy",
    "accuracy_delta_vs_other_stopping_policy_pp",
    "unlabeled_count", "mean_degree",
    "reciprocal_arc_fraction", "candidate_production_no_stop_parity",
)


def production_run(graph, degrees, labels, seed, cap, early_stop):
    model = ParticleCompetitionAndCooperation(impl="numpy")
    # Critical: original build_graph uses k_nn=10, not max symmetrized degree.
    model.set_graph(graph, degrees, k_nn=10)
    start = time.perf_counter()
    prediction, info = model.fit_predict(
        labels.copy(), max_iter=cap, seed=seed,
        early_stop=early_stop, es_chk=2000,
        stop_criterion="potential", return_info=True)
    elapsed = time.perf_counter() - start
    return prediction, elapsed, info


def compare(dataset, seed, cap):
    graph, degrees, labels, truth, stats = canonical_prepare(
        dataset, seed, k=10, labeled_per_class=2)
    unknown = labels == -1
    reference, reference_seconds, ref_info = production_run(
        graph, degrees, labels, seed, cap, early_stop=False)
    if ref_info["effective_iterations"] != cap:
        raise AssertionError("Production no-stop reference ended early")
    candidate_ref, _, candidate_ref_info = run(
        graph, degrees, labels, seed, cap, 0, 1000)
    parity = bool(np.array_equal(reference, candidate_ref))
    if (candidate_ref_info["effective_iterations"] != cap or
            not parity):
        raise AssertionError(
            f"Production/candidate no-stop prediction parity failure: "
            f"{dataset} seed={seed}")

    production_stop, prod_seconds, prod_info = production_run(
        graph, degrees, labels, seed, cap, early_stop=True)
    candidate_stop, candidate_seconds, candidate_info = run(
        graph, degrees, labels, seed, cap, 10, 1000,
        criterion="label_change", tolerance=0.0)

    policies = [
        ("production_no_stop", reference, reference_seconds, ref_info),
        ("production_potential_default", production_stop,
         prod_seconds, prod_info),
        ("experimental_label_patience_10", candidate_stop,
         candidate_seconds, candidate_info),
    ]
    ref_accuracy = float(np.mean(reference[unknown] == truth[unknown]))
    rows = []
    for name, prediction, elapsed, info in policies:
        iterations = int(info["effective_iterations"])
        if not 0 < iterations <= cap:
            raise AssertionError("Invalid stop iteration count")
        if name == "production_potential_default":
            other = candidate_stop
        elif name == "experimental_label_patience_10":
            other = production_stop
        else:
            other = reference
        accuracy = float(np.mean(prediction[unknown] == truth[unknown]))
        other_accuracy = float(np.mean(other[unknown] == truth[unknown]))
        rows.append(dict(
            dataset=dataset, seed=seed, policy=name, cap=cap,
            iterations=iterations, early_stop=(iterations < cap),
            saved_iterations_fraction=(cap - iterations) / cap,
            accuracy=accuracy, reference_accuracy=ref_accuracy,
            accuracy_delta_pp=100 * (accuracy - ref_accuracy),
            agreement_with_reference=float(np.mean(
                prediction[unknown] == reference[unknown])),
            elapsed_seconds=float(elapsed),
            reference_seconds=float(reference_seconds),
            saved_wall_time_fraction=1.0 - elapsed / reference_seconds,
            agreement_with_other_stopping_policy=float(np.mean(
                prediction[unknown] == other[unknown])),
            accuracy_delta_vs_other_stopping_policy_pp=100.0 * (
                accuracy - other_accuracy),
            unlabeled_count=int(np.count_nonzero(unknown)),
            mean_degree=stats["mean_degree"],
            reciprocal_arc_fraction=stats["reciprocal_arc_fraction"],
            candidate_production_no_stop_parity=parity,
        ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[3407, 3511, 3607])
    parser.add_argument("--cap", type=int, default=100000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.cap < 1):
        parser.error("Invalid fixed comparison protocol")
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            rows = compare(args.dataset, seed, args.cap)
            writer.writerows(rows)
            print(
                f"{args.dataset}, seed={seed}: " +
                ", ".join(
                    f"{row['policy']}={row['iterations']}"
                    for row in rows),
                flush=True)


if __name__ == "__main__":
    main()
