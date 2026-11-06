"""Predeclared, experimental full-state backend parity audit.

See backend_choice_controlled_parity_protocol.md. This executes actual
production backends on (a) a choice-forced synthetic graph and
(b) canonical undirected 10-NN datasets. It does NOT modify backend kernels.
"""
import argparse
import csv

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.audit_canonical_undirected_lockout import canonical_prepare

SEEDS = (4409, 4513)
SINGLE_CHOICE_CAPS = (1, 2, 10, 100, 500)
CANONICAL_CAPS = (1, 2, 5, 10, 50, 200, 1000)
P_GREEDY = (0.0, 1.0)
DATASETS = ("iris", "wine", "breast_cancer", "digits")
BACKEND_NAMES = {
    "numpy": "pcc_propagate_numpy",
    "numba": "pcc_propagate_numba",
    "cython": "pcc_propagate",
}
FLOAT_FIELDS = ("strength", "dominance", "owndeg")
DISCRETE_FIELDS = ("prediction", "position", "distance")
FIELDS = (
    "graph_type", "dataset", "seed", "p_grd", "cap",
    "backend_a", "backend_b", "n_nodes", "n_particles",
    "prediction_agreement", "prediction_equal",
    "position_equal", "position_mismatches",
    "distance_equal", "distance_mismatches",
    "max_abs_strength", "max_abs_dominance", "max_abs_owndeg",
    "strength_close", "dominance_close", "owndeg_close",
    "full_state_close",
)


def single_choice_graph():
    """Undirected, degree-one pairs: no RNG can change destination choice."""
    n = 16
    graph = (np.arange(n, dtype=np.int64) ^ 1).reshape(n, 1)
    degrees = np.ones(n, dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    labels[::2] = np.arange(n // 2, dtype=np.int64) % 2
    if (not np.all(graph[:, 0] != np.arange(n))
            or not np.array_equal(graph[graph[:, 0], 0], np.arange(n))
            or not np.all(degrees == 1)):
        raise AssertionError("Invalid undirected forced-choice control graph")
    return graph, degrees, labels


def fit(graph, degrees, labels, seed, cap, backend, p_grd, k_nn):
    model = ParticleCompetitionAndCooperation(impl=backend)
    actual = model._get_backend_fn().__name__
    if actual != BACKEND_NAMES[backend]:
        raise AssertionError(f"Silent backend fallback {backend} -> {actual}")
    model.set_graph(graph, degrees, k_nn=k_nn)
    pred, info = model.fit_predict(
        labels.copy(), p_grd=p_grd,
        max_iter=cap, early_stop=False, seed=seed, return_info=True)
    if int(info["effective_iterations"]) != cap:
        raise AssertionError("Unexpected stop or iteration count")
    if np.any(pred[labels >= 0] != labels[labels >= 0]):
        raise AssertionError("A home label changed")
    state = {
        "prediction": np.asarray(pred).copy(),
        "position": model.part.curnode.copy(),
        "distance": model.part.dist_table.copy(),
        "strength": model.part.strength.copy(),
        "dominance": model.node.dominance.copy(),
        "owndeg": model.owndeg.copy(),
    }
    for field in FLOAT_FIELDS:
        if not np.all(np.isfinite(state[field])):
            raise AssertionError(f"Nonfinite {field}: {backend}")
    return state


def compare(left, right, graph_type, dataset, seed, p_grd,
            cap, backend_a, backend_b):
    for name in DISCRETE_FIELDS + FLOAT_FIELDS:
        if left[name].shape != right[name].shape:
            raise AssertionError(f"Mismatched {name} shape")
    record = dict(
        graph_type=graph_type, dataset=dataset,
        seed=seed, p_grd=p_grd, cap=cap,
        backend_a=backend_a, backend_b=backend_b,
        n_nodes=left["prediction"].size,
        n_particles=left["position"].size,
        prediction_agreement=float(np.mean(
            left["prediction"] == right["prediction"])),
    )
    for field in DISCRETE_FIELDS:
        record[f"{field}_equal"] = bool(
            np.array_equal(left[field], right[field]))
        if field != "prediction":
            record[f"{field}_mismatches"] = int(np.count_nonzero(
                left[field] != right[field]))
    for field in FLOAT_FIELDS:
        record[f"max_abs_{field}"] = float(np.max(
            np.abs(left[field] - right[field])))
        record[f"{field}_close"] = bool(np.allclose(
            left[field], right[field], rtol=0.0, atol=1e-12))
    record["full_state_close"] = bool(all(
        record[f"{f}_equal"] for f in DISCRETE_FIELDS) and all(
        record[f"{f}_close"] for f in FLOAT_FIELDS))
    return record


def forced_choice_rows():
    graph, degrees, labels = single_choice_graph()
    rows = []
    for seed in SEEDS:
        for p_grd in P_GREEDY:
            for cap in SINGLE_CHOICE_CAPS:
                states = {
                    backend: fit(graph, degrees, labels, seed, cap,
                                 backend, p_grd, k_nn=1)
                    for backend in BACKEND_NAMES
                }
                for a, b in (("numpy", "numba"),
                             ("numpy", "cython"),
                             ("numba", "cython")):
                    row = compare(states[a], states[b], "forced_degree_one",
                                  "synthetic_pairs", seed, p_grd, cap, a, b)
                    rows.append(row)
                    if not row["full_state_close"]:
                        raise AssertionError(
                            f"Forced-choice state mismatch: {a}/{b} "
                            f"seed={seed}, greedy={p_grd}, cap={cap}: {row}")
    return rows


def canonical_rows(dataset):
    rows = []
    for seed in SEEDS:
        graph, degrees, labels, _truth, stats = canonical_prepare(
            dataset, seed, k=10, labeled_per_class=2)
        if stats["reciprocal_arc_fraction"] != 1.0:
            raise AssertionError("Noncanonical directed graph")
        for p_grd in P_GREEDY:
            for cap in CANONICAL_CAPS:
                left = fit(graph, degrees, labels, seed, cap,
                           "numpy", p_grd, k_nn=10)
                right = fit(graph, degrees, labels, seed, cap,
                            "numba", p_grd, k_nn=10)
                rows.append(compare(left, right, "canonical_undirected_10nn",
                                    dataset, seed, p_grd, cap,
                                    "numpy", "numba"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = forced_choice_rows() if args.dataset == "iris" else []
    rows.extend(canonical_rows(args.dataset))
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    canonical = [r for r in rows if r["graph_type"] == "canonical_undirected_10nn"]
    for p_grd in P_GREEDY:
        cohort = [r for r in canonical if r["p_grd"] == p_grd]
        mismatches = [r for r in cohort if not r["full_state_close"]]
        print(f"{args.dataset} p_grd={p_grd}: {len(cohort)} checkpoints; "
              f"{len(mismatches)} full-state mismatches", flush=True)
        if mismatches:
            first = mismatches[0]
            print(f"  First sampled mismatch: seed={first['seed']}, "
                  f"cap={first['cap']}, pos_mismatch={first['position_mismatches']}, "
                  f"distance_mismatch={first['distance_mismatches']}, "
                  f"max_dom_diff={first['max_abs_dominance']}", flush=True)
    if args.dataset == "iris":
        print("Forced-choice NumPy/Numba/Cython parity passed at all "
              "60 state comparisons", flush=True)


if __name__ == "__main__":
    main()
