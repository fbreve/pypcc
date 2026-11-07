"""Low-resolution seeded prefix replay on untouched production PCC backends.

Every checkpoint is a fresh fit from iteration zero, not a snapshot during
one continuous fit. No production kernel or stopping policy is modified.
Protocol: experiments/seeded_prefix_replay_protocol.md.
"""
import argparse
import csv
import hashlib
import json
import time

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph

DATASETS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}
BACKENDS = {
    "numpy": "pcc_propagate_numpy",
    "numba": "pcc_propagate_numba",
    "cython": "pcc_propagate",
}
SEEDS = (11003, 11117)
MILESTONES = (1, 11, 101, 501, 2001)
FIELDS = (
    "dataset", "backend", "seed", "iteration", "n_nodes", "classes",
    "particles", "graph_digest", "labels_digest", "reciprocal_graph",
    "actual_backend", "requested_early_stop", "reported_iterations",
    "reported_stop_reason", "reported_checks",
    "accuracy_unlabeled_retrospective", "class_changes_since_previous_sample",
    "mean_max_dominance", "mean_margin", "state_digests",
    "terminal_original_model_parity", "elapsed_seconds_exploratory",
)
ARRAYS = (
    "predictions", "positions", "strengths", "distances",
    "dominance", "ownership",
)


def digest(array):
    arr = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.dtype.str.encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


class InfoObservedPCC(PCC):
    """Capture backend return_info without changing its propagation kernel."""

    def _get_backend_fn(self):
        native = super()._get_backend_fn()

        def capture(*args, **kwargs):
            if "return_info" in kwargs:
                raise AssertionError("Unexpected duplicate return_info")
            self.captured_propagation_info = dict(
                native(*args, return_info=True, **kwargs))
            return None

        # __genNodes checks the dispatched backend function's name when
        # choosing the dominance array layout (C vs Fortran order).
        capture.__name__ = native.__name__
        return capture


def canonical(dataset, seed):
    loader = DATASETS[dataset]
    loaded = loader()
    truth = np.asarray(loaded.target, dtype=np.int64)
    features = StandardScaler().fit_transform(
        np.asarray(loaded.data, dtype=np.float64))
    graph, degrees = build_knn_graph(features, k_nn=10)
    neighbors = [
        list(map(int, graph[i, :int(degrees[i])]))
        for i in range(len(truth))
    ]
    sets = [set(row) for row in neighbors]
    if any(len(row) != len(sets[i]) or i in row for i,row in enumerate(neighbors)):
        raise AssertionError("Duplicate/self neighbor in canonical graph")
    if any(i not in sets[j] for i,adj in enumerate(sets) for j in adj):
        raise AssertionError("Directed graph found; required union-symmetric 10NN")
    labels = np.full(len(truth), -1, dtype=np.int64)
    selector = np.random.default_rng(seed)
    classes = np.unique(truth)
    for cls in classes:
        candidate = np.flatnonzero(truth == cls)
        if len(candidate) < 2:
            raise AssertionError("Not enough per-class labeled homes")
        labels[selector.choice(candidate, 2, replace=False)] = cls
    if np.count_nonzero(labels != -1) != 2 * len(classes):
        raise AssertionError("Wrong known-node count")
    return graph, degrees, labels, truth


def run_fresh(backend, graph, degrees, labels, seed, cap, observed):
    model_type = InfoObservedPCC if observed else PCC
    model = model_type(impl=backend)
    actual = (PCC._get_backend_fn(model)).__name__
    if actual != BACKENDS[backend]:
        raise AssertionError(f"Backend fallback: {backend}->{actual}")
    model.set_graph(graph, degrees, k_nn=10)
    t0 = time.perf_counter()
    prediction = model.fit_predict(
        labels.copy(), seed=seed, max_iter=cap, early_stop=False,
        p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0, es_chk=2000,
    )
    elapsed = time.perf_counter() - t0
    if (not np.array_equal(prediction[labels >= 0], labels[labels >= 0])
            or model.c != len(np.unique(labels[labels >= 0]))):
        raise AssertionError("Labeled home or class mapping changed")
    result = dict(
        predictions=np.asarray(prediction).copy(),
        positions=model.part.curnode.copy(),
        strengths=model.part.strength.copy(),
        distances=model.part.dist_table.copy(),
        dominance=model.node.dominance.copy(),
        ownership=model.owndeg.copy(),
    )
    for key, array in result.items():
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise AssertionError(f"Nonfinite state {key}")
    info = getattr(model, "captured_propagation_info", None)
    if observed:
        if not isinstance(info, dict):
            raise AssertionError("No backend return_info captured")
        if (int(info["effective_iterations"]) != cap or
                info["stop_reason"] != "max_iter" or int(info["checks"]) != 0):
            raise AssertionError(f"Unexpected native propagation metrics: {info}")
    return result, info, elapsed, actual


def evaluate(dataset, backend, output):
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        file.flush()
        for seed in SEEDS:
            graph, degrees, labels, truth = canonical(dataset, seed)
            graph_digest = digest(graph) + ":" + digest(degrees)
            labels_digest = digest(labels)
            prior = None
            observations = []
            for cap in MILESTONES:
                arrays, info, secs, actual = run_fresh(
                    backend, graph, degrees, labels, seed, cap, observed=True)
                changes = (None if prior is None else int(np.count_nonzero(
                    arrays["predictions"] != prior)))
                unknown = labels == -1
                dom = arrays["dominance"]
                top2 = np.sort(dom, axis=1)[:, -2:]
                row = dict(
                    dataset=dataset,backend=backend,seed=seed,iteration=cap,
                    n_nodes=len(labels),classes=len(np.unique(truth)),
                    particles=int(np.count_nonzero(labels >= 0)),
                    graph_digest=graph_digest,labels_digest=labels_digest,
                    reciprocal_graph=True,actual_backend=actual,
                    requested_early_stop=False,
                    reported_iterations=info["effective_iterations"],
                    reported_stop_reason=info["stop_reason"],
                    reported_checks=info["checks"],
                    accuracy_unlabeled_retrospective=float(np.mean(
                        arrays["predictions"][unknown] == truth[unknown])),
                    class_changes_since_previous_sample=changes,
                    mean_max_dominance=float(np.max(dom,axis=1).mean()),
                    mean_margin=float(np.mean(top2[:,1]-top2[:,0])),
                    state_digests=json.dumps({
                        k:digest(arrays[k]) for k in ARRAYS},
                        sort_keys=True,separators=(",",":")),
                    terminal_original_model_parity=False,
                    elapsed_seconds_exploratory=secs,
                )
                observations.append(row)
                prior = arrays["predictions"]
                if cap == MILESTONES[-1]:
                    # Real ordinary public API, no observer adapter: compare
                    # bitwise all six arrays, including the high-level final
                    # prediction. The duplicate is never included in timings.
                    control, _, _, _ = run_fresh(
                        backend, graph, degrees, labels, seed, cap, observed=False)
                    diff = [k for k in ARRAYS if not np.array_equal(
                        arrays[k],control[k])]
                    if diff:
                        raise AssertionError(
                            f"Seeded replay diverged from original API "
                            f"{dataset}/{backend}/{seed}: {diff}")
            for row in observations:
                row["terminal_original_model_parity"] = True
                writer.writerow(row)
            file.flush()
            print(f"{dataset}/{backend}/{seed}: {len(observations)} "
                  "separately replayed milestones + exact terminal control "
                  "PASS (not a continuous-run trace)", flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    p.add_argument("--backend", required=True, choices=tuple(BACKENDS))
    p.add_argument("--output", required=True)
    args=p.parse_args(argv)
    evaluate(args.dataset,args.backend,args.output)


if __name__ == "__main__":
    main()
