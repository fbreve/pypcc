"""Reproducible synthetic *complete fit_predict* CPU A/B pilot.

Research-only. No real data, CPU superiority claim, kernel replacement, or
unobserved experimental fit is carried out by importing this module.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.live_numpy_milestones import observe_numpy_sequential


def fixture(nodes=200):
    if type(nodes) is not int or nodes < 24:
        raise ValueError("nodes must be an integer >= 24")
    neighbors = np.full((nodes, 16), -1, dtype=np.int64)
    offsets = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
    for node in range(nodes):
        for j, offset in enumerate(offsets):
            neighbors[node, j] = (node + offset) % nodes
    degrees = np.full(nodes, len(offsets), dtype=np.int64)
    labels = np.full(nodes, -1, dtype=np.int64)
    labels[[0, 2, 4]] = 0
    labels[[nodes // 2, nodes // 2 + 2, nodes // 2 + 4]] = 1
    return neighbors, degrees, labels


def fingerprint(state):
    """Bytewise canonical digest of a small set of final mutable fit arrays."""
    h = hashlib.sha256()
    for x in (state.node.dominance, state.node.label, state.part.curnode,
              state.part.label, state.part.strength, state.part.dist_table,
              state.owndeg):
        data = np.ascontiguousarray(x)
        h.update(str(data.shape).encode("ascii"))
        h.update(data.dtype.str.encode("ascii"))
        h.update(data.tobytes())
    return h.hexdigest()


def rng_digest():
    state = np.random.get_state()
    h = hashlib.sha256()
    h.update(state[0].encode("ascii"))
    h.update(state[1].astype("<u4", copy=False).tobytes())
    h.update(int(state[2]).to_bytes(8, "little", signed=True))
    h.update(str(state[3:]).encode("ascii"))
    return h.hexdigest()


def run_one(neighbors, degrees, labels, iterations, milestones, observed):
    model = ParticleCompetitionAndCooperation(impl="numpy")
    model.set_graph(neighbors, degrees, k_nn=10)
    info = []

    if observed:
        def experimental_propagate(*args, **kwargs):
            observation = observe_numpy_sequential(
                *args, **kwargs, milestones=milestones,
            )
            info.append(observation)
            return None
        # Instance-local test dispatch, not a globally modified backend.
        model._get_backend_fn = lambda: experimental_propagate

    start = time.process_time_ns()
    predictions = model.fit_predict(
        labels, max_iter=iterations, early_stop=False, seed=71,
    )
    elapsed_ns = time.process_time_ns() - start
    if observed:
        if len(info) != 1 or info[0].observed_iterations != milestones:
            raise AssertionError("missing or malformed requested observations")

    return dict(
        cpu_ns=elapsed_ns, prediction_sha256=hashlib.sha256(
            np.asarray(predictions, dtype="<i8").tobytes()).hexdigest(),
        full_state_sha256=fingerprint(model), rng_sha256=rng_digest(),
        bytes_copied=info[0].bytes_copied if observed else 0,
        saved_milestones=len(info[0].milestones) if observed else 0,
    )


def benchmark(nodes=200, iterations=1200, repeats=3):
    if type(iterations) is not int or iterations < 100 or iterations > 5000:
        raise ValueError("iterations must be an integer in 100..5000")
    if type(repeats) is not int or repeats < 2 or repeats > 8:
        raise ValueError("repeats must be an integer in 2..8")
    neighbors, degrees, labels = fixture(nodes)
    milestones = tuple(sorted(set(
        i for i in (0, 1, 10, 50, 100, 250, 500, iterations)
        if i <= iterations
    )))
    initial_rng = np.random.get_state()
    try:
        # Identical warmups, excluded from paired CPU medians.
        a = run_one(neighbors, degrees, labels, iterations, milestones, False)
        b = run_one(neighbors, degrees, labels, iterations, milestones, True)
        for key in ("prediction_sha256", "full_state_sha256", "rng_sha256"):
            if a[key] != b[key]:
                raise AssertionError(f"observed model differs on warmup: {key}")
        samples = {"baseline": [], "observed": []}
        for rep in range(repeats):
            # Alternating AB/BA order avoids systematic warm/hot ordering.
            sequence = (False, True) if rep % 2 == 0 else (True, False)
            pair = {}
            for observed in sequence:
                result = run_one(neighbors, degrees, labels,
                                 iterations, milestones, observed)
                name = "observed" if observed else "baseline"
                samples[name].append(result["cpu_ns"])
                pair[name] = result
            for key in ("prediction_sha256", "full_state_sha256",
                        "rng_sha256"):
                if pair["baseline"][key] != pair["observed"][key]:
                    raise AssertionError(
                        f"pair {rep} observed model differs: {key}"
                    )
        baseline = statistics.median(samples["baseline"])
        observed = statistics.median(samples["observed"])
        if baseline <= 0:
            raise AssertionError("invalid zero CPU sample")
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        return {
            "schema": 1,
            "scope": "synthetic full fit_predict CPU, not real-data performance",
            "source_commit": sha,
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "nodes": nodes,
            "particles": 6,
            "neighbor_active_degree": 10,
            "neighbor_allocated_width": 16,
            "iterations": iterations,
            "early_stop": False,
            "propagation_seed": 71,
            "milestones": list(milestones),
            "repeats": repeats,
            "samples_cpu_ns": samples,
            "median_baseline_cpu_s": baseline / 1e9,
            "median_observed_cpu_s": observed / 1e9,
            "observed_over_baseline_cpu_ratio": observed / baseline,
            "ratio_minus_one": observed / baseline - 1,
            "snapshot_bytes_per_observed_fit": b["bytes_copied"],
            "no_external_data_or_warmup_samples_in_medians": True,
            "limitations": [
                "One synthetic graph/label fixture and platform-bound CPU timings",
                "Order alternated but OS load and allocator noise remain",
                "Includes entire fit_predict, not graph construction",
                "Only NumPy instrumented; no native Cython/Numba result",
                "Not an estimate of cost for real C2 datasets",
            ],
        }
    finally:
        np.random.set_state(initial_rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = benchmark(args.nodes, args.iterations, args.repeats)
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
