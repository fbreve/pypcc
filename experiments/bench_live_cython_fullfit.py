"""Paired, synthetic, complete-fit Cython CPU feasibility benchmark.

Production Cython reference versus isolated native observer; two native hotloop
paths are measured separately. No external datasets and no graph-build timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import pcc_step

from pcc import ParticleCompetitionAndCooperation
from experiments.live_cython_milestones import observe_cython_sequential


def fixture(nodes=500, per_class=12):
    if type(nodes) is not int or nodes < 48:
        raise ValueError("nodes must be at least 48")
    if type(per_class) is not int or per_class < 3 or 4 * per_class > nodes:
        raise ValueError("invalid labeled node count")
    neighbors = np.full((nodes, 16), -1, dtype=np.int64)
    offsets = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
    for i in range(nodes):
        for j, offset in enumerate(offsets):
            neighbors[i, j] = (i + offset) % nodes
    degree = np.full(nodes, 10, dtype=np.int64)
    labels = np.full(nodes, -1, dtype=np.int64)
    known = np.arange(per_class, dtype=np.int64) * 2
    labels[known] = 0
    labels[nodes // 2 + known] = 1
    return neighbors, degree, labels


def digest(arrays):
    sha = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        sha.update(value.dtype.str.encode("ascii"))
        sha.update(str(value.shape).encode("ascii"))
        sha.update(value.tobytes())
    return sha.hexdigest()


def full_state_digest(model):
    return digest((
        model.node.dominance, model.node.label, model.owndeg,
    )) if hasattr(model, "node") and hasattr(model.node, "owndeg") else digest((
        model.node.dominance, model.node.label, model.owndeg
    ))


def fingerprint(model):
    return digest((
        model.node.dominance, model.node.label, model.part.homenode,
        model.part.curnode, model.part.label, model.part.strength,
        model.part.dist_table, model.owndeg,
    ))


def prepare(neighbors, degrees, labels, mode, milestones, observed):
    model = ParticleCompetitionAndCooperation(impl="cython")
    model.set_graph(neighbors, degrees, k_nn=10)
    captured = []
    early_stop = mode != "no-stop"
    use_cache = mode == "early-cached"

    def selected_backend(*args, **kwargs):
        # The model's production optimization for canonical initialization
        # only recognizes the original backend FUNCTION identity. Because
        # this benchmark injects a function on BOTH arms, explicitly pass
        # the exact canonical setting selected by the default model.
        kwargs["experimental_rowmax_cache"] = use_cache
        kwargs["experimental_canonical_rowmax_init"] = use_cache and early_stop
        if observed:
            report = observe_cython_sequential(
                *args, **kwargs, milestones=milestones,
            )
            captured.append(report)
            return None
        return pcc_step.pcc_propagate(*args, **kwargs)

    model._get_backend_fn = lambda: selected_backend
    return (model, labels.copy(), captured)


def batch(neighbors, degrees, labels, mode, milestones,
          observed, fits_per_sample, iterations):
    prepared = [
        prepare(neighbors, degrees, labels, mode, milestones, observed)
        for _ in range(fits_per_sample)
    ]
    start = time.process_time_ns()
    predictions = []
    for model, known, _ in prepared:
        predictions.append(model.fit_predict(
            known, max_iter=iterations, early_stop=mode != "no-stop", seed=71,
        ))
    cpu_ns = time.process_time_ns() - start

    fingerprints = []
    snapshot_bytes = 0
    counts = []
    stop_info = []
    for (model, _, captured), labels_out in zip(prepared, predictions):
        fingerprints.append((
            hashlib.sha256(np.asarray(labels_out, dtype="<i8").tobytes()).hexdigest(),
            fingerprint(model),
        ))
        if observed:
            if len(captured) != 1 or captured[0].observed_iterations[0] != 0:
                raise AssertionError("missing Cython live trace")
            snapshot_bytes += captured[0].bytes_copied
            counts.append(len(captured[0].milestones))
            stop_info.append(dict(captured[0].info))
    if len(set(fingerprints)) != 1:
        raise AssertionError("deterministic fit replay differed inside batch")
    if observed:
        if len(set(counts)) != 1 or any(x != stop_info[0] for x in stop_info):
            raise AssertionError("identically seeded observed iterations differed")

    return {
        "cpu_ns": cpu_ns,
        "fits": fits_per_sample,
        "prediction_sha256": fingerprints[0][0],
        "state_sha256": fingerprints[0][1],
        "snapshots_bytes_all_fits": snapshot_bytes,
        "observed_milestone_count": counts[0] if observed else 0,
        "effective_iterations": stop_info[0]["effective_iterations"] if observed else None,
    }


def benchmark(mode, nodes=500, iterations=5000, per_class=12,
              repeats=5, fits_per_sample=400):
    if mode not in ("no-stop", "early-cached", "early-legacy"):
        raise ValueError("unknown Cython kernel mode")
    if type(iterations) is not int or not 100 <= iterations <= 10000:
        raise ValueError("invalid iteration count")
    if type(repeats) is not int or not 2 <= repeats <= 8:
        raise ValueError("invalid paired repetition count")
    if type(fits_per_sample) is not int or not 10 <= fits_per_sample <= 1000:
        raise ValueError("invalid fits_per_sample")
    neighbors, degrees, labels = fixture(nodes, per_class)
    milestones = tuple(sorted(set(
        value for value in (0, 1, 10, 50, 100, 250, 500, iterations)
        if value <= iterations
    )))
    baseline_warm = batch(neighbors, degrees, labels, mode,
                          milestones, False, 1, iterations)
    observed_warm = batch(neighbors, degrees, labels, mode,
                          milestones, True, 1, iterations)
    keys = ("prediction_sha256", "state_sha256")
    for key in keys:
        if baseline_warm[key] != observed_warm[key]:
            raise AssertionError(f"warmup parity failure: {key}")

    pairs = []
    for rep in range(repeats):
        order = (False, True) if rep % 2 == 0 else (True, False)
        runs = {}
        for observed in order:
            name = "observed" if observed else "baseline"
            runs[name] = batch(
                neighbors, degrees, labels, mode, milestones,
                observed, fits_per_sample, iterations,
            )
        for key in keys:
            if runs["baseline"][key] != runs["observed"][key]:
                raise AssertionError(f"paired fit state divergence {rep}: {key}")
        pairs.append({
            "pair": rep,
            "order": ["observed" if x else "baseline" for x in order],
            **runs,
        })
    a = [pair["baseline"]["cpu_ns"] for pair in pairs]
    b = [pair["observed"]["cpu_ns"] for pair in pairs]
    if min(a + b) <= 0:
        raise AssertionError("CPU samples under clock resolution")
    med_a = statistics.median(a)
    med_b = statistics.median(b)
    ratios = [obs / base for base, obs in zip(a, b)]
    return {
        "schema": 1,
        "scope": "single synthetic complete fit_predict Cython benchmark",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mode": mode,
        "nodes": nodes,
        "particles": 2 * per_class,
        "iterations_limit": iterations,
        "active_degree": 10,
        "dense_width": 16,
        "milestones": list(milestones),
        "seed": 71,
        "fits_per_sample": fits_per_sample,
        "pairs": pairs,
        "median_baseline_cpu_s_per_fit": med_a / (fits_per_sample * 1e9),
        "median_observed_cpu_s_per_fit": med_b / (fits_per_sample * 1e9),
        "overhead_percent_ratio_of_medians": 100.0 * (med_b / med_a - 1),
        "median_paired_overhead_percent": 100.0 * (statistics.median(ratios) - 1),
        "paired_ratio_each": ratios,
        "snapshot_bytes_per_fit": observed_warm["snapshots_bytes_all_fits"],
        "observed_milestone_count": observed_warm["observed_milestone_count"],
        "observed_effective_iterations": observed_warm["effective_iterations"],
        "compilation_and_graph_excluded": True,
        "limitations": [
            "Synthetic source only, not a C2 real-data performance estimate",
            "CPU process-time may be quantized or affected by runner noise",
            "No disk export timing; snapshots retained in memory",
            "Full fit_predict measured, graph set before CPU timing",
            "Compare within each cached/legacy Cython mode, not across algorithms",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("no-stop", "early-cached", "early-legacy"), required=True)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fits-per-sample", type=int, default=400)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = benchmark(
        args.mode, args.nodes, args.iterations,
        args.per_class, args.repeats, args.fits_per_sample,
    )
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
