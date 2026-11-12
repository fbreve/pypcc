"""Synthetic paired complete-fit CPU benchmark for compiled Numba observation.

Research only: 500-node undirected padded graph, fixed labels and seed.
CPU timing excludes graph setup, hashes, and first-time JIT compilation.
The timing unit is a batch of COMPLETE fit_predict calls to reduce Windows
process-time clock quantization. Each call still includes its fit initialization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
from numba import njit

from pcc import ParticleCompetitionAndCooperation
from experiments.live_numba_milestones import observe_numba_sequential, _observe_compiled


@njit(cache=True)
def _next_native_draw():
    return np.random.random()


def fixture(nodes=500, labels_per_class=12):
    if type(nodes) is not int or nodes < 48:
        raise ValueError("nodes must be integer >= 48")
    if (type(labels_per_class) is not int or labels_per_class < 3
            or 4 * labels_per_class > nodes):
        raise ValueError("invalid labels_per_class")
    neighbors = np.full((nodes, 16), -1, dtype=np.int64)
    offsets = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
    for node in range(nodes):
        for j, offset in enumerate(offsets):
            neighbors[node, j] = (node + offset) % nodes
    degrees = np.full(nodes, len(offsets), dtype=np.int64)
    labels = np.full(nodes, -1, dtype=np.int64)
    homes = np.arange(labels_per_class, dtype=np.int64) * 2
    labels[homes] = 0
    labels[nodes // 2 + homes] = 1
    return neighbors, degrees, labels


def fingerprint(model):
    h = hashlib.sha256()
    for array in (
        model.node.dominance, model.node.label,
        model.part.homenode, model.part.curnode,
        model.part.label, model.part.strength,
        model.part.dist_table, model.owndeg,
    ):
        data = np.ascontiguousarray(array)
        h.update(str(data.shape).encode("ascii"))
        h.update(data.dtype.str.encode("ascii"))
        h.update(data.tobytes())
    return h.hexdigest()


def prepare(neighbors, degrees, labels, observed, milestones):
    model = ParticleCompetitionAndCooperation(impl="numba")
    model.set_graph(neighbors, degrees, k_nn=10)
    captures = []
    if observed:
        def dispatch(*args, **kwargs):
            report = observe_numba_sequential(
                *args, **kwargs, milestones=milestones,
            )
            captures.append(report)
            return None
        # Deliberately instance-only research dispatch; no production edits.
        model._get_backend_fn = lambda: dispatch
    return model, labels.copy(), captures


def run_batch(neighbors, degrees, labels, iterations, milestones,
              observed, fits_per_sample):
    prepared = [
        prepare(neighbors, degrees, labels, observed, milestones)
        for _ in range(fits_per_sample)
    ]
    # Each observation is measured across the entire full-fit batch. Avoid
    # individual sub-tick process CPU timestamps on Windows.
    cpu_start = time.process_time_ns()
    predictions = []
    for model, current_labels, _ in prepared:
        predictions.append(model.fit_predict(
            current_labels, max_iter=iterations, early_stop=False, seed=71,
        ))
    cpu_ns = time.process_time_ns() - cpu_start
    native_next = _next_native_draw()

    digests = []
    total_bytes = 0
    for (model, _, captures), prediction in zip(prepared, predictions):
        prediction_hash = hashlib.sha256(
            np.asarray(prediction, dtype="<i8").tobytes()
        ).hexdigest()
        state_hash = fingerprint(model)
        if observed:
            if len(captures) != 1 or captures[0].observed_iterations != milestones:
                raise AssertionError("incomplete live Numba observation batch")
            total_bytes += captures[0].bytes_copied
        digests.append((prediction_hash, state_hash))
    if len(set(digests)) != 1:
        raise AssertionError("identically seeded fits diverged within batch")
    return {
        "cpu_ns": cpu_ns,
        "prediction_sha256": digests[0][0],
        "full_state_sha256": digests[0][1],
        "next_native_rng_bits": struct.pack("<d", native_next).hex(),
        "snapshot_bytes_total": total_bytes,
        "fits": fits_per_sample,
    }


def benchmark(nodes=500, iterations=5000, repeats=5,
              fits_per_sample=100, labels_per_class=12):
    if type(iterations) is not int or not 100 <= iterations <= 50000:
        raise ValueError("iterations must be int in 100..50000")
    if type(repeats) is not int or not 2 <= repeats <= 8:
        raise ValueError("repeats must be int in 2..8")
    if type(fits_per_sample) is not int or not 1 <= fits_per_sample <= 200:
        raise ValueError("fits_per_sample must be int in 1..200")
    neighbors, degrees, labels = fixture(nodes, labels_per_class)
    milestones = tuple(sorted(set(
        x for x in (0, 1, 10, 50, 100, 250, 500, iterations)
        if x <= iterations
    )))
    # Warm both code paths and the native RNG draw OUTSIDE timing.
    a = run_batch(neighbors, degrees, labels, iterations,
                  milestones, False, 1)
    b = run_batch(neighbors, degrees, labels, iterations,
                  milestones, True, 1)
    keys = ("prediction_sha256", "full_state_sha256", "next_native_rng_bits")
    if any(a[k] != b[k] for k in keys):
        raise AssertionError("observed and baseline warmups diverged")
    if not _observe_compiled.nopython_signatures:
        raise AssertionError("Numba observation did not compile in nopython mode")

    pairs = []
    for repeat in range(repeats):
        # Alternate AB/BA to control temporal order within the same OS job.
        order = (False, True) if repeat % 2 == 0 else (True, False)
        pair = {}
        for observed in order:
            name = "observed" if observed else "baseline"
            pair[name] = run_batch(
                neighbors, degrees, labels, iterations,
                milestones, observed, fits_per_sample,
            )
        for k in keys:
            if pair["baseline"][k] != pair["observed"][k]:
                raise AssertionError(f"pair {repeat} diverged: {k}")
        pairs.append({"pair": repeat, "order": [
            "observed" if x else "baseline" for x in order
        ], **pair})

    baseline_times = [x["baseline"]["cpu_ns"] for x in pairs]
    observed_times = [x["observed"]["cpu_ns"] for x in pairs]
    if min(baseline_times) <= 0 or min(observed_times) <= 0:
        raise AssertionError("clock did not resolve CPU samples")
    med_a = statistics.median(baseline_times)
    med_b = statistics.median(observed_times)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    return {
        "schema": 1,
        "scope": "synthetic full fit_predict batch CPU; no real C2 data",
        "source_commit": commit,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "nodes": nodes,
        "particles": 2 * labels_per_class,
        "active_degree": 10,
        "dense_neighbor_width": 16,
        "iterations_per_fit": iterations,
        "early_stop": False,
        "seed": 71,
        "milestones": list(milestones),
        "fits_per_batch": fits_per_sample,
        "paired_repeats": repeats,
        "pairs": pairs,
        "median_baseline_cpu_s_per_fit": med_a / (1e9 * fits_per_sample),
        "median_observed_cpu_s_per_fit": med_b / (1e9 * fits_per_sample),
        "observed_over_baseline_ratio": med_b / med_a,
        "ratio_minus_one": med_b / med_a - 1.0,
        "snapshot_bytes_per_fit": b["snapshot_bytes_total"],
        "warmups_excluded": True,
        "nopython_observer_verified": True,
        "limitations": [
            "Single synthetic graph/label configuration",
            "Batch CPU includes complete fit_predict but not graph creation",
            "Batch warmups excluded; residual scheduler/thermal noise remains",
            "Snapshot data held in memory: no disk serialization measured",
            "No real data or Cython observer used",
            "Observed vs baseline provenance is within Numba only",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fits-per-sample", type=int, default=100)
    parser.add_argument("--labels-per-class", type=int, default=12)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = benchmark(
        args.nodes, args.iterations, args.repeats,
        args.fits_per_sample, args.labels_per_class,
    )
    result = json.dumps(report, sort_keys=True, indent=2) + "\n"
    print(result)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
