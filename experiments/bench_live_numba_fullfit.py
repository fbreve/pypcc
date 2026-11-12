"""Source-pinned synthetic complete-fit CPU A/B for compiled Numba snapshots.

Research-only. No real datasets, no background jobs, no speedup assertion.
The comparison always verifies final model/prediction/native RNG first.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

from numba import _helperlib
import numba
import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.live_numba_milestones import observe_numba_sequential


def fixture(nodes=500, labels_per_class=12):
    if (type(nodes) is not int or nodes < 48
            or type(labels_per_class) is not int
            or labels_per_class < 3 or 4 * labels_per_class > nodes):
        raise ValueError("bad synthetic fixture shape")
    edges = np.full((nodes, 16), -1, dtype=np.int64)
    offsets = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
    for i in range(nodes):
        for j, delta in enumerate(offsets):
            edges[i, j] = (i + delta) % nodes
    degree = np.full(nodes, 10, dtype=np.int64)
    labels = np.full(nodes, -1, dtype=np.int64)
    homes = np.arange(labels_per_class) * 2
    labels[homes] = 0
    labels[nodes // 2 + homes] = 1
    return edges, degree, labels


def native_rng_digest():
    state = _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())
    h = hashlib.sha256()
    h.update(int(state[0]).to_bytes(8, "little"))
    h.update(np.asarray(state[1], dtype="<u4").tobytes())
    return h.hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for value in (model.node.dominance, model.node.label,
                  model.part.curnode, model.part.label,
                  model.part.strength, model.part.dist_table, model.owndeg):
        a = np.ascontiguousarray(value)
        digest.update(str(a.shape).encode("ascii"))
        digest.update(a.dtype.str.encode("ascii"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def run_one(edges, degrees, labels, iterations, milestones, observed):
    model = ParticleCompetitionAndCooperation(impl="numba")
    model.set_graph(edges, degrees, k_nn=10)
    captures = []
    if observed:
        def instrumented(*args, **kwargs):
            result = observe_numba_sequential(
                *args, **kwargs, milestones=milestones
            )
            captures.append(result)
            return None
        model._get_backend_fn = lambda: instrumented

    start_cpu = time.process_time_ns()
    start_wall = time.perf_counter_ns()
    predictions = model.fit_predict(
        labels, max_iter=iterations, early_stop=False, seed=71
    )
    wall_ns = time.perf_counter_ns() - start_wall
    cpu_ns = time.process_time_ns() - start_cpu

    if observed and (
        len(captures) != 1
        or captures[0].observed_iterations != milestones
    ):
        raise AssertionError("requested native milestones were not captured")
    return {
        "cpu_ns": cpu_ns, "wall_ns": wall_ns,
        "predictions_sha": hashlib.sha256(
            np.asarray(predictions, dtype="<i8").tobytes()
        ).hexdigest(),
        "full_state_sha": model_digest(model),
        "native_rng_sha": native_rng_digest(),
        "bytes_copied": captures[0].bytes_copied if observed else 0,
        "bytes_reserved": captures[0].reserved_bytes if observed else 0,
    }


def benchmark(nodes=500, iterations=1200000, repeats=5,
              labels_per_class=12):
    if type(iterations) is not int or not 5000 <= iterations <= 2000000:
        raise ValueError("iterations must be integer in 5000..2000000")
    if type(repeats) is not int or not 3 <= repeats <= 8:
        raise ValueError("repeats must be integer in 3..8")
    graph, degree, labels = fixture(nodes, labels_per_class)
    milestones = (0, 1, 10, 50, 100, 250, 500, 1000, 10000, iterations)
    native_ptr = _helperlib.rnd_get_np_state_ptr()
    saved_native = _helperlib.rnd_get_state(native_ptr)
    saved_python = np.random.get_state()
    try:
        # Separate JIT compilation/physical-memory warmup from all samples.
        plain = run_one(graph, degree, labels, iterations,
                        milestones, False)
        traced = run_one(graph, degree, labels, iterations,
                         milestones, True)
        for key in ("predictions_sha", "full_state_sha", "native_rng_sha"):
            if plain[key] != traced[key]:
                raise AssertionError("JIT warmup parity mismatch: " + key)

        samples = {"normal_cpu_ns": [], "observed_cpu_ns": [],
                   "normal_wall_ns": [], "observed_wall_ns": []}
        paired_ratios = []
        for repetition in range(repeats):
            order = (False, True) if repetition % 2 == 0 else (True, False)
            pair = {}
            for observation in order:
                result = run_one(graph, degree, labels, iterations,
                                 milestones, observation)
                label = "observed" if observation else "normal"
                samples[label + "_cpu_ns"].append(result["cpu_ns"])
                samples[label + "_wall_ns"].append(result["wall_ns"])
                pair[label] = result
            for key in ("predictions_sha", "full_state_sha",
                        "native_rng_sha"):
                if pair["normal"][key] != pair["observed"][key]:
                    raise AssertionError(
                        f"pair {repetition}: compiled observation changed {key}"
                    )
            if pair["normal"]["cpu_ns"] <= 0:
                raise AssertionError("zero CPU measurement")
            paired_ratios.append(pair["observed"]["cpu_ns"]
                                 / pair["normal"]["cpu_ns"])

        baseline = statistics.median(samples["normal_cpu_ns"])
        monitored = statistics.median(samples["observed_cpu_ns"])
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        return {
            "schema": 1,
            "scope": "paired synthetic full fit_predict compiled Numba CPU",
            "source_commit": sha,
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__, "numba": numba.__version__,
            "nodes": nodes, "particle_count": labels_per_class * 2,
            "dense_neighbor_width": 16, "active_neighbor_degree": 10,
            "iterations": iterations, "early_stop": False,
            "seed": 71, "milestones": list(milestones),
            "repeats": repeats, "samples_ns": samples,
            "paired_cpu_ratios": paired_ratios,
            "median_baseline_cpu_s": baseline / 1e9,
            "median_observed_cpu_s": monitored / 1e9,
            "ratio_of_cpu_medians": monitored / baseline,
            "median_paired_cpu_ratio": statistics.median(paired_ratios),
            "observed_reserved_bytes": traced["bytes_reserved"],
            "observed_copied_bytes": traced["bytes_copied"],
            "both_jit_warmups_excluded": True,
            "all_prediction_state_native_rng_pairs_exact": True,
            "limitations": [
                "Synthetic undirected ring only; not real C2 datasets",
                "Complete fit_predict CPU excludes graph construction",
                "Python Windows process-time can be quantized",
                "Five pairs do not rule out small overhead or host noise",
                "Numba only; Cython/NumPy RNG streams are not comparable",
            ],
        }
    finally:
        _helperlib.rnd_set_state(native_ptr, saved_native)
        np.random.set_state(saved_python)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=1200000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--labels-per-class", type=int, default=12)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    data = benchmark(args.nodes, args.iterations, args.repeats,
                     args.labels_per_class)
    encoded = json.dumps(data, sort_keys=True, indent=2) + "\n"
    print(encoded)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
