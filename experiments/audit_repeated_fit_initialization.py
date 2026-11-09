"""Original-source, same-model repeated PCC fit initialization attribution.

Scientific observation only. See repeated_fit_initialization_protocol.md.
No production module is replaced and no candidate optimization is introduced.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from numba import njit
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from benchmarks.fixtures import array_hash
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph
from benchmarks.runner import expected_backend

BASE = "b4c100bd22eeff1979a1a581bc73682b8c3705b8"
PROTECTED = ("pcc.py", "pcc_graph.py", "pcc_numba.py", "pcc_numpy.py",
             "pcc_step.pyx", "pcc_validation.py", "setup.py")
MASK_SEEDS = (82013, 82117)
BACKENDS = ("numpy", "numba", "cython")
N_BATCHES, FITS_PER_BATCH = 4, 8
ARRAY_NAMES = ("predictions", "dominance", "positions", "strengths",
               "distances", "owndeg")
PARAMS = dict(p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
              es_chk=2000, early_stop=False, seed=29)


@njit
def _next_numba():
    return np.random.random()


class TracedPCC(PCC):
    """Time outer Python boundaries only; call actual parent methods/backend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase_ns = {"particles": [], "nodes": [], "labelprop": [], "native": []}

    def _ParticleCompetitionAndCooperation__genParticles(self):
        begin = time.perf_counter_ns()
        result = super()._ParticleCompetitionAndCooperation__genParticles()
        self.phase_ns["particles"].append(time.perf_counter_ns() - begin)
        return result

    def _ParticleCompetitionAndCooperation__genNodes(self):
        begin = time.perf_counter_ns()
        result = super()._ParticleCompetitionAndCooperation__genNodes()
        self.phase_ns["nodes"].append(time.perf_counter_ns() - begin)
        return result

    def _ParticleCompetitionAndCooperation__labelPropagation(self, seed=None):
        begin = time.perf_counter_ns()
        result = super()._ParticleCompetitionAndCooperation__labelPropagation(seed=seed)
        self.phase_ns["labelprop"].append(time.perf_counter_ns() - begin)
        return result

    def _get_backend_fn(self):
        original = super()._get_backend_fn()

        @functools.wraps(original)
        def observed(*args, **kwargs):
            begin = time.perf_counter_ns()
            result = original(*args, **kwargs)
            self.phase_ns["native"].append(time.perf_counter_ns() - begin)
            return result

        # functools.wraps preserves __name__, used by real __genNodes for Numba C layout.
        return observed


def check_sources():
    for path in PROTECTED:
        original = subprocess.check_output(
            ["git", "show", f"{BASE}:{path}"], stderr=subprocess.DEVNULL)
        assert (Path(__file__).resolve().parents[1] / path).read_bytes() == original, path
    changes = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD", "--", *PROTECTED],
        text=True).strip()
    assert not changes, f"unexpected production changes: {changes}"


def labels_for(truth, seed):
    selected = np.full(len(truth), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for value in np.unique(truth):
        indices = np.flatnonzero(truth == value)
        assert len(indices) > 3
        chosen = rng.choice(indices, 3, replace=False)
        selected[chosen] = value
    return selected


def fixture(name):
    if name in ("wine", "breast_cancer", "digits"):
        data = getattr(datasets, "load_" + name)()
        truth = np.asarray(data.target, dtype=np.int64)
        features = StandardScaler().fit_transform(data.data)
        neighbors, degrees = build_knn_graph(features, 10, n_jobs=1)
        k_nn = 10
    else:
        _, n, c = name.split("_")
        n, c = int(n), int(c)
        truth = np.arange(n, dtype=np.int64) % c
        indices = np.arange(n, dtype=np.int64)
        neighbors = np.column_stack(((indices - 1) % n, (indices + 1) % n))
        degrees = np.full(n, 2, dtype=np.int64)
        k_nn = 2
        assert np.array_equal(neighbors[neighbors[:, 0], 1], indices)
        assert np.array_equal(neighbors[neighbors[:, 1], 0], indices)
    masks = [labels_for(truth, seed) for seed in MASK_SEEDS]
    assert all(np.unique(m[m != -1]).size == np.unique(truth).size for m in masks)
    return neighbors, degrees, masks, k_nn


def fit_state(model, prediction):
    # Capture only REFERENCES during timed batches. Hash/copy outside timing.
    return (prediction, model.node.dominance, model.part.curnode,
            model.part.strength, model.part.dist_table, model.owndeg)


def fingerprints(states):
    return [{name: array_hash(value) for name, value in zip(ARRAY_NAMES, state)}
            for state in states]


def last_dimensions(model):
    return dict(dist_table_bytes=int(model.part.dist_table.nbytes),
                dominance_bytes=int(model.node.dominance.nbytes),
                owndeg_bytes=int(model.owndeg.nbytes),
                scratch_bytes=sum(int(getattr(model, name).nbytes)
                    for name in ("zerovec", "dist_weights", "buf_dom_row", "buf_reduc",
                                 "buf_dom_list", "buf_dist_list", "buf_prob",
                                 "buf_slices")),
                dominance_c_contiguous=bool(model.node.dominance.flags.c_contiguous),
                dominance_f_contiguous=bool(model.node.dominance.flags.f_contiguous))


def make_model(cls, backend, graph, degrees, k_nn):
    model = cls(impl=backend)
    model.set_graph(graph, degrees, k_nn=k_nn)
    actual = model._get_backend_fn()
    if isinstance(model, TracedPCC):
        actual = actual.__wrapped__
    assert actual is expected_backend(backend), (backend, actual)
    return model


def run_batch(model, masks, iterations):
    retained = []
    begin_cpu = time.process_time_ns()
    begin_wall = time.perf_counter_ns()
    for j in range(FITS_PER_BATCH):
        labels = masks[j % 2]
        prediction = model.fit_predict(labels, max_iter=iterations, **PARAMS)
        retained.append(fit_state(model, prediction))
    wall_ns = time.perf_counter_ns() - begin_wall
    cpu_ns = time.process_time_ns() - begin_cpu
    # Time is stopped BEFORE SHA, state inspection and RNG replay.
    hashes = fingerprints(retained)
    for a, b in zip(retained[:-1], retained[1:]):
        for pos in (1, 4, 5):
            assert a[pos] is not b[pos], "previous-fit state mutated/aliased"
    dim = last_dimensions(model)
    return dict(cpu_ns=cpu_ns, wall_ns=wall_ns, hashes=hashes, dimensions=dim)


def next_rng(backend):
    if backend == "numba":
        return float(_next_numba())
    if backend == "numpy":
        return float(np.random.random())
    return None


def parity_replay(control, traced, masks, iterations, expected_hashes):
    """Outside the timer, repeat original fits for true native RNG endpoints."""
    rngs = []
    for i, labels in enumerate(masks):
        a = control.fit_predict(labels, max_iter=iterations, **PARAMS)
        ha = fingerprints([fit_state(control, a)])[0]
        ra = next_rng(control.impl)
        b = traced.fit_predict(labels, max_iter=iterations, **PARAMS)
        hb = fingerprints([fit_state(traced, b)])[0]
        rb = next_rng(traced.impl)
        assert ha == hb == expected_hashes[i], "native state or replay mismatch"
        assert ra == rb, "post-fit RNG mismatch"
        rngs.append(ra)
    # Invalid-seed preflight must preserve all fitted buffers and identities.
    for model in (control, traced):
        old = fit_state(model, np.asarray(masks[-1]))
        old_hash = fingerprints([old])[0]
        try:
            model.fit_predict(masks[-1], max_iter=iterations, seed=-1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed accepted")
        new = fit_state(model, old[0])
        assert all(a is b for a, b in zip(old[1:], new[1:]))
        assert old_hash == fingerprints([new])[0]
    return rngs


def one_cohort(name, backend, iterations, smoke=False):
    graph, degrees, masks, k_nn = fixture(name)
    graph_hash, degree_hash = array_hash(graph), array_hash(degrees)
    control = make_model(PCC, backend, graph, degrees, k_nn)
    traced = make_model(TracedPCC, backend, graph, degrees, k_nn)
    # True original compilation warmup, excluded from measured CPU.
    if backend == "numba":
        for model in (control, traced):
            model.fit_predict(masks[0], max_iter=2, **PARAMS)
        _next_numba()  # compile RNG observer outside timed batches
    n_batches = 1 if smoke else N_BATCHES
    paired = []
    for rep in range(n_batches):
        # Each batch consists of eight back-to-back fits on the same model.
        # Alternate arm order, and do not hash within the CPU/wall timer.
        if rep % 2 == 0:
            a = run_batch(control, masks, iterations)
            begin_phases = {k: len(v) for k, v in traced.phase_ns.items()}
            b = run_batch(traced, masks, iterations)
        else:
            begin_phases = {k: len(v) for k, v in traced.phase_ns.items()}
            b = run_batch(traced, masks, iterations)
            a = run_batch(control, masks, iterations)
        assert a["hashes"] == b["hashes"], "paired states differ"
        assert a["dimensions"] == b["dimensions"], "source layout/bytes differ"
        # Verify same graph remains installed, with no rebuild or set_graph call.
        assert array_hash(control.neib_list) == array_hash(traced.neib_list)
        assert array_hash(control.neib_qt) == array_hash(traced.neib_qt)
        observed = {k: v[begin_phases[k]:begin_phases[k] + FITS_PER_BATCH]
                    for k, v in traced.phase_ns.items()}
        assert all(len(v) == FITS_PER_BATCH for v in observed.values()), observed
        for native, labelprop in zip(observed["native"], observed["labelprop"]):
            assert 0 <= native <= labelprop
        # First and second mask are independent; replay after the timed batches.
        replay_rng = parity_replay(control, traced, masks, iterations, a["hashes"])
        paired.append(dict(repetition=rep, control_cpu_ns=a["cpu_ns"],
                           traced_cpu_ns=b["cpu_ns"], control_wall_ns=a["wall_ns"],
                           traced_wall_ns=b["wall_ns"], traced_phases_ns=observed,
                           state_hashes=a["hashes"], replay_next_rng=replay_rng))
        print(f"{name} {backend} steps={iterations} pair={rep+1}/{n_batches} "
              f"control_cpu_ms={a['cpu_ns']/1e6:.3f} traced_cpu_ms={b['cpu_ns']/1e6:.3f}",
              flush=True)
    return dict(case=name, backend=backend, steps=iterations,
                graph_sha=graph_hash, degree_sha=degree_hash,
                mask_shas=[array_hash(x) for x in masks],
                dimensions=a["dimensions"], pairs=paired)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_sources()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        assert os.environ.get(key) == "1", f"must pin {key}=1"
    cases = ["wine"] if args.smoke else [
        "wine", "breast_cancer", "digits",
        "ring_30000_2", "ring_30000_10", "ring_30000_50",
        "ring_100000_2", "ring_100000_10", "ring_100000_50"]
    cohorts = []
    for name in cases:
        for backend in (("numba",) if args.smoke else BACKENDS):
            steps = (0,) if args.smoke else ((0, 5, 500) if
                       name in ("wine", "breast_cancer", "digits") and
                       backend in ("numba", "cython") else (0, 5))
            for count in steps:
                cohorts.append(one_cohort(name, backend, count, smoke=args.smoke))
    payload = dict(schema=1, base=BASE, os=sys.platform,
                   platform=platform.platform(), python=sys.version,
                   scipy_and_sklearn="production native graph on real cases",
                   batches=1 if args.smoke else N_BATCHES,
                   fits_per_batch=FITS_PER_BATCH,
                   cases=cases, cohorts=cohorts, protected_sources_exact=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(f"PROFILE COMPLETE {sys.platform}: {len(cohorts)} cohorts, "
          f"{len(cohorts)*payload['batches']*FITS_PER_BATCH*2} timed native fits",
          flush=True)


if __name__ == "__main__":
    main()
