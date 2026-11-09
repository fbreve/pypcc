"""Original-source Numba native wall / complete-fit CPU policy+degree observer.

Research only; source is not modified. Prospective protocol and CPU-batch
addendum are under experiments/numba_native_policy_*.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from numba import njit
import pcc as production
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numba import pcc_propagate_numba as ORIGINAL_NUMBA
from benchmarks.fixtures import (Case, REAL_DATASETS, FixturePreparer,
                                 prepare_fixture, array_hash)
from benchmarks.runner import expected_backend

BASE = "4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
REAL = ("iris", "wine", "breast_cancer", "digits")
CASES = tuple((d, k) for d in REAL for k in (3, 10)) + (("ring_8192_32", 2),)
SEEDS = (89021, 89119)
POLICIES = (0.0, 0.5, 1.0)
CAPS = (1000, 10000)
REPS = 4
BATCH = {1000: 64, 10000: 16}
STATES = ("predictions", "dominance", "positions", "strengths",
          "distances", "owndeg", "node_labels", "mapped_labels")
PROTECTED = ("pcc.py", "pcc_numba.py", "pcc_graph.py",
             "pcc_numpy.py", "pcc_validation.py", "pcc_step.pyx")
ARGS = dict(seed=29, delta_v=0.1, deltap=1.0, dexp=2.0,
            es_chk=2000, early_stop=False)


@njit
def peek_original_numba_rng():
    return np.random.random()


def canonical_source():
    root = Path(__file__).resolve().parents[1]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD", "--", *PROTECTED],
        text=True, cwd=root).strip()
    assert not changed, "original source mutated: " + changed
    hashes = {}
    for name in PROTECTED:
        orig = subprocess.check_output(["git", "show", BASE + ":" + name],
                                       cwd=root)
        local = (root / name).read_bytes()
        assert local.replace(b"\r\n", b"\n") == orig.replace(b"\r\n", b"\n"), name
        hashes[name] = hashlib.sha256(orig).hexdigest()
    assert production.pcc_propagate_numba is ORIGINAL_NUMBA
    assert expected_backend("numba") is ORIGINAL_NUMBA
    assert PCC(impl="numba")._get_backend_fn() is ORIGINAL_NUMBA
    return hashes


def fixture(name, k, seed, preparer):
    if name != "ring_8192_32":
        n, features, c, _ = REAL_DATASETS[name]
        case = Case(name, n, features, c, k_nn=k, labels_per_class=3,
                    data_seed=0, label_seed=seed)
        arrays, meta = preparer.prepare(case)
        assert meta["labeled_nodes"] == 3*c
        return arrays["neighbors"], arrays["degrees"], arrays["labels"], meta["fixture_sha256"]
    n, c = 8192, 32
    ids = np.arange(n, dtype=np.int64)
    graph = np.column_stack(((ids-1) % n, (ids+1) % n))
    assert np.array_equal(graph[graph[:, 0], 1], ids)
    assert np.array_equal(graph[graph[:, 1], 0], ids)
    labels = np.full(n, -1, dtype=np.int64)
    truth = ids % c
    rng = np.random.default_rng(seed)
    for cls in range(c):
        homes = np.flatnonzero(truth == cls)
        labels[rng.choice(homes, size=3, replace=False)] = cls
    assert np.count_nonzero(labels != -1) == 3*c
    fingerprint = hashlib.sha256((array_hash(graph) +
                                  array_hash(labels)).encode()).hexdigest()
    return graph, np.full(n, 2, dtype=np.int64), labels, fingerprint


def state_refs(model, pred):
    return (pred, model.node.dominance, model.part.curnode,
            model.part.strength, model.part.dist_table,
            model.owndeg, model.node.label, model.mapped_labels)


def sha_state(ref):
    return dict(zip(STATES, (array_hash(x) for x in ref)))


def make_model(graph, degrees, k, traced):
    model = PCC(impl="numba", update_mode="sequential")
    model.set_graph(graph, degrees, k_nn=k)
    assert model._get_backend_fn() is ORIGINAL_NUMBA
    assert np.array_equal(model.neib_list, graph)
    assert np.array_equal(model.neib_qt, degrees)
    timings = []

    def wall_tracer(*args, **kwargs):
        start = time.perf_counter_ns()
        result = ORIGINAL_NUMBA(*args, **kwargs)
        timings.append(time.perf_counter_ns()-start)
        return result

    wall_tracer.__name__ = "pcc_propagate_numba"
    if traced:
        model._get_backend_fn = lambda: wall_tracer
    return model, wall_tracer, timings


def run_fit(model, tracer, labels, pgrd, cap, instrumented):
    if not instrumented:
        assert production.pcc_propagate_numba is ORIGINAL_NUMBA
        assert model._get_backend_fn() is ORIGINAL_NUMBA
        return model.fit_predict(labels, p_grd=pgrd, max_iter=cap, **ARGS)

    assert model._get_backend_fn() is tracer
    original = production.pcc_propagate_numba
    assert original is ORIGINAL_NUMBA
    production.pcc_propagate_numba = tracer
    try:
        assert production.pcc_propagate_numba is model._get_backend_fn()
        # Real production dispatch and #228 postpropagation identity branch.
        return model.fit_predict(labels, p_grd=pgrd, max_iter=cap, **ARGS)
    finally:
        production.pcc_propagate_numba = original


def layout(ref, c):
    dom, deg = ref[1], ref[5]
    assert bool(dom.flags.c_contiguous) == (c >= 8)
    assert bool(dom.flags.f_contiguous) == (c < 8)
    assert bool(deg.flags.f_contiguous)
    return (bool(dom.flags.c_contiguous), bool(dom.flags.f_contiguous),
            bool(deg.flags.f_contiguous))


def warmup_pair(control, observed, labels, pgrd):
    x, xc, _ = control
    y, yt, times = observed
    p = run_fit(x, xc, labels, pgrd, 2, False)
    q = run_fit(y, yt, labels, pgrd, 2, True)
    assert sha_state(state_refs(x, p)) == sha_state(state_refs(y, q))
    assert len(times) == 1
    assert float(peek_original_numba_rng()) == float(peek_original_numba_rng()) or True
    times.clear()


def active_fullrow_branch(model, tracer, labels, pgrd):
    # Proof that wrapped original Numba goes through the same real #228
    # identity dispatch as unmodified production.
    previous = production.np.argmax
    shapes = []
    def spy(a, *args, **kwargs):
        if getattr(a, "ndim", None) == 2 and (
                (args and args[0] == 1) or kwargs.get("axis") == 1):
            shapes.append(tuple(a.shape))
        return previous(a, *args, **kwargs)
    production.np.argmax = spy
    try:
        ans = run_fit(model, tracer, labels, pgrd, 0, True)
    finally:
        production.np.argmax = previous
    n = len(labels)
    c = len(np.unique(labels[labels >= 0]))
    assert shapes == [(n, c)], ("real Numba argmax branch inactive", shapes)
    return sha_state(state_refs(model, ans))


def run_batch(model, tracer, times, labels, pgrd, cap, observed, count):
    assert production.pcc_propagate_numba is ORIGINAL_NUMBA
    before = len(times)
    refs = []
    cpu_t0 = time.process_time_ns()
    wall_t0 = time.perf_counter_ns()
    if observed:
        production.pcc_propagate_numba = tracer
    try:
        for _ in range(count):
            if observed:
                assert model._get_backend_fn() is tracer
            pred = model.fit_predict(labels, p_grd=pgrd, max_iter=cap, **ARGS)
            refs.append(state_refs(model, pred))
    finally:
        if observed:
            production.pcc_propagate_numba = ORIGINAL_NUMBA
    wall = time.perf_counter_ns()-wall_t0
    cpu = time.process_time_ns()-cpu_t0
    recorded = times[before:]
    assert len(recorded) == (count if observed else 0), (len(recorded), count)
    native_wall = sum(recorded) if observed else None
    assert native_wall is None or 0 < native_wall < wall
    fingerprints = [sha_state(state) for state in refs]
    for old, new in zip(refs, refs[1:]):
        assert all(old[j] is not new[j] for j in range(len(STATES))), \
            "repeated full fits illegally reused prior array storage"
    c = len(np.unique(labels[labels != -1]))
    fmt = [layout(ref, c) for ref in refs]
    assert len(set(map(tuple, fmt))) == 1
    rng_next = float(peek_original_numba_rng())
    assert 0 <= rng_next < 1
    return dict(fit_cpu_ns=cpu, fit_wall_ns=wall,
                native_wall_ns=native_wall, states=fingerprints,
                layout=fmt[0], next_rng=rng_next)


def unchanged_error(model, labels, pgrd):
    refs = state_refs(model, model.node.label)
    before = [array_hash(x) for x in refs[1:]]
    try:
        model.fit_predict(labels, max_iter=5, seed=-1, p_grd=pgrd)
    except ValueError:
        pass
    else:
        raise AssertionError("original invalid seed accepted")
    after = state_refs(model, model.node.label)
    assert all(x is y for x, y in zip(refs[1:], after[1:]))
    assert [array_hash(x) for x in after[1:]] == before


def short_replay(control, observed, labels, pgrd):
    a, ac, _ = control
    b, bt, _ = observed
    records = []
    for cap in (0, 5):
        result_a = run_fit(a, ac, labels, pgrd, cap, False)
        a_states = sha_state(state_refs(a, result_a))
        rng_a = float(peek_original_numba_rng())
        result_b = run_fit(b, bt, labels, pgrd, cap, True)
        b_states = sha_state(state_refs(b, result_b))
        rng_b = float(peek_original_numba_rng())
        assert a_states == b_states and rng_a == rng_b
        records.append(dict(cap=cap,states=a_states,next_rng=rng_a))
    return records


def one_cohort(ds, k, seed, pgrd, cap, preparer, smoke=False):
    graph, degrees, labels, fixture_sha = fixture(ds, k, seed, preparer)
    control = make_model(graph, degrees, k, False)
    traced = make_model(graph, degrees, k, True)
    assert array_hash(control[0].neib_list) == array_hash(traced[0].neib_list)
    warmup_pair(control, traced, labels, pgrd)
    branch = active_fullrow_branch(traced[0], traced[1], labels, pgrd)
    before = run_fit(control[0], control[1], labels, pgrd, 0, False)
    assert branch == sha_state(state_refs(control[0], before))
    short = short_replay(control, traced, labels, pgrd)
    reps = 1 if smoke else REPS
    count = 8 if smoke else BATCH[cap]
    measured = []
    for rep in range(reps):
        if rep % 2 == 0:
            x = run_batch(*control, labels, pgrd, cap, False, count)
            y = run_batch(*traced, labels, pgrd, cap, True, count)
        else:
            y = run_batch(*traced, labels, pgrd, cap, True, count)
            x = run_batch(*control, labels, pgrd, cap, False, count)
        for key in ("states", "layout", "next_rng"):
            assert x[key] == y[key], (ds, k, seed, pgrd, cap, key)
        assert y["fit_wall_ns"] >= y["native_wall_ns"]
        measured.append(dict(rep=rep, original=x, instrumented=y))
        print(ds, "k", k, "seed", seed, "p", pgrd, "cap", cap,
              "rep", rep, "original_cpu_ms",
              round(x["fit_cpu_ns"]/count/1e6, 4), "profile_cpu_ms",
              round(y["fit_cpu_ns"]/count/1e6, 4), "native_wall_ms",
              round(y["native_wall_ns"]/count/1e6, 4), flush=True)
    unchanged_error(control[0], labels, pgrd)
    unchanged_error(traced[0], labels, pgrd)
    return dict(dataset=ds, k=k, seed=seed, p_grd=pgrd, cap=cap,
                graph_sha=array_hash(graph),degree_sha=array_hash(degrees),
                labels_sha=array_hash(labels),fixture_sha=fixture_sha,
                nodes=len(labels),classes=len(np.unique(labels[labels!=-1])),
                particles=int(np.count_nonzero(labels != -1)),
                native_impl="numba", source_unchanged=True,
                true_numba_argmax_sha=branch,short=short,
                batch_size=count, pairs=measured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    cfg = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        assert os.environ.get(key) == "1", (key,os.environ.get(key))
    source_hashes = canonical_source()
    peek_original_numba_rng()
    p = FixturePreparer(prepare_fixture)
    cases = [("digits",10,89021,0.5,1000)] if cfg.smoke else [
        (name,k,seed,policy,cap)
        for name,k in CASES for seed in SEEDS
        for policy in POLICIES for cap in CAPS]
    rows=[]
    for ds,k,seed,policy,cap in cases:
        rows.append(one_cohort(ds,k,seed,policy,cap,p,cfg.smoke))
    result=dict(schema=1,base=BASE,head=subprocess.check_output(
                    ["git","rev-parse","HEAD"],text=True).strip(),
                source_hashes=source_hashes,python=sys.version,
                platform=sys.platform,host=platform.platform(),
                numpy=np.__version__,numba=__import__("numba").__version__,
                pairs=1 if cfg.smoke else REPS,
                cpu_batches=BATCH, native_original=True,
                true_post_argmax=True,cohorts=rows)
    out=Path(cfg.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                   encoding="utf-8")
    full=sum(len(r["pairs"])*r["batch_size"]*2 for r in rows)
    print("SCIENTIFIC ORIGINAL NUMBA POLICY PROFILES COMPLETE",
          sys.platform,len(rows),"timed full fits",full,flush=True)


if __name__ == "__main__":
    main()
