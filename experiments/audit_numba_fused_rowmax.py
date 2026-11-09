"""Pinned-original Numba fused dominance row-max scientific A/B, not production."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
import types
import warnings

import numpy as np
from numba import njit
from benchmarks.fixtures import (Case, REAL_DATASETS, FixturePreparer,
                                 prepare_fixture, array_hash)
import pcc as production
from pcc import ParticleCompetitionAndCooperation as PCC

BASE = "4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
DATASETS = ("iris", "wine", "breast_cancer", "digits", "ring_8192_32")
SEEDS = (87523, 87697)
MODES = (True, False)
REPS = 5
BATCH = 8
CAP = 10000
STATE = ("predictions", "dominance", "positions", "strengths",
         "distances", "owndeg", "node_labels", "mapped_labels")
PROTECTED = ("pcc.py", "pcc_numba.py", "pcc_numpy.py", "pcc_graph.py",
             "pcc_step.pyx", "pcc_validation.py", "setup.py",
             "benchmarks/runner.py", "benchmarks/fixtures.py")
ARGS = dict(seed=0, p_grd=.5, delta_v=.1, deltap=1., dexp=2., es_chk=2000)


@njit
def peek_rng():
    return np.random.random()


def uniq(src, old, new):
    assert src.count(old) == 1, (old[:60], src.count(old))
    return src.replace(old, new, 1)


def source_pair():
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD", "--", *PROTECTED],
        text=True).strip()
    assert not changed, "protected real production changed: " + changed
    for name in PROTECTED:
        raw = subprocess.check_output(["git", "show", BASE + ":" + name])
        disk = Path(name).read_bytes()
        assert disk.replace(b"\r\n", b"\n") == raw.replace(b"\r\n", b"\n"), name
    original_raw = subprocess.check_output(["git", "show", BASE + ":pcc_numba.py"])
    original = original_raw.decode("utf-8")
    start = "@njit(cache=True)\ndef _pcc_step_numba_sequential("
    next_start = "@njit(cache=True)\ndef _pcc_calc_mmpot_sequential("
    assert original.count(start) == original.count(next_start) == 1
    step = original[original.index(start):original.index(next_start)]
    new = step

    # All original arithmetic and statement order are preserved. Only an
    # additional maximum comparison is inserted inside the existing loop.
    new = uniq(new,
        "                sum_reduc = 0.0\n                for i in range(c):\n",
        "                sum_reduc = 0.0\n                reduced_max = 0.0\n"
        "                for i in range(c):\n")
    new = uniq(new,
        "                    dominance[next_node, i] = current - reduc_val\n"
        "                    sum_reduc += reduc_val\n",
        "                    dominance[next_node, i] = current - reduc_val\n"
        "                    sum_reduc += reduc_val\n"
        "                    reduced = dominance[next_node, i]\n"
        "                    if i == 0 or reduced > reduced_max:\n"
        "                        reduced_max = reduced\n")
    new = uniq(new,
        "                dominance[next_node, label] += sum_reduc\n",
        "                dominance[next_node, label] += sum_reduc\n"
        "                max_dom = reduced_max\n"
        "                if dominance[next_node, label] > max_dom:\n"
        "                    max_dom = dominance[next_node, label]\n")
    original_max = (
        "            max_dom = dominance[next_node, 0]\n"
        "            for i in range(1, c):\n"
        "                if dominance[next_node, i] > max_dom:\n"
        "                    max_dom = dominance[next_node, i]\n"
    )
    labeled_only_max = (
        "            if labels[next_node] != -1:\n"
        "                max_dom = dominance[next_node, 0]\n"
        "                for i in range(1, c):\n"
        "                    if dominance[next_node, i] > max_dom:\n"
        "                        max_dom = dominance[next_node, i]\n"
    )
    new = uniq(new, original_max, labeled_only_max)
    candidate = uniq(original, step, new)
    assert candidate != original
    # Ensure the ONLY full-file edit is one JIT step and no altered wrapper
    assert candidate.replace(new, step, 1) == original
    # Production #216 explicitly uses @njit(cache=True). Numba requires a
    # REAL on-disk Python source path as its disk-cache locator; a synthetic
    # Git-blob compile filename is invalid (initial smoke erratum in PR #230).
    # Only the scientific runner writes these two exact source copies.
    staging = Path(".cache/numba_fused_rowmax_source").resolve()
    def load_source(name, contents):
        directory = staging / name
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / "pcc_numba.py"
        file.write_bytes(contents.encode("utf-8"))
        mod_name = "_pcc_science_fused_rowmax_" + name
        mod = types.ModuleType(mod_name)
        mod.__file__ = str(file)
        sys.modules[mod_name] = mod
        exec(compile(contents, str(file), "exec"), mod.__dict__)
        return mod
    original_mod = load_source("original", original)
    candidate_mod = load_source("candidate", candidate)
    assert original_mod.pcc_propagate_numba.__name__ == "pcc_propagate_numba"
    assert candidate_mod.pcc_propagate_numba.__name__ == "pcc_propagate_numba"
    return (original_mod.pcc_propagate_numba,
            candidate_mod.pcc_propagate_numba,
            hashlib.sha256(original_raw).hexdigest(),
            hashlib.sha256(candidate.encode()).hexdigest())


def fixture(dataset, seed, preparer):
    if dataset != "ring_8192_32":
        n, features, classes, _ = REAL_DATASETS[dataset]
        case = Case(dataset, n, features, classes, k_nn=10,
                    labels_per_class=3, label_seed=seed, data_seed=0)
        arrays, meta = preparer.prepare(case)
        assert meta["labeled_nodes"] == classes * 3
        return arrays, 10, meta["fixture_sha256"]
    n, c = 8192, 32
    nodes = np.arange(n, dtype=np.int64)
    graph = np.column_stack(((nodes-1) % n, (nodes+1) % n))
    assert np.array_equal(graph[graph[:, 0], 1], nodes)
    assert np.array_equal(graph[graph[:, 1], 0], nodes)
    truth = nodes % c
    labels = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for cls in range(c):
        eligible = np.flatnonzero(truth == cls)
        labels[rng.choice(eligible, 3, replace=False)] = cls
    assert np.count_nonzero(labels != -1) == c * 3
    arrays = dict(neighbors=graph, degrees=np.full(n, 2, dtype=np.int64),
                  labels=labels)
    return arrays, 2, array_hash(labels)


def state_arrays(model, prediction):
    return (prediction, model.node.dominance, model.part.curnode,
            model.part.strength, model.part.dist_table, model.owndeg,
            model.node.label, model.mapped_labels)


def fingerprints(arrays):
    return dict(zip(STATE, (array_hash(x) for x in arrays)))


def make_model(arrays, k, fn):
    model = PCC(impl="numba", update_mode="sequential")
    model.set_graph(arrays["neighbors"], arrays["degrees"], k_nn=k)
    infos = []
    def native_wrapper(*args, **kwargs):
        kwargs["return_info"] = True
        info = fn(*args, **kwargs)
        assert isinstance(info, dict)
        infos.append(info)
        return None
    native_wrapper.__name__ = "pcc_propagate_numba"
    model._get_backend_fn = lambda: native_wrapper
    return model, native_wrapper, infos


def invoke(model, wrapper, labels, mode, cap, pgrd=.5):
    # The actual production master #228 checks the *identity* of the true
    # propagator to select full-row Numba argmax. The original-source
    # experimental per-instance dispatcher must be the same identity.
    # Scoped patch only in the science runner; restores master immediately.
    saved = production.pcc_propagate_numba
    production.pcc_propagate_numba = wrapper
    assert model._get_backend_fn() is production.pcc_propagate_numba
    try:
        return model.fit_predict(labels, max_iter=cap, early_stop=mode,
                                 seed=ARGS["seed"], p_grd=pgrd,
                                 delta_v=ARGS["delta_v"], deltap=ARGS["deltap"],
                                 dexp=ARGS["dexp"], es_chk=ARGS["es_chk"])
    finally:
        production.pcc_propagate_numba = saved


def branch_smoke(arrays, k, fn):
    m, wrapper, _ = make_model(arrays, k, fn)
    actual = np.argmax
    observed = []
    def spy(a, *args, **kwargs):
        if getattr(a, "ndim", 0) == 2:
            observed.append(tuple(a.shape))
        return actual(a, *args, **kwargs)
    production.np.argmax = spy
    try:
        result = invoke(m, wrapper, arrays["labels"], False, 0)
    finally:
        production.np.argmax = actual
    n = len(arrays["labels"])
    classes = len(np.unique(arrays["labels"][arrays["labels"] != -1]))
    assert observed == [(n, classes)], (observed, n, classes)
    assert m.node.dominance.flags.c_contiguous == (classes >= 8)
    assert m.owndeg.flags.f_contiguous
    return fingerprints(state_arrays(m, result))


def validate_single(result, mode, cap):
    info = result["stop"]
    assert info["effective_iterations"] <= cap
    assert info["stop_reason"] in ("max_iter", "early_stop")
    if not mode:
        assert info["effective_iterations"] == cap
        assert info["stop_reason"] == "max_iter"
    assert result["dom_layout"] in ("C", "F")
    assert result["own_layout"] == "F"


def single(arrays, k, fn, mode, cap, trace=False, pgrd=.5):
    m, wrapper, infos = make_model(arrays, k, fn)
    if trace:
        tracemalloc.start()
        tracemalloc.reset_peak()
    try:
        pred = invoke(m, wrapper, arrays["labels"], mode, cap, pgrd)
        peak = tracemalloc.get_traced_memory()[1] if trace else None
    finally:
        if trace:
            tracemalloc.stop()
    assert len(infos) == 1
    st = state_arrays(m, pred)
    data = dict(state=fingerprints(st), stop=infos[-1],
                next_rng=float(peek_rng()),
                dom_layout="C" if st[1].flags.c_contiguous else "F",
                own_layout="F" if st[5].flags.f_contiguous else "C",
                peak=peak)
    if cap:
        validate_single(data, mode, cap)
    return data


def batch(arrays, k, model, wrapper, infos, mode, cap):
    refs = []
    start_cpu = time.process_time_ns()
    start_wall = time.perf_counter_ns()
    for _ in range(BATCH):
        prediction = invoke(model, wrapper, arrays["labels"], mode, cap)
        refs.append(state_arrays(model, prediction))
    wall = time.perf_counter_ns() - start_wall
    cpu = time.process_time_ns() - start_cpu
    assert len(infos) >= BATCH
    info_batch = [dict(x) for x in infos[-BATCH:]]
    keys = [fingerprints(ref) for ref in refs]
    for old, new in zip(refs, refs[1:]):
        # Every fit owns fresh arrays, even for reused model/graph.
        assert all(old[i] is not new[i] for i in (0,1,4,5,6,7))
    layouts = [(ref[1].flags.c_contiguous, ref[1].flags.f_contiguous,
                ref[5].flags.f_contiguous) for ref in refs]
    assert all(x[2] for x in layouts)
    return dict(cpu_ns=cpu, wall_ns=wall, states=keys, infos=info_batch,
                layouts=layouts, next_rng=float(peek_rng()))


def rollback(model, wrapper, labels, cap):
    old_arrays = state_arrays(model, model.node.label)
    before = fingerprints(old_arrays)
    try:
        model.fit_predict(labels, max_iter=cap, seed=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid negative seed was accepted")
    newer = state_arrays(model, old_arrays[0])
    assert all(a is b for a,b in zip(old_arrays[1:],newer[1:]))
    assert fingerprints(newer) == before


def run_one(dataset, seed, arrays, k, original, candidate, smoke=False):
    a, aw, ai = make_model(arrays, k, original)
    b, bw, bi = make_model(arrays, k, candidate)
    # JIT both modes and kernels BEFORE timed region.
    for m, w in ((a, aw),(b,bw)):
        for mode in MODES:
            invoke(m, w, arrays["labels"], mode, 2)
    tests = []
    nrep = 1 if smoke else REPS
    for mode in MODES:
        pairs = []
        for rep in range(nrep):
            if rep % 2 == 0:
                o = batch(arrays, k, a, aw, ai, mode, CAP)
                c = batch(arrays, k, b, bw, bi, mode, CAP)
            else:
                c = batch(arrays, k, b, bw, bi, mode, CAP)
                o = batch(arrays, k, a, aw, ai, mode, CAP)
            for name in ("states","infos","layouts","next_rng"):
                assert o[name] == c[name], (dataset, seed, mode, rep, name)
            assert o["cpu_ns"] >= 0 and c["cpu_ns"] >= 0
            assert o["wall_ns"] > 0 and c["wall_ns"] > 0
            pairs.append(dict(rep=rep, original=o, candidate=c))
            print(dataset, seed, mode, rep,
                  "orig_cpu_ms", round(o["cpu_ns"]/1e6,3),
                  "cand_cpu_ms", round(c["cpu_ns"]/1e6,3), flush=True)
        tests.append(dict(mode="monitored" if mode else "fixed", pairs=pairs))
    rollback(a, aw, arrays["labels"], CAP)
    rollback(b, bw, arrays["labels"], CAP)

    if smoke:
        return dict(dataset=dataset,seed=seed,graph_shas=[
            array_hash(arrays["neighbors"]),array_hash(arrays["degrees"])],
            label_sha=array_hash(arrays["labels"]),tests=tests,
            warmup_complete=True, memory=[], forced=[], prefix=None)
    memory=[]
    for mode in MODES:
        samples=[]
        for i in range(3):
            if i%2==0:
                so=single(arrays,k,original,mode,CAP,True)
                sc=single(arrays,k,candidate,mode,CAP,True)
            else:
                sc=single(arrays,k,candidate,mode,CAP,True)
                so=single(arrays,k,original,mode,CAP,True)
            assert all(so[x]==sc[x] for x in ("state","stop","next_rng",
                                              "dom_layout","own_layout"))
            samples.append(dict(rep=i,baseline=so["peak"],candidate=sc["peak"]))
        memory.append(dict(mode="monitored" if mode else "fixed",samples=samples))
    prefix=[]
    for mode, steps in ((True,2001),):
        so=single(arrays,k,original,mode,steps)
        sc=single(arrays,k,candidate,mode,steps)
        assert so==sc
        prefix.append(dict(mode="monitored",steps=steps,state=so))
    forced=[]
    for pgrd in (0.,1.):
        for steps in (2,1001):
            so=single(arrays,k,original,False,steps,pgrd=pgrd)
            sc=single(arrays,k,candidate,False,steps,pgrd=pgrd)
            assert so==sc
            forced.append(dict(p_grd=pgrd,steps=steps,state=so))
    return dict(dataset=dataset,seed=seed,graph_shas=[
        array_hash(arrays["neighbors"]),array_hash(arrays["degrees"])],
        label_sha=array_hash(arrays["labels"]),tests=tests,
        warmup_complete=True,memory=memory,forced=forced,prefix=prefix)


def main():
    for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                     "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        assert os.environ.get(variable) == "1", (variable,os.environ.get(variable))
    original,candidate,orig_sha,cand_sha = source_pair()
    preparer=FixturePreparer(prepare_fixture)
    smoke = "--smoke" in sys.argv
    output = sys.argv[sys.argv.index("--output")+1]
    rows=[]
    cases = [("digits",87523)] if smoke else [
        (ds,seed) for ds in DATASETS for seed in SEEDS]
    for ds,seed in cases:
        arrays,k,fixture_sha = fixture(ds,seed,preparer)
        if not rows:
            # Directly prove original #228 post-argmax remains ACTIVE.
            x = branch_smoke(arrays,k,original)
            y = branch_smoke(arrays,k,candidate)
            assert x == y, "original/experimental real Numba argmax branch mismatch"
        case=run_one(ds,seed,arrays,k,original,candidate,smoke)
        case["fixture_sha"]=fixture_sha
        rows.append(case)
    result=dict(schema=1,base=BASE,head=subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),
        original_sha=orig_sha,candidate_sha=cand_sha,
        platform=sys.platform,system=platform.platform(),
        python=sys.version, cases=rows, batch=BATCH,
        repetitions=1 if smoke else REPS,
        source_protected=True,true_numba_argmax_active=True)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                            encoding="utf-8")
    print("FROZEN NUMBA FUSED ROWMAX FINISHED",sys.platform,
          "cohorts",len(rows)*2,"full_timed_fits",
          len(rows)*2*(1 if smoke else REPS)*2*BATCH,flush=True)


if __name__=="__main__":
    main()
