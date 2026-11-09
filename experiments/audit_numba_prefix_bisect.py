"""Original-source Numba prefix bisect with exact first-greater roulette semantics."""
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
DATASETS = ("iris_k10", "wine_k10", "breast_cancer_k10", "digits_k3", "digits_k10", "ring4096_d2", "ring4096_d32")
GEOMETRIES = {"iris_k10": ("iris", 10), "wine_k10": ("wine", 10),
              "breast_cancer_k10": ("breast_cancer", 10),
              "digits_k3": ("digits", 3), "digits_k10": ("digits", 10),
              "ring4096_d2": ("ring", 2), "ring4096_d32": ("ring", 32)}
POLICIES = (0.0, 0.5, 1.0)
SEEDS = (89021, 89119)
MODES = (True, False)
REPS = 4
BATCH = 8
CAP = 10000
STATE = ("predictions", "dominance", "positions", "strengths",
         "distances", "owndeg", "node_labels", "mapped_labels")
PROTECTED = ("pcc.py", "pcc_numba.py", "pcc_numpy.py", "pcc_graph.py",
             "pcc_step.pyx", "pcc_validation.py", "setup.py",
             "benchmarks/runner.py", "benchmarks/fixtures.py")
ARGS = dict(seed=29, delta_v=.1, deltap=1., dexp=2., es_chk=2000)


@njit
def peek_rng():
    return np.random.random()


def uniq(src, old, new):
    assert src.count(old) == 1, (old[:60], src.count(old))
    return src.replace(old, new, 1)



@njit
def original_prefix_choice(prefix,rand_val):
    choice=0
    for i in range(len(prefix)):
        if rand_val < prefix[i]:
            choice=i
            break
    return choice


@njit
def binary_prefix_choice(prefix,rand_val):
    k=len(prefix)
    monotone=True
    previous=0.0
    for i in range(k):
        if not (prefix[i]>=previous):
            monotone=False
        previous=prefix[i]
    choice=0
    if k>=8 and monotone:
        lo=0
        hi=k
        while lo<hi:
            mid=(lo+hi)//2
            if rand_val < prefix[mid]:
                hi=mid
            else:
                lo=mid+1
        if lo<k:
            choice=lo
    else:
        for i in range(k):
            if rand_val<prefix[i]:
                choice=i
                break
    return choice


def characterize_roulette():
    rng=np.random.default_rng(928337)
    degrees=(1,2,3,7,8,9,16,32,64)
    results=[]
    for t in range(1500):
        k=degrees[t%len(degrees)]
        x=rng.random(k).astype(np.float64)
        style=t%10
        if style in (0,1,2):
            x[::2]=0.
        if style==3:
            x[:]=0.
        if style==4:
            x[:]=1.0
        if style==5:
            x[::3]=-2.0
        if style==6:
            x[::3]=0.
        prefix=np.cumsum(x,dtype=np.float64)
        candidates=[0.,prefix[0],prefix[-1],
                    np.nextafter(prefix[-1],-np.inf),
                    np.nextafter(prefix[min(1,k-1)],np.inf),
                    float(rng.random())*max(0.,float(prefix[-1]))]
        val=float(candidates[t%len(candidates)])
        orig=int(original_prefix_choice(prefix,val))
        bisect=int(binary_prefix_choice(prefix,val))
        assert orig==bisect,(t,k,style,val,orig,bisect,prefix)
        results.append(orig)
    digest=hashlib.sha256(np.asarray(results,dtype=np.int64).tobytes()).hexdigest()
    return dict(cases=len(results),original_equals_binary=True,
                choice_sha=digest)


def source_pair():
    changed=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        text=True).strip()
    assert not changed, "protected production changed: "+changed
    for name in PROTECTED:
        raw=subprocess.check_output(["git","show",BASE+":"+name])
        disk=Path(name).read_bytes()
        assert disk.replace(b"\r\n",b"\n")==raw.replace(b"\r\n",b"\n"),name
    original_raw=subprocess.check_output(["git","show",BASE+":pcc_numba.py"])
    original=original_raw.decode("utf-8")
    start="@njit(cache=True)\ndef _pcc_step_numba_sequential("
    end="@njit(cache=True)\ndef _pcc_calc_mmpot_sequential("
    assert original.count(start)==original.count(end)==1
    step=original[original.index(start):original.index(end)]
    new=step
    new=uniq(new,
        "            total = 0.0\n            for i in range(k):\n"
        "                total += prob[i]\n                slices[i] = total\n",
        "            total = 0.0\n            monotone = True\n"
        "            prev_total = 0.0\n"
        "            for i in range(k):\n"
        "                total += prob[i]\n"
        "                slices[i] = total\n"
        "                if not (total >= prev_total):\n"
        "                    monotone = False\n"
        "                prev_total = total\n")
    new=uniq(new,
        "                total = 0.0\n                for i in range(k):\n"
        "                    prob[i] /= max_weight\n"
        "                    total += prob[i]\n"
        "                    slices[i] = total\n",
        "                total = 0.0\n                monotone = True\n"
        "                prev_total = 0.0\n"
        "                for i in range(k):\n"
        "                    prob[i] /= max_weight\n"
        "                    total += prob[i]\n"
        "                    slices[i] = total\n"
        "                    if not (total >= prev_total):\n"
        "                        monotone = False\n"
        "                    prev_total = total\n")
    new=uniq(new,
        "                choice = 0\n                for i in range(k):\n"
        "                    if rand_val < slices[i]:\n"
        "                        choice = i\n"
        "                        break\n",
        "                choice = 0\n"
        "                if k >= 8 and monotone:\n"
        "                    lo = 0\n"
        "                    hi = k\n"
        "                    while lo < hi:\n"
        "                        mid = (lo + hi) // 2\n"
        "                        if rand_val < slices[mid]:\n"
        "                            hi = mid\n"
        "                        else:\n"
        "                            lo = mid + 1\n"
        "                    if lo < k:\n"
        "                        choice = lo\n"
        "                else:\n"
        "                    for i in range(k):\n"
        "                        if rand_val < slices[i]:\n"
        "                            choice = i\n"
        "                            break\n")
    candidate=uniq(original,step,new)
    assert candidate.replace(new,step,1)==original
    staging=Path(".cache/numba_prefix_bisect_source").resolve()
    def load(name,src):
        directory=staging/name
        directory.mkdir(parents=True,exist_ok=True)
        file=directory/"pcc_numba.py"
        file.write_bytes(src.encode("utf-8"))
        module=types.ModuleType("_scientific_prefix_bisect_"+name)
        module.__file__=str(file)
        sys.modules[module.__name__]=module
        exec(compile(src,str(file),"exec"),module.__dict__)
        return module
    a=load("original",original)
    b=load("candidate",candidate)
    assert a.pcc_propagate_numba.__name__==b.pcc_propagate_numba.__name__=="pcc_propagate_numba"
    return (a.pcc_propagate_numba,b.pcc_propagate_numba,
            hashlib.sha256(original_raw).hexdigest(),
            hashlib.sha256(candidate.encode()).hexdigest())

def fixture(dataset,seed,preparer):
    name,k=GEOMETRIES[dataset]
    if name!="ring":
        n,features,c,_=REAL_DATASETS[name]
        arrays,meta=preparer.prepare(
            Case(name,n,features,c,k_nn=k,labels_per_class=3,
                 label_seed=seed,data_seed=0))
        assert meta["labeled_nodes"]==c*3
        return arrays,k,meta["fixture_sha256"]
    n,c=4096,32
    nodes=np.arange(n,dtype=np.int64)
    offsets=np.array([shift for j in range(1,k//2+1)
                      for shift in (-j,j)],dtype=np.int64)
    graph=(nodes[:,None]+offsets[None,:])%n
    assert graph.shape==(n,k) and np.unique(graph[0]).size==k
    assert np.array_equal(graph[(nodes+1)%n],(graph+1)%n)
    labels=np.full(n,-1,dtype=np.int64)
    rng=np.random.default_rng(seed)
    truth=nodes%c
    for cls in range(c):
        eligible=np.flatnonzero(truth==cls)
        labels[rng.choice(eligible,3,replace=False)]=cls
    assert np.count_nonzero(labels!=-1)==3*c
    arrays=dict(neighbors=graph,degrees=np.full(n,k,dtype=np.int64),
                labels=labels)
    return arrays,k,hashlib.sha256(
        (array_hash(graph)+array_hash(labels)).encode()).hexdigest()

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


def batch(arrays, k, model, wrapper, infos, mode, cap, pgrd):
    refs = []
    start_cpu = time.process_time_ns()
    start_wall = time.perf_counter_ns()
    for _ in range(BATCH):
        prediction = invoke(model, wrapper, arrays["labels"], mode, cap, pgrd)
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


def run_one(dataset,seed,pgrd,arrays,k,original,candidate,smoke=False):
    a,aw,ai=make_model(arrays,k,original)
    b,bw,bi=make_model(arrays,k,candidate)
    for m,w in ((a,aw),(b,bw)):
        for mode in MODES:
            invoke(m,w,arrays["labels"],mode,2,pgrd)
    tests=[]
    for mode in MODES:
        pairs=[]
        for rep in range(1 if smoke else REPS):
            if rep%2==0:
                o=batch(arrays,k,a,aw,ai,mode,CAP,pgrd)
                c=batch(arrays,k,b,bw,bi,mode,CAP,pgrd)
            else:
                c=batch(arrays,k,b,bw,bi,mode,CAP,pgrd)
                o=batch(arrays,k,a,aw,ai,mode,CAP,pgrd)
            for key in ("states","infos","layouts","next_rng"):
                assert o[key]==c[key],(dataset,seed,pgrd,mode,rep,key)
            assert o["cpu_ns"]>0 and c["cpu_ns"]>0
            assert o["wall_ns"]>0 and c["wall_ns"]>0
            pairs.append(dict(rep=rep,original=o,candidate=c))
            print(dataset,seed,pgrd,mode,rep,
                  "original_cpu_fit_ms",round(o["cpu_ns"]/8e6,4),
                  "binary_cpu_fit_ms",round(c["cpu_ns"]/8e6,4),flush=True)
        tests.append(dict(mode="monitored" if mode else "fixed",pairs=pairs))
    rollback(a,aw,arrays["labels"],CAP)
    rollback(b,bw,arrays["labels"],CAP)
    if smoke:
        return dict(dataset=dataset,seed=seed,p_grd=pgrd,
                    graph_shas=[array_hash(arrays["neighbors"]),
                                array_hash(arrays["degrees"])],
                    label_sha=array_hash(arrays["labels"]),
                    tests=tests,warmup_complete=True,memory=[],
                    forced=[],prefix=[])
    memory=[]
    for mode in MODES:
        samples=[]
        for i in range(3):
            if i%2==0:
                so=single(arrays,k,original,mode,CAP,True,pgrd)
                sc=single(arrays,k,candidate,mode,CAP,True,pgrd)
            else:
                sc=single(arrays,k,candidate,mode,CAP,True,pgrd)
                so=single(arrays,k,original,mode,CAP,True,pgrd)
            assert all(so[x]==sc[x] for x in (
                "state","stop","next_rng","dom_layout","own_layout"))
            samples.append(dict(rep=i,baseline=so["peak"],candidate=sc["peak"]))
        memory.append(dict(mode="monitored" if mode else "fixed",samples=samples))
    prefix=[]
    so=single(arrays,k,original,True,2001,False,pgrd)
    sc=single(arrays,k,candidate,True,2001,False,pgrd)
    assert so==sc
    prefix.append(dict(mode="monitored",steps=2001,state=so))
    forced=[]
    for cap in (0,5):
        so=single(arrays,k,original,False,cap,False,pgrd)
        sc=single(arrays,k,candidate,False,cap,False,pgrd)
        assert so==sc
        forced.append(dict(p_grd=pgrd,steps=cap,state=so))
    return dict(dataset=dataset,seed=seed,p_grd=pgrd,
                graph_shas=[array_hash(arrays["neighbors"]),
                            array_hash(arrays["degrees"])],
                label_sha=array_hash(arrays["labels"]),
                tests=tests,warmup_complete=True,memory=memory,
                forced=forced,prefix=prefix)

def main():
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        assert os.environ.get(key)=="1",(key,os.environ.get(key))
    original,candidate,orig_sha,cand_sha=source_pair()
    characterization=characterize_roulette()
    preparer=FixturePreparer(prepare_fixture)
    smoke="--smoke" in sys.argv
    target=sys.argv[sys.argv.index("--output")+1]
    cases=[("digits_k10",89021,.5)] if smoke else [
        (ds,seed,p) for ds in DATASETS for seed in SEEDS for p in POLICIES]
    rows=[]
    for ds,seed,p in cases:
        arrays,k,fixture_sha=fixture(ds,seed,preparer)
        if not rows:
            aa=branch_smoke(arrays,k,original)
            bb=branch_smoke(arrays,k,candidate)
            assert aa==bb,"actual current production #228 Numba path mismatch"
        result=run_one(ds,seed,p,arrays,k,original,candidate,smoke)
        result["fixture_sha"]=fixture_sha
        rows.append(result)
    report=dict(schema=1,base=BASE,head=subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),
        original_sha=orig_sha,candidate_sha=cand_sha,
        platform=sys.platform,system=platform.platform(),
        python=sys.version,characterization=characterization,
        cases=rows,batch=BATCH,repetitions=1 if smoke else REPS,
        source_protected=True,true_numba_argmax_active=True)
    dest=Path(target);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")
    print("FROZEN ORIGINAL NUMBA GREEDY PREFIX BISECT COMPLETE",
          sys.platform,"cohorts",len(rows)*2,"full_timed_fits",
          len(rows)*2*(1 if smoke else REPS)*2*BATCH,flush=True)


if __name__=="__main__":
    main()
