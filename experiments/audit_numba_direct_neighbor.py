"""Source-pinned Numba native direct-neighbor access CPU science (NO production edits).

Loads actual ORIGINAL pcc_numba.py Git blob twice: original reference and an
independently Numba-compiled one-hunk-equivalent candidate generated with
unique guarded anchors. Runs full ORIGINAL high-level PCC and native RNG.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

import numpy as np
from numba import njit
from benchmarks.fixtures import Case, REAL_DATASETS, FixturePreparer, prepare_fixture, array_hash
from pcc import ParticleCompetitionAndCooperation as PCC

BASE="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("iris","wine","breast_cancer","digits")
SEEDS=(64013,64117)
BACKENDS=("original","candidate")
MODES=(True,False)
REPS=6
TRACES=3
MAX_ITER=30000
PROTECTED=("pcc.py","pcc_numba.py","pcc_numpy.py","pcc_graph.py",
           "pcc_step.pyx","pcc_validation.py","setup.py","benchmarks")


def uniq_replace(s,old,new):
    if s.count(old)!=1:
        raise AssertionError(f"source drift: expected one guarded anchor {old[:85]!r}, found {s.count(old)}")
    return s.replace(old,new,1)


def source_modules():
    subprocess.check_call(
        ["git","cat-file","-e",BASE+"^{commit}"],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    unchanged=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        text=True).strip()
    if unchanged:
        raise AssertionError("protected production/native source diff: "+unchanged)
    raw=subprocess.check_output(["git","show",BASE+":pcc_numba.py"])
    original=raw.decode("utf-8").replace("\r\n","\n")
    disk=Path("pcc_numba.py").read_text(encoding="utf-8").replace("\r\n","\n")
    if disk!=original:
        raise AssertionError("production worktree differs from canonical original Git")
    start="@njit\ndef _pcc_step_numba_sequential("
    next_one="@njit\ndef _pcc_calc_mmpot_sequential("
    if original.count(start)!=1 or original.count(next_one)!=1:
        raise AssertionError("unclear Numba original step boundaries")
    step=original[original.index(start):original.index(next_one)]
    replaced=step
    for old,new,expected in (
        ("        neighbors = neib_list[curnode, :k]\n","",1),
        ("next_node = neighbors[i]","next_node = neib_list[curnode, i]",1),
        ("next_node = neighbors[choice]","next_node = neib_list[curnode, choice]",1),
        ("next_node = neighbors[idx]","next_node = neib_list[curnode, idx]",2),
    ):
        if replaced.count(old)!=expected:
            raise AssertionError(
                "Numba frozen original neighbor access unexpectedly changed "+
                repr((old,expected,replaced.count(old))))
        replaced=replaced.replace(old,new)
    if "neighbors" in replaced:
        raise AssertionError("remaining slice-view alias in direct candidate")
    candidate=original.replace(step,replaced,1)
    if candidate==original or candidate.count(replaced)!=1:
        raise AssertionError("candidate failed unique single-step transformation")
    original_mod=types.ModuleType("frozen_original_numba")
    candidate_mod=types.ModuleType("frozen_candidate_direct_neighbor_numba")
    exec(compile(original,BASE+":pcc_numba.py","exec"),original_mod.__dict__)
    exec(compile(candidate,BASE+":direct_neighbor","exec"),candidate_mod.__dict__)
    for mod in (original_mod,candidate_mod):
        if mod.pcc_propagate_numba.__name__!="pcc_propagate_numba":
            raise AssertionError("native Numba dispatch identity lost")
    return (original_mod,candidate_mod,hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(candidate.encode("utf-8")).hexdigest())


@njit
def peek_numba_rng():
    return np.random.random()


def digest_rng():
    # Used exclusively *after* the completed timed native fit.
    return float(peek_numba_rng())


def model_fit(arrays,case,fn,mode,cap=MAX_ITER,trace=False,pgrd=.5):
    model=PCC(impl="numba",update_mode="sequential")
    model.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
    # Inject ONLY this instance's direct compiled-from-frozen-Git native
    # backend. No module-wide monkeypatch and no kernel observer.
    if fn.__name__ not in ("pcc_propagate_numba","pcc_propagate_numba_verified"):
        raise AssertionError("not original Numba source native function")
    observed={}
    def native_info_wrapper(*args,**kwargs):
        kwargs["return_info"]=True
        begin=time.perf_counter_ns()
        info=fn(*args,**kwargs)
        observed.update(info)
        observed["propagation_elapsed_ns"]=time.perf_counter_ns()-begin
        return None
    # Preserve actual Numba C-order selection with >=8 classes (#203).
    native_info_wrapper.__name__="pcc_propagate_numba"
    model._get_backend_fn=lambda:native_info_wrapper
    if trace:
        gc.collect()
        tracemalloc.start()
    try:
        start=time.perf_counter_ns()
        predictions=model.fit_predict(
            arrays["labels"],max_iter=cap,early_stop=mode,
            seed=0,p_grd=pgrd,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)
        elapsed_ns=time.perf_counter_ns()-start
        peak=tracemalloc.get_traced_memory()[1] if trace else None
    finally:
        if trace:
            tracemalloc.stop()
    dom=model.node.dominance
    expected_layout=(dom.flags.c_contiguous if case.classes>=8
                     else dom.flags.f_contiguous)
    if not expected_layout:
        raise AssertionError("Numba dominance layout differs from production native policy")
    if not 0<observed["effective_iterations"]<=cap:
        raise AssertionError("invalid original Numba stop cap")
    if not mode and observed["effective_iterations"]!=cap:
        raise AssertionError("fixed path truncated early")
    if observed["stop_reason"] not in ("early_stop","max_iter"):
        raise AssertionError("Numba stop reason absent")
    state=dict(predictions=predictions,
               dominance=dom,positions=model.part.curnode,
               strengths=model.part.strength,
               distances=model.part.dist_table,
               owndeg=model.owndeg)
    states={name:array_hash(data) for name,data in state.items()}
    return dict(elapsed_ns=elapsed_ns,peak_bytes=peak,
                iteration_cap=cap,mode="monitored" if mode else "fixed",
                effective_iterations=observed["effective_iterations"],
                stop_reason=observed["stop_reason"],checks=observed["checks"],
                stalled_checks=observed["stalled_checks"],
                stop_max=observed["stop_max"],
                propagation_elapsed_ns=observed["propagation_elapsed_ns"],
                state_hashes=states,
                predicted_hash=array_hash(predictions),
                rng_next=digest_rng(),
                mapped_labels_hash=array_hash(model.mapped_labels))


def exactly_equal(a,b,tag):
    for key in ("state_hashes","rng_next","mapped_labels_hash",
                "effective_iterations","stop_reason","checks",
                "stalled_checks","stop_max"):
        if a[key]!=b[key]:
            raise AssertionError("source-paired Numba direct neighbor drift "+
                                 repr((tag,key)))
    return True


def run_case(case,arrays,original,candidate):
    o=original.pcc_propagate_numba
    c=candidate.pcc_propagate_numba
    warmups=[]
    cpu=[]
    traced=[]
    for variant,func in (("original",o),("candidate",c)):
        for mode in MODES:
            # Warm both true specialized JIT modes and C/F layout outside CPU.
            warmup=model_fit(arrays,case,func,mode,cap=2)
            if warmup["effective_iterations"]!=2:
                raise AssertionError("invalid 2-step warmup")
            warmups.append(dict(variant=variant,**warmup))
    for mode in MODES:
        saved_by_variant={}
        for rep in range(REPS):
            order=(("candidate","original") if rep%2==0
                   else ("original","candidate"))
            pair={}
            for variant in order:
                model_func=o if variant=="original" else c
                result=model_fit(arrays,case,model_func,mode)
                pair[variant]=result
                statekey=(result["state_hashes"],result["rng_next"],
                          result["mapped_labels_hash"],
                          result["effective_iterations"],
                          result["stop_reason"],result["checks"],
                          result["stalled_checks"],result["stop_max"])
                if variant in saved_by_variant and saved_by_variant[variant]!=statekey:
                    raise AssertionError("same original/candidate repeated fit drift "+
                                         repr((case.case_id,variant,mode)))
                saved_by_variant.setdefault(variant,statekey)
            exactly_equal(pair["original"],pair["candidate"],
                          (case.case_id,mode,rep))
            cpu.append(dict(mode="monitored" if mode else "fixed",rep=rep,
                            order=list(order),original=pair["original"],
                            candidate=pair["candidate"]))
        for rep in range(TRACES):
            order=(("candidate","original") if rep%2==0
                   else ("original","candidate"))
            pair={}
            for variant in order:
                func=o if variant=="original" else c
                pair[variant]=model_fit(arrays,case,func,mode,trace=True)
            exactly_equal(pair["original"],pair["candidate"],
                          (case.case_id,mode,"traced",rep))
            traced.append(dict(mode="monitored" if mode else "fixed",rep=rep,
                               original=pair["original"],candidate=pair["candidate"]))

    # One fixed known-prefix monitored run per seed/dataset, after all 30k pairs.
    short_o=model_fit(arrays,case,o,True,cap=2001)
    short_c=model_fit(arrays,case,c,True,cap=2001)
    exactly_equal(short_o,short_c,(case.case_id,"short-prefix"))
    short=dict(original=short_o,candidate=short_c,cap=2001)

    # Test each extreme of the same public greedy policy without altering RNG.
    boundary=[]
    for pgrd in (0.0,1.0):
        a=model_fit(arrays,case,o,False,cap=2,pgrd=pgrd)
        b=model_fit(arrays,case,c,False,cap=2,pgrd=pgrd)
        exactly_equal(a,b,(case.case_id,"pgrd",pgrd))
        boundary.append(dict(p_grd=pgrd,original=a,candidate=b))
    return dict(dataset=case.name,seed=case.label_seed,case_id=case.case_id,
                labels_sha=array_hash(arrays["labels"]),
                graph_sha=[array_hash(arrays[k]) for k in ("neighbors","degrees")],
                samples=cpu,peaks=traced,warmups=warmups,
                short_prefix=short,forced_choices=boundary)


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    controls=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
              "MKL_NUM_THREADS","NUMBA_NUM_THREADS")
    if any(os.environ.get(v)!="1" for v in controls):
        raise AssertionError("native CPU thread controls not all 1")
    original,candidate,source_sha,candidate_sha=source_modules()
    preparer=FixturePreparer(prepare_fixture)
    rows=[]
    for dataset in DATASETS:
        nodes,features,classes,_=REAL_DATASETS[dataset]
        for seed in SEEDS:
            case=Case(dataset,nodes,features,classes,
                      k_nn=10,labels_per_class=3,data_seed=0,label_seed=seed)
            arrays,metadata=preparer.prepare(case)
            if metadata["labeled_nodes"]!=3*classes:
                raise AssertionError("fixture label population differs")
            print("START",case.case_id,flush=True)
            row=run_case(case,arrays,original,candidate)
            row.update(fixture_sha=metadata["fixture_sha256"],
                       graph_reused=metadata["preparation"]["reused"],
                       nodes=nodes,classes=classes,
                       input_sha=array_hash(arrays["x"]))
            rows.append(row)
            monitored=[p for p in row["samples"] if p["mode"]=="monitored"]
            fixed=[p for p in row["samples"] if p["mode"]=="fixed"]
            print("DONE",case.case_id,"full CPU percent saved",
                  round(100*(1-statistics.median(
                       p["candidate"]["elapsed_ns"]/p["original"]["elapsed_ns"]
                       for p in monitored)),3),
                  round(100*(1-statistics.median(
                       p["candidate"]["elapsed_ns"]/p["original"]["elapsed_ns"]
                       for p in fixed)),3),flush=True)
    if len(rows)!=8:raise AssertionError("missing original fixture cases")
    corpus=dict(schema=1,base=BASE,
        branch_head=subprocess.check_output(
            ["git","rev-parse","HEAD"],text=True).strip(),
        platform=sys.platform,host=platform.platform(),cpu=platform.processor(),
        python=sys.version,numpy=np.__version__,
        numba=__import__("numba").__version__,
        sklearn=__import__("sklearn").__version__,
        source_sha256=source_sha,candidate_sha256=candidate_sha,
        threads={v:os.environ[v] for v in controls},
        cases=rows)
    Path("numba_neighbor_direct_results.json").write_text(
        json.dumps(corpus,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("COMPLETE direct Numba neighbor: 96 timed pairs, 48 traced pairs,"
          " 16 JIT warmup pairs, 8 exact short-prefix pairs and"
          " 16 forced-choice pairs ON THIS OS",flush=True)


if __name__=="__main__":
    main()
