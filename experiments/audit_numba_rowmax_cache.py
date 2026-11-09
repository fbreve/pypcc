"""Source-pinned Numba native row-maximum potential cache science (NO production edits).

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
SEEDS=(62011,62137)
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
    subprocess.check_call(["git","cat-file","-e",BASE+"^{commit}"],
                          stdout=subprocess.DEVNULL)
    diff=subprocess.check_output(["git","diff","--name-only",BASE,"HEAD",
                                  "--",*PROTECTED],text=True).strip()
    if diff:
        raise AssertionError("production/benchmark source differs from frozen base: "+diff)
    raw=subprocess.check_output(["git","show",BASE+":pcc_numba.py"])
    original=raw.decode("utf-8").replace("\r\n","\n")
    if Path("pcc_numba.py").read_text(encoding="utf-8").replace("\r\n","\n")!=original:
        raise AssertionError("local production source is not original Numba Git blob")

    step_start="@njit\ndef _pcc_step_numba_sequential("
    pot_start="@njit\ndef _pcc_calc_mmpot_sequential("
    prop_start="@njit\ndef _pcc_propagate_numba_sequential("
    standalone_start="def pcc_step_numba("
    wrap_start="def pcc_propagate_numba("
    step=original[original.index(step_start):original.index(pot_start)]
    prop=original[original.index(prop_start):original.index(standalone_start)]
    wrapper=original[original.index(wrap_start):]
    if not wrapper.endswith("        return propagation_info(iterations, early_stop, max_iter, stop_cnt)\n"):
        # Original signature has stop_max between max_iter and stop_cnt.
        assert wrapper.rstrip().endswith(
            "return propagation_info(iterations, early_stop, max_iter, stop_max, stop_cnt)")

    cached_step=uniq_replace(
        step,"def _pcc_step_numba_sequential(","def _pcc_step_numba_sequential_cached(")
    cached_step=uniq_replace(
        cached_step,"                               dist_weights):",
        "                               dist_weights, row_max_cache):")
    cached_step=uniq_replace(
        cached_step,"            if dominance[next_node, label] == max_dom:\n",
        "            if labels[next_node] == -1:\n"
        "                row_max_cache[next_node] = max_dom\n"
        "            if dominance[next_node, label] == max_dom:\n")
    helper=StringHelpers
    cached_prop=uniq_replace(
        prop,"def _pcc_propagate_numba_sequential(",
        "def _pcc_propagate_numba_sequential_cached(")
    cached_prop=uniq_replace(
        cached_prop,"max_iter, early_stop, es_chk, stop_max, return_info=False):",
        "max_iter, early_stop, es_chk, stop_max, return_info=False, row_max_cache=None):")
    cached_prop=uniq_replace(
        cached_prop,"        _pcc_step_numba_sequential(",
        "        _pcc_step_numba_sequential_cached(")
    cached_prop=uniq_replace(
        cached_prop,"                                   dist_weights)\n",
        "                                   dist_weights, row_max_cache)\n")
    cached_prop=uniq_replace(
        cached_prop,"            mmpot = _pcc_calc_mmpot_sequential(dominance, n_nodes, c)\n",
        "            mmpot = _pcc_calc_mmpot_from_rowmax(row_max_cache, n_nodes)\n")

    # Generate a distinct check-only compiled clone; NO extra condition is
    # placed inside the measured candidate's normal hot propagation loop.
    checked_prop=uniq_replace(
        cached_prop,"def _pcc_propagate_numba_sequential_cached(",
        "def _pcc_propagate_numba_sequential_checked(")
    checked_prop=uniq_replace(
        checked_prop,"stop_max, return_info=False, row_max_cache=None):",
        "stop_max, return_info=False, row_max_cache=None, audit_counts=None):")
    checked_prop=uniq_replace(
        checked_prop,"            mmpot = _pcc_calc_mmpot_from_rowmax(row_max_cache, n_nodes)\n",
        "            mmpot = _pcc_calc_mmpot_from_rowmax(row_max_cache, n_nodes)\n"
        "            if mmpot != _pcc_calc_mmpot_sequential(dominance, n_nodes, c):\n"
        "                raise AssertionError('cached Numba checkpoint potential differs bitwise')\n"
        "            audit_counts[0] += 1\n")

    # Preserve frozen wrapper validation, seed, native execution contract.
    old_call="""    metrics = _pcc_propagate_numba_sequential(
        neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
        part_curnode, part_label, part_strength, dist_table, dominance, owndeg,
        deltap, dexp, dom_row, reduc, dom_list, dist_list, prob, slices,
        dist_weights, max_iter, early_stop, es_chk, stop_max, return_info)"""
    new_call="""    if early_stop:
        row_max_cache = np.empty(neib_list.shape[0], dtype=np.float64)
        _pcc_initialize_rowmax(dominance, neib_list.shape[0], c, row_max_cache)
        metrics = _pcc_propagate_numba_sequential_cached(
            neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
            part_curnode, part_label, part_strength, dist_table, dominance, owndeg,
            deltap, dexp, dom_row, reduc, dom_list, dist_list, prob, slices,
            dist_weights, max_iter, early_stop, es_chk, stop_max, return_info,
            row_max_cache)
    else:
        metrics = _pcc_propagate_numba_sequential(
            neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
            part_curnode, part_label, part_strength, dist_table, dominance, owndeg,
            deltap, dexp, dom_row, reduc, dom_list, dist_list, prob, slices,
            dist_weights, max_iter, early_stop, es_chk, stop_max, return_info)"""
    candidate_wrap=uniq_replace(wrapper,old_call,new_call)
    # Experiment-only verification callable; preserves all wrapper preflight
    # while evaluating *every* original checkpoint from the checked kernel.
    verified_wrap=uniq_replace(
        candidate_wrap,"def pcc_propagate_numba(","def pcc_propagate_numba_verified(")
    verified_wrap=uniq_replace(
        verified_wrap,"        metrics = _pcc_propagate_numba_sequential_cached(",
        "        audit_counts = np.zeros(1, dtype=np.int64)\n"
        "        metrics = _pcc_propagate_numba_sequential_checked(")
    verified_wrap=uniq_replace(
        verified_wrap,"            row_max_cache)\n",
        "            row_max_cache, audit_counts)\n"
        "        CHECKPOINT_AUDIT.append(int(audit_counts[0]))\n")

    original_mod=types.ModuleType("frozen_original_numba")
    candidate_mod=types.ModuleType("scientific_cached_numba")
    candidate_mod.CHECKPOINT_AUDIT=[]
    exec(compile(original,BASE+":pcc_numba.py","exec"),original_mod.__dict__)
    candidate_source=original.replace(wrapper,candidate_wrap,1)
    candidate_source+="\n\n"+cached_step+"\n\n"+helper+"\n\n"+cached_prop+"\n\n"+checked_prop+"\n\n"+verified_wrap+"\n"
    exec(compile(candidate_source,BASE+":derived-cache-candidate","exec"),
         candidate_mod.__dict__)
    for mod in (original_mod,candidate_mod):
        if mod.pcc_propagate_numba.__name__!="pcc_propagate_numba":
            raise AssertionError("Numba dominance C-order dispatch identity altered")
    original_hash=hashlib.sha256(raw).hexdigest()
    return original_mod,candidate_mod,original_hash,hashlib.sha256(
        candidate_source.encode()).hexdigest()


StringHelpers = """
@njit
def _pcc_initialize_rowmax(dominance, n_nodes, c, row_max_cache):
    for i in range(n_nodes):
        row_max = dominance[i, 0]
        for j in range(1, c):
            if dominance[i, j] > row_max:
                row_max = dominance[i, j]
        row_max_cache[i] = row_max


@njit
def _pcc_calc_mmpot_from_rowmax(row_max_cache, n_nodes):
    total_max = 0.0
    for i in range(n_nodes):
        total_max += row_max_cache[i]
    return total_max / n_nodes
"""


@njit
def peek_numba_rng():
    return np.random.random()


def digest_rng():
    # Used exclusively *after* the completed timed native fit.
    return float(peek_numba_rng())


def model_fit(arrays,case,fn,mode,cap=MAX_ITER,trace=False):
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
            seed=0,p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)
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


def run_case(case,arrays,original,candidate):
    results=[]
    timings=[]
    peaks=[]
    warmups=[]
    labels_sha=array_hash(arrays["labels"])
    graph_sha=[array_hash(arrays[n]) for n in ("neighbors","degrees")]
    for name,func in (("original",original.pcc_propagate_numba),
                      ("candidate",candidate.pcc_propagate_numba)):
        # Warm BOTH distinct compiled propagators per variant. The candidate
        # uses cached specialization ON, original specialization OFF; omitting
        # the fixed warmup would time the first Numba JIT and fake a speedup.
        for mode in MODES:
            data=model_fit(arrays,case,func,mode,cap=2)
            if data["effective_iterations"]!=2:
                raise AssertionError("warmup did not execute two original updates")
            warmups.append(dict(variant=name,**data))
    for mode in MODES:
        first=None
        for rep in range(REPS):
            pair={}
            for key,func in ((("original",original.pcc_propagate_numba),
                              ("candidate",candidate.pcc_propagate_numba))
                             if rep%2 else
                             (("candidate",candidate.pcc_propagate_numba),
                              ("original",original.pcc_propagate_numba))):
                data=model_fit(arrays,case,func,mode)
                pair[key]=data
                reference=(data["state_hashes"],data["rng_next"],
                           data["mapped_labels_hash"])
                if first is None and key=="original":
                    first=reference
                if key=="original" and reference!=first:
                    raise AssertionError("repeated original native fit not deterministic")
            for key in ("state_hashes","rng_next","mapped_labels_hash",
                        "effective_iterations","stop_reason","checks",
                        "stalled_checks","stop_max"):
                if pair["original"][key]!=pair["candidate"][key]:
                    raise AssertionError("source-paired full Numba state/RNG/stop differs "+
                                         repr((case.case_id,mode,rep,key)))
            timings.append(dict(case_id=case.case_id,mode="monitored" if mode else "fixed",
                                repetition=rep,order=list(pair),
                                original=pair["original"],
                                candidate=pair["candidate"]))
        for idx in range(TRACES):
            tracepair={}
            for key,func in ((("original",original.pcc_propagate_numba),
                              ("candidate",candidate.pcc_propagate_numba))
                             if idx%2 else
                             (("candidate",candidate.pcc_propagate_numba),
                              ("original",original.pcc_propagate_numba))):
                tracepair[key]=model_fit(arrays,case,func,mode,trace=True)
            if tracepair["original"]["state_hashes"]!=tracepair["candidate"]["state_hashes"]:
                raise AssertionError("traced source-paired states differ")
            peaks.append(dict(case_id=case.case_id,mode="monitored" if mode else "fixed",
                              repetition=idx,original=tracepair["original"],
                              candidate=tracepair["candidate"]))
    # Check-only native kernel: 2001 steps => 201 original it%10==0 checks.
    # Any numerical deviation raises inside a separately compiled verifier.
    checked=candidate.pcc_propagate_numba_verified
    before=len(candidate.CHECKPOINT_AUDIT)
    verified=model_fit(arrays,case,checked,True,cap=2001)
    checks=candidate.CHECKPOINT_AUDIT[before:]
    if checks!=[201]:
        raise AssertionError("incomplete numeric check counts "+repr(checks))
    baseline_short=model_fit(arrays,case,original.pcc_propagate_numba,True,cap=2001)
    if (verified["state_hashes"]!=baseline_short["state_hashes"] or
            verified["rng_next"]!=baseline_short["rng_next"]):
        raise AssertionError("verified 201-check Numba trajectory does not match original")
    return dict(case_id=case.case_id,dataset=case.name,seed=case.label_seed,
                labels_sha=labels_sha,graph_sha=graph_sha,
                samples=timings,peaks=peaks,warmups=warmups,
                audited_checkpoints=checks[0],
                verified_state_hashes=verified["state_hashes"],
                verified_rng_next=verified["rng_next"])


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    thread_vars=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                 "MKL_NUM_THREADS","NUMBA_NUM_THREADS")
    if any(os.environ.get(name)!="1" for name in thread_vars):
        raise AssertionError("benchmark threads must be configured to one")
    original,candidate,raw_sha,candidate_sha=source_modules()
    prepared=[]
    cases=[]
    fixtures=FixturePreparer(prepare_fixture)
    for name in DATASETS:
        nodes,features,classes,_=REAL_DATASETS[name]
        for seed in SEEDS:
            case=Case(name,nodes,features,classes,
                      labels_per_class=3,k_nn=10,label_seed=seed,data_seed=0)
            arrays,meta=fixtures.prepare(case)
            if meta["labeled_nodes"]!=3*classes:
                raise AssertionError("class-balanced labeled population drift")
            print("START",case.case_id,flush=True)
            result=run_case(case,arrays,original,candidate)
            result.update(fixture_sha=meta["fixture_sha256"],
                          graph_reused=meta["preparation"]["reused"],
                          nodes=nodes,classes=classes,
                          input_sha=array_hash(arrays["x"]))
            cases.append(result)
            print("DONE",case.case_id,"monitored pair medians",
                  statistics.median(1-row["candidate"]["elapsed_ns"]/row["original"]["elapsed_ns"]
                    for row in result["samples"] if row["mode"]=="monitored"),flush=True)
    if len(cases)!=8:
        raise AssertionError("missing 8 precommitted real-dataset scenarios")
    output=dict(schema=1,base=BASE,
                branch_head=subprocess.check_output(["git","rev-parse","HEAD"],
                                                    text=True).strip(),
                platform=sys.platform,system=platform.platform(),
                machine=platform.processor(),
                python=sys.version,numpy=np.__version__,
                numba=__import__("numba").__version__,
                sklearn=__import__("sklearn").__version__,
                source_sha256=raw_sha,candidate_sha256=candidate_sha,
                threads={v:os.environ[v] for v in thread_vars},
                cases=cases)
    Path("numba_rowmax_results.json").write_text(
        json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("PASS 96 original/candidate full-fit CPU pairs, 48 tracemalloc pairs,"
          " 8*201 bitwise potential checkpoints on this host, 16 warmup pairs",
          flush=True)


if __name__=="__main__":
    main()
