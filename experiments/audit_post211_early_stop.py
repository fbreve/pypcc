"""Reproducible ORIGINAL production PCC stop-on/fixed native CPU characterization.

No kernel edits, stop retuning, in-loop callbacks or model monkeypatching.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from benchmarks.fixtures import (Case, FixturePreparer, REAL_DATASETS,
                                 array_hash, prepare_fixture)
from benchmarks.runner import (PARAMETERS, environment_manifest, expected_backend,
                               measure, signatures)

BASE="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("wine","breast_cancer","digits")
SEEDS=(45013,45117)
BACKENDS=("numpy","numba","cython")
MODES=("early_stopping","fixed")
REPS=3
ITERATIONS=30000
PROTECTED=("pcc.py","pcc_graph.py","pcc_numpy.py","pcc_numba.py",
           "pcc_step.pyx","pcc_validation.py","setup.py","benchmarks")
EXPECTED_IDENTITY={
    "numpy":"pcc_numpy.pcc_propagate_numpy",
    "numba":"pcc_numba.pcc_propagate_numba",
    "cython":"pcc_step.pcc_propagate",
}


def sha_numpy_rng():
    st=np.random.get_state()
    return hashlib.sha256(
        st[0].encode() + st[1].tobytes() +
        f"{st[2]}|{st[3]}|{st[4]:.17g}".encode()
    ).hexdigest()


def original_source_guard():
    root=Path(__file__).resolve().parents[1]
    subprocess.run(["git","cat-file","-e",BASE+"^{commit}"],cwd=root,
                   stdout=subprocess.DEVNULL,check=True)
    changed=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        cwd=root,text=True).strip()
    if changed:
        raise AssertionError("Production source or benchmark drift: "+changed)
    for name in BACKENDS:
        expected_backend(name)
    if PARAMETERS!=dict(p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000):
        raise AssertionError("Existing PCC benchmark parameters changed")
    if not all(os.environ.get(v)=="1" for v in
               ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","NUMBA_NUM_THREADS")):
        raise AssertionError("Native thread controls not set to one")
    return root


def native_measure(arrays,case,backend,mode,cap=ITERATIONS,warm=False):
    result=measure(arrays,case,backend,cap,mode=="early_stopping",
                   seed=0,steady_state=not warm)
    if result["backend_identity"]!=EXPECTED_IDENTITY[backend]:
        raise AssertionError("actual backend fallback")
    if result["mode"]!=mode or result["max_iter"]!=cap:
        raise AssertionError("incorrect original run mode or cap")
    if result["propagation_seed_effective"]!=(123456789 if backend=="cython" else 0):
        raise AssertionError("native RNG seed changed")
    if result["stop_reason"] not in ("max_iter","early_stop"):
        raise AssertionError("unrecognized stop reason")
    if not 0<result["effective_iterations"]<=cap:
        raise AssertionError("stop exceeded native cap")
    if not warm and mode=="fixed":
        if (result["effective_iterations"]!=cap or
                result["stop_reason"]!="max_iter"):
            raise AssertionError("fixed trajectory truncated")
    if not warm and mode=="early_stopping" and result["effective_iterations"]<cap:
        if result["stop_reason"]!="early_stop":
            raise AssertionError("premature original stop without reason")
    if warm and result["effective_iterations"]!=2:
        raise AssertionError("short warmup was not 2 steps")
    if set(result["state_hashes"])!={
        "predictions","dominance","positions","strengths","distances","owndeg"}:
        raise AssertionError("native 6-array state missing")
    rng=sha_numpy_rng() if backend=="numpy" else None
    return result,rng


def check_fingerprint(value):
    return isinstance(value,str) and len(value)==64 and all(
        c in "0123456789abcdef" for c in value)


def assess_replay(arrays,case,backend,stopped,stopped_rng,fixed,fixed_rng):
    actual=stopped["effective_iterations"]
    if actual==ITERATIONS:
        candidate,candidate_rng=fixed,fixed_rng
        reason="same_cap"
    else:
        candidate,candidate_rng=native_measure(
            arrays,case,backend,"fixed",cap=actual)
        reason="seeded_prefix_replay"
    for field in ("state_hashes","effective_iterations","accuracy_unlabeled"):
        if stopped[field]!=candidate[field]:
            raise AssertionError(
                "native stopped/fixed-prefix state divergence: "
                +repr((case.case_id,backend,reason,field)))
    if backend=="numpy" and stopped_rng!=candidate_rng:
        raise AssertionError(
            "NumPy next-RNG-state changed after equivalent native prefix")
    return dict(case_id=case.case_id,backend=backend,
                reached_cap=bool(actual==ITERATIONS),
                method=reason,stop_reason=stopped["stop_reason"],
                effective_iterations=actual,
                original_native_prefix_parity=True,
                state_hashes=stopped["state_hashes"],
                fixed_prefix_hashes=candidate["state_hashes"],
                numpy_rng_sha=stopped_rng,
                fixed_prefix_numpy_rng_sha=candidate_rng,
                replay_only_elapsed_s=(
                    candidate["fit_predict_s"] if reason=="seeded_prefix_replay"
                    else None))


def native_layout_probe(arrays,case):
    from pcc import ParticleCompetitionAndCooperation as PCC
    probe={}
    for backend in BACKENDS:
        model=PCC(impl=backend)
        model.set_graph(arrays["neighbors"],arrays["degrees"],k_nn=10)
        actual=model._get_backend_fn()
        if actual.__module__+"."+actual.__name__!=EXPECTED_IDENTITY[backend]:
            raise AssertionError("native backend identity differs in layout probe")
        labels=arrays["labels"]
        # Use the production path without a benchmark wrapper as an independent
        # cross-check for the true native Numba C-order class policy.
        model.fit_predict(labels,max_iter=1,early_stop=True,seed=0)
        dom=model.node.dominance
        layout="C" if dom.flags.c_contiguous else "F"
        expected=("C" if backend=="numba" and case.classes>=8 else "F")
        if layout!=expected:
            raise AssertionError("unexpected native dominance memory order "
                                 +repr((case.name,backend,layout,expected)))
        probe[backend]=dict(backend_identity=EXPECTED_IDENTITY[backend],
                            dominance_order=layout)
    return probe


def run():
    repo=original_source_guard()
    preparer=FixturePreparer(prepare_fixture)
    started=time.perf_counter()
    all_measured=[]
    all_warmups=[]
    parity=[]
    prepared=[]
    layout=[]
    for dataset in DATASETS:
        nodes,features,classes,_=REAL_DATASETS[dataset]
        for label_seed in SEEDS:
            case=Case(dataset,nodes,features,classes,
                      k_nn=10,labels_per_class=3,
                      data_seed=0,label_seed=label_seed)
            arrays,metadata=preparer.prepare(case)
            fp=metadata["fixture_sha256"]
            if metadata["labeled_nodes"]!=3*classes:
                raise AssertionError("wrong label policy or particle count")
            if not np.array_equal(arrays["labels"][arrays["labels"]!=-1],
                                  arrays["truth"][arrays["labels"]!=-1]):
                raise AssertionError("label input source mismatch")
            prepared.append(dict(
                case_id=case.case_id,dataset=dataset,seed=label_seed,
                nodes=nodes,classes=classes,fixture_sha=fp,
                neighbors_sha=array_hash(arrays["neighbors"]),
                degrees_sha=array_hash(arrays["degrees"]),
                truth_sha=array_hash(arrays["truth"]),
                labels_sha=array_hash(arrays["labels"]),
                input_sha=array_hash(arrays["x"]),
                graph_reused=bool(metadata["preparation"]["reused"]),
                original_graph_build_s=metadata["timings"]["graph_build_s"]))
            if label_seed==SEEDS[0]:
                layout.append(dict(dataset=dataset,
                                   native_layout=native_layout_probe(arrays,case)))
            for backend in BACKENDS:
                for first in range(2):
                    w,rng=native_measure(arrays,case,backend,"fixed",cap=2,
                                         warm=True)
                    all_warmups.append(dict(case_id=case.case_id,backend=backend,
                                            index=first,info=w,numpy_rng_sha=rng))
            first_rows={}
            for rep in range(REPS):
                # Latin rotation fully balances which backend executes first.
                order=BACKENDS[rep:]+BACKENDS[:rep]
                mode_order=(MODES if rep%2==0 else MODES[::-1])
                for backend_index,backend in enumerate(order):
                    for mode_index,mode in enumerate(mode_order):
                        info,rng=native_measure(arrays,case,backend,mode)
                        row=dict(case_id=case.case_id,dataset=dataset,
                                 seed=label_seed,backend=backend,mode=mode,
                                 rep=rep,backend_order=list(order),
                                 mode_order=list(mode_order),
                                 backend_order_index=backend_index,
                                 mode_order_index=mode_index,
                                 fixture_sha=fp,numpy_rng_sha=rng,
                                 **info)
                        key=(backend,mode)
                        compare=(info["state_hashes"],info["effective_iterations"],
                                 info["stop_reason"],info["checks"],
                                 info["stalled_checks"],info["stop_max"],
                                 info["accuracy_unlabeled"],rng)
                        if key in first_rows and compare!=first_rows[key]:
                            raise AssertionError("nonreproducible measured fit: "
                                                 +repr((case.case_id,key)))
                        first_rows.setdefault(key,compare)
                        all_measured.append(row)
                        print(f"{case.case_id} {backend} {mode} rep={rep}: "
                              f"iter={info['effective_iterations']} "
                              f"reason={info['stop_reason']} "
                              f"full_ms={1000*info['fit_predict_s']:.3f}",flush=True)
            for backend in BACKENDS:
                selected={(r["mode"],r["rep"]):r for r in all_measured
                          if r["case_id"]==case.case_id and r["backend"]==backend}
                stopped=selected[("early_stopping",0)]
                fixed=selected[("fixed",0)]
                parity.append(assess_replay(
                    arrays,case,backend,stopped,stopped["numpy_rng_sha"],
                    fixed,fixed["numpy_rng_sha"]))
            print(f"PASS {case.case_id}: 18 full native measured fits,"
                  f" 6 warmups, 3 stopped-prefix assessments",flush=True)
    if (len(all_measured)!=108 or len(all_warmups)!=36
            or len(parity)!=18 or len(prepared)!=6 or len(layout)!=3):
        raise AssertionError("incomplete frozen current-master cohort")
    manifest=environment_manifest(dict(
        experiment="post211_early_stop_native_baseline_v1",
        base_commit=BASE,datasets=DATASETS,
        seeds=SEEDS,backends=BACKENDS,
        iterations=ITERATIONS,es_chk=2000,
        modes=MODES,repeats=REPS,parameters=PARAMETERS),
        list(BACKENDS))
    if manifest["repository"]["commit"]==BASE:
        raise AssertionError("experiment must run source-pinned research branch")
    if any(not check_fingerprint(a["fixture_sha"]) for a in prepared):
        raise AssertionError("incomplete fixture hash")
    output=dict(schema=1,base_commit=BASE,
                branch_head=subprocess.check_output(
                    ["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
                os=sys.platform,system=platform.platform(),
                python=sys.version,numpy=np.__version__,
                manifest=manifest,prepared=prepared,
                layout_probes=layout,measured=all_measured,
                warmups=all_warmups,parity=parity,
                experiment_wall_seconds=time.perf_counter()-started)
    filename=Path("post211_early_stop_result.json")
    filename.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("COMPLETE CURRENT-MASTER NATIVE EARLY-STOP BASELINE:",
          len(all_measured),"timed fits",len(all_warmups),"warmups",
          len(parity),"seeded prefix checks",flush=True)


if __name__=="__main__":
    run()
