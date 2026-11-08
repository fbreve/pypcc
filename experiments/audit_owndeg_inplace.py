"""Preregistered exact original PCC full fit versus in-place owndeg postprocess.

All output files are plain JSON and an exact provenance package inventory.
Original model source comes from the actual immutable master Git blob, not
a copied approximation. Neither graph nor native propagation is changed.
"""
import argparse
import gc
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

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import benchmarks.runner as runner
from benchmarks.fixtures import Case, array_hash, select_labels
from pcc_graph import build_knn_graph
from pcc import ParticleCompetitionAndCooperation as CandidatePCC

MASTER="ababc48986b40f083d5930e922f844aa2ea418b2"
DATASETS=("iris","wine","breast_cancer","digits")
LABEL_SEEDS=(32003,32027)
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
SIZES=((0,2),(178,3),(569,2),(1797,10),(30000,10),(30000,50))
MICRO_SEED=31991
TIMING_REPS=40
MEMORY_REPS=5
ITERATIONS=1000
SHAS=("predictions","dominance","positions","strengths","distances","owndeg")


def check(ok,why):
    if not ok:
        raise AssertionError(why)


def root():
    return Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git",*args],cwd=root(),text=True).strip()


def original_model():
    code=subprocess.check_output(
        ["git","show",f"{MASTER}:pcc.py"],cwd=root(),text=True,encoding="utf-8")
    old="        self.owndeg = self.owndeg / row_sums"
    new="        np.divide(self.owndeg, row_sums, out=self.owndeg)"
    check(code.count(old)==1,
          "Frozen original master's owndeg postprocessing changed")
    actual=(root()/"pcc.py").read_text(encoding="utf-8")
    check(actual==code.replace(old,new),
          "The candidate must contain precisely the one-line approved edit")
    untouched=("pcc_graph.py","pcc_numpy.py","pcc_numba.py",
               "pcc_step.pyx","pcc_validation.py","benchmarks","setup.py")
    check(not git("diff","--name-only",MASTER,"HEAD","--",*untouched),
          "Original graph/kernel/benchmark/packaging source changed")
    module=types.ModuleType("_frozen_pcc_original_normalization")
    exec(compile(code,f"{MASTER}:pcc.py","exec"),module.__dict__)
    return module.ParticleCompetitionAndCooperation,hashlib.sha256(
        code.encode("utf-8")).hexdigest()


def real_fixtures():
    result={}
    for name in DATASETS:
        raw={
            "iris":datasets.load_iris,"wine":datasets.load_wine,
            "breast_cancer":datasets.load_breast_cancer,
            "digits":datasets.load_digits,
        }[name]()
        x=StandardScaler().fit_transform(
            np.asarray(raw.data,dtype=np.float64))
        truth=np.asarray(raw.target,dtype=np.int64)
        graph=build_knn_graph(x,10,n_jobs=1)
        result[name]=dict(
            x=x,truth=truth,neighbors=graph[0],degrees=graph[1],
            data_hash=array_hash(x),graph_hashes=dict(
                neighbors=array_hash(graph[0]),degrees=array_hash(graph[1])))
    return result


def pair_fits(Original,fixtures):
    old_runner_model=runner.PCC
    rows=[]
    try:
        for dataset in DATASETS:
            raw=fixtures[dataset]
            for label_seed in LABEL_SEEDS:
                case=Case(dataset,len(raw["x"]),raw["x"].shape[1],
                          int(len(np.unique(raw["truth"]))),k_nn=10,
                          labels_per_class=3,data_seed=0,
                          label_seed=label_seed)
                labels=select_labels(raw["truth"],case)
                arrays=dict(truth=raw["truth"],labels=labels,
                            neighbors=raw["neighbors"],degrees=raw["degrees"])
                for backend in BACKENDS:
                    if backend=="numba":
                        runner.PCC=CandidatePCC
                        runner.measure(arrays,case,backend,2,False,0,
                                       steady_state=False)
                    for mode in MODES:
                        early=(mode=="early_stopping")
                        old=lambda:runner.measure(
                            arrays,case,backend,ITERATIONS,early,0,
                            steady_state=True)
                        # Swap only the class constructor in unmodified
                        # existing timing runner. Every run creates a
                        # fresh separate model and re-seeds the backend.
                        first_is_old=(len(rows)%2==0)
                        def call(Model):
                            runner.PCC=Model
                            return old()
                        a=call(Original if first_is_old else CandidatePCC)
                        b=call(CandidatePCC if first_is_old else Original)
                        original,candidate=(a,b) if first_is_old else (b,a)
                        fields=("state_hashes","effective_iterations",
                                "stop_reason","checks","stalled_checks",
                                "stop_max","backend_identity",
                                "accuracy_unlabeled")
                        check(all(original.get(f)==candidate.get(f)
                                  for f in fields),
                              f"State mismatch: {dataset}/{label_seed}/{backend}/{mode}")
                        check(set(original["state_hashes"])==set(SHAS),
                              "Missing original six end-state arrays")
                        timings=("initialization_and_dispatch_s",
                                 "propagation_s","postprocessing_s",
                                 "fit_predict_s")
                        row=dict(
                            dataset=dataset,seed=label_seed,backend=backend,
                            mode=mode,order="original_first"
                            if first_is_old else "candidate_first",
                            label_sha256=array_hash(labels),
                            graph_sha256=raw["graph_hashes"],
                            iterations=original["effective_iterations"],
                            stop_reason=original["stop_reason"],
                            backend_identity=original["backend_identity"],
                            accuracy=original["accuracy_unlabeled"],
                            state_hashes=original["state_hashes"],
                            original_times={f:original[f] for f in timings},
                            candidate_times={f:candidate[f] for f in timings},
                            exact=True)
                        check(all(row[x][f]>0 for x in (
                            "original_times","candidate_times") for f in timings),
                            "Nonpositive recorded full fit phase")
                        rows.append(row)
            print("full state/fit",dataset,
                  sum(r["dataset"]==dataset for r in rows),
                  "matched backend pairs PASS",flush=True)
    finally:
        runner.PCC=old_runner_model
    check(len(rows)==48,"Missing 48 original native fit state pairs")
    return rows


def make_inputs():
    rng=np.random.default_rng(MICRO_SEED)
    base={}
    for n,c in SIZES:
        arr=np.asfortranarray(rng.random((n,c),dtype=np.float64))
        arr[::7,:]=0.0
        # Retain original F-layout and a mix of rows with zero total.
        check(arr.dtype==np.float64 and arr.flags.f_contiguous,
              "Incorrect original normalization input shape or dtype")
        base[f"{n}x{c}"]=arr
    return base


def normalize(data,inplace):
    x=data.copy(order="F")
    row_sums=np.sum(x,axis=1,keepdims=True)
    row_sums[row_sums==0.0]=1.0
    if inplace:
        np.divide(x,row_sums,out=x)
        return x
    return x/row_sums


def one_trace(fn):
    gc.collect()
    tracemalloc.start()
    try:
        result=fn()
        current,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    check(isinstance(result,np.ndarray) and peak>=current>=0,
          "Invalid independently traced output peak")
    out=dict(
        peak_bytes=int(peak),current_bytes=int(current),
        returned_bytes=int(result.nbytes),
        result_sha256=array_hash(result))
    del result
    return out


def micro():
    inputs=make_inputs()
    timings=[]; memory=[]; fixture=[]
    for name,x in inputs.items():
        original=lambda x=x:normalize(x,False)
        candidate=lambda x=x:normalize(x,True)
        warm_a,warm_b=original(),candidate()
        check(np.array_equal(warm_a,warm_b)
              and warm_a.dtype==warm_b.dtype
              and warm_a.flags.f_contiguous==warm_b.flags.f_contiguous,
              f"Normalization numeric/layout mismatch {name}")
        fingerprint=array_hash(warm_a)
        fixture.append(dict(name=name,shape=list(x.shape),
                            input_sha256=array_hash(x),
                            output_sha256=fingerprint,
                            returned_bytes=int(warm_a.nbytes)))
        for rep in range(TIMING_REPS):
            funcs=(original,candidate) if rep%2==0 else (candidate,original)
            t0=time.perf_counter_ns()
            a=funcs[0]()
            t1=time.perf_counter_ns()
            b=funcs[1]()
            t2=time.perf_counter_ns()
            ref,cand=(a,b) if rep%2==0 else (b,a)
            check(np.array_equal(ref,cand)
                  and array_hash(ref)==fingerprint
                  and array_hash(cand)==fingerprint,
                  f"Timed normalization changed output: {name}/{rep}")
            orig_ns,alt_ns=((t1-t0,t2-t1) if rep%2==0 else (t2-t1,t1-t0))
            check(orig_ns>0 and alt_ns>0,
                  f"Invalid elapsed time: {name}/{rep}")
            timings.append(dict(name=name,repeat=rep,
                                order="original_first" if rep%2==0
                                else "candidate_first",
                                original_ns=orig_ns,candidate_ns=alt_ns))
        for rep in range(MEMORY_REPS):
            funcs=(original,candidate) if rep%2==0 else (candidate,original)
            a,b=one_trace(funcs[0]),one_trace(funcs[1])
            ref,cand=(a,b) if rep%2==0 else (b,a)
            check(ref["result_sha256"]==cand["result_sha256"]==fingerprint,
                  f"Traced output diverged: {name}/{rep}")
            memory.append(dict(name=name,repeat=rep,
                               order="original_first" if rep%2==0
                               else "candidate_first",
                               original=ref,candidate=cand))
        print("normalization",name,
              "40 exact timed + 5 exact traced paired cases PASS",flush=True)
    check(len(timings)==len(SIZES)*TIMING_REPS and
          len(memory)==len(SIZES)*MEMORY_REPS,
          "Incomplete microbenchmark workload")
    return fixture,timings,memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    Original,src_hash=original_model()
    fixtures=real_fixtures()
    fitted=pair_fits(Original,fixtures)
    micro_fixtures,times,peaks=micro()
    result=dict(
        protocol="owndeg_inplace_v1",
        master=MASTER,head=git("rev-parse","HEAD"),
        original_pcc_sha256=src_hash,
        host=dict(os=platform.platform(),python=sys.version,
                  cpu=platform.processor(),numpy=np.__version__,
                  thread_env={x:os.getenv(x) for x in (
                      "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                      "MKL_NUM_THREADS","NUMBA_NUM_THREADS")}),
        datasets=list(DATASETS),label_seeds=list(LABEL_SEEDS),
        backends=list(BACKENDS),modes=list(MODES),
        iterations=ITERATIONS,micro_seed=MICRO_SEED,
        micro_sizes=[list(x) for x in SIZES],
        timing_repetitions=TIMING_REPS,
        memory_repetitions=MEMORY_REPS,
        real_inputs={
            name:{k:v for k,v in d.items()
                  if k in ("data_hash","graph_hashes")}
            for name,d in fixtures.items()},
        native_fit_pairs=fitted,micro_fixtures=micro_fixtures,
        micro_time_pairs=times,micro_memory_pairs=peaks)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,
                                      allow_nan=False)+"\n",encoding="utf-8")
    print("COMPLETE: 48 complete exact native fits, 240 micro CPU pairs, "
          "30 isolated peak memory pairs",flush=True)


if __name__=="__main__":
    main()
