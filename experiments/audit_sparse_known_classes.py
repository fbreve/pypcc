"""Real original-Git PCC known-class discovery: source-paired native CPU audit.

No production changes. Actual pcc.py from pinned Git and uniquely derived
single-hunk candidate are imported independently as full PCC classes.
Windows CPU samples use preregistered full repeated batches.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import tracemalloc
import warnings

import numpy as np
from numba import njit
from benchmarks.fixtures import (Case, REAL_DATASETS,
                                 FixturePreparer, prepare_fixture, array_hash)

BASE="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
PROTECTED=("pcc.py","pcc_numba.py","pcc_numpy.py","pcc_graph.py",
           "pcc_step.pyx","pcc_validation.py","setup.py","benchmarks")
OLD_BLOCK=("        prepared_classes = np.unique(prepared_labels)\n"
           "        prepared_classes = prepared_classes[prepared_classes != -1]")
NEW_BLOCK=("        if prepared_labels.size >= 1024:\n"
           "            prepared_classes = np.unique(prepared_labels[prepared_labels != -1])\n"
           "        else:\n"
           "            prepared_classes = np.unique(prepared_labels)\n"
           "            prepared_classes = prepared_classes[prepared_classes != -1]")
SEEDS=(78013,78117)
BACKENDS=("numpy","numba","cython")
CPU_PAIRS_MICRO=25
PEAK_PAIRS_MICRO=5
FULL_PAIRS_REAL=2
FULL_FITS_REAL={"numpy":4,"numba":32,"cython":64}
FULL_PAIRS_RING=2
FULL_FITS_RING=32
FULL_FITS_DENSE=128
RING_SIZES=(30000,100000)
RING_CLASSES=(2,10,50)
RING_CAPS=(0,5)
RNG_INFO=("effective_iterations","stop_reason","checks",
          "stalled_checks","stop_max")
STATE_KEYS=("predictions","dominance","positions","strengths","distances","owndeg")

@njit
def numba_rng_next():
    return np.random.random()


def ensure(ok,why):
    if not ok:
        raise AssertionError(why)


def guard_source():
    subprocess.run(["git","cat-file","-e",BASE+"^{commit}"],
                   stdout=subprocess.DEVNULL,check=True)
    drift=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        text=True).strip()
    ensure(not drift,"science changed original production "+drift)
    original=subprocess.check_output(
        ["git","show",BASE+":pcc.py"]).decode("utf-8").replace("\r\n","\n")
    disk=Path("pcc.py").read_text(encoding="utf-8").replace("\r\n","\n")
    ensure(disk==original,"original production PCC does not match frozen Git master")
    ensure(original.count(OLD_BLOCK)==1 and NEW_BLOCK not in original,
           "original two-line np.unique source block not unique")
    candidate=original.replace(OLD_BLOCK,NEW_BLOCK)
    ensure(candidate.replace(NEW_BLOCK,OLD_BLOCK)==original
           and candidate.count(NEW_BLOCK)==1,
           "candidate source changed more than one guarded pcc.py fragment")
    return original,candidate


def load_true_pcc(path,module_name):
    spec=importlib.util.spec_from_file_location(module_name,str(path.resolve()))
    ensure(spec is not None and spec.loader is not None,
           "not a real source-backed original PCC module")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=mod
    spec.loader.exec_module(mod)
    return mod.ParticleCompetitionAndCooperation


def compiled_phase(source):
    block=NEW_BLOCK if NEW_BLOCK in source else OLD_BLOCK
    fragment=textwrap.dedent(block)
    return compile(fragment,"<exact-original-source-class-discovery>","exec")


def label_values(c):
    # -1 is reserved for unknown; test negative/zero/large non-sentinel.
    return np.asarray([
        (-10 if i==0 else 0 if i==1 else (1 << 40)+11*i)
        for i in range(c)],dtype=np.int64)


def deterministic_labels(n,c,kind):
    labels=np.full(n,-1,dtype=np.int64)
    if kind=="unlabeled":
        return labels
    codes=label_values(c)
    if kind=="single":
        labels[n//3]=codes[0]
        return labels
    if kind=="dense":
        count=n//2
    else:
        count=3*c
    ensure(count<=n and count>=c,
           "invalid prespecified sparse/dense labels")
    places=np.linspace(0,n-1,count,dtype=np.int64)
    ensure(np.unique(places).size==count,"duplicate deterministic homes")
    labels[places]=codes[np.arange(count)%c]
    ensure(np.unique(labels[labels!=-1]).size==c,
           "not each original class represented")
    return labels


MICRO_FIXTURES=(
    (1023,2,"sparse"),(1024,2,"sparse"),(1797,10,"sparse"),
    (30000,10,"sparse"),(30000,50,"sparse"),
    (100000,10,"sparse"),(100000,50,"sparse"),
    (300000,10,"sparse"),(300000,50,"sparse"),
    (100000,10,"dense"),(100000,10,"unlabeled"),
    (100000,10,"single"),
)


def measure_micro(original, candidate, source_sha):
    controls={"original":compiled_phase(original),
              "candidate":compiled_phase(candidate)}
    result=[]
    for n,c,kind in MICRO_FIXTURES:
        values=deterministic_labels(n,c,kind)
        checksum=array_hash(values)
        repetitions=2048 if n<=1797 else 256
        timers=[]
        peaks=[]
        for pair in range(CPU_PAIRS_MICRO):
            order=("candidate","original") if pair%2==0 else (
                "original","candidate")
            records={}
            for variant in order:
                namespace={"np":np,"prepared_labels":values}
                before=time.process_time_ns()
                for _ in range(repetitions):
                    exec(controls[variant],namespace)
                ns=time.process_time_ns()-before
                out=namespace["prepared_classes"]
                check_expected=np.unique(values)
                check_expected=check_expected[check_expected!=-1]
                ensure(np.array_equal(out,check_expected) and
                       out.dtype==np.int64 and
                       out.flags.c_contiguous,
                       "source compiled original/candidate classes/dtype diverged")
                ensure(ns>0,"Windows CPU counter did not resolve preregistered batch")
                records[variant]=dict(total_cpu_ns=ns,
                                      per_class_discovery_cpu_ns=ns/repetitions,
                                      classes_sha=array_hash(out),
                                      count=int(out.size))
            ensure(records["original"]["classes_sha"]==
                   records["candidate"]["classes_sha"],
                   "source paired complete class mapping changed")
            timers.append(dict(rep=pair,order=list(order),**records))
        for pair in range(PEAK_PAIRS_MICRO):
            track={}
            for variant in (("candidate","original") if pair%2==0
                            else ("original","candidate")):
                namespace={"np":np,"prepared_labels":values}
                tracemalloc.start()
                try:
                    exec(controls[variant],namespace)
                    peak=tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                ensure(peak>=0,"invalid Python traced class extraction peak")
                track[variant]=int(peak)
            ensure(track["candidate"]-track["original"]<=
                   16*n+1048576,
                   "unexpected unbounded CPU-for-RAM class extraction")
            peaks.append(dict(rep=pair,**track))
        ratio=statistics.median(
            p["candidate"]["total_cpu_ns"]/
            p["original"]["total_cpu_ns"] for p in timers)
        print(f"MICRO n={n} c={c} kind={kind} paired CPU saved "
              f"{100*(1-ratio):+.2f}%",flush=True)
        ensure(array_hash(values)==checksum,
               "class discovery mutated original labels")
        result.append(dict(n=n,classes=c,kind=kind,
                           input_sha=checksum,
                           outputs_sha=timers[0]["original"]["classes_sha"],
                           repetitions=repetitions,
                           timings=timers,peak_pairs=peaks))
    return result


def ring_graph(n):
    indices=np.arange(n,dtype=np.int64)
    adj=np.empty((n,2),dtype=np.int64)
    adj[:,0]=(indices-1)%n
    adj[:,1]=(indices+1)%n
    degree=np.full(n,2,dtype=np.int64)
    ensure(n>2 and
           np.array_equal(adj[(indices+1)%n,0],indices) and
           np.array_equal(adj[(indices-1)%n,1],indices),
           "ring must be undirected/reciprocal")
    return adj,degree


def make_model(PCC,neighbors,degrees,k,backend):
    model=PCC(impl=backend,update_mode="sequential")
    model.set_graph(neighbors,degrees,k_nn=k)
    native=model._get_backend_fn()
    ensure(native.__name__=={
        "numpy":"pcc_propagate_numpy",
        "numba":"pcc_propagate_numba",
        "cython":"pcc_propagate"
    }[backend],"original PCC native backend silently fell back")
    info={}
    def observer(*args,**kwargs):
        kwargs["return_info"]=True
        details=native(*args,**kwargs)
        ensure(isinstance(details,dict),"native backend did not report original stop information")
        info.clear()
        info.update(details)
        return None
    observer.__name__=native.__name__
    model._get_backend_fn=lambda:observer
    return model,info,native.__name__


def result_states(model,pred,info,backend,cap,mode):
    expected=(("C" if backend=="numba" and model.c>=8
              else "F"))
    layout=("C" if model.node.dominance.flags.c_contiguous
            else "F")
    ensure(layout==expected,"Numba >=8 classes original C/F dominance policy changed")
    ensure(info["stop_reason"] in ("max_iter","early_stop") and
           isinstance(info["effective_iterations"],int) and
           0<=info["effective_iterations"]<=cap,
           "invalid original native stop records")
    ensure(info["effective_iterations"]==cap if not mode else True,
           "fixed native original iterations truncated")
    ensure(info["effective_iterations"]==0 if cap==0 else
           info["effective_iterations"]>0,
           "incorrect zero / positive iteration count")
    if backend=="numba":
        rng=float(numba_rng_next())
    elif backend=="numpy":
        rng=float(np.random.random())
    else:
        rng=None
    state={
        "predictions":array_hash(pred),
        "dominance":array_hash(model.node.dominance),
        "positions":array_hash(model.part.curnode),
        "strengths":array_hash(model.part.strength),
        "distances":array_hash(model.part.dist_table),
        "owndeg":array_hash(model.owndeg),
    }
    return dict(
        state_hashes=state,
        mapped_labels_sha=array_hash(model.mapped_labels),
        sorted_known_classes_sha=array_hash(model.unique_labels),
        label_to_idx={str(int(k)):int(v) for k,v in model.label_to_idx.items()},
        info={key:info[key] for key in RNG_INFO},
        numba_or_numpy_next_rng=rng,
        native_dominance_layout=layout,
        n_particles=int(model.part.amount),
        native_backend=backend,cap=cap,
        monitored=mode)


def run_batch(PCC,neighbors,degrees,k,labels,backend,cap,mode,count):
    # Create all eight independently initialized objects/graphs BEFORE timer,
    # then discard their large fit state one by one to bound peak RAM.
    models=[make_model(PCC,neighbors,degrees,k,backend)
            for _ in range(count)]
    preds=[]
    sampled_stop=[]
    last_model=None
    last_info=None
    t_cpu=time.process_time_ns()
    t_wall=time.perf_counter_ns()
    for i in range(count):
        model,info,native_name=models[i]
        predictions=model.fit_predict(
            labels,seed=0,max_iter=cap,early_stop=mode,
            p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)
        preds.append(predictions)
        sampled_stop.append(dict(info))
        if i==count-1:
            last_model,last_info=model,info
        # Drop the previous full dominance, distance, owndeg immediately.
        models[i]=None
        del model,info
    elapsed_wall=time.perf_counter_ns()-t_wall
    elapsed_cpu=time.process_time_ns()-t_cpu
    ensure(elapsed_wall>0 and elapsed_cpu>0,
           "actual CPU/wall 4/8-full-fit source batch must be measurable")
    last=result_states(last_model,preds[-1],last_info,
                       backend,cap,mode)
    all_predictions=[array_hash(p) for p in preds]
    ensure(all(x==last["state_hashes"]["predictions"]
               for x in all_predictions),
           "independently initialized original fits returned different labels")
    ensure(all(s==sampled_stop[0] for s in sampled_stop),
           "independently initialized original native stop counters differed")
    return dict(count=count,wall_ns=elapsed_wall,cpu_ns=elapsed_cpu,
                cpu_per_fit_ns=elapsed_cpu/count,
                wall_per_fit_ns=elapsed_wall/count,
                all_predictions_sha=all_predictions,
                last=last)


def exact_pair(a,b,description):
    # Compare every full native state array, class mapping, backend RNG and
    # actual propagation stop counters, independent of timing/memory.
    ensure(a["last"]==b["last"] and
           a["all_predictions_sha"]==b["all_predictions_sha"],
           "different source full fit state, RNG or stop "+repr(description))


def warm_backend(PCC,neighbors,degrees,k,labels,backend):
    model,info,_=make_model(PCC,neighbors,degrees,k,backend)
    model.fit_predict(labels,max_iter=2,early_stop=False,seed=0)
    ensure(info["effective_iterations"]==2,
           "original JIT/native backend wasn't warmed")


def real_native_trials(classes):
    preparer=FixturePreparer(prepare_fixture)
    rows=[]
    for ds in ("iris","wine","breast_cancer","digits"):
        n,f,c,_=REAL_DATASETS[ds]
        for seed in SEEDS:
            case=Case(ds,n,f,c,k_nn=10,labels_per_class=3,
                      label_seed=seed,data_seed=0)
            arr,meta=preparer.prepare(case)
            ensure(meta["labeled_nodes"]==3*c,"not original 3 labels per class")
            for backend in BACKENDS:
                for mode in (False,True):
                    # Include backend warmup outside measured fit.
                    for PCC in classes.values():
                        warm_backend(PCC,arr["neighbors"],arr["degrees"],
                                     10,arr["labels"],backend)
                    for rep in range(FULL_PAIRS_REAL):
                        order=("candidate","original") if rep%2==0 else (
                            "original","candidate")
                        results={}
                        for variant in order:
                            results[variant]=run_batch(
                                classes[variant],arr["neighbors"],
                                arr["degrees"],10,arr["labels"],
                                backend,1000,mode,FULL_FITS_REAL[backend])
                        exact_pair(results["original"],results["candidate"],
                                   (ds,seed,backend,mode,rep))
                        rows.append(dict(dataset=ds,seed=seed,kind="real",
                            nodes=n,classes=c,backend=backend,cap=1000,
                            monitored=mode,rep=rep,order=list(order),
                            fixture_sha=meta["fixture_sha256"],
                            graph_sha=[array_hash(arr[x])
                                       for x in ("neighbors","degrees")],
                            labels_sha=array_hash(arr["labels"]),
                            **results))
            print("REAL COMPLETE",case.case_id,flush=True)
    return rows


def synthetic_native_trials(classes):
    rows=[]
    for n in RING_SIZES:
        neighbors,degrees=ring_graph(n)
        graph_sha=[array_hash(neighbors),array_hash(degrees)]
        for c in RING_CLASSES:
            labels=deterministic_labels(n,c,"sparse")
            for backend in ("numpy","numba"):
                for cap in RING_CAPS:
                    # Original backend JIT uses separate F/C specialization;
                    # all signatures warmed before original CPU sampling.
                    if backend=="numba":
                        for PCC in classes.values():
                            warm_backend(PCC,neighbors,degrees,2,labels,backend)
                    for rep in range(FULL_PAIRS_RING):
                        order=("candidate","original") if rep%2==0 else (
                            "original","candidate")
                        results={}
                        for variant in order:
                            results[variant]=run_batch(
                                classes[variant],neighbors,degrees,2,labels,
                                backend,cap,False,FULL_FITS_RING)
                        exact_pair(results["original"],results["candidate"],
                                   ("ring",n,c,backend,cap,rep))
                        rows.append(dict(dataset=f"ring_{n}_{c}",seed=0,
                            kind="ring",nodes=n,classes=c,backend=backend,
                            cap=cap,monitored=False,rep=rep,order=list(order),
                            graph_sha=graph_sha,labels_sha=array_hash(labels),
                            **results))
            print("RING COMPLETE",n,c,flush=True)
    return rows


def sparse_and_dense_control(classes):
    rows=[]
    for n in (1024,1797):
        neighbors,degrees=ring_graph(n)
        labels=deterministic_labels(n,2,"dense")
        for variant,PCC in classes.items():
            warm_backend(PCC,neighbors,degrees,2,labels,"numba")
        results={}
        for variant in ("candidate","original"):
            results[variant]=run_batch(
                classes[variant],neighbors,degrees,2,labels,
                "numba",0,False,FULL_FITS_DENSE)
        exact_pair(results["original"],results["candidate"],
                   ("small-dense",n))
        rows.append(dict(n=n,kind="dense",count=FULL_FITS_DENSE,
                         graph_sha=[array_hash(neighbors),array_hash(degrees)],
                         labels_sha=array_hash(labels),**results))
    # Original no-fit must return SAME user's labels object, warn exactly
    # once, and clear old state. Both source classes must do so identically.
    no_class=[]
    n=100000
    neighbors,degrees=ring_graph(n)
    for kind in ("unlabeled","single"):
        labels=deterministic_labels(n,10,kind)
        evidence=[]
        for variant,PCC in classes.items():
            model=PCC(impl="numba")
            model.set_graph(neighbors,degrees,k_nn=2)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                pred=model.fit_predict(labels,max_iter=0,seed=0)
            ensure(pred is labels and len(captured)==1 and
                   model.node is None and model.part is None,
                   "no-class original public identity/warning/clear state changed")
            evidence.append(dict(variant=variant,returned_same_input=True,
                                 warning=str(captured[0].message),
                                 node_none=model.node is None,
                                 particle_none=model.part is None))
        ensure(evidence[0]["warning"]==evidence[1]["warning"],
               "no-label original/candidate warning text drift")
        no_class.append(dict(kind=kind,n=n,labels_sha=array_hash(labels),
                             both=evidence))
    return rows,no_class


def preflight():
    """Cheap structural and actual native smoke check before full campaign."""
    original,candidate=guard_source()
    source_funcs={"original":compiled_phase(original),
                  "candidate":compiled_phase(candidate)}
    for n,c,kind in ((24,2,"sparse"),(1023,2,"sparse"),
                     (1024,10,"sparse"),(1024,10,"unlabeled"),
                     (1024,10,"single")):
        values=deterministic_labels(n,c,kind)
        seen={}
        for name,block in source_funcs.items():
            env=dict(np=np,prepared_labels=values)
            exec(block,env)
            seen[name]=env["prepared_classes"]
        ensure(np.array_equal(seen["original"],seen["candidate"]),
               "source-derived sparse class microphase smoke mismatch")
    with tempfile.TemporaryDirectory(prefix="pypcc_classes_smoke_") as tmp:
        modules={}
        for name,source in (("original",original),("candidate",candidate)):
            path=Path(tmp)/(name+".py")
            path.write_text(source,encoding="utf-8",newline="\n")
            modules[name]=load_true_pcc(path,"science_smoke_"+name)
        adj,degree=ring_graph(24)
        labels=deterministic_labels(24,2,"sparse")
        for backend in BACKENDS:
            pair={}
            for name,PCC in modules.items():
                pair[name]=run_batch(PCC,adj,degree,2,labels,backend,2,
                                     False,1)
            exact_pair(pair["original"],pair["candidate"],
                       ("smoke",backend))
    print("SMOKE ORIGINAL GIT PCC SOURCE PAIR and real 3-backend "
          "two-iteration six native states/RNG/stop PASS",flush=True)


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    required=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
              "MKL_NUM_THREADS","NUMBA_NUM_THREADS")
    ensure(all(os.environ.get(x)=="1" for x in required),
           "single-thread NumPy and native Cython/Numba required")
    old,new=guard_source()
    with tempfile.TemporaryDirectory(prefix="pypcc_class_unique_") as temp:
        classes={}
        source_sha={}
        for name,source in (("original",old),("candidate",new)):
            path=Path(temp)/name
            path.mkdir()
            target=path/f"pcc_{name}_class_unique.py"
            target.write_text(source,encoding="utf-8",newline="\n")
            source_sha[name]=hashlib.sha256(target.read_bytes()).hexdigest()
            ensure(source_sha[name]==hashlib.sha256(source.encode()).hexdigest(),
                   "candidate/original source disk SHA differs from exact Git source")
            classes[name]=load_true_pcc(target,f"experiment_{name}_pcc_class")
        micro=measure_micro(old,new,source_sha)
        real=real_native_trials(classes)
        rings=synthetic_native_trials(classes)
        dense,noclass=sparse_and_dense_control(classes)
    ensure(len(micro)==12 and len(real)==96 and len(rings)==48
           and len(dense)==2 and len(noclass)==2,
           "missing one of the prespecified per-OS source-paired cohorts")
    payload=dict(schema=1,base=BASE,
        head=subprocess.check_output(["git","rev-parse","HEAD"],
                                      text=True).strip(),
        platform=sys.platform,host=platform.platform(),
        python=sys.version,
        numpy=np.__version__,numba=__import__("numba").__version__,
        source_sha256=source_sha["original"],
        candidate_sha256=source_sha["candidate"],
        threads={key:os.environ[key] for key in required},
        micro_cases=micro,real_pairs=real,ring_pairs=rings,
        dense_controls=dense,no_class_controls=noclass)
    Path("sparse_known_classes_results.json").write_text(
        json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("STRICT SOURCE PAIRED HOST COMPLETE: 12 micro fixtures×25 CPU "
          "pairs + 5 peak pairs, 96 real native source pairs, "
          "48 0/5-step ring model native source pairs, "
          "2 dense real full CPU controls and 2 zero/one-class "
          "identity-return controls PASS",flush=True)


if __name__=="__main__":
    if "--smoke-only" in sys.argv:
        preflight()
    else:
        main()
