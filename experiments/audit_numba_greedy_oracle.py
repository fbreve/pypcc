"""Source-exact original Numba greedy-event recording and oracle replay.

Scientific counterfactual only: tape is recorded OUTSIDE timing from actual
weighted greedy path, while the timed oracle still consumes exactly the actual
original Numba RNG draws and preserves full PCC state. No production change.
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
import types

import numpy as np
from numba import njit
import pcc as prod
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numba import pcc_propagate_numba as ORIGINAL
from benchmarks.fixtures import (Case, REAL_DATASETS, FixturePreparer,
                                 prepare_fixture, array_hash)
from benchmarks.runner import expected_backend

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
PROTECTED=("pcc.py","pcc_numba.py","pcc_graph.py","pcc_numpy.py",
           "pcc_validation.py","pcc_step.pyx")
GEOMETRIES=(("breast_cancer",10),("digits",3),("digits",10),
            ("ring_8192_32",2))
SEEDS=(89021,89119)
POLICIES=(0.,.5,1.)
MODES=(("fixed",False),("monitored",True))
CAP=10000
BATCH=8
REPS=4
ARGS=dict(seed=29,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000)
STATES=("predictions","dominance","positions","strengths",
        "distances","owndeg","node_labels","mapped_labels")
EXTRA="science_meta, science_total, science_uniform, science_weighted, science_cursor"


@njit
def after_rng():
    return np.random.random()


def unique(source,old,new):
    count=source.count(old)
    assert count==1,("source-derived guarded anchor count",count,old[:95])
    return source.replace(old,new,1)


def git_guard_and_clones():
    root=Path(__file__).resolve().parents[1]
    modified=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        cwd=root,text=True).strip()
    assert not modified,modified
    source_hashes={}
    for path in PROTECTED:
        disk=(root/path).read_bytes()
        original=subprocess.check_output(["git","show",BASE+":"+path],
                                         cwd=root)
        assert disk.replace(b"\r\n",b"\n")==original.replace(b"\r\n",b"\n"),path
        source_hashes[path]=hashlib.sha256(original).hexdigest()
    assert prod.pcc_propagate_numba is ORIGINAL
    assert PCC(impl="numba")._get_backend_fn() is expected_backend("numba") is ORIGINAL
    original=subprocess.check_output(
        ["git","show",BASE+":pcc_numba.py"],cwd=root).decode("utf-8")
    begin="@njit(cache=True)\ndef _pcc_step_numba_sequential("
    finish="@njit(cache=True)\ndef _pcc_calc_mmpot_sequential("
    assert original.count(begin)==original.count(finish)==1
    step=original[original.index(begin):original.index(finish)]
    common=unique(step,
        "                               dist_weights):\n    n_particles",
        "                               dist_weights, "+EXTRA+"):\n    n_particles")
    recorder=unique(common,
        "        if use_greedy:\n            # greedy:",
        "        if use_greedy:\n            choice = -1\n            rand_val = -1.0\n            # greedy:")
    snippet=(
        "        if use_greedy:\n"
        "            science_i = science_cursor[0]\n"
        "            if science_i >= science_meta.shape[0]:\n"
        "                raise ValueError('greedy tape overflow')\n"
        "            science_meta[science_i, 0] = p_i\n"
        "            science_meta[science_i, 1] = curnode\n"
        "            science_meta[science_i, 2] = k\n"
        "            science_meta[science_i, 3] = choice if greedy else idx\n"
        "            science_total[science_i] = total\n"
        "            science_uniform[science_i] = rand_val if greedy else -1.0\n"
        "            science_weighted[science_i] = 1 if greedy else 0\n"
        "            science_cursor[0] = science_i + 1\n"
    )
    recorder=unique(recorder,
        "\n        # update dominance / força / dist_table / posição\n",
        "\n"+snippet+"\n        # update dominance / força / dist_table / posição\n")

    original_greedy_start="        if use_greedy:\n"
    original_random_start="        else:\n            # random\n"
    assert common.count(original_greedy_start)==1
    assert common.count(original_random_start)==1
    before=common.index(original_greedy_start)
    after=common.index(original_random_start,before)
    oracle_block=(
        "        if use_greedy:\n"
        "            science_i = science_cursor[0]\n"
        "            if science_i >= science_meta.shape[0]:\n"
        "                raise ValueError('greedy tape cursor overflow')\n"
        "            if (science_meta[science_i, 0] != p_i or\n"
        "                science_meta[science_i, 1] != curnode or\n"
        "                science_meta[science_i, 2] != k):\n"
        "                raise ValueError('greedy recorded particle/graph drift')\n"
        "            total = science_total[science_i]\n"
        "            if science_weighted[science_i] != 0:\n"
        "                if not total > 0.0:\n"
        "                    raise ValueError('greedy nonpositive weighted total')\n"
        "                rand_val = np.random.uniform(0.0, total)\n"
        "                if rand_val != science_uniform[science_i]:\n"
        "                    raise ValueError('greedy native uniform MT drift')\n"
        "                choice = science_meta[science_i, 3]\n"
        "                if choice < 0 or choice >= k:\n"
        "                    raise ValueError('greedy choice outside original neighbors')\n"
        "                next_node = neighbors[choice]\n"
        "                greedy = True\n"
        "            else:\n"
        "                if total > 0.0:\n"
        "                    raise ValueError('greedy invalid fallback mode')\n"
        "                idx = np.random.randint(k)\n"
        "                if idx != science_meta[science_i, 3]:\n"
        "                    raise ValueError('greedy fallback MT drift')\n"
        "                next_node = neighbors[idx]\n"
        "                greedy = False\n"
        "            science_cursor[0] = science_i + 1\n"
    )
    oracle=common[:before]+oracle_block+common[after:]
    assert oracle.count("np.random.uniform(0.0, total)")==1
    assert oracle.count("science_cursor[0] = science_i + 1")==1
    assert recorder.count("science_cursor[0] = science_i + 1")==1

    def transform(step_variant):
        source=unique(original,step,step_variant)
        source=unique(source,
            "                        max_iter, early_stop, es_chk, stop_max, return_info=False):",
            "                        max_iter, early_stop, es_chk, stop_max, return_info=False, "+
            EXTRA.replace(", ", "=None, ")+"=None):")
        source=unique(source,
            "                                   dist_weights)\n        \n        if early_stop",
            "                                   dist_weights, "+EXTRA+")\n        \n        if early_stop")
        # Private direct-step function is retained as a compilable callable
        # with its own scratch tape (not used for this scientific comparison).
        source=unique(source,
            "                                       dist_weights)\n\n\ndef pcc_propagate_numba(",
            "                                       dist_weights, np.empty((1,4),dtype=np.int64), "+
            "np.empty(1), np.empty(1), np.empty(1,dtype=np.int8), "+
            "np.zeros(1,dtype=np.int64))\n\n\ndef pcc_propagate_numba(")
        source=unique(source,
            "                        max_iter, early_stop, es_chk, stop_max, return_info=False, seed=None):",
            "                        max_iter, early_stop, es_chk, stop_max, return_info=False, seed=None, "+
            EXTRA.replace(", ", "=None, ")+"=None):")
        source=unique(source,
            "        dist_weights, max_iter, early_stop, es_chk, stop_max, return_info)\n    if return_info:",
            "        dist_weights, max_iter, early_stop, es_chk, stop_max, return_info, "+
            EXTRA+")\n    if return_info:")
        return source

    modules=[]
    shas=[]
    for kind,step_source in (("record",recorder),("oracle",oracle)):
        transformed=transform(step_source)
        directory=root/".cache"/"numba_greedy_oracle_exact"/kind
        directory.mkdir(parents=True,exist_ok=True)
        source_path=directory/"pcc_numba.py"
        source_path.write_bytes(transformed.encode("utf-8"))
        mod=types.ModuleType("_numba_greedy_oracle_science_"+kind)
        mod.__file__=str(source_path)
        sys.modules[mod.__name__]=mod
        exec(compile(transformed,str(source_path),"exec"),mod.__dict__)
        assert mod.pcc_propagate_numba.__name__=="pcc_propagate_numba"
        assert mod.pcc_propagate_numba is not ORIGINAL
        modules.append(mod.pcc_propagate_numba)
        shas.append(hashlib.sha256(transformed.encode("utf-8")).hexdigest())
    return (*modules,source_hashes,shas)


def fixture(ds,k,seed,preparer):
    if ds!="ring_8192_32":
        n,d,c,_=REAL_DATASETS[ds]
        arrays,meta=preparer.prepare(Case(ds,n,d,c,k_nn=k,labels_per_class=3,
                                           data_seed=0,label_seed=seed))
        assert meta["labeled_nodes"]==3*c
        return arrays["neighbors"],arrays["degrees"],arrays["labels"],meta["fixture_sha256"]
    n,c=8192,32
    nodes=np.arange(n,dtype=np.int64)
    graph=np.column_stack(((nodes-1)%n,(nodes+1)%n))
    assert np.array_equal(graph[graph[:,0],1],nodes)
    assert np.array_equal(graph[graph[:,1],0],nodes)
    labels=np.full(n,-1,dtype=np.int64)
    rng=np.random.default_rng(seed)
    for cl in range(c):
        eligible=np.flatnonzero(nodes%c==cl)
        labels[rng.choice(eligible,3,replace=False)]=cl
    fx=hashlib.sha256((array_hash(graph)+array_hash(labels)).encode()).hexdigest()
    return graph,np.full(n,2,dtype=np.int64),labels,fx


def snapshot(model,pred):
    return (pred,model.node.dominance,model.part.curnode,
            model.part.strength,model.part.dist_table,model.owndeg,
            model.node.label,model.mapped_labels)


def hashes(values):
    return dict(zip(STATES,(array_hash(a) for a in values)))


def tape_new(cap,n):
    slots=max(1,cap*n)
    return (np.empty((slots,4),dtype=np.int64),
            np.empty(slots,dtype=np.float64),
            np.empty(slots,dtype=np.float64),
            np.empty(slots,dtype=np.int8),
            np.zeros(1,dtype=np.int64))


def make_model(graph,degree,k,kind,fn,tape=None):
    obj=PCC(impl="numba",update_mode="sequential")
    obj.set_graph(graph,degree,k_nn=k)
    assert obj.neib_list.dtype==np.int64 and obj.neib_qt.dtype==np.int64
    assert np.array_equal(obj.neib_qt,degree)
    active=np.arange(graph.shape[1])[None,:]<degree[:,None]
    assert np.array_equal(obj.neib_list[active],graph[active])
    assert not np.any(obj.neib_list[~active])
    events=[]

    def wrapper(*args,**kwargs):
        kwargs["return_info"]=True
        if kind!="original":
            tape[-1][0]=0
            kwargs.update(dict(science_meta=tape[0],science_total=tape[1],
                               science_uniform=tape[2],
                               science_weighted=tape[3],
                               science_cursor=tape[4]))
        info=fn(*args,**kwargs)
        assert isinstance(info,dict)
        events.append(dict(info=info,
                           cursor=0 if tape is None else int(tape[4][0])))
        return None
    wrapper.__name__="pcc_propagate_numba"
    obj._get_backend_fn=lambda:wrapper
    return obj,wrapper,events


def fit(model,wrapped,events,labels,pgrd,cap,stop,tape_expected=None):
    assert prod.pcc_propagate_numba is ORIGINAL
    old=prod.pcc_propagate_numba
    prod.pcc_propagate_numba=wrapped
    previous=len(events)
    try:
        assert model._get_backend_fn() is prod.pcc_propagate_numba
        pred=model.fit_predict(labels,p_grd=pgrd,max_iter=cap,
                               early_stop=stop,seed=ARGS["seed"],
                               delta_v=ARGS["delta_v"],deltap=ARGS["deltap"],
                               dexp=ARGS["dexp"],es_chk=ARGS["es_chk"])
    finally:
        prod.pcc_propagate_numba=old
    assert len(events)==previous+1
    data=events[-1]
    if tape_expected is not None:
        assert data["cursor"]==tape_expected,(
            "true original greedy tape not consumed exactly",data["cursor"],tape_expected)
    result=snapshot(model,pred)
    classes=len(np.unique(labels[labels!=-1]))
    assert result[1].flags.c_contiguous == (classes>=8)
    assert result[1].flags.f_contiguous == (classes<8)
    assert result[5].flags.f_contiguous
    assert result[4].dtype==np.uint8
    assert np.array_equal(pred[labels!=-1],labels[labels!=-1])
    assert np.isfinite(result[1]).all()
    assert -1e-12<=result[1].min() and result[1].max()<=1.+1e-12
    stop_data=data["info"]
    assert 0<=stop_data["effective_iterations"]<=cap
    assert stop_data["stop_reason"] in ("max_iter","early_stop")
    if not stop:
        assert stop_data["effective_iterations"]==cap
    return dict(state=hashes(result),
                next_rng=float(after_rng()),
                stop=stop_data,
                refs=result)


def assert_semantics(a,b,context):
    for key in ("state","stop","next_rng"):
        assert a[key]==b[key],(context,key,a[key],b[key])


def tape_summary(tape,count,policy):
    meta,total,uniform,weighted,cursor=tape
    assert int(cursor[0])==count
    if policy==0:
        assert count==0
    if count:
        weights=weighted[:count]
        assert np.all((weights==0)|(weights==1))
        assert np.all(meta[:count,2]>0)
        assert np.all(meta[:count,3]>=0)
        assert np.all(meta[:count,3]<meta[:count,2])
        assert np.all((total[:count]>0)==(weights==1))
        assert np.all(np.isfinite(uniform[:count][weights==1]))
        assert np.all((uniform[:count][weights==1]>=0))
    return dict(events=count,weighted=int(np.sum(weighted[:count])),
                fallback=count-int(np.sum(weighted[:count])),
                max_degree=int(meta[:count,2].max()) if count else 0,
                tape_allocated_bytes=sum(x.nbytes for x in tape),
                tape_used_bytes=count*(4*8+8+8+1)+8,
                event_meta_sha=hashlib.sha256(meta[:count].tobytes()).hexdigest(),
                event_total_sha=hashlib.sha256(total[:count].tobytes()).hexdigest(),
                event_uniform_sha=hashlib.sha256(uniform[:count].tobytes()).hexdigest(),
                event_weighted_sha=hashlib.sha256(weighted[:count].tobytes()).hexdigest())


def batch(model,wrapped,events,labels,policy,stop,tape_count):
    refs=[]
    t_cpu=time.process_time_ns()
    t_wall=time.perf_counter_ns()
    for _ in range(BATCH):
        assert prod.pcc_propagate_numba is ORIGINAL
        prod.pcc_propagate_numba=wrapped
        n_before=len(events)
        try:
            pred=model.fit_predict(labels,p_grd=policy,max_iter=CAP,
                                   early_stop=stop,seed=ARGS["seed"],
                                   delta_v=ARGS["delta_v"],deltap=ARGS["deltap"],
                                   dexp=ARGS["dexp"],es_chk=ARGS["es_chk"])
        finally:
            prod.pcc_propagate_numba=ORIGINAL
        assert len(events)==n_before+1
        if tape_count is not None:
            assert events[-1]["cursor"]==tape_count
        refs.append(snapshot(model,pred))
    wall=time.perf_counter_ns()-t_wall
    cpu=time.process_time_ns()-t_cpu
    assert cpu>0 and wall>0
    states=[hashes(z) for z in refs]
    for old,new in zip(refs,refs[1:]):
        assert all(old[i] is not new[i] for i in range(1,len(STATES)))
        assert all(hashlib.sha256(np.ascontiguousarray(old[i]).tobytes()).hexdigest()
                   ==hashlib.sha256(np.ascontiguousarray(old[i]).tobytes()).hexdigest()
                   for i in (1,2,3,4,5,6,7))
    return dict(cpu_ns=cpu,wall_ns=wall,states=states,
                stops=[x["info"] for x in events[-BATCH:]],
                next_rng=float(after_rng()))


def rollback(model,labels):
    old=snapshot(model,model.node.label)
    checks=hashes(old)
    try:
        model.fit_predict(labels,seed=-1,max_iter=5)
    except ValueError:
        pass
    else:
        raise AssertionError("original invalid seed not rejected")
    current=snapshot(model,old[0])
    assert all(x is y for x,y in zip(old[1:],current[1:]))
    assert checks==hashes(current)


def one(ds,k,seed,policy,stop,record_fn,oracle_fn,preparer,smoke=False):
    graph,degree,labels,fx=fixture(ds,k,seed,preparer)
    n_particles=int(np.count_nonzero(labels!=-1))
    tape=tape_new(0 if policy==0 else CAP,n_particles)
    control=make_model(graph,degree,k,"original",ORIGINAL)
    recorded=make_model(graph,degree,k,"record",record_fn,tape)
    oracle=make_model(graph,degree,k,"oracle",oracle_fn,tape)
    # Actual source and native signatures JIT compiled off-clock. For small
    # smoke, exercise the exact parameterization, not a toy standalone kernel.
    initial=fit(*control,labels,policy,CAP,stop)
    source=fit(*recorded,labels,policy,CAP,stop)
    assert_semantics(initial,source,("original vs complete tape recorder",ds,k,seed,policy,stop))
    count=recorded[2][-1]["cursor"]
    tape_info=tape_summary(tape,count,policy)
    oracle_fit=fit(*oracle,labels,policy,CAP,stop,tape_expected=count)
    assert_semantics(initial,oracle_fit,("original vs oracle replay",ds,k,seed,policy,stop))
    # Retain the original branch only as a control and the counterfactual as
    # a nonproduction full-fit oracle: no public API modifications.
    nrep=1 if smoke else REPS
    measurements=[]
    for r in range(nrep):
        if r%2==0:
            a=batch(*control,labels,policy,stop,None)
            b=batch(*oracle,labels,policy,stop,count)
        else:
            b=batch(*oracle,labels,policy,stop,count)
            a=batch(*control,labels,policy,stop,None)
        for key in ("states","stops","next_rng"):
            assert a[key]==b[key],(ds,k,seed,policy,stop,r,key)
        measurements.append(dict(rep=r,original=a,oracle=b))
        print(ds,k,seed,policy,stop,r,
              "original_cpu_fit_ms",round(a["cpu_ns"]/BATCH/1e6,3),
              "oracle_cpu_fit_ms",round(b["cpu_ns"]/BATCH/1e6,3),
              "recorded_greedy",count,flush=True)
    for m in (control,oracle):
        rollback(m[0],labels)
    return dict(dataset=ds,k=k,seed=seed,p_grd=policy,
                stop_mode="monitored" if stop else "fixed",
                cap=CAP,graph_sha=array_hash(graph),
                degree_sha=array_hash(degree),
                label_sha=array_hash(labels),fixture_sha=fx,
                particles=n_particles,classes=len(np.unique(labels[labels!=-1])),
                baseline=dict(state=initial["state"],stop=initial["stop"],
                              next_rng=initial["next_rng"]),
                recorder=dict(state=source["state"],stop=source["stop"],
                              next_rng=source["next_rng"]),
                oracle=dict(state=oracle_fit["state"],stop=oracle_fit["stop"],
                            next_rng=oracle_fit["next_rng"]),
                tape=tape_info,pairs=measurements,
                rollback_exact=True,prior_fit_independent=True)


def zero_step_controls(record_fn,oracle_fn,preparer):
    rows=[]
    for ds,k,seed in (("digits",10,89021),("ring_8192_32",2,89119)):
        graph,degree,labels,fx=fixture(ds,k,seed,preparer)
        for policy in POLICIES:
            tape=tape_new(0,len(labels))
            original=make_model(graph,degree,k,"original",ORIGINAL)
            recorded=make_model(graph,degree,k,"record",record_fn,tape)
            oracle=make_model(graph,degree,k,"oracle",oracle_fn,tape)
            for cap in (0,5):
                if cap==5:
                    tape=tape_new(cap,np.count_nonzero(labels!=-1))
                    recorded=make_model(graph,degree,k,"record",record_fn,tape)
                    oracle=make_model(graph,degree,k,"oracle",oracle_fn,tape)
                a=fit(*original,labels,policy,cap,False)
                c=fit(*recorded,labels,policy,cap,False)
                assert_semantics(a,c,("short recorder",ds,policy,cap))
                count=recorded[2][-1]["cursor"]
                b=fit(*oracle,labels,policy,cap,False,tape_expected=count)
                assert_semantics(a,b,("short oracle",ds,policy,cap))
                rows.append(dict(dataset=ds,k=k,seed=seed,p_grd=policy,
                                 cap=cap,state=a["state"],
                                 next_rng=a["next_rng"],stop=a["stop"],
                                 tape_count=count))
    assert len(rows)==12
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        assert os.environ.get(key)=="1",(key,os.environ.get(key))
    record_fn,oracle_fn,source_hashes,clone_hashes=git_guard_and_clones()
    after_rng()
    preparer=FixturePreparer(prepare_fixture)
    edges=zero_step_controls(record_fn,oracle_fn,preparer)
    cases=[(ds,k,seed,p,mode)
           for ds,k in GEOMETRIES for seed in SEEDS for p in POLICIES
           for name,mode in MODES]
    if args.smoke:
        cases=[("digits",10,89021,.5,False)]
    rows=[]
    for ds,k,seed,policy,stop in cases:
        rows.append(one(ds,k,seed,policy,stop,record_fn,oracle_fn,preparer,args.smoke))
    root=Path(__file__).resolve().parents[1]
    output=dict(schema=1,base=BASE,
                head=subprocess.check_output(["git","rev-parse","HEAD"],
                                             cwd=root,text=True).strip(),
                platform=sys.platform,host=platform.platform(),
                source_hashes=source_hashes,
                clone_hashes=clone_hashes,
                oracle_cannot_run_without_prior_tape=True,
                original_pcc_source_unchanged=True,
                no_achievable_optimization_go=True,
                batch=BATCH,repetitions=1 if args.smoke else REPS,
                short_controls=edges,cohorts=rows)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",
                      encoding="utf-8")
    print("COMPLETE ORIGINAL EXACT-TRAJECTORY ORACLE SCIENCE",
          sys.platform,len(rows),"timed complete fits",
          len(rows)*2*BATCH*(1 if args.smoke else REPS),flush=True)


if __name__=="__main__":
    main()
