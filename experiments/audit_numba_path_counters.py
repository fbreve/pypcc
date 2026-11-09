"""Original-source Numba particle path counters: separate untimed science copy.

This script does NOT change production pcc_numba.py or benchmark counter time.
It runs true original PCC.fit_predict twice (original vs compiled Numba clone
with pure int64 counter side effects), enforcing eight-state/RNG/stop parity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types

import numpy as np
from numba import njit
import pcc as prod
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numba import pcc_propagate_numba as ORIGINAL
from benchmarks.fixtures import (Case, REAL_DATASETS, FixturePreparer,
                                 prepare_fixture, array_hash)
from benchmarks.runner import expected_backend

BASE = "4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
PROTECTED = ("pcc.py","pcc_numba.py","pcc_graph.py","pcc_numpy.py",
             "pcc_validation.py","pcc_step.pyx")
GEOM = tuple((name,k) for name in
             ("iris","wine","breast_cancer","digits") for k in (3,10)) + (
             ("ring_8192_32",2),)
SEEDS = (89021,89119)
POLICIES = (0.,.5,1.)
MODES = (("fixed_1000",1000,False),
         ("fixed_10000",10000,False),
         ("monitored_10000",10000,True))
STAT = ("attempts","invalid_current","no_degree","invalid_label",
        "greedy_attempt","greedy_infinite_weight","greedy_weighted",
        "greedy_fallback","direct_random","valid_dest","unlabeled_update",
        "owndeg_random_increment","accepted","rejected","invalid_dest",
        "labeled_dest","distance_relax","position_changed",
        "weighted_greedy_accepted","random_accepted","eligible")
STATE = ("predictions","dominance","positions","strengths","distances",
         "owndeg","node_labels","mapped_labels")
ARGS = dict(seed=29,p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,
            es_chk=2000)


@njit
def next_rng():
    return np.random.random()


def replace_one(source,old,new):
    if source.count(old)!=1:
        raise AssertionError(f"Original scientific source anchor count {source.count(old)}: {old!r}")
    return source.replace(old,new,1)


def source_clone():
    root=Path(__file__).resolve().parents[1]
    delta=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        cwd=root,text=True).strip()
    assert not delta, "real production source was changed: "+delta
    shas={}
    for path in PROTECTED:
        original=subprocess.check_output(["git","show",BASE+":"+path],cwd=root)
        local=(root/path).read_bytes()
        assert original.replace(b"\r\n",b"\n")==local.replace(b"\r\n",b"\n"),path
        shas[path]=hashlib.sha256(original).hexdigest()
    assert prod.pcc_propagate_numba is ORIGINAL
    assert PCC(impl="numba")._get_backend_fn() is expected_backend("numba") is ORIGINAL
    source=subprocess.check_output(
        ["git","show",BASE+":pcc_numba.py"],cwd=root).decode("utf-8")
    start="@njit(cache=True)\ndef _pcc_step_numba_sequential("
    end="@njit(cache=True)\ndef _pcc_calc_mmpot_sequential("
    assert source.count(start)==source.count(end)==1
    step=source[source.index(start):source.index(end)]
    # Changes only record extra integers, never replace an original native
    # branch or arithmetic/MT19937 operation.
    changes=(
       ("                               dist_weights):\n    n_particles",
        "                               dist_weights, science_counts):\n    n_particles"),
       ("    for p_i in range(n_particles):\n        curnode",
        "    for p_i in range(n_particles):\n        science_counts[0] += 1\n        curnode"),
       ("        if curnode < 0 or curnode >= n_nodes:\n            continue",
        "        if curnode < 0 or curnode >= n_nodes:\n            science_counts[1] += 1\n            continue"),
       ("        if k <= 0:\n            continue",
        "        if k <= 0:\n            science_counts[2] += 1\n            continue"),
       ("        if label < 0 or label >= c:\n            continue",
        "        if label < 0 or label >= c:\n            science_counts[3] += 1\n            continue"),
       ("        use_greedy = np.random.random() < p_grd",
        "        science_counts[20] += 1\n        use_greedy = np.random.random() < p_grd"),
       ("        if use_greedy:\n            # greedy:",
        "        if use_greedy:\n            science_counts[4] += 1\n            # greedy:"),
       ("            if np.isinf(total):\n                max_weight",
        "            if np.isinf(total):\n                science_counts[5] += 1\n                max_weight"),
       ("            if total > 0.0:\n                rand_val",
        "            if total > 0.0:\n                science_counts[6] += 1\n                rand_val"),
       ("            else:\n                idx = np.random.randint(k)\n                next_node = neighbors[idx]\n                greedy = False",
        "            else:\n                science_counts[7] += 1\n                idx = np.random.randint(k)\n                next_node = neighbors[idx]\n                greedy = False"),
       ("        else:\n            # random\n            idx = np.random.randint(k)",
        "        else:\n            science_counts[8] += 1\n            # random\n            idx = np.random.randint(k)"),
       ("        if next_node >= 0 and next_node < n_nodes and label >= 0 and label < c:\n            if labels[next_node] == -1:",
        "        if next_node >= 0 and next_node < n_nodes and label >= 0 and label < c:\n            science_counts[9] += 1\n            if labels[next_node] == -1:\n                science_counts[10] += 1"),
       ("                dominance[next_node, label] += sum_reduc\n                ",
        "                dominance[next_node, label] += sum_reduc\n            else:\n                science_counts[15] += 1\n                "),
       ("                    dist_table[next_node, p_i] = cur_d + 1",
        "                    dist_table[next_node, p_i] = cur_d + 1\n                    science_counts[16] += 1"),
       ("            if not greedy:\n                owndeg",
        "            if not greedy:\n                science_counts[11] += 1\n                owndeg"),
       ("            if dominance[next_node, label] == max_dom:\n                part_curnode[p_i] = next_node",
        "            if dominance[next_node, label] == max_dom:\n                science_counts[12] += 1\n                if next_node != curnode:\n                    science_counts[17] += 1\n                if greedy:\n                    science_counts[18] += 1\n                else:\n                    science_counts[19] += 1\n                part_curnode[p_i] = next_node\n            else:\n                science_counts[13] += 1\n        else:\n            science_counts[14] += 1"),
    )
    changed=step
    for old,new in changes:
        changed=replace_one(changed,old,new)
    assert changed != step and changed.count("science_counts[") >= 20
    candidate=replace_one(source,step,changed)

    # Thread the private 21-int sidecar ONLY through the science clone,
    # preserving every originally executed call/seed/stop expression.
    candidate=replace_one(candidate,
        "                        max_iter, early_stop, es_chk, stop_max, return_info=False):",
        "                        max_iter, early_stop, es_chk, stop_max, return_info=False, science_counts=None):")
    candidate=replace_one(candidate,
        "                                   dist_weights)\n        \n        if early_stop",
        "                                   dist_weights, science_counts)\n        \n        if early_stop")
    # The original pcc_step_numba is preserved for provenance but its
    # direct one-step call needs a private scratch counter to compile.
    candidate=replace_one(candidate,
        "                                       dist_weights)\n\n\ndef pcc_propagate_numba(",
        "                                       dist_weights, np.zeros(21, dtype=np.int64))\n\n\ndef pcc_propagate_numba(")
    candidate=replace_one(candidate,
        "                        max_iter, early_stop, es_chk, stop_max, return_info=False, seed=None):",
        "                        max_iter, early_stop, es_chk, stop_max, return_info=False, seed=None, science_counts=None):")
    candidate=replace_one(candidate,
        "        dist_weights, max_iter, early_stop, es_chk, stop_max, return_info)\n    if return_info:",
        "        dist_weights, max_iter, early_stop, es_chk, stop_max, return_info, science_counts)\n    if return_info:")

    # Guarantee source edit fully reversible by replacing the complete
    # modified original step plus the four modified dispatch spans.
    assert candidate.count("science_counts[20]") == 1
    out=root/".cache"/"numba_path_counter_original_source"/"pcc_numba.py"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(candidate.encode("utf-8"))
    name="_scientific_numba_path_counter_clone"
    module=types.ModuleType(name)
    module.__file__=str(out)
    sys.modules[name]=module
    exec(compile(candidate,str(out),"exec"),module.__dict__)
    assert module.pcc_propagate_numba.__name__=="pcc_propagate_numba"
    assert module.pcc_propagate_numba is not ORIGINAL
    return module.pcc_propagate_numba, shas, hashlib.sha256(candidate.encode()).hexdigest()


def fixture(ds,k,seed,preparer):
    if ds!="ring_8192_32":
        n,features,classes,_=REAL_DATASETS[ds]
        case=Case(ds,n,features,classes,k_nn=k,labels_per_class=3,
                  label_seed=seed,data_seed=0)
        arrays,meta=preparer.prepare(case)
        assert meta["labeled_nodes"]==3*classes
        return (arrays["neighbors"],arrays["degrees"],arrays["labels"],
                meta["fixture_sha256"])
    n,c=8192,32
    idx=np.arange(n,dtype=np.int64)
    graph=np.column_stack(((idx-1)%n,(idx+1)%n))
    assert np.array_equal(graph[graph[:,0],1],idx)
    assert np.array_equal(graph[graph[:,1],0],idx)
    labels=np.full(n,-1,dtype=np.int64)
    rng=np.random.default_rng(seed)
    truth=idx%c
    for cls in range(c):
        homes=np.flatnonzero(truth==cls)
        labels[rng.choice(homes,3,replace=False)]=cls
    digest=hashlib.sha256((array_hash(graph)+array_hash(labels)).encode()).hexdigest()
    return graph,np.full(n,2,dtype=np.int64),labels,digest


def refs(model,pred):
    return (pred,model.node.dominance,model.part.curnode,
            model.part.strength,model.part.dist_table,model.owndeg,
            model.node.label,model.mapped_labels)


def sha_state(seq):
    return dict(zip(STATE,(array_hash(x) for x in seq)))


def assert_counters(raw,iterations,n_particles):
    assert raw.dtype==np.int64 and raw.shape==(len(STAT),)
    assert np.all(raw>=0)
    a=[int(x) for x in raw]
    assert a[0]==iterations*n_particles,(a[0],iterations,n_particles)
    assert a[0]==a[1]+a[2]+a[3]+a[20],("eligibility",a)
    assert a[20]==a[4]+a[8]==a[6]+a[7]+a[8],("policy coverage",a)
    assert a[4]==a[6]+a[7],("weighted attempt/fallback",a)
    assert a[20]==a[9]+a[14],("destination",a)
    assert a[9]==a[10]+a[15]==a[12]+a[13],("destination category/acceptance",a)
    assert a[11]<=a[9],("non-greedy owner increments bounded by valid visits",a)
    assert a[12]==a[18]+a[19],("accepted branch composition",a)
    assert a[17]<=a[12] and a[16]<=a[9],("movement/distance bounds",a)
    assert a[5]<=a[4] and a[18]<=a[6] and a[19]<=a[7]+a[8]
    assert a[11]<=a[9] and a[11]<=a[7]+a[8],("ownership updates",a)
    return dict(zip(STAT,a))


def make_model(graph,degree,k,backend,counters=None):
    model=PCC(impl="numba",update_mode="sequential")
    model.set_graph(graph,degree,k_nn=k)
    assert model.neib_list.dtype==np.int64 and model.neib_qt.dtype==np.int64
    assert np.array_equal(model.neib_qt,degree)
    active=np.arange(graph.shape[1])[None,:]<degree[:,None]
    assert np.array_equal(model.neib_list[active],graph[active])
    assert not np.any(model.neib_list[~active])
    capture=[]

    def wrapped(*args,**kwargs):
        kwargs["return_info"]=True
        if counters is not None:
            counters.fill(0)
            kwargs["science_counts"]=counters
        info=backend(*args,**kwargs)
        assert isinstance(info,dict)
        capture.append(dict(info=info,counts=None if counters is None
                            else counters.copy()))
        return None
    wrapped.__name__="pcc_propagate_numba"
    model._get_backend_fn=lambda:wrapped
    return model,wrapped,capture


def one_fit(model,fn,capture,labels,pgrd,cap,stop):
    prev=prod.pcc_propagate_numba
    assert prev is ORIGINAL
    prod.pcc_propagate_numba=fn
    mark=len(capture)
    try:
        assert model._get_backend_fn() is prod.pcc_propagate_numba
        prediction=model.fit_predict(labels,p_grd=pgrd,max_iter=cap,
                                     early_stop=stop,seed=ARGS["seed"],
                                     delta_v=ARGS["delta_v"],deltap=ARGS["deltap"],
                                     dexp=ARGS["dexp"],es_chk=ARGS["es_chk"])
    finally:
        prod.pcc_propagate_numba=prev
    assert len(capture)==mark+1,"effective original Numba dispatcher not called"
    m=capture[-1]
    arr=refs(model,prediction)
    classes=len(np.unique(labels[labels!=-1]))
    assert arr[1].flags.c_contiguous==(classes>=8)
    assert arr[1].flags.f_contiguous==(classes<8)
    assert arr[5].flags.f_contiguous
    assert arr[4].dtype==np.uint8
    assert np.array_equal(prediction[labels!=-1],labels[labels!=-1])
    assert np.all(np.isfinite(arr[1]))
    assert np.min(arr[1]) >= -1e-12 and np.max(arr[1]) <= 1+1e-12
    assert 0<=m["info"]["effective_iterations"]<=cap
    assert m["info"]["stop_reason"] in ("max_iter","early_stop")
    if not stop:
        assert m["info"]["effective_iterations"]==cap
    if m["counts"] is not None:
        result=assert_counters(m["counts"],m["info"]["effective_iterations"],
                               np.count_nonzero(labels!=-1))
    else:
        result=None
    return dict(state=sha_state(arr),next_rng=float(next_rng()),
                stop=m["info"],counts=result)


def no_op_error(model,labels,pgrd):
    old=refs(model,model.node.label)
    hashes=sha_state(old)
    try:
        model.fit_predict(labels,max_iter=5,seed=-1,p_grd=pgrd)
    except ValueError:
        pass
    else:
        raise AssertionError("negative seed accepted by original")
    now=refs(model,old[0])
    assert hashes==sha_state(now)
    assert all(a is b for a,b in zip(old[1:],now[1:]))


def run_cohort(ds,k,seed,pgrd,mode,cap,stop,source,preparer):
    graph,degree,labels,fx=fixture(ds,k,seed,preparer)
    orig=make_model(graph,degree,k,ORIGINAL)
    counted=make_model(graph,degree,k,source,np.zeros(len(STAT),dtype=np.int64))
    results=[]
    previous_refs=None
    for trial in range(2 if ds in ("digits","ring_8192_32") and
                       mode=="fixed_10000" else 1):
        oa=one_fit(*orig,labels,pgrd,cap,stop)
        cb=one_fit(*counted,labels,pgrd,cap,stop)
        if previous_refs is not None:
            for model,oldref,oldsha in previous_refs:
                current=refs(model,model.node.label)
                assert all(oldref[i] is not current[i] for i in range(1,len(STATE)))
                assert [array_hash(x) for x in oldref[1:]]==oldsha, "old fitted arrays mutated"
        previous_refs=[
            (model,list(refs(model,model.node.label)),
             [array_hash(x) for x in refs(model,model.node.label)[1:]])
            for model in (orig[0],counted[0])
        ]
        for field in ("state","stop","next_rng"):
            assert oa[field]==cb[field],(ds,k,seed,pgrd,mode,trial,field)
        results.append(dict(original=oa,counter=cb))
    no_op_error(orig[0],labels,pgrd)
    no_op_error(counted[0],labels,pgrd)
    assert prod.pcc_propagate_numba is ORIGINAL
    return dict(dataset=ds,k=k,seed=seed,p_grd=pgrd,mode=mode,cap=cap,
                early_stop=stop,nodes=len(labels),classes=3 if ds in
                ("iris","wine") else 2 if ds=="breast_cancer" else
                10 if ds=="digits" else 32,
                particles=int(np.count_nonzero(labels!=-1)),
                graph_sha=array_hash(graph),degree_sha=array_hash(degree),
                label_sha=array_hash(labels),fixture_sha=fx,
                results=results,rollback_exact=True,post_argmax_active=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        assert os.environ.get(key)=="1",(key,os.environ.get(key))
    clone,protected,candidate_sha=source_clone()
    next_rng()  # JIT compile outside science replay.
    prep=FixturePreparer(prepare_fixture)
    allcases=[(ds,k,seed,pgrd,mode,cap,stop)
              for ds,k in GEOM for seed in SEEDS for pgrd in POLICIES
              for mode,cap,stop in MODES]
    chosen=[("digits",10,89021,.5,"fixed_1000",1000,False)] if args.smoke else allcases
    rows=[]
    for ds,k,seed,pgrd,mode,cap,stop in chosen:
        row=run_cohort(ds,k,seed,pgrd,mode,cap,stop,clone,prep)
        rows.append(row)
        trial=row["results"][0]["counter"]
        cnt=trial["counts"]
        print(f"{ds} k={k} seed={seed} p={pgrd} mode={mode} "
              f"iter={trial['stop']['effective_iterations']} "
              f"greedy={cnt['greedy_attempt']} weighted={cnt['greedy_weighted']} "
              f"fallback={cnt['greedy_fallback']} "
              f"unlabeled={cnt['unlabeled_update']} "
              f"accepted={cnt['accepted']}",flush=True)
    root=Path(__file__).resolve().parents[1]
    report=dict(schema=1,base=BASE,head=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        platform=sys.platform,environment=platform.platform(),
        source_hashes=protected,candidate_sha=candidate_sha,
        sidecar=STAT,scoped_identity=True,no_native_timing=True,
        cohort_count=len(rows),cohorts=rows)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")
    print("SOURCE-EXACT NUMBA PATH COUNTERS COMPLETE",sys.platform,len(rows),flush=True)


if __name__=="__main__":
    main()
