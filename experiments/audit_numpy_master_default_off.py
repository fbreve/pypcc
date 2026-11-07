"""Independent source-tree benchmark: unchanged master vs observer-default-off.

Every measured worker is a *new Python subprocess* running with its own
checkout directory first on sys.path. The optional observer is NEVER passed.
The same genuine high-level NumPy fit is timed in both versions, not module
imports, launcher overhead or graph construction.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DATASETS=("iris","breast_cancer","digits")
OSES=("ubuntu-latest","windows-latest")
SEEDS=(15013,15101)
REPEATS=2
VARIANTS=("master","candidate_off")
CAP=6001
BASE_SHA="8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128"
STATE_KEYS=(
    "predictions","positions","strengths","distances",
    "dominance","ownership","rng_keys","rng_pos",
    "rng_has_gauss","rng_cached_gaussian","rng_next16",
)
FIELDS=(
    "runner_os","dataset","seed","repeat","variant_order",
    "original_seconds","candidate_seconds","saving_percent",
    "baseline_commit","candidate_commit",
    "baseline_source_sha256","candidate_source_sha256",
    "baseline_import_path","candidate_import_path",
    "all_state_and_rng_exact","both_normal_public_api",
    "actual_graph_reciprocal","both_distinct_checkouts",
)


def worker(args):
    checkout=Path.cwd().resolve()
    sys.path.insert(0,str(checkout))
    import pcc
    import pcc_numpy
    from pcc_graph import build_knn_graph
    from sklearn.datasets import load_iris,load_breast_cancer,load_digits
    from sklearn.preprocessing import StandardScaler
    from inspect import signature

    model_file=Path(pcc.__file__).resolve()
    kernel_file=Path(pcc_numpy.__file__).resolve()
    if model_file.parent!=checkout or kernel_file.parent!=checkout:
        raise AssertionError(
            f"Wrong checkout module: {model_file}, {kernel_file}; expected {checkout}")
    has_observer="observer" in signature(pcc_numpy.pcc_propagate_numpy).parameters
    if has_observer!=(args.variant=="candidate_off"):
        raise AssertionError("Imported wrong master/candidate propagation signature")
    commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=checkout,text=True).strip()
    if args.variant=="master" and commit!=BASE_SHA:
        raise AssertionError("Baseline revision drifted from pinned original master")
    if args.variant=="candidate_off" and commit==BASE_SHA:
        raise AssertionError("Candidate unexpectedly equal to original master")

    load={"iris":load_iris,"breast_cancer":load_breast_cancer,
          "digits":load_digits}[args.dataset]
    ds=load()
    truth=np.asarray(ds.target,dtype=np.int64)
    features=StandardScaler().fit_transform(
        np.asarray(ds.data,dtype=np.float64))
    graph,degrees=build_knn_graph(features,k_nn=10)
    neighbors=[set(map(int,graph[i,:int(degrees[i])]))
               for i in range(len(truth))]
    if any(len(neighbors[i])!=int(degrees[i]) or i in neighbors[i] or
           any(i not in neighbors[j] for j in neighbors[i])
           for i in range(len(truth))):
        raise AssertionError("Graph not canonical undirected union-10NN")
    labels=np.full(len(truth),-1,dtype=np.int64)
    rng=np.random.default_rng(args.seed)
    for cls in np.unique(truth):
        labels[rng.choice(np.flatnonzero(truth==cls),2,replace=False)]=cls
    if np.count_nonzero(labels!=-1)!=2*len(np.unique(truth)):
        raise AssertionError("Incorrect frozen initial label subset")

    model=pcc.ParticleCompetitionAndCooperation(impl="numpy")
    if model._get_backend_fn() is not pcc_numpy.pcc_propagate_numpy:
        raise AssertionError("Requested NumPy backend is not actually selected")
    model.set_graph(graph,degrees,k_nn=10)
    np.random.seed(args.seed)
    started=time.perf_counter()
    prediction=model.fit_predict(
        labels.copy(),seed=args.seed,max_iter=CAP,early_stop=False,
        es_chk=2000,p_grd=.5,delta_v=.1,deltap=1.,dexp=2.)
    elapsed=time.perf_counter()-started
    if not elapsed>0 or not np.array_equal(
            np.asarray(prediction)[labels!=-1],labels[labels!=-1]):
        raise AssertionError("Native NumPy fit failed or changed labeled homes")
    rng_after=np.random.get_state()
    next16=np.random.random(16)
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(
        predictions=np.asarray(prediction).copy(),
        positions=model.part.curnode.copy(),
        strengths=model.part.strength.copy(),
        distances=model.part.dist_table.copy(),
        dominance=model.node.dominance.copy(),
        ownership=model.owndeg.copy(),
        rng_keys=rng_after[1].copy(),
        rng_pos=np.array([rng_after[2]],dtype=np.int64),
        rng_has_gauss=np.array([rng_after[3]],dtype=np.int64),
        rng_cached_gaussian=np.array([rng_after[4]],dtype=np.float64),
        rng_next16=next16.copy(),
    )
    if not all(np.isfinite(v).all() for v in arrays.values()
               if v.dtype.kind=="f"):
        raise AssertionError("Nonfinite full-state array")
    np.savez_compressed(output.with_suffix(".npz"),**arrays)
    metadata=dict(
        variant=args.variant,dataset=args.dataset,seed=args.seed,
        elapsed_seconds=elapsed,source_commit=commit,
        kernel_import_path=str(kernel_file),model_import_path=str(model_file),
        kernel_source_sha256=hashlib.sha256(
            kernel_file.read_bytes()).hexdigest(),
        graph_reciprocal=True,ordinary_public_fit=True,
        actual_backend="pcc_propagate_numpy",
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata,sort_keys=True,indent=2),encoding="utf-8")
    print(f"{args.variant}/{args.dataset}/{args.seed}: "
          f"fit_predict={elapsed:.4f}s, full state/RNG captured",
          flush=True)


def compare_and_write(baseline,candidate):
    metadata={}
    for variant,prefix in (("master",baseline),("candidate_off",candidate)):
        metadata[variant]=json.loads(
            prefix.with_suffix(".json").read_text(encoding="utf-8"))
    a,b=metadata["master"],metadata["candidate_off"]
    if a["source_commit"]!=BASE_SHA or b["source_commit"]==BASE_SHA:
        raise AssertionError("Incorrect independent source revisions")
    if (a["kernel_source_sha256"]==b["kernel_source_sha256"]
            or Path(a["kernel_import_path"]).resolve().parent==
               Path(b["kernel_import_path"]).resolve().parent):
        raise AssertionError("Master/candidate source not independently imported")
    with np.load(baseline.with_suffix(".npz"),allow_pickle=False) as state_a:
        with np.load(candidate.with_suffix(".npz"),allow_pickle=False) as state_b:
            for key in STATE_KEYS:
                if key not in state_a or key not in state_b or not (
                        np.array_equal(state_a[key],state_b[key])):
                    raise AssertionError(
                        f"Independent master/candidate state/RNG mismatch: "
                        f"{a['dataset']}/{a['seed']}/{key}")
    return a,b


def orchestrate(args):
    master=Path(args.master_dir).resolve()
    candidate=Path(args.candidate_dir).resolve()
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    script=(candidate/"experiments"/
            "audit_numpy_master_default_off.py").resolve()
    if not script.exists() or master==candidate:
        raise AssertionError("Missing independent checkouts")
    with open(args.csv,"w",newline="",encoding="utf-8") as file:
        w=csv.DictWriter(file,fieldnames=FIELDS)
        w.writeheader()
        for seed_index,seed in enumerate(SEEDS):
            for repeat in range(REPEATS):
                order=(VARIANTS if (seed_index+repeat)%2==0
                       else tuple(reversed(VARIANTS)))
                results={}
                for variant in order:
                    cwd=master if variant=="master" else candidate
                    prefix=output / (
                        f"{args.runner_os}_{args.dataset}_{seed}_"
                        f"block{repeat}_{variant}")
                    cmd=[sys.executable,str(script),"worker","--variant",variant,
                         "--dataset",args.dataset,"--seed",str(seed),
                         "--output",str(prefix)]
                    proc=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True)
                    if proc.returncode!=0:
                        raise RuntimeError(
                            f"Native source worker failed {cmd}: "
                            f"stdout={proc.stdout[-3000:]} stderr={proc.stderr[-4000:]}")
                    results[variant]=prefix
                a,b=compare_and_write(
                    results["master"],results["candidate_off"])
                saving=100*(1-float(b["elapsed_seconds"])/
                           float(a["elapsed_seconds"]))
                w.writerow(dict(
                    runner_os=args.runner_os,dataset=args.dataset,
                    seed=seed,repeat=repeat,
                    variant_order="/".join(order),
                    original_seconds=a["elapsed_seconds"],
                    candidate_seconds=b["elapsed_seconds"],
                    saving_percent=saving,
                    baseline_commit=a["source_commit"],
                    candidate_commit=b["source_commit"],
                    baseline_source_sha256=a["kernel_source_sha256"],
                    candidate_source_sha256=b["kernel_source_sha256"],
                    baseline_import_path=a["kernel_import_path"],
                    candidate_import_path=b["kernel_import_path"],
                    all_state_and_rng_exact=True,
                    both_normal_public_api=(
                        a["ordinary_public_fit"] and b["ordinary_public_fit"]),
                    actual_graph_reciprocal=(
                        a["graph_reciprocal"] and b["graph_reciprocal"]),
                    both_distinct_checkouts=True,
                ))
                file.flush()
                print(f"{args.runner_os}/{args.dataset}/{seed}/"
                      f"block{repeat}: master={a['elapsed_seconds']:.4f}s "
                      f"candidate_off={b['elapsed_seconds']:.4f}s "
                      f"saving={saving:+.2f}% exact state/RNG PASS",flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    worker_args=sub.add_parser("worker")
    worker_args.add_argument("--variant",choices=VARIANTS,required=True)
    worker_args.add_argument("--dataset",choices=DATASETS,required=True)
    worker_args.add_argument("--seed",type=int,choices=SEEDS,required=True)
    worker_args.add_argument("--output",required=True)
    orch=sub.add_parser("orchestrate")
    orch.add_argument("--runner-os",choices=OSES,required=True)
    orch.add_argument("--dataset",choices=DATASETS,required=True)
    orch.add_argument("--master-dir",required=True)
    orch.add_argument("--candidate-dir",required=True)
    orch.add_argument("--output-dir",required=True)
    orch.add_argument("--csv",required=True)
    a=p.parse_args(argv)
    if a.command=="worker":
        worker(a)
    else:
        orchestrate(a)


if __name__=="__main__":
    main()
