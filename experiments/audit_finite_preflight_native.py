"""Clean-master vs finite-preflight Cython binary comparison on real graphs.

Each worker imports only its own checkout's compiled extension. The parent
orchestrator compares actual NPZ arrays, complete stopping metadata and SHA-256
fingerprints before interpreting any timing. See the frozen protocol.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.audit_clean_master_cython_cache import (
    canonical_direct_kernel_input, expected_stop_max, STATE,
)

DATASETS=("breast_cancer","digits")
OSES=("ubuntu-latest","windows-latest")
SEEDS=(9503,9601,9709)
MODES=(
    ("forced_potential", True, 10000000),
    ("no_stop", False, 2000),
)
VARIANTS=("master","preflight")
CAP=200000
REPEATS=2
FIELDS=(
    "runner_os","dataset","seed","mode","repeat","variant","order",
    "elapsed_seconds","effective_iterations","stop_reason","checks",
    "stop_max","stalled_checks","source_commit","extension_sha256",
    "extension_file","baseline_seconds","preflight_seconds",
    "preflight_saving_percent","all_state_arrays_exact",
    "all_stop_metadata_exact","canonical_reciprocal_graph",
)


def worker(args):
    # This process is launched with cwd set to the independently compiled
    # checkout. Never combine the modules from two builds in one process.
    checkout=Path.cwd().resolve()
    sys.path.insert(0,str(checkout))
    import pcc_step
    binary=Path(pcc_step.__file__).resolve()
    if binary.parent!=checkout or binary.suffix.lower() not in (".so",".pyd"):
        raise AssertionError(f"Loaded Cython from the wrong checkout: {binary}")
    x,state,labels,truth,degrees=canonical_direct_kernel_input(
        args.dataset,args.seed)
    mode=next((m for m in MODES if m[0]==args.mode),None)
    if mode is None:
        raise AssertionError("Wrong frozen regime")
    _,monitor,es_chk=mode
    stop_max=expected_stop_max(
        len(labels),len(x[7]),10,es_chk,degrees)
    if monitor and stop_max<=CAP//10+1:
        raise AssertionError("The fixed monitoring workload could stop")
    start=time.perf_counter()
    info=pcc_step.pcc_propagate(
        *x,CAP,int(monitor),es_chk,stop_max,
        return_info=True,seed=args.seed)
    secs=time.perf_counter()-start
    info=dict(info)
    if int(info["effective_iterations"])!=CAP or secs<=0:
        raise AssertionError(f"Unexpected iteration count or timing: {info}")
    pred=np.argmax(state["dominance"],axis=1)
    pred[labels>=0]=labels[labels>=0]
    state["predictions"]=pred
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        str(output.with_suffix(".npz")),
        **{k:np.asarray(v) for k,v in state.items()})
    sha=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=checkout,text=True).strip()
    meta=dict(
        dataset=args.dataset,seed=args.seed,mode=args.mode,
        variant=args.variant,elapsed_seconds=secs,
        info=info,
        source_commit=sha,
        extension_file=str(binary),
        extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        n_nodes=len(labels),n_classes=len(np.unique(truth)),
        canonical_reciprocal_graph=True,
    )
    output.with_suffix(".json").write_text(
        json.dumps(meta,sort_keys=True,indent=2),encoding="utf-8")


def check_exact(baseline,candidate):
    oldfile,old=baseline
    newfile,new=candidate
    if old["info"]!=new["info"]:
        raise AssertionError(f"Different native stopping metadata {old['info']} vs {new['info']}")
    if old["source_commit"]==new["source_commit"]:
        raise AssertionError("Baseline and candidate checkouts are identical")
    if old["extension_sha256"]==new["extension_sha256"]:
        raise AssertionError("Identical loaded native binary hashes")
    with np.load(oldfile,allow_pickle=False) as a:
        with np.load(newfile,allow_pickle=False) as b:
            for key in STATE:
                if key not in a or key not in b or not (
                        np.array_equal(a[key],b[key])):
                    raise AssertionError(
                        f"Finite validation changed state {key} "
                        f"for {new['dataset']}/{new['seed']}/{new['mode']}")
    return old["info"]


def orchestrate(args):
    checkouts=dict(
        master=Path(args.baseline_dir).resolve(),
        preflight=Path(args.candidate_dir).resolve(),
    )
    if checkouts["master"]==checkouts["preflight"]:
        raise AssertionError("Two real source checkouts are mandatory")
    script=checkouts["preflight"]/"experiments"/"audit_finite_preflight_native.py"
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with open(args.csv,"w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader()
        file.flush()
        for i,seed in enumerate(SEEDS):
            for mode,monitor,es_chk in MODES:
                for repeat in range(REPEATS):
                    order=VARIANTS if (i+repeat)%2==0 else VARIANTS[::-1]
                    results={}
                    for index,variant in enumerate(order):
                        checkout=checkouts[variant]
                        path=output/f"{args.runner_os}_{args.dataset}_{seed}_{mode}_{repeat}_{variant}"
                        cmd=[
                            sys.executable,str(script),"worker",
                            "--dataset",args.dataset,"--seed",str(seed),
                            "--mode",mode,"--variant",variant,
                            "--output",str(path)
                        ]
                        result=subprocess.run(
                            cmd,cwd=checkout,capture_output=True,text=True)
                        if result.returncode:
                            raise AssertionError(
                                f"Native worker failed: {cmd} "
                                f"stdout={result.stdout[-1500:]} "
                                f"stderr={result.stderr[-1500:]}")
                        results[variant]=(
                            path.with_suffix(".npz"),
                            json.loads(path.with_suffix(".json").read_text(
                                encoding="utf-8")),
                            index)
                    info=check_exact(results["master"][:2],
                                     results["preflight"][:2])
                    baseline=results["master"][1]["elapsed_seconds"]
                    candidate=results["preflight"][1]["elapsed_seconds"]
                    saving=100*(1-candidate/baseline)
                    for variant in VARIANTS:
                        meta=results[variant][1]
                        writer.writerow(dict(
                            runner_os=args.runner_os,dataset=args.dataset,
                            seed=seed,mode=mode,repeat=repeat,
                            variant=variant,order=results[variant][2],
                            elapsed_seconds=meta["elapsed_seconds"],
                            effective_iterations=info["effective_iterations"],
                            stop_reason=info["stop_reason"],
                            checks=info["checks"],stop_max=info["stop_max"],
                            stalled_checks=info["stalled_checks"],
                            source_commit=meta["source_commit"],
                            extension_sha256=meta["extension_sha256"],
                            extension_file=meta["extension_file"],
                            baseline_seconds=baseline,
                            preflight_seconds=candidate,
                            preflight_saving_percent=saving,
                            all_state_arrays_exact=True,
                            all_stop_metadata_exact=True,
                            canonical_reciprocal_graph=True,
                        ))
                    file.flush()
                    print(
                        f"{args.runner_os}/{args.dataset}/{seed}/{mode}/"
                        f"repeat{repeat}: original={baseline*1000:.3f}ms "
                        f"finite-preflight={candidate*1000:.3f}ms "
                        f"saving={saving:+.2f}% full-state parity PASS",
                        flush=True,
                    )


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    worker_parser=sub.add_parser("worker")
    worker_parser.add_argument("--dataset",choices=DATASETS,required=True)
    worker_parser.add_argument("--seed",choices=SEEDS,type=int,required=True)
    worker_parser.add_argument("--mode",choices=[m[0] for m in MODES],required=True)
    worker_parser.add_argument("--variant",choices=VARIANTS,required=True)
    worker_parser.add_argument("--output",required=True)
    runner=sub.add_parser("orchestrate")
    runner.add_argument("--dataset",choices=DATASETS,required=True)
    runner.add_argument("--runner-os",choices=OSES,required=True)
    runner.add_argument("--baseline-dir",required=True)
    runner.add_argument("--candidate-dir",required=True)
    runner.add_argument("--output-dir",required=True)
    runner.add_argument("--csv",required=True)
    args=parser.parse_args(argv)
    worker(args) if args.command=="worker" else orchestrate(args)


if __name__=="__main__":
    main()
