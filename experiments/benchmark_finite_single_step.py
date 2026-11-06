"""Independent native master vs candidate repeated pcc_step performance audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.audit_clean_master_cython_cache import (  # noqa: E402
    canonical_direct_kernel_input, STATE,
)

OSES=("ubuntu-latest","windows-latest")
DATASETS=("breast_cancer","digits")
SEEDS=(9803,9901,10007)
VARIANTS=("master","preflight")
STEPS=5000
FIELDS=(
    "runner_os","dataset","seed","variant","order",
    "steps","elapsed_seconds","baseline_seconds","preflight_seconds",
    "slowdown_percent","source_commit","extension_sha256",
    "extension_file","exact_all_seven_state_arrays","canonical_graph",
)


def worker(args):
    checkout=Path.cwd().resolve()
    sys.path.insert(0,str(checkout))
    import pcc_step
    ext=Path(pcc_step.__file__).resolve()
    if ext.parent!=checkout or ext.suffix.lower() not in (".pyd",".so"):
        raise AssertionError(f"Not the independently compiled Cython extension: {ext}")
    x,state,labels,truth,degrees=canonical_direct_kernel_input(
        args.dataset,args.seed)
    step_args=tuple(x[:15])+(x[19],x[20],x[21])
    start=time.perf_counter()
    for _ in range(STEPS):
        pcc_step.pcc_step(*step_args)
    elapsed=time.perf_counter()-start
    if elapsed<=0:
        raise AssertionError("Unmeasurable direct step workload")
    pred=np.argmax(state["dominance"],axis=1)
    pred[labels>=0]=labels[labels>=0]
    state["predictions"]=pred
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(output.with_suffix(".npz")),
                        **{k:np.asarray(v) for k,v in state.items()})
    sha=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=checkout,text=True).strip()
    payload=dict(
        dataset=args.dataset,seed=args.seed,variant=args.variant,
        steps=STEPS,elapsed_seconds=elapsed,source_commit=sha,
        extension_sha256=hashlib.sha256(ext.read_bytes()).hexdigest(),
        extension_file=str(ext),canonical_graph=True,
    )
    output.with_suffix(".json").write_text(
        json.dumps(payload,sort_keys=True,indent=2),encoding="utf-8")


def compare(a,b):
    if a[1]["source_commit"]==b[1]["source_commit"]:
        raise AssertionError("Master and candidate source are not independent")
    if a[1]["extension_sha256"]==b[1]["extension_sha256"]:
        raise AssertionError("Master and candidate loaded same native binary")
    with np.load(a[0],allow_pickle=False) as base:
        with np.load(b[0],allow_pickle=False) as candidate:
            for key in STATE:
                if (key not in base or key not in candidate
                        or not np.array_equal(base[key],candidate[key])):
                    raise AssertionError(
                        f"Public pcc_step state mismatch: "
                        f"{a[1]['dataset']}/{a[1]['seed']}/{key}")


def orchestrate(args):
    roots=dict(
        master=Path(args.baseline_dir).resolve(),
        preflight=Path(args.candidate_dir).resolve(),
    )
    script=roots["preflight"]/"experiments"/"benchmark_finite_single_step.py"
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if roots["master"]==roots["preflight"] or not script.exists():
        raise AssertionError("Source checkouts not independent")
    with open(args.csv,"w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader()
        file.flush()
        for i,seed in enumerate(SEEDS):
            order=VARIANTS if i%2==0 else VARIANTS[::-1]
            results={}
            for index,variant in enumerate(order):
                checkout=roots[variant]
                output_prefix=output/f"{args.runner_os}_{args.dataset}_{seed}_{variant}"
                cmd=[
                    sys.executable,str(script),"worker",
                    "--dataset",args.dataset,"--seed",str(seed),
                    "--variant",variant,"--output",str(output_prefix)
                ]
                result=subprocess.run(
                    cmd,cwd=checkout,capture_output=True,text=True)
                if result.returncode:
                    raise AssertionError(
                        f"Cython public step failed: {cmd}\n"
                        f"stdout={result.stdout[-2500:]}\n"
                        f"stderr={result.stderr[-2500:]}")
                results[variant]=(
                    output_prefix.with_suffix(".npz"),
                    json.loads(output_prefix.with_suffix(".json").read_text(
                        encoding="utf-8")),
                    index,
                )
            compare(results["master"][:2],results["preflight"][:2])
            base=float(results["master"][1]["elapsed_seconds"])
            new=float(results["preflight"][1]["elapsed_seconds"])
            pct=100*(new/base-1)
            for variant in VARIANTS:
                payload=results[variant][1]
                writer.writerow(dict(
                    runner_os=args.runner_os,dataset=args.dataset,
                    seed=seed,variant=variant,order=results[variant][2],
                    steps=STEPS,elapsed_seconds=payload["elapsed_seconds"],
                    baseline_seconds=base,preflight_seconds=new,
                    slowdown_percent=pct,
                    source_commit=payload["source_commit"],
                    extension_sha256=payload["extension_sha256"],
                    extension_file=payload["extension_file"],
                    exact_all_seven_state_arrays=True,
                    canonical_graph=True,
                ))
            file.flush()
            print(
                f"{args.runner_os}/{args.dataset} seed={seed}: "
                f"{STEPS} direct steps master={base:.4f}s "
                f"preflight={new:.4f}s overhead={pct:+.2f}% "
                "all state exact",
                flush=True,
            )


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    w=sub.add_parser("worker")
    w.add_argument("--dataset",choices=DATASETS,required=True)
    w.add_argument("--seed",type=int,choices=SEEDS,required=True)
    w.add_argument("--variant",choices=VARIANTS,required=True)
    w.add_argument("--output",required=True)
    o=sub.add_parser("orchestrate")
    o.add_argument("--dataset",choices=DATASETS,required=True)
    o.add_argument("--runner-os",choices=OSES,required=True)
    o.add_argument("--baseline-dir",required=True)
    o.add_argument("--candidate-dir",required=True)
    o.add_argument("--output-dir",required=True)
    o.add_argument("--csv",required=True)
    args=p.parse_args(argv)
    worker(args) if args.command=="worker" else orchestrate(args)


if __name__=="__main__":
    main()
