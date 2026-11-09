"""Post-#207 original production master phase and graph-preparation observation; monitor RSS.

Never times kernels or changes an algorithm. All scientific choices and sample
counts are fixed in current_master_performance_protocol.md before outcomes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


FROZEN_MASTER="c1073ae53ea785031de9e474b678e94f44a2124d"
PRODUCTION_FILES=(
    "pcc.py","pcc_numpy.py","pcc_numba.py","pcc_step.pyx",
    "pcc_graph.py","pcc_validation.py","setup.py","benchmarks",
)
DATASETS=("wine","breast_cancer","digits")
SEEDS=(25013,25117)
BACKENDS=("numpy","numba","cython")


def frozen_command():
    return [
        sys.executable,"-m","benchmarks.run",
        "--datasets",*DATASETS,
        "--label-seeds",*[str(s) for s in SEEDS],
        "--backends",*BACKENDS,
        "--mode","fixed","--iterations","10000","--repeats","6",
        "--order-policy","permutations","--order-seed","1901",
        "--labels-per-class","3","--data-seed","0",
        "--propagation-seed","0","--k-nn","10",
    ]


def verify_original_master_source():
    root=Path(__file__).resolve().parents[1]
    subprocess.run(
        ["git","cat-file","-e",FROZEN_MASTER+"^{commit}"],
        cwd=root,check=True,stdout=subprocess.DEVNULL)
    diff=subprocess.run(
        ["git","diff","--name-only",FROZEN_MASTER,"HEAD",
         "--",*PRODUCTION_FILES],
        cwd=root,check=True,capture_output=True,text=True)
    if diff.stdout.strip():
        raise AssertionError(
            "Benchmark source changed vs frozen original master:\n"+diff.stdout)
    return root


def validate_plan(plan):
    if (plan.get("case_count")!=6 or plan.get("preparation_groups")!=3
            or plan.get("measured_executions")!=108
            or plan.get("warmup_executions")!=36
            or plan.get("measured_iteration_cap")!=1080000
            or plan.get("particle_update_cap")!=16200000
            or plan.get("backends")!=list(BACKENDS)
            or plan.get("mode")!="fixed"
            or plan.get("repetitions")!=6
            or plan.get("order_policy")!="permutations"
            or plan.get("order_seed")!=1901
            or plan.get("requires_expensive_opt_in") is not False):
        raise AssertionError("Frozen pre-execution cost or scheduling differs")
    expected_cases=[
        f"{ds}-labels{seed}" for ds in DATASETS for seed in SEEDS]
    if [r.get("case_id") for r in plan["cases"]]!=expected_cases:
        raise AssertionError("Frozen six input fixtures differ")
    return plan


def execute(rootdir,os_label):
    # Optional external monitoring dependency must not be required for
    # importing protocol constants in the repository's minimal regression CI.
    import psutil
    repo=verify_original_master_source()
    rootdir=Path(rootdir).resolve()
    if rootdir.exists():
        raise FileExistsError(f"Output must be a NEW directory: {rootdir}")
    if str(rootdir).startswith(str(repo)+os.sep):
        raise ValueError("Benchmark outputs must live OUTSIDE checkout")
    rootdir.mkdir(parents=True)
    command=frozen_command()
    dryrun=subprocess.run(
        [*command,"--dry-run"],cwd=repo,check=True,text=True,
        capture_output=True)
    (rootdir/"dry-run.json").write_text(dryrun.stdout,encoding="utf-8")
    validate_plan(json.loads(dryrun.stdout))
    requirements=subprocess.check_output(
        [sys.executable,"-m","pip","freeze"],
        cwd=repo,text=True)
    (rootdir/"pip-freeze.txt").write_text(
        requirements,encoding="utf-8")
    prefix=rootdir/"records"
    native_command=[
        *command,"--execute","--output",str(prefix),
        "--session-id",f"master-native-{os_label}",
    ]
    (rootdir/"invocation.json").write_text(
        json.dumps(dict(
            schema=1,baseline_master_sha=FROZEN_MASTER,
            host_label=os_label,command=native_command,
            monitoring_interval_s=.05,
            rss_measurement_type="RSS sampled on benchmark subprocess, all phases",
            subprocess_stdout="execution.log",
        ),indent=2)+"\n",encoding="utf-8")
    samples=0
    peak=0
    first=None
    started=time.perf_counter()
    with (rootdir/"execution.log").open("w",encoding="utf-8") as log:
        child=subprocess.Popen(
            native_command,cwd=repo,stdout=log,
            stderr=subprocess.STDOUT)
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss=int(proc.memory_info().rss)
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                rss=None
            if rss is not None:
                if first is None:
                    first=rss
                peak=max(peak,rss)
                samples+=1
            time.sleep(.05)
        returncode=child.wait()
    duration=time.perf_counter()-started
    stats=dict(schema=1,host_label=os_label,child_pid=child.pid,
               exit_code=returncode,
               sample_interval_seconds=.05,n_samples=samples,
               sampled_first_rss_bytes=first,
               sampled_peak_rss_bytes=peak,
               subprocess_wall_seconds=duration,
               estimate_limits="Whole child-process resident memory sampled every 50ms; "
                   "not precise native peak nor per-fit allocations; misses brief peaks.")
    (rootdir/"memory.json").write_text(
        json.dumps(stats,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    if returncode!=0:
        tail=(rootdir/"execution.log").read_text(
            encoding="utf-8",errors="replace").splitlines()[-100:]
        print("\n".join(tail),file=sys.stderr,flush=True)
        raise RuntimeError(f"Native baseline exited {returncode}; preserved logs")
    print(f"HOST={os_label}; 108 measured real native fits; "
          f"sampled max child RSS={peak} bytes over {samples} samples; "
          f"subprocess wall={duration:.2f}s",flush=True)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",required=True,type=Path)
    p.add_argument("--os-label",choices=("linux","windows"),required=True)
    args=p.parse_args()
    execute(args.output_root,args.os_label)


if __name__=="__main__":
    main()
