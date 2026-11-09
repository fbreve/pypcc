"""48 fresh-process original-vs-production Numba amortization trials.

One graph per process, 20 successive public fits with alternating label masks,
timestamps before any diagnostics, then 20 exact independently replayed fits.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

OLD="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
PROD="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
DATASETS=("wine","breast_cancer","digits")
SEEDS=(70013,70117)
PHASES=("cold","warm")
VARIANTS=("original","production")
CHECKPOINTS=(1,2,5,10,20)
PROTECTED=("pcc.py","pcc_numba.py","pcc_numpy.py","pcc_graph.py",
           "pcc_step.pyx","pcc_validation.py","setup.py","benchmarks")


def freeze_source():
    for commit in (OLD,PROD):
        subprocess.run(["git","cat-file","-e",commit+"^{commit}"],
                       check=True,stdout=subprocess.DEVNULL)
    drift=subprocess.check_output(
        ["git","diff","--name-only",PROD,"HEAD","--",*PROTECTED],
        text=True).strip()
    if drift:
        raise AssertionError("science branch changes original production: "+drift)
    raw={}
    for variant,commit in (("original",OLD),("production",PROD)):
        raw[variant]=subprocess.check_output(
            ["git","show",commit+":pcc_numba.py"]
        ).decode("utf-8").replace("\r\n","\n")
    old,new=raw["original"],raw["production"]
    if old.count("@njit\n")!=4 or new.count("@njit(cache=True)\n")!=4 or (
            old.replace("@njit\n","@njit(cache=True)\n")!=new):
        raise AssertionError("Numba cached production differs from original in more than 4 decorators")
    if new!=Path("pcc_numba.py").read_text(encoding="utf-8").replace("\r\n","\n"):
        raise AssertionError("science source is not merged master pcc_numba.py")
    return raw


def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def launch(source,variant,phase,ds,seed,out,cache):
    env=os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        env[key]="1"
    env["NUMBA_DEBUG_CACHE"]="1"
    env["NUMBA_CACHE_DIR"]=str(cache.resolve())
    cmd=[sys.executable,"-m","experiments.run_post216_persistent_child",
         "--module",str(source.resolve()),"--variant",variant,
         "--phase",phase,"--dataset",ds,"--first-seed",str(seed),
         "--out",str(out.resolve())]
    before=time.perf_counter_ns()
    proc=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=240)
    after=time.perf_counter_ns()
    if proc.returncode:
        raise RuntimeError(
            f"fresh 20-fit process failed: {ds}/{seed}/{variant}/{phase}"
            +f" rc={proc.returncode}\nstdout:\n{proc.stdout[-8000:]}"
            +f"\nstderr:\n{proc.stderr[-8000:]}")
    if not out.is_file():
        raise AssertionError("fresh persistent process produced no archived JSON")
    report=json.loads(out.read_text(encoding="utf-8"))
    if report["source_sha256"]!=digest(source.read_text(encoding="utf-8")):
        raise AssertionError("child source differs from original archived Git blob")
    checkpoints={}
    cpu={}
    last_wall=0
    last_cpu=0
    for num in CHECKPOINTS:
        key=str(num)
        end=report["checkpoints_end_ns"][key]
        proc_cpu=report["checkpoints_process_cpu_ns"][key]
        actual_wall=end-before
        if (not 0<actual_wall<after-before or
                actual_wall<=last_wall or
                proc_cpu<=last_cpu or
                report["imported_ns"]<=before or
                report["script_entry_ns"]<=before):
            raise AssertionError("parent/child globally monotonic persistent-program clocks invalid")
        checkpoints[key]=actual_wall
        cpu[key]=proc_cpu
        last_wall,last_cpu=actual_wall,proc_cpu
    if report["fit20_done_ns"]<report["checkpoints_end_ns"]["20"]:
        raise AssertionError("20th finish marked before its checkpoint")
    report["app_wall_ns_at_fit_count"]=checkpoints
    report["app_cpu_ns_at_fit_count"]=cpu
    report["full_subprocess_wall_ns"]=after-before
    report["postfit_replay_and_process_exit_wall_ns"]=after-report["fit20_done_ns"]
    if report["postfit_replay_and_process_exit_wall_ns"]<=0:
        raise AssertionError("20 full replay fits incorrectly charged to timed sequence")

    logs=[
        line for line in (proc.stdout+"\n"+proc.stderr).splitlines()
        if "[cache]" in line and "pcc_numba" in line
    ]
    loaded=sum("data loaded" in line for line in logs)
    saved=sum("data saved" in line for line in logs)
    if variant=="original" and logs:
        raise AssertionError("original no-cache source unexpectedly saved/loaded compiled code")
    if variant=="production":
        if phase=="cold" and saved<1:
            raise AssertionError("first cached process failed to SAVE real Numba compiled machine code")
        if phase=="warm" and loaded<1:
            raise AssertionError("second independent cached process failed to LOAD Numba compiled code")
    report.update(cache_trace=logs,cache_data_loaded=loaded,
                  cache_data_saved=saved)
    print(f"{ds}-first{seed} {variant} {phase}: "
          f"N=1 {checkpoints['1']/1e6:.2f}ms, "
          f"N=5 {checkpoints['5']/1e6:.2f}ms, "
          f"N=20 {checkpoints['20']/1e6:.2f}ms, "
          f"native code loads={loaded}, saves={saved}",flush=True)
    return report


def cache_bytes(cache):
    return sum(f.stat().st_size for f in cache.rglob("*")
               if f.is_file() and f.suffix in (".nbc",".nbi"))


def exact_states(a,b,where):
    for key in ("input_sha","graph_hashes","labels_sha","fixture_sha",
                "final_timed_state_hashes","final_timed_mapped_sha",
                "final_timed_rng_next","replay_outcomes",
                "model_graph_stable","original_graph_prepared_once"):
        if a[key]!=b[key]:
            raise AssertionError("source matched 20-fit native state/graph/RNG diverged "
                                 +repr((where,key)))


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    frozen=freeze_source()
    results=[]
    with tempfile.TemporaryDirectory(prefix="pypcc_persist20_") as temporary:
        for ds in DATASETS:
            for first_seed in SEEDS:
                work=Path(temporary)/f"{ds}_first{first_seed}"
                work.mkdir()
                paths={}
                for variant,s in frozen.items():
                    sub=work/variant
                    sub.mkdir()
                    paths[variant]=sub/"pcc_numba.py"
                    paths[variant].write_text(s,encoding="utf-8",newline="\n")
                order=VARIANTS if first_seed==SEEDS[0] else VARIANTS[::-1]
                variants={}
                reference=None
                for variant in order:
                    cache=work/variant/"numba_disk_cache"
                    cache.mkdir()
                    launches=[]
                    for phase in PHASES:
                        out=work/variant/f"{phase}.json"
                        x=launch(paths[variant],variant,phase,ds,first_seed,
                                 out,cache)
                        if reference is None:
                            reference=x
                        exact_states(reference,x,(ds,first_seed,variant,phase))
                        launches.append(x)
                    footprint=cache_bytes(cache)
                    if variant=="original" and footprint!=0:
                        raise AssertionError("original JIT unexpectedly cached compiled source")
                    if variant=="production" and not 0<footprint<=32*1024*1024:
                        raise AssertionError("Numba compiled code cache disk bound exceeded")
                    variants[variant]=dict(
                        launches=launches,source_sha256=digest(frozen[variant]),
                        disk_cache_bytes=footprint)
                results.append(dict(dataset=ds,first_seed=first_seed,
                                    variant_order=list(order),variants=variants))
    if len(results)!=6:
        raise AssertionError("not six independent first-mask+dataset scenarios")
    payload=dict(
        schema=1,original=OLD,production=PROD,
        branch_head=subprocess.check_output(
            ["git","rev-parse","HEAD"],text=True).strip(),
        platform=sys.platform,system=platform.platform(),
        python=sys.version,original_source_sha256=digest(frozen["original"]),
        production_source_sha256=digest(frozen["production"]),
        cases=results)
    Path("post216_persistent_amortization_results.json").write_text(
        json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("COMPLETE 24 fresh Python processes, 480 timed native PCC fits, "
          "480 posttiming exact native verified replays; "
          "one original graph per interpreter; 5 full-program checkpoints; "
          "actual Numba compiled disk cache provenance all PASS on this OS",flush=True)


if __name__=="__main__":
    main()
