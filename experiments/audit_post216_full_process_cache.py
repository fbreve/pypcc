"""Original source vs ACTUAL merged production Numba cache: whole Python process.

No code in production is changed. Keep all per-process data, including cold,
and keep diagnostic hashing strictly OUTSIDE parent-to-first-prediction clocks.
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
SEEDS=(68013,68117)
PHASES=("cold","warm1","warm2","warm3")
VARIANTS=("original","production")
PROTECTED=("pcc.py","pcc_numba.py","pcc_numpy.py","pcc_graph.py",
           "pcc_step.pyx","pcc_validation.py","setup.py","benchmarks")


def canonical_sources():
    for rev in (OLD,PROD):
        subprocess.run(["git","cat-file","-e",rev+"^{commit}"],
                       check=True,stdout=subprocess.DEVNULL)
    changed=subprocess.check_output(
        ["git","diff","--name-only",PROD,"HEAD","--",*PROTECTED],
        text=True).strip()
    if changed:
        raise AssertionError("read-only science changed production source "+changed)
    original=subprocess.check_output(
        ["git","show",OLD+":pcc_numba.py"]).decode("utf-8")
    production=subprocess.check_output(
        ["git","show",PROD+":pcc_numba.py"]).decode("utf-8")
    original=original.replace("\r\n","\n")
    production=production.replace("\r\n","\n")
    if (original.count("@njit\n")!=4 or
            production.count("@njit(cache=True)\n")!=4 or
            original.replace("@njit\n","@njit(cache=True)\n")!=production):
        raise AssertionError("post-#216 Numba source differs by more than 4 decorators")
    from_disk=Path("pcc_numba.py").read_text(
        encoding="utf-8").replace("\r\n","\n")
    if production!=from_disk:
        raise AssertionError("new master installed Numba source does not match Git")
    return {
        "original":original,
        "production":production,
    }


def file_digest(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_child(source,variant,phase,dataset,seed,out,cache):
    env=os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="1",
        NUMBA_DEBUG_CACHE="1",NUMBA_CACHE_DIR=str(cache.resolve()),
    )
    command=[
        sys.executable,"-m","experiments.run_post216_full_process_child",
        "--source",str(source.resolve()),"--variant",variant,
        "--phase",phase,"--dataset",dataset,
        "--seed",str(seed),"--result",str(out.resolve())]
    parent_start_ns=time.perf_counter_ns()
    cp=subprocess.run(
        command,capture_output=True,text=True,timeout=160,env=env)
    parent_end_ns=time.perf_counter_ns()
    if cp.returncode!=0:
        raise RuntimeError(
            "Fresh original/production complete PCC application FAILED "
            +repr((variant,phase,dataset,seed,cp.returncode))
            +"\nstdout:\n"+cp.stdout[-7000:]+"\nstderr:\n"+cp.stderr[-7000:])
    if not out.is_file():
        raise AssertionError("fresh process did not save first-fit state evidence")
    data=json.loads(out.read_text(encoding="utf-8"))
    timers=data["phase_timing"]
    first_result_end_ns=timers["fit_end_ns"]
    true_total=first_result_end_ns-parent_start_ns
    total_process=parent_end_ns-parent_start_ns
    if (not 0<true_total<total_process or
            not 0<timers["total_cpu_to_first_fit_ns"] or
            timers["imported_ns"]<=parent_start_ns):
        raise AssertionError(
            "inconsistent globally monotonic parent/child Python performance clocks")
    # Includes Python interpreter creation (starts BEFORE child import), all
    # package imports, original sklearn load/scale/kNN, set_graph + first fit.
    data["app_first_result_wall_ns"]=true_total
    data["app_first_result_cpu_ns"]=timers["total_cpu_to_first_fit_ns"]
    data["python_spawn_and_import_wall_ns"]=timers["imported_ns"]-parent_start_ns
    data["total_process_including_postfit_checks_wall_ns"]=total_process
    data["postfit_verification_and_process_exit_ns"]=parent_end_ns-first_result_end_ns

    # Genuine native Numba cache debug logs from the REAL pcc_numba source.
    relevant=[
        line for line in (cp.stdout+"\n"+cp.stderr).splitlines()
        if "[cache]" in line and "pcc_numba" in line
    ]
    loaded=sum("data loaded" in line for line in relevant)
    saved=sum("data saved" in line for line in relevant)
    data.update(cache_trace=relevant,cache_load_count=loaded,
                cache_save_count=saved)
    if variant=="original" and relevant:
        raise AssertionError("no-cache original unexpectedly read/wrote disk native code")
    if variant=="production":
        if phase=="cold" and saved<1:
            raise AssertionError("production first process did not SAVE Numba machine code")
        if phase!="cold" and loaded<1:
            raise AssertionError("production new process did not LOAD Numba machine code")
    print(f"{dataset}-labels{seed} {variant} {phase}: "
          f"full-app-wall={true_total/1e6:.2f}ms "
          f"full-app-cpu={data['app_first_result_cpu_ns']/1e6:.2f}ms "
          f"fit={timers['fit_wall_ns']/1e6:.2f}ms "
          f"loaded={loaded} saved={saved} "
          f"iters={data['native_stop']['effective_iterations']}",flush=True)
    return data


def native_disk_bytes(cache):
    return sum(f.stat().st_size for f in cache.rglob("*")
               if f.is_file() and f.suffix in (".nbi",".nbc"))


def identical_science(first,other):
    for item in ("fixture_sha","graph_hashes","input_hash","label_hash",
                 "state_hashes","mapped_labels_hash","native_rng_endpoint",
                 "native_stop","real_nodes","classes","true_original_pcc_dispatch"):
        if first[item]!=other[item]:
            raise AssertionError("whole-app fresh-process Numba source/stop/RNG drift "+
                                 repr((other["dataset"],other["seed"],item)))


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    canonical=canonical_sources()
    record=[]
    with tempfile.TemporaryDirectory(prefix="post216_original_full_app_") as tmp:
        for ds in DATASETS:
            for seed in SEEDS:
                work=Path(tmp)/f"{ds}_labels{seed}"
                work.mkdir()
                modules={}
                for variant,source_text in canonical.items():
                    path=work/variant
                    path.mkdir()
                    src=path/"pcc_numba.py"
                    src.write_text(source_text,encoding="utf-8",newline="\n")
                    modules[variant]=src
                    if file_digest(source_text)!=hashlib.sha256(
                            src.read_bytes()).hexdigest():
                        raise AssertionError("filesystem native source changed unexpectedly")
                order=(VARIANTS if seed==SEEDS[0] else VARIANTS[::-1])
                variants={}
                control=None
                for variant in order:
                    cache=work/variant/"numba_private_disk"
                    cache.mkdir()
                    launches=[]
                    for phase in PHASES:
                        out=work/variant/f"{phase}.json"
                        d=parse_child(modules[variant],variant,phase,ds,seed,out,cache)
                        if d["source_sha256"]!=file_digest(canonical[variant]):
                            raise AssertionError("incorrect original/production source load")
                        if control is None:control=d
                        identical_science(control,d)
                        launches.append(d)
                    footprint=native_disk_bytes(cache)
                    if variant=="original" and footprint:
                        raise AssertionError("original JIT unexpectedly writes cached Numba data")
                    if variant=="production" and not (0<footprint<=32*1024*1024):
                        raise AssertionError("production Numba compiled code disk cache out of budget")
                    variants[variant]=dict(launches=launches,
                                           disk_code_cache_bytes=footprint,
                                           source_sha256=file_digest(canonical[variant]))
                record.append(dict(dataset=ds,seed=seed,
                                   order=list(order),variants=variants))
    if len(record)!=len(DATASETS)*len(SEEDS):
        raise AssertionError("missing original 6 natural data/mask cases")
    payload=dict(schema=1,production=PROD,original=OLD,
                 branch_head=subprocess.check_output(
                     ["git","rev-parse","HEAD"],text=True).strip(),
                 platform=sys.platform,host=platform.platform(),
                 python=sys.version,os_cpu=platform.processor(),
                 original_source_sha256=file_digest(canonical["original"]),
                 production_source_sha256=file_digest(canonical["production"]),
                 scientific_no_production_diff=True,cases=record)
    Path("post216_full_app_cache_results.json").write_text(
        json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("COMPLETE 48 real end-to-end fresh Python interpreter launches "
          "(3 datasets × 2 masks × 2 source versions × 4 launches), "
          "all native source/stop/6-state/RNG cache-load parity PASS on this OS",
          flush=True)


if __name__=="__main__":
    main()
