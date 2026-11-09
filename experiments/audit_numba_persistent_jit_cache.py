"""Frozen Git Numba no-cache vs four-decorator persistent-cache experiment.

Every measurement runs an independent Python interpreter and an actual
original PCC.fit_predict, not a fake one-function Numba microbenchmark.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

BASE="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("wine","breast_cancer","digits")
SEEDS=(66013,66117)
VARIANTS=("original","candidate")
PHASES=("cold","fresh_reuse")
PROTECTED=("pcc.py","pcc_numba.py","pcc_numpy.py","pcc_graph.py",
           "pcc_step.pyx","pcc_validation.py","setup.py","benchmarks")
DECORATOR="@njit\n"
CACHED="@njit(cache=True)\n"


def ensure_source():
    subprocess.run(["git","cat-file","-e",BASE+"^{commit}"],
                   stdout=subprocess.DEVNULL,check=True)
    changed=subprocess.check_output(
        ["git","diff","--name-only",BASE,"HEAD","--",*PROTECTED],
        text=True).strip()
    if changed:
        raise AssertionError("production drift during cache experiment: "+changed)
    raw=subprocess.check_output(["git","show",BASE+":pcc_numba.py"])
    original=raw.decode("utf-8").replace("\r\n","\n")
    current=Path("pcc_numba.py").read_text(encoding="utf-8").replace("\r\n","\n")
    if current!=original:
        raise AssertionError("not the canonical pinned Numba source")
    if original.count(DECORATOR)!=4:
        raise AssertionError(
            "four original exact source @njit signatures not found; STOP")
    candidate=original.replace(DECORATOR,CACHED)
    if candidate.count(CACHED)!=4 or candidate.replace(CACHED,DECORATOR)!=original:
        raise AssertionError("candidate changed more than four disk-cache decorators")
    return (original,candidate,
            hashlib.sha256(original.encode()).hexdigest(),
            hashlib.sha256(candidate.encode()).hexdigest())


def launch(pyfile,variant,dataset,seed,phase,result_path,cache_dir):
    env=os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
        env[key]="1"
    env["NUMBA_DEBUG_CACHE"]="1"
    env["NUMBA_CACHE_DIR"]=str(cache_dir)
    proc=subprocess.run(
        [sys.executable,"-m","experiments.audit_numba_jit_cache_child",
         "--module",str(pyfile),"--dataset",dataset,
         "--seed",str(seed),"--variant",variant,
         "--phase",phase,"--out",str(result_path)],
        text=True,capture_output=True,env=env,timeout=180)
    if proc.returncode:
        raise RuntimeError(
            "native fresh process failed "
            +repr((variant,dataset,seed,phase,proc.returncode))
            +"\nstdout:\n"+proc.stdout[-7000:]
            +"\nstderr:\n"+proc.stderr[-7000:])
    if not result_path.exists():
        raise AssertionError("fresh process did not return JSON evidence")
    data=json.loads(result_path.read_text(encoding="utf-8"))
    log=proc.stdout+"\n"+proc.stderr
    relevant=[
        line for line in log.splitlines()
        if "[cache]" in line and pyfile.stem in line
    ]
    loads=sum("data loaded" in line for line in relevant)
    saves=sum("data saved" in line for line in relevant)
    if variant=="original" and relevant:
        raise AssertionError("source-original unexpectedly used Numba disk cache")
    if variant=="candidate":
        if phase=="cold" and saves<1:
            raise AssertionError("cold candidate failed to create compiled Numba cache")
        if phase=="fresh_reuse" and loads<1:
            raise AssertionError("fresh Python process did not really LOAD Numba cache")
    data["fresh_process_cache_trace"]=relevant
    data["cache_loaded_count"]=loads
    data["cache_saved_count"]=saves
    print(f"{dataset}-labels{seed} {variant} {phase}: "
          f"CPU={data['first']['cpu_ns']/1e6:.2f}ms "
          f"hot={data['hot']['cpu_ns']/1e6:.2f}ms "
          f"cache loaded={loads} saved={saves}",flush=True)
    return data


def cache_bytes(directory):
    p=Path(directory)
    return sum(f.stat().st_size for f in p.rglob("*")
               if f.is_file() and f.suffix in (".nbi",".nbc"))


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    original,candidate,base_sha,candidate_sha=ensure_source()
    rows=[]
    with tempfile.TemporaryDirectory(prefix="pypcc_jit_repro_") as tmp:
        for dataset in DATASETS:
            for seed in SEEDS:
                working=Path(tmp)/f"{dataset}_labels{seed}"
                working.mkdir()
                modules={}
                for variant,source in (("original",original),("candidate",candidate)):
                    pyfile=working/f"frozen_{variant}_numba.py"
                    pyfile.write_text(source,encoding="utf-8",newline="\n")
                    modules[variant]=pyfile
                source_hash={
                    k:hashlib.sha256(v.read_bytes()).hexdigest()
                    for k,v in modules.items()}
                if source_hash!={"original":base_sha,"candidate":candidate_sha}:
                    raise AssertionError("fresh module source bytes mismatch")
                # Counterbalance which variant compiles first by seed.
                order=("original","candidate") if seed==SEEDS[0] else (
                    "candidate","original")
                records={}
                for variant in order:
                    cache_dir=working/f"{variant}_disk_cache"
                    cache_dir.mkdir()
                    cold_file=working/f"{variant}_cold.json"
                    warm_file=working/f"{variant}_fresh_reuse.json"
                    first=launch(modules[variant],variant,dataset,seed,
                                 "cold",cold_file,cache_dir)
                    second=launch(modules[variant],variant,dataset,seed,
                                  "fresh_reuse",warm_file,cache_dir)
                    records[variant]=dict(cold=first,fresh_reuse=second,
                                          cache_size_bytes=cache_bytes(cache_dir),
                                          source_sha256=source_hash[variant])
                # Verify native state and original seeded MT19937 in both
                # variants and both distinct fresh Python sessions.
                a,b=records["original"],records["candidate"]
                for variant,run in records.items():
                    for phase in PHASES:
                        item=run[phase]
                        for f in ("fixture_sha","input_sha","ordered_graph_sha",
                                  "labels_sha"):
                            if item[f]!=a["cold"][f]:
                                raise AssertionError("native source/label graph mismatch")
                        for key in ("state_hashes","native_rng_next","stop_info","layout"):
                            if item["first"][key]!=a["cold"]["first"][key]:
                                raise AssertionError("original-candidate cache alters native "
                                                     +repr((dataset,seed,variant,phase,key)))
                rows.append(dict(dataset=dataset,seed=seed,
                                 phase_order=list(order),variants=records))
    if len(rows)!=6:
        raise AssertionError("missing complete 3 real datasets by 2 independent masks")
    results=dict(schema=1,base=BASE,platform=sys.platform,
                 head=subprocess.check_output(
                     ["git","rev-parse","HEAD"],text=True).strip(),
                 host=platform.platform(),python=sys.version,
                 numba=__import__("numba").__version__,
                 source_sha256=base_sha,candidate_sha256=candidate_sha,
                 thread_limit="1",cases=rows)
    Path("numba_persistent_cache_results.json").write_text(
        json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("PASS 24 independent fresh Python processes and 48 native whole fits "
          "with exact source/cache/state/RNG evidence ON THIS OS",flush=True)


if __name__=="__main__":
    main()
