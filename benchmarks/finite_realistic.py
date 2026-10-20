"""Measure finite-preflight thresholds with realistic particle populations."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=(ROOT/"pcc_step.pyx").read_text()
NEEDLE="values.shape[0] * values.shape[1] >= 4096"
THRESHOLDS=(1024,4096,16384)
SCENARIOS=((256,4,64),(1024,8,256),(4096,8,1024),(10000,10,2000))
SAMPLES=5
MAX_ITER=500

def make_args(n,c,p):
    neighbors=np.column_stack(((np.arange(n)-1)%n,(np.arange(n)+1)%n)).astype(np.int64)
    degree=np.full(n,2,dtype=np.int64)
    labels=np.full(n,-1,dtype=np.int64)
    pos=np.arange(p,dtype=np.int64)%n
    classes=np.arange(p,dtype=np.int64)%c
    strength=np.ones(p,dtype=np.float64)
    dist=np.zeros((n,p),dtype=np.uint8)
    dom=np.full((n,c),1.0/c,dtype=np.float64)
    own=np.zeros((n,c),dtype=np.float64)
    return [neighbors,degree,labels,0.5,0.1,c,np.zeros(c),pos,classes,strength,dist,dom,own]

def run_case(module,n,c,p):
    values=[]
    for _ in range(SAMPLES):
        args=make_args(n,c,p)
        scratch=[np.zeros(c),np.zeros(c),np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2)]
        weights=np.ones(257,dtype=np.float64)
        start=time.perf_counter_ns()
        module.pcc_propagate(*args,1.0,2.0,*scratch,weights,MAX_ITER,0,10,1)
        values.append((time.perf_counter_ns()-start)/1e6)
    return values

if "--worker" in sys.argv:
    threshold=int(sys.argv[sys.argv.index("--worker")+1])
    if SOURCE.count(NEEDLE)!=1:
        raise RuntimeError("Original Cython threshold expression not found")
    (ROOT/"pcc_step.pyx").write_text(SOURCE.replace(NEEDLE,f"values.shape[0] * values.shape[1] >= {threshold}"))
    subprocess.run([sys.executable,"setup.py","build_ext","--inplace"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    import pcc_step
    results=[]
    for n,c,p in SCENARIOS:
        vals=run_case(pcc_step,n,c,p)
        results.append(dict(n=n,c=c,particles=p,samples_ms=vals,median_ms=statistics.median(vals)))
    print("BENCH_JSON="+json.dumps(results))
else:
    results={}
    try:
        for threshold in THRESHOLDS:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",str(threshold)],cwd=ROOT,capture_output=True,text=True)
            (ROOT/"pcc_step.pyx").write_text(SOURCE)
            if proc.returncode:
                print(proc.stdout,file=sys.stderr)
                print(proc.stderr,file=sys.stderr)
                raise RuntimeError(f"Threshold {threshold} failed: {proc.returncode}")
            results[str(threshold)]=json.loads(next(line.split("=",1)[1] for line in proc.stdout.splitlines() if line.startswith("BENCH_JSON=")))
    finally:
        (ROOT/"pcc_step.pyx").write_text(SOURCE)
    Path("finite-realistic-result.json").write_text(json.dumps(dict(os=sys.platform,max_iter=MAX_ITER,results=results),indent=2))
    print("| Nodes | Classes | Particles | 1024 ms | 4096 ms | 16384 ms |")
    print("|---:|---:|---:|---:|---:|---:|")
    for i,(n,c,p) in enumerate(SCENARIOS):
        vals=[results[str(t)][i]["median_ms"] for t in THRESHOLDS]
        print(f"| {n} | {c} | {p} | "+ " | ".join(f"{v:.3f}" for v in vals)+" |")
