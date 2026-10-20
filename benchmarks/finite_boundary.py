"""Cross-platform Cython finite-preflight crossover benchmark (diagnostic only)."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np

THRESHOLDS = (1024, 4096, 16384)
SIZES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
ROUNDS = 7
REPEATS = 150
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = (ROOT / "pcc_step.pyx").read_text()
NEEDLE = "values.shape[0] * values.shape[1] >= 4096"
assert SOURCE.count(NEEDLE) == 1, "Expected one threshold expression"

def build(threshold):
    (ROOT / "pcc_step.pyx").write_text(SOURCE.replace(NEEDLE, f"values.shape[0] * values.shape[1] >= {threshold}"))
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    import importlib, pcc_step
    return importlib.reload(pcc_step)

def measure(mod, n, c=4):
    # Fresh mutable arrays for every call; isolate public-entry validation cost.
    neighbors=np.column_stack(((np.arange(n)-1)%n,(np.arange(n)+1)%n)).astype(np.int64)
    degree=np.full(n,2,dtype=np.int64)
    labels=np.full(n,-1,dtype=np.int64)
    pos=np.array([0],dtype=np.int64)
    classes=np.array([0],dtype=np.int64)
    strength=np.ones(1,dtype=np.float64)
    distance=np.zeros((n,1),dtype=np.uint8)
    dominance=np.full((n,c),1/c,dtype=np.float64)
    args=[neighbors,degree,labels,0.5,0.1,c,np.zeros(c),pos,classes,strength,distance,dominance,np.zeros((n,c))]
    kw=dict(prob=np.zeros(2),slices=np.zeros(2),dist_weights=np.ones(257))
    for _ in range(10): mod.pcc_step(*args,**kw)
    start=time.perf_counter_ns()
    for _ in range(REPEATS): mod.pcc_step(*args,**kw)
    return (time.perf_counter_ns()-start)/REPEATS/1e3

# Separate processes ensure CPython never reuses a loaded extension across builds.
if "--worker" in sys.argv:
    threshold=int(sys.argv[sys.argv.index("--worker")+1])
    module=build(threshold)
    measurements={str(n):[measure(module,n) for _ in range(ROUNDS)] for n in SIZES}
    print("BENCH_JSON="+json.dumps(dict(threshold=threshold,measurements=measurements)))
else:
    # Each variant is measured in an isolated interpreter, with repeated samples.
    # Runs are grouped by variant; repeat the workflow to assess runner drift.
    result={}
    try:
        for threshold in THRESHOLDS:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",str(threshold)],cwd=ROOT,capture_output=True,text=True)
            if proc.returncode:
                print(proc.stdout, file=sys.stderr)
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"Threshold {threshold} worker failed with exit code {proc.returncode}")
            line=next(x for x in proc.stdout.splitlines() if x.startswith("BENCH_JSON="))
            result[str(threshold)]=json.loads(line.split("=",1)[1])["measurements"]
    finally:
        (ROOT/"pcc_step.pyx").write_text(SOURCE)
    output=dict(units="microseconds per pcc_step",rounds=ROUNDS,repeats=REPEATS,results=result)
    (ROOT/"finite-boundary-result.json").write_text(json.dumps(output,indent=2))
    print("| n | elements | 1024 us | 4096 us | 16384 us |")
    print("|---:|---:|---:|---:|---:|")
    for n in SIZES:
        vals=[statistics.median(result[str(t)][str(n)]) for t in THRESHOLDS]
        print(f"| {n} | {n*4} | "+ " | ".join(f"{v:.3f}" for v in vals)+" |")
