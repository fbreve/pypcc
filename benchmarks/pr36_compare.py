"""Compare Cython entry-point overhead at fixed revisions. No package imports across builds."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
from pcc_step import pcc_step, pcc_propagate

def args(n, c):
    # A ring, two neighbors per node; no labels are updated.
    neighbors = np.column_stack(((np.arange(n)-1)%n,(np.arange(n)+1)%n)).astype(np.int64)
    degree = np.full(n,2,dtype=np.int64)
    labels = np.full(n,-1,dtype=np.int64)
    pos = np.arange(n,dtype=np.int64)
    classes = np.arange(n,dtype=np.int64)%c
    strength = np.ones(n,dtype=np.float64)
    distance = np.zeros((n,n),dtype=np.uint8) if n<=1000 else np.zeros((n,1),dtype=np.uint8)
    # For large n use only one particle to bound memory.
    if n>1000:
        pos=pos[:1].copy(); classes=classes[:1].copy(); strength=strength[:1].copy()
    dominance = np.full((n,c),1.0/c,dtype=np.float64)
    return [neighbors,degree,labels,0.5,0.1,c,np.zeros(c),pos,classes,strength,distance,dominance,np.zeros((n,c))]

def measure(fn, repeat=7):
    fn()
    vals=[]
    for _ in range(repeat):
        start=time.perf_counter_ns(); fn(); vals.append((time.perf_counter_ns()-start)/1e6)
    return vals

results=[]
for n,c in [(150,3),(1000,10),(10000,10),(30000,10)]:
    a=args(n,c)
    weights=np.ones(257,dtype=np.float64)
    prob=np.zeros(2); slices=np.zeros(2)
    def steps():
        for _ in range(1000):
            pcc_step(*a,prob=prob,slices=slices,dist_weights=weights)
    samples=measure(steps)
    results.append(dict(entry="step_1000",n=n,c=c,samples_ms=samples,median_ms=statistics.median(samples)))
    if n<=1000:
        # Propagation takes a different argument order, with explicit scratch buffers.
        b=[np.zeros(c),np.zeros(c),np.zeros(2),np.zeros(2),prob,slices]
        def propagation():
            pcc_propagate(*a,1.0,2.0,*b,weights,10000,0,10,1)
        samples=measure(propagation,5)
        results.append(dict(entry="propagate_10000",n=n,c=c,samples_ms=samples,median_ms=statistics.median(samples)))
out=dict(commit=os.environ.get("BENCH_REF"),python=sys.version,numpy=np.__version__,results=results)
Path("benchmark-result.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
