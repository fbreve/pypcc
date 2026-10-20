"""Differential state comparison of baseline and fused Cython kernels."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "10313ab01207d04ccf3b556bd7701d941b87c020"
SOURCE = ROOT / "pcc_step.pyx"
ORIGINAL = SOURCE.read_text()
WORKER = r"""
import json,sys
import numpy as np
from pcc_step import pcc_propagate
out={}
for seed in range(12):
    rng=np.random.default_rng(seed)
    n=48+seed*5; c=2+seed%7; p=12+seed*3; degree=2+seed%5
    neighbors=np.empty((n,degree),dtype=np.int64)
    for i in range(n):
        neighbors[i]=rng.choice(n,size=degree,replace=False)
    counts=np.full(n,degree,dtype=np.int64)
    labels=np.full(n,-1,dtype=np.int64)
    labels[rng.choice(n,size=n//4,replace=False)]=rng.integers(0,c,size=n//4)
    positions=rng.integers(0,n,size=p,dtype=np.int64)
    classes=rng.integers(0,c,size=p,dtype=np.int64)
    strength=rng.random(p,dtype=np.float64)
    distance=np.zeros((n,p),dtype=np.uint8)
    dominance=rng.dirichlet(np.ones(c),size=n).astype(np.float64)
    ownership=np.zeros((n,c),dtype=np.float64)
    zero=np.zeros(c,dtype=np.float64)
    scratch=[np.zeros(c),np.zeros(c),np.zeros(degree),np.zeros(degree),np.zeros(degree),np.zeros(degree)]
    result=pcc_propagate(neighbors,counts,labels,0.5,0.1,c,zero,positions,classes,strength,distance,dominance,ownership,1.0,2.0,*scratch,np.ones(257),150,0,10,seed+1)
    for name,value in (("positions",positions),("classes",classes),("strength",strength),("distance",distance),("dominance",dominance),("ownership",ownership)):
        out[f"{seed}_{name}"]=value
np.savez(sys.argv[1],**out)
"""
def build(source):
    SOURCE.write_text(source)
    proc=subprocess.run([sys.executable,"setup.py","build_ext","--inplace"],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stdout+"\n"+proc.stderr)
def run(output):
    proc=subprocess.run([sys.executable,"-c",WORKER,str(output)],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stdout+"\n"+proc.stderr)
try:
    baseline=subprocess.run(["git","show",f"{BASE}:pcc_step.pyx"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    with tempfile.TemporaryDirectory() as d:
        a=Path(d)/"baseline.npz"; b=Path(d)/"fused.npz"
        build(baseline); run(a)
        build(ORIGINAL); run(b)
        with np.load(a) as original, np.load(b) as fused:
            assert original.files==fused.files
            for key in original.files:
                if not np.array_equal(original[key],fused[key],equal_nan=True):
                    raise AssertionError(f"Mismatch in {key}: max absolute error {np.max(np.abs(original[key].astype(float)-fused[key].astype(float)))}")
        print(f"PASS: {len(original.files)} state arrays identical across 12 seeded scenarios")
finally:
    SOURCE.write_text(ORIGINAL)
