"""Compare baseline and fused kernels on the same runner in isolated interpreters."""
import json, os, statistics, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE="perf/cython-hybrid-finite-preflight"
VARIANTS=("baseline","fused")
ORDER=("baseline","fused","fused","baseline")
SCRIPT=ROOT/"benchmarks"/"kernel_scaling.py"

def run_variant(variant, iteration):
    if variant=="baseline":
        source=subprocess.run(["git","show",f"{BASE}:pcc_step.pyx"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    else:
        source=ORIGINAL
    (ROOT/"pcc_step.pyx").write_text(source)
    build=subprocess.run([sys.executable,"setup.py","build_ext","--inplace"],cwd=ROOT,capture_output=True,text=True)
    if build.returncode:
        raise RuntimeError(build.stdout+"\n"+build.stderr)
    proc=subprocess.run([sys.executable,str(SCRIPT)],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stdout+"\n"+proc.stderr)
    result=json.loads((ROOT/"kernel-scaling-result.json").read_text())
    return result["results"]

ORIGINAL=(ROOT/"pcc_step.pyx").read_text()
results=[]
try:
    for iteration,variant in enumerate(ORDER):
        rows=run_variant(variant,iteration)
        results.append(dict(variant=variant,order=iteration,rows=rows))
finally:
    (ROOT/"pcc_step.pyx").write_text(ORIGINAL)
output=dict(order=ORDER,results=results)
(ROOT/"fused-matched-result.json").write_text(json.dumps(output,indent=2))
print("| Dimension | Nodes | Classes | Particles | Baseline ms | Fused ms | Change |")
print("|---|---:|---:|---:|---:|---:|---:|")
for i,r in enumerate(results[0]["rows"]):
    base=statistics.median([run["rows"][i]["median_ms"] for run in results if run["variant"]=="baseline"])
    fused=statistics.median([run["rows"][i]["median_ms"] for run in results if run["variant"]=="fused"])
    print(f'| {r["dimension"]} | {r["nodes"]} | {r["classes"]} | {r["particles"]} | {base:.3f} | {fused:.3f} | {(fused/base-1)*100:+.1f}% |')
