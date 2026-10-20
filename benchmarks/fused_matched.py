"""Compare baseline and fused kernels on the same runner in isolated interpreters."""
import json, os, statistics, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE="10313ab01207d04ccf3b556bd7701d941b87c020"
VARIANTS=("baseline","fused")
ORDER=("baseline","fused","fused","baseline") * 5
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

# Paired log-ratio intervals across independent ABBA blocks.
import math
from scipy.stats import t as student_t
intervals=[]
for i,row in enumerate(results[0]["rows"]):
    ratios=[]
    for block in range(len(ORDER)//4):
        group=results[4*block:4*block+4]
        baseline=statistics.geometric_mean(x["rows"][i]["median_ms"] for x in group if x["variant"]=="baseline")
        fused=statistics.geometric_mean(x["rows"][i]["median_ms"] for x in group if x["variant"]=="fused")
        ratios.append(math.log(fused/baseline))
    mean=statistics.mean(ratios)
    margin=student_t.ppf(0.975,len(ratios)-1)*statistics.stdev(ratios)/math.sqrt(len(ratios))
    intervals.append(dict(dimension=row["dimension"],nodes=row["nodes"],classes=row["classes"],particles=row["particles"],blocks=len(ratios),change_pct=100*math.expm1(mean),ci95_low=100*math.expm1(mean-margin),ci95_high=100*math.expm1(mean+margin),log_ratios=ratios))
output["confidence_intervals"]=intervals
(ROOT/"fused-matched-result.json").write_text(json.dumps(output,indent=2))
print("\\n| Dimension | Nodes | Classes | Particles | Fused change | 95% CI |")
print("|---|---:|---:|---:|---:|---:|")
for x in intervals:
    print(f'| {x["dimension"]} | {x["nodes"]} | {x["classes"]} | {x["particles"]} | {x["change_pct"]:+.2f}% | [{x["ci95_low"]:+.2f}%, {x["ci95_high"]:+.2f}%] |')
