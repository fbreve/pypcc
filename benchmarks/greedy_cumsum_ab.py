"""Paired greedy cumulative-sum benchmark with exact state checks."""
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from graph_topology import build
from pcc_step import pcc_propagate
NAMES=("original","on_demand")
def run(case,mode):
    degree,fraction,layout,placement,seed=case
    args,scratch,weights=build(degree,fraction,512,layout,placement,seed)
    start=time.perf_counter_ns()
    pcc_propagate(*args,1.0,2.0,*scratch,weights,300,0,10,1,greedy_cumsum_on_demand=bool(mode))
    elapsed=(time.perf_counter_ns()-start)/1e6
    state=[np.array(x,copy=True) for x in (args[7],args[9],args[10],args[11],args[12])]
    return elapsed,state
def main():
    results=[]
    for degree in (2,8,32):
        for fraction in (.1,.25,.5):
            for layout in ("prefix","random"):
                for placement in ("stratified","random"):
                    for seed in (101,202,303):
                        case=(degree,fraction,layout,placement,seed)
                        samples=[[],[]]
                        for rep in range(7):
                            states={}
                            for mode in ((0,1),(1,0))[rep%2]:
                                elapsed,state=run(case,mode)
                                samples[mode].append(elapsed)
                                states[mode]=state
                            for mode in (1,):
                                for a,z in zip(states[0],states[mode]):
                                    if not np.array_equal(a,z,equal_nan=True):
                                        raise AssertionError(f"State divergence: {case}, mode={mode}")
                        medians=[statistics.median(x) for x in samples]
                        row=dict(degree=degree,fraction=fraction,layout=layout,placement=placement,seed=seed,original_ms=medians[0],on_demand_ms=medians[1],on_demand_change_pct=(medians[1]/medians[0]-1)*100)
                        print(json.dumps(row),flush=True)
                        results.append(row)
    summary=dict(cases=len(results))
    for name in NAMES[1:]:
        changes=[row[name+"_change_pct"] for row in results]
        summary[name]=dict(median_change_pct=statistics.median(changes),mean_change_pct=statistics.mean(changes),faster_cases=sum(v<0 for v in changes))
    print("SUMMARY "+json.dumps(summary),flush=True)
if __name__=="__main__":
    main()
