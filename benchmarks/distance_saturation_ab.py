"""Paired destination distance-read benchmark with saturation controls."""
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graph_topology import build
from pcc_step import pcc_propagate

def run(case,mode):
    degree,fraction,seed,saturation=case
    args,scratch,weights=build(degree,fraction,512,"random","random",seed)
    if saturation == "full":
        args[10][:] = 255
    elif saturation == "half":
        args[10][:,::2] = 255
    elif saturation != "zero":
        raise ValueError(saturation)
    start=time.perf_counter_ns()
    pcc_propagate(*args,1.0,2.0,*scratch,weights,300,0,10,1,
                  benchmark_force_distance_read=(mode==0))
    elapsed=(time.perf_counter_ns()-start)/1e6
    state=[np.array(x,copy=True) for x in (args[7],args[9],args[10],args[11],args[12])]
    return elapsed,state

def main():
    rows=[]
    for saturation in ("zero","half","full"):
        for degree in (2,8,32):
            for fraction in (.1,.25,.5):
                for seed in (101,202,303):
                    case=(degree,fraction,seed,saturation)
                    samples=[[],[]]
                    for rep in range(7):
                        states={}
                        for mode in ((0,1),(1,0))[rep%2]:
                            elapsed,state=run(case,mode)
                            samples[mode].append(elapsed)
                            states[mode]=state
                        for a,z in zip(states[0],states[1]):
                            if not np.array_equal(a,z,equal_nan=True):
                                raise AssertionError(f"State divergence: {case}")
                    old,new=(statistics.median(x) for x in samples)
                    row=dict(saturation=saturation,degree=degree,fraction=fraction,seed=seed,
                             original_ms=old,optimized_ms=new,change_pct=(new/old-1)*100)
                    print(json.dumps(row),flush=True)
                    rows.append(row)
    summary={"cases":len(rows),"by_saturation":{}}
    for saturation in ("zero","half","full"):
        changes=[r["change_pct"] for r in rows if r["saturation"]==saturation]
        summary["by_saturation"][saturation]=dict(
            median_change_pct=statistics.median(changes),
            mean_change_pct=statistics.mean(changes),
            faster_cases=sum(v<0 for v in changes))
    print("SUMMARY "+json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
