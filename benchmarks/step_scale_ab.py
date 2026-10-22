"""Paired A/B of original per-particle division versus hoisted step scale."""
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graph_topology import build
from pcc_step import pcc_propagate

def run(case,original):
    degree,fraction,seed=case
    args,scratch,weights=build(degree,fraction,512,"random","random",seed)
    start=time.perf_counter_ns()
    pcc_propagate(*args,1.0,2.0,*scratch,weights,300,0,10,1,
                  benchmark_original_step_scale=bool(original))
    elapsed=(time.perf_counter_ns()-start)/1e6
    state=[np.array(x,copy=True) for x in (args[7],args[9],args[10],args[11],args[12])]
    return elapsed,state

def main():
    rows=[]
    for degree in (2,8,32):
        for fraction in (0.0,.1,.25,.5,.75):
            for seed in (101,202,303,404,505):
                case=(degree,fraction,seed)
                samples=[[],[]]
                for rep in range(7):
                    states={}
                    for original in ((0,1),(1,0))[rep%2]:
                        elapsed,state=run(case,original)
                        samples[original].append(elapsed)
                        states[original]=state
                    for x,y in zip(states[0],states[1]):
                        if not np.array_equal(x,y,equal_nan=True):
                            raise AssertionError(f"State divergence: {case}")
                optimized=statistics.median(samples[0])
                original=statistics.median(samples[1])
                row=dict(degree=degree,fraction=fraction,seed=seed,
                         original_ms=original,optimized_ms=optimized,
                         change_pct=(optimized/original-1)*100)
                print(json.dumps(row),flush=True)
                rows.append(row)
    changes=[row["change_pct"] for row in rows]
    summary=dict(cases=len(rows),median_change_pct=statistics.median(changes),
                 mean_change_pct=statistics.mean(changes),
                 faster_cases=sum(x<0 for x in changes))
    print("SUMMARY "+json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
