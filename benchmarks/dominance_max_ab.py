"""Paired original-vs-early-reject Cython propagation benchmark."""
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graph_topology import build
from pcc_step import pcc_propagate

def run(case, fast):
    degree, fraction, layout, placement, seed = case
    args, scratch, weights = build(degree, fraction, 512, layout, placement, seed)
    t0 = time.perf_counter_ns()
    pcc_propagate(*args, 1.0, 2.0, *scratch, weights, 300, 0, 10, 1, max_early_reject=fast)
    elapsed = (time.perf_counter_ns()-t0)/1e6
    return elapsed, [np.array(x,copy=True) for x in (args[7],args[9],args[10],args[11],args[12])]

def main():
    results=[]
    for degree in (2,8,32):
        for fraction in (.1,.25,.5):
            for layout in ("prefix","random"):
                for placement in ("stratified","random"):
                    for seed in (101,202,303):
                        case=(degree,fraction,layout,placement,seed)
                        samples={False:[],True:[]}
                        for rep in range(7):
                            order=(False,True) if rep%2==0 else (True,False)
                            states={}
                            for fast in order:
                                elapsed,state=run(case,fast)
                                samples[fast].append(elapsed)
                                states[fast]=state
                            for a,z in zip(states[False],states[True]):
                                if not np.array_equal(a,z,equal_nan=True):
                                    raise AssertionError(f"State divergence: {case}")
                        old=statistics.median(samples[False])
                        new=statistics.median(samples[True])
                        result=dict(degree=degree,fraction=fraction,layout=layout,placement=placement,seed=seed,baseline_ms=old,early_reject_ms=new,change_pct=(new/old-1)*100)
                        print(json.dumps(result),flush=True)
                        results.append(result)
    changes=[x["change_pct"] for x in results]
    print("SUMMARY "+json.dumps(dict(cases=len(results),median_change_pct=statistics.median(changes),mean_change_pct=statistics.mean(changes),faster_cases=sum(v<0 for v in changes))),flush=True)

if __name__=="__main__":
    main()
