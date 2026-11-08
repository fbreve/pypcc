"""Fail-closed two-host scientific completeness and frozen CPU/memory decision."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

DATASETS=("iris","wine","breast_cancer","digits")
SEEDS=(32003,32027)
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
SIZES=((0,2),(178,3),(569,2),(1797,10),(30000,10),(30000,50))
SHAS={"predictions","dominance","positions","strengths","distances","owndeg"}
MASTER="ababc48986b40f083d5930e922f844aa2ea418b2"
TIMING_REPS=40
MEM_REPS=5
HOSTS=("linux","windows")


def require(ok,message):
    if not ok:
        raise AssertionError(message)


def sha(x,n=64):
    return isinstance(x,str) and len(x)==n and all(
        c in "0123456789abcdef" for c in x.lower())


def time_pos(x):
    return type(x) in (int,float) and 0<x<1e12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(folder,host):
    folder=Path(folder)
    data=read(folder/"result.json")
    freeze=(folder/"pip-freeze.txt").read_text(encoding="utf-8")
    require(len(freeze)>100 and
            data.get("protocol")=="owndeg_inplace_v1" and
            data.get("master")==MASTER and
            sha(data.get("head"),40) and
            sha(data.get("original_pcc_sha256")) and
            data.get("datasets")==list(DATASETS) and
            data.get("label_seeds")==list(SEEDS) and
            data.get("backends")==list(BACKENDS) and
            data.get("modes")==list(MODES) and
            data.get("iterations")==1000 and
            data.get("micro_seed")==31991 and
            data.get("micro_sizes")==[list(s) for s in SIZES] and
            data.get("timing_repetitions")==TIMING_REPS and
            data.get("memory_repetitions")==MEM_REPS,
            f"Frozen version/workload provenance invalid: {host}")
    h=data.get("host",{})
    require(isinstance(h.get("os"),str) and bool(h["os"])
            and isinstance(h.get("python"),str) and
            isinstance(h.get("numpy"),str) and all(
                h.get("thread_env",{}).get(k)=="1" for k in (
                    "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                    "MKL_NUM_THREADS","NUMBA_NUM_THREADS")),
            f"Thread or environment inventory invalid: {host}")
    inputs=data.get("real_inputs",{})
    require(set(inputs)==set(DATASETS) and all(
        sha(x.get("data_hash")) and
        set(x.get("graph_hashes",{}))=={"neighbors","degrees"} and
        all(sha(v) for v in x["graph_hashes"].values())
        for x in inputs.values()),f"Original input/graph hashes missing: {host}")

    rows=data.get("native_fit_pairs",[])
    require(len(rows)==48,f"Expected 48 full original native fits: {host}")
    native=defaultdict(list)
    for r in rows:
        ds,seed,backend,mode=(r.get("dataset"),r.get("seed"),
                              r.get("backend"),r.get("mode"))
        key=(ds,seed,backend,mode)
        require(ds in DATASETS and seed in SEEDS and backend in BACKENDS
                and mode in MODES and r.get("exact") is True
                and sha(r.get("label_sha256"))
                and r.get("graph_sha256")==inputs[ds]["graph_hashes"]
                and type(r.get("iterations"))==int
                and 0<r["iterations"]<=1000
                and r.get("stop_reason") in ("max_iter","early_stop")
                and r.get("backend_identity")=={
                    "numpy":"pcc_numpy.pcc_propagate_numpy",
                    "numba":"pcc_numba.pcc_propagate_numba",
                    "cython":"pcc_step.pcc_propagate"}[backend]
                and 0<=r.get("accuracy",-1)<=1
                and set(r.get("state_hashes",{}))==SHAS
                and all(sha(v) for v in r["state_hashes"].values()),
                f"Native original/candidate state check invalid: {host}/{key}")
        for timed in ("original_times","candidate_times"):
            record=r.get(timed,{})
            require(set(record)=={
                "initialization_and_dispatch_s","propagation_s",
                "postprocessing_s","fit_predict_s"} and
                all(time_pos(v) for v in record.values()),
                f"Native phase timings missing: {host}/{key}/{timed}")
        native[key].append(r)
    require(set(native)=={(ds,seed,backend,mode) for ds in DATASETS
                         for seed in SEEDS for backend in BACKENDS
                         for mode in MODES}
            and all(len(v)==1 for v in native.values()),
            f"Missing/repeated native backend pair: {host}")

    fixture=data.get("micro_fixtures",[])
    f={}
    for x in fixture:
        name=x.get("name")
        require(name in {f"{n}x{c}" for n,c in SIZES}
                and name not in f and
                x.get("shape")==[int(z) for z in name.split("x")]
                and sha(x.get("input_sha256")) and
                sha(x.get("output_sha256"))
                and x.get("returned_bytes")==8*int(name.split("x")[0])*
                    int(name.split("x")[1]),
                f"Invalid micro input/output fixture: {host}/{name}")
        f[name]=x
    require(len(f)==len(SIZES),"Missing a micro normalization fixture")
    micro=data.get("micro_time_pairs",[])
    memory=data.get("micro_memory_pairs",[])
    require(len(micro)==len(SIZES)*TIMING_REPS
            and len(memory)==len(SIZES)*MEM_REPS,
            f"Incomplete isolated normalization timings or peak trace: {host}")
    timing=defaultdict(list)
    for r in micro:
        key=(r.get("name"),r.get("repeat"))
        name,rep=key
        require(name in f and type(rep)==int and
                0<=rep<TIMING_REPS and
                r.get("order")==(
                    "original_first" if rep%2==0 else "candidate_first")
                and time_pos(r.get("original_ns"))
                and time_pos(r.get("candidate_ns")),
                f"Unbalanced/invalid micro timing {host}/{key}")
        timing[key].append(r)
    require(set(timing)=={(name,rep) for name in f
                          for rep in range(TIMING_REPS)}
            and all(len(x)==1 for x in timing.values()),
            f"Missing/duplicate micro timed sample {host}")
    mem=defaultdict(list)
    for r in memory:
        key=(r.get("name"),r.get("repeat"))
        name,rep=key
        require(name in f and type(rep)==int and
                0<=rep<MEM_REPS and
                r.get("order")==(
                    "original_first" if rep%2==0 else "candidate_first"),
                f"Invalid micro memory order {host}/{key}")
        for label in ("original","candidate"):
            entry=r.get(label,{})
            require(type(entry.get("peak_bytes"))==int
                    and type(entry.get("current_bytes"))==int
                    and type(entry.get("returned_bytes"))==int
                    and 0<=entry["current_bytes"]<=entry["peak_bytes"]<1e12
                    and entry["peak_bytes"]>0
                    and entry["returned_bytes"]==f[name]["returned_bytes"]
                    and entry.get("result_sha256")==f[name]["output_sha256"],
                    f"Output/peak sample invalid {host}/{key}/{label}")
        mem[key].append(r)
    require(set(mem)=={(name,rep) for name in f
                       for rep in range(MEM_REPS)}
            and all(len(x)==1 for x in mem.values()),
            f"Missing/duplicate traced micro peak {host}")
    return data


def report(by_host):
    names=[f"{n}x{c}" for n,c in SIZES]
    lines=[
        "# Original versus in-place owndeg normalization: exact two-OS audit",
        "",
        "Complete: **96/96 exact original PCC six-state native fit pairs**,",
        "**480/480 balanced untraced normalization timing pairs** and",
        "**60/60 independent tracemalloc tracked peak pairs**. The",
        "original model is loaded from the immutable production Git blob,",
        "and candidate source differs in precisely one post-propagation",
        "normalization expression. Dense graph arrays and native kernels",
        "were not changed. The only object-identity difference is that",
        "the candidate retains the internally allocated owndeg array",
        "instead of assigning a newly allocated normalized array.",
        "",
        "## Isolated normalization time and Python/NumPy tracked peak",
        "",
        "Each row uses 40 balanced paired CPU calls without tracing and",
        "separate 5 matched pairs with tracemalloc tracking the complete",
        "normalization function including a fresh mutable input copy",
        "and its returned array. These peaks are **not** full process RSS",
        "or exact external native malloc totals. Never pool OS timing.",
        "",
        "| OS | Input shape | Original CPU (us) | Candidate CPU (us) | Median paired CPU change | Original peak KiB | Candidate peak KiB | Median paired peak change |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    result={}
    for host,d in by_host.items():
        for name in names:
            t=[x for x in d["micro_time_pairs"] if x["name"]==name]
            m=[x for x in d["micro_memory_pairs"] if x["name"]==name]
            ratio_time=median([x["candidate_ns"]/x["original_ns"]
                               for x in t])
            ratio_memory=median([x["candidate"]["peak_bytes"]/
                                 x["original"]["peak_bytes"] for x in m])
            result[host,name]=(ratio_time,ratio_memory)
            lines.append(
                f"| {host} | {name} | "
                f"{median(x['original_ns'] for x in t)/1e3:.2f} | "
                f"{median(x['candidate_ns'] for x in t)/1e3:.2f} | "
                f"{100*(1-ratio_time):+.2f}% | "
                f"{median(x['original']['peak_bytes'] for x in m)/1024:.1f} | "
                f"{median(x['candidate']['peak_bytes'] for x in m)/1024:.1f} | "
                f"{100*(1-ratio_memory):+.2f}% |")
    lines.extend(["",
        "## Complete original PCC fit timing (two fixed label seeds per row)",
        "",
        "The whole original 1000-iteration PCC includes model initialization,",
        "original propagation and postprocessing; ratios are for",
        "original vs candidate on the **same** backend, dataset, stop mode",
        "and original graph. Both predicted labels and six final model",
        "arrays are required exact. These only have two seed pairs each",
        "and are descriptive; avoid universal latency claims.",
        "",
        "| OS | Dataset | Backend | Mode | Median full-fit candidate/original | Median postprocessing candidate/original |",
        "|---|---|---|---|---:|---:|",
    ])
    full_ratios={}
    for host,d in by_host.items():
        for ds in DATASETS:
            for b in BACKENDS:
                for mode in MODES:
                    vals=[x for x in d["native_fit_pairs"]
                          if x["dataset"]==ds and x["backend"]==b
                          and x["mode"]==mode]
                    ratios=[
                        x["candidate_times"]["fit_predict_s"]/
                        x["original_times"]["fit_predict_s"] for x in vals]
                    post=[
                        x["candidate_times"]["postprocessing_s"]/
                        x["original_times"]["postprocessing_s"]
                        for x in vals]
                    full_ratios[host,ds,b,mode]=median(ratios)
                    lines.append(f"| {host} | {ds} | {b} | {mode} | "
                                 f"{median(ratios):.3f}× | "
                                 f"{median(post):.3f}× |")
    large=[name for name in names if name in ("30000x10","30000x50")]
    small=[name for name in names if name not in large]
    large_cpu={
        host:sum(result[host,name][0]<=.75 for name in large)
        for host in HOSTS}
    large_mem={
        host:sum(result[host,name][1]<=.80 for name in large)
        for host in HOSTS}
    small_reg=[(host,name,r[0]) for host in HOSTS
               for name in small if (r:=result[host,name])[0]>1.08]
    full_reg=[(host,ds,b,mode,r) for (host,ds,b,mode),r
              in full_ratios.items() if r>1.10]
    eligible=all(z==2 for z in large_cpu.values()) and all(
        z==2 for z in large_mem.values()) and not small_reg and not full_reg
    lines.extend(["",
        "## Precommitted production-candidacy decision",
        "",
        "- Require >=25% CPU saving for both 30k-row matrix sizes on",
        "  each OS, and >=20% lower traced peak for both matrices;",
        "  no >8% CPU slowdown for the other four micro sizes;",
        "  no >10% median paired full-fit regression for any",
        "  dataset/backend/stop-mode group on either OS.",
        f"- Large matrix CPU PASS counts: Linux **{large_cpu['linux']}/2**, "
        f"Windows **{large_cpu['windows']}/2**.",
        f"- Large matrix memory PASS counts: Linux **{large_mem['linux']}/2**, "
        f"Windows **{large_mem['windows']}/2**.",
        f"- Small micro CPU regressions >8%: **{len(small_reg)}**.",
        f"- Full PCC fit regressions >10%: **{len(full_reg)}**.",
        f"- **Decision: {'PASSES all frozen candidacy gates' if eligible else 'NO-GO under the frozen rules'}**.",
        "",
        "Scientific exact-state and completeness success does not by",
        "itself justify merging; any production edit requires a",
        "separate minimal PR and full original Linux/Windows wheel/sdist CI.",
        "This experiment intentionally preserves the dense neighbor",
        "array used to reduce CPU cycles inside particle propagation.",
        "Local host CPU measurements and 2-seed fit medians are not",
        "generally representative of all environments.",
    ])
    if small_reg:
        lines.append(f"- Small regressions: {small_reg!r}.")
    if full_reg:
        lines.append(f"- Full-fit regressions: {full_reg!r}.")
    return "\n".join(lines)+"\n",eligible


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--linux",required=True)
    p.add_argument("--windows",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    by_host={host:verify(getattr(a,host),host) for host in HOSTS}
    require(by_host["linux"]["head"]==by_host["windows"]["head"] and
            by_host["linux"]["original_pcc_sha256"]==
            by_host["windows"]["original_pcc_sha256"],
            "Original source/candidate commit differed across OS")
    for ds in DATASETS:
        require(by_host["linux"]["real_inputs"][ds]==
                by_host["windows"]["real_inputs"][ds],
                f"Original standardized dataset or graph differs across OS: {ds}")
    one={x["name"]:x for x in by_host["linux"]["micro_fixtures"]}
    two={x["name"]:x for x in by_host["windows"]["micro_fixtures"]}
    require(one==two,"Fixed micro input/output changed between operating systems")
    out,decision=report(by_host)
    Path(a.output).write_text(out,encoding="utf-8")
    print(out,flush=True)
    print("Complete semantic and full sample gates PASS; "
          "candidate performance decision="+("PASS" if decision else "NO-GO"),
          flush=True)


if __name__=="__main__":
    main()
