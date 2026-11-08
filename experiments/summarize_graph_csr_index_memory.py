"""Fail-closed two-host verification of complete CSR-index memory experiment.

Keep negative memory and timing outcomes: only scientific completeness and
same-backend equivalence are CI gates; the frozen production decision is shown
in the report rather than used to filter results.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

REAL=("iris","wine","breast_cancer","digits")
KS=(1,3,10)
SYNTH=("empty_10000","isolated_30000_1000",
       "sparse_10000_1000","medium_1000_3000",
       "dense_1000_30000","dense_10000_100000",
       "dense_30000_100000","loops_1000_3000",
       "duplicates_1000_3000")
HIGH_EDGE=("dense_1000_30000","dense_10000_100000",
           "dense_30000_100000")
HOSTS=("linux","windows")
SEEDS=(28639,28711)
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
STATES={"predictions","dominance","positions","strengths",
        "distances","owndeg"}
MASTER="5a8cab9978cf367b8e02341ac856f297b7f2c2c5"
TIME_REPS=20
MEM_REPS=5
CASE_ORDER=[
    f"{name}-k{k}-{source}"
    for name in REAL for k in KS for source in ("knn","edge_index")
]+list(SYNTH)


def require(ok,message):
    if not ok:
        raise AssertionError(message)


def sha(x,n=64):
    return isinstance(x,str) and len(x)==n and all(
        c in "0123456789abcdef" for c in x.lower())


def hashes(x):
    return isinstance(x,dict) and set(x)=={"neighbors","degrees"} and all(
        sha(v) for v in x.values())


def measured_ns(x):
    return type(x) in (float,int) and 0<x<10**13


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(root,host):
    root=Path(root)
    d=read(root/"results.json")
    freeze=(root/"pip-freeze.txt").read_text(encoding="utf-8")
    require(
        len(freeze)>100 and d.get("protocol")=="csr_index_memory_v1"
        and d.get("master")==MASTER and sha(d.get("head"),40)
        and sha(d.get("original_source_sha256"))
        and d.get("datasets")==list(REAL) and d.get("ks")==list(KS)
        and d.get("synthetic")==list(SYNTH)
        and d.get("timing_repetitions")==TIME_REPS
        and d.get("memory_repetitions")==MEM_REPS
        and d.get("label_seeds")==list(SEEDS)
        and d.get("backends")==list(BACKENDS)
        and d.get("modes")==list(MODES)
        and d.get("max_iter")==1000,
        f"Frozen protocol or Python environment incomplete: {host}")
    env=d.get("host",{})
    require(isinstance(env.get("os"),str)
            and isinstance(env.get("python"),str)
            and isinstance(env.get("numpy"),str)
            and all(env.get("thread_env",{}).get(x)=="1" for x in
                    ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                     "MKL_NUM_THREADS","NUMBA_NUM_THREADS")),
            f"Missing fixed host thread settings: {host}")
    conditions=d.get("conditions",[])
    times=d.get("full_builder_timings",[])
    mem=d.get("traced_memory_pairs",[])
    native=d.get("native_pairs",[])
    ctrl=d.get("edge_and_tie_controls",[])
    require(len(conditions)==33 and len(times)==660
            and len(mem)==165 and len(native)==96 and len(ctrl)==20,
            f"Missing scientific observations: {host}")
    check={}
    for c in conditions:
        name=c.get("case")
        require(name in CASE_ORDER and name not in check
                and c.get("kind")==("real" if name not in SYNTH else "synthetic")
                and sha(c.get("input_sha256"))
                and hashes(c.get("original_hashes"))
                and isinstance(c.get("returned_bytes"),int)
                and c["returned_bytes"]>0,
                f"Graph fixture incomplete or duplicated: {host}/{name}")
        if name not in SYNTH:
            ds,k,source=name.rsplit("-k",1)[0],None,None
            # The last split removes the -k... suffix; assert stored
            # dataset, k and construction source independently.
            actual_ds=c.get("dataset")
            actual_k=c.get("k")
            actual_source=c.get("source")
            require(actual_ds in REAL and actual_k in KS
                    and actual_source in ("knn","edge_index")
                    and name==f"{actual_ds}-k{actual_k}-{actual_source}"
                    and c.get("n_nodes") in (150,178,569,1797)
                    and isinstance(c.get("n_input_edges"),int),
                    f"Real kNN graph condition wrong: {host}/{name}")
        else:
            require(type(c.get("n_nodes"))==int and
                    type(c.get("n_input_edges"))==int and
                    c["n_nodes"]>0 and c["n_input_edges"]>=0 and
                    type(c.get("self_loops"))==bool and
                    c["self_loops"]==(name=="loops_1000_3000"),
                    f"Synthetic edge-index case wrong: {host}/{name}")
        check[name]=c
    require(set(check)==set(CASE_ORDER),
            f"Unregistered graph case: {host}")
    groups=defaultdict(list)
    for t in times:
        name,rep=t.get("case"),t.get("repeat")
        key=(name,rep)
        require(name in check and type(rep)==int
                and 0<=rep<TIME_REPS
                and t.get("order")==(
                    "original_first" if rep%2==0 else "candidate_first")
                and measured_ns(t.get("reference_ns"))
                and measured_ns(t.get("candidate_ns")),
                f"Invalid paired complete graph time: {host}/{key}")
        groups[key].append(t)
    require(set(groups)=={(name,r) for name in CASE_ORDER
                          for r in range(TIME_REPS)}
            and all(len(v)==1 for v in groups.values()),
            f"Missing/duplicate 660 graph timing pairs: {host}")
    mgroups=defaultdict(list)
    for m in mem:
        name,rep=m.get("case"),m.get("repeat")
        key=(name,rep)
        require(name in check and type(rep)==int
                and 0<=rep<MEM_REPS
                and m.get("order")==(
                    "original_first" if rep%2==0 else "candidate_first"),
                f"Unplanned isolated allocation pair: {host}/{key}")
        for mode in ("reference","candidate"):
            x=m.get(mode,{})
            require(type(x.get("current_traced_bytes"))==int
                    and type(x.get("peak_traced_bytes"))==int
                    and type(x.get("returned_bytes"))==int
                    and 0<x["current_traced_bytes"]<=x["peak_traced_bytes"]<10**12
                    and x["returned_bytes"]==check[name]["returned_bytes"],
                    f"Invalid tracemalloc peak or changed output: {host}/{key}/{mode}")
        mgroups[key].append(m)
    require(set(mgroups)=={(name,r) for name in CASE_ORDER
                          for r in range(MEM_REPS)}
            and all(len(v)==1 for v in mgroups.values()),
            f"Missing/duplicate 165 isolated traced-memory pairs: {host}")
    ngroups=defaultdict(list)
    for n in native:
        ds,source,seed,backend,mode=(
            n.get("dataset"),n.get("source"),n.get("seed"),
            n.get("backend"),n.get("mode"))
        key=(ds,source,seed,backend,mode)
        require(ds in REAL and source in ("knn","edge_index")
                and seed in SEEDS and backend in BACKENDS
                and mode in MODES and n.get("exact") is True
                and n.get("iteration_cap")==1000
                and type(n.get("effective_iterations"))==int
                and 0<n["effective_iterations"]<=1000
                and n.get("stop_reason") in ("max_iter","early_stop")
                and n.get("backend_identity")=={
                    "numpy":"pcc_numpy.pcc_propagate_numpy",
                    "numba":"pcc_numba.pcc_propagate_numba",
                    "cython":"pcc_step.pcc_propagate"}[backend]
                and 0<=n.get("accuracy_unlabeled",-1)<=1
                and set(n.get("state_hashes",{}))==STATES
                and all(sha(v) for v in n["state_hashes"].values()),
                f"Original backend state fails: {host}/{key}")
        ngroups[key].append(n)
    require(set(ngroups)=={(ds,source,seed,b,m) for ds in REAL
        for source in ("knn","edge_index")
        for seed in SEEDS for b in BACKENDS for m in MODES}
        and all(len(v)==1 for v in ngroups.values()),
        f"Missing/duplicate 96 full native state pairs: {host}")
    controls_expected={
        "n0","n0_has_edge","bad_shape","bad_negative_nodecount",
        "bad_negative_vertex","bad_oob_vertex","loops_only",
        "noncontiguous_edges",
    }
    controls_expected|={
        f"{kind}-k{k}-p{p}"
        for kind in ("duplicate_ties","fortran_ties")
        for k in (1,3,9) for p in (1,2)}
    require(len(controls_expected)==20 and
            {x.get("case") for x in ctrl}==controls_expected
            and len({x["case"] for x in ctrl})==20
            and all(x.get("exact") is True
                    and sha(x.get("input_sha256"))
                    and x.get("outcome",{}).get("status") in ("success","error")
                    for x in ctrl),
            f"Missing adversarial/invalid original behavior controls: {host}")
    return d,check


def quantile(seq,p):
    values=sorted(seq)
    v=(len(values)-1)*p
    i=int(v)
    return values[i]*(1-(v-i))+values[min(i+1,len(values)-1)]*(v-i)


def summarize(both):
    lines=[
        "# Native CSR index dtype: exact full graph, native state, runtime and traced allocation",
        "",
        "Strict **two-OS science completed**: 1,320/1,320 complete paired",
        "graph-build timings; 330/330 separately traced isolated memory pairs",
        "(660 individual original/candidate peak samples); 192/192 original",
        "PCC six-array native state/stopping pairs; and 40/40 original",
        "adversarial/invalid behavior controls. No production change",
        "is authorized by CI success alone.",
        "",
        "## Complete graph runtime and tracemalloc peak by original input",
        "",
        "All values are medians on **one host**, 20 alternating full-builder",
        "time pairs, 5 independent traced peak pairs per condition.",
        "Peak includes the retained public output in the measured call;",
        "it is a tracemalloc-reported **Python/NumPy tracked allocation peak**",
        "not process RSS, precise native peak, or Cython/Numba memory.",
        "Separate timing does not run with tracing. Ratios are medians",
        "of paired candidate/reference values; zero observations dropped.",
        "",
        "| Host | Case | Public output MiB | Original peak MiB | Candidate peak MiB | Paired peak reduction | Paired time reduction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ratios={}
    for host,(d,cases) in both.items():
        for case in CASE_ORDER:
            ts=[x for x in d["full_builder_timings"]
                if x["case"]==case]
            mem=[x for x in d["traced_memory_pairs"]
                 if x["case"]==case]
            peak=[x["candidate"]["peak_traced_bytes"]/
                  x["reference"]["peak_traced_bytes"] for x in mem]
            duration=[x["candidate_ns"]/x["reference_ns"] for x in ts]
            ratios[host,case]=(median(peak),median(duration))
            lines.append(
                f"| {host} | {case} | "
                f"{cases[case]['returned_bytes']/1048576:.3f} | "
                f"{median(x['reference']['peak_traced_bytes'] for x in mem)/1048576:.3f} | "
                f"{median(x['candidate']['peak_traced_bytes'] for x in mem)/1048576:.3f} | "
                f"{100*(1-median(peak)):+.2f}% | "
                f"{100*(1-median(duration)):+.2f}% |")
    good={host:sum(ratios[host,name][0]<=.90 for name in HIGH_EDGE)
          for host in HOSTS}
    time_reg=[(host,name,ratios[host,name][1])
              for host in HOSTS for name in CASE_ORDER
              if ratios[host,name][1]>1.10]
    mem_reg=[(host,name,ratios[host,name][0])
             for host in HOSTS for name in CASE_ORDER
             if ratios[host,name][0]>1.05]
    eligible=(all(v==len(HIGH_EDGE) for v in good.values())
              and not time_reg and not mem_reg)
    lines.extend([
        "",
        "## Frozen production promotion decision",
        "",
        "Required all real/synthetic graph outputs and native six-state",
        "arrays **exact** on both OS (already gated), >=10% traced",
        "peak reduction in each of three preselected high-edge cases",
        "on each OS, no >10% full-builder median time regression",
        "in any case, and no >5% traced peak regression in any case.",
        f"- High-edge memory conditions >=10% reduction: "
        f"Linux **{good['linux']}/3**, Windows **{good['windows']}/3**.",
        f"- Full-builder timing regressions >10%: **{len(time_reg)}**.",
        f"- Traced-peak regressions >5%: **{len(mem_reg)}**.",
        f"- **Decision: {'MEETS the frozen candidate gate' if eligible else 'NO-GO under the frozen gate'}**.",
        "",
        "## Scope and limitations",
        "",
        "- The return values remain **dense padded int64 arrays**;",
        "  this optimization only reduces transient CSR index/neighbor",
        "  conversion copies when SciPy uses smaller native indices.",
        "- The index width is taken from `adj.indices.dtype` and",
        "  `adj.indptr.dtype`, **never hard-coded to int32**;",
        "  where SciPy requires int64 index storage, it stays int64.",
        "- Tracemalloc is a sampled Python/NumPy tracing mechanism",
        "  and must not be portrayed as a whole-process OS RSS or",
        "  exact peak of non-Python native allocations. The returned",
        "  output bytes were reported separately and did not change.",
        "- This does not change sklearn `NearestNeighbors('auto')`,",
        "  stop criteria, graph topology, node order, backend semantics",
        "  or the public API. The benchmark is performed on GitHub-hosted",
        "  runners with differing CPU/OS/NumPy builds.",
        "- Zero samples were filtered, and timing and memory results",
        "  are separate paired measurements. A negative outcome should",
        "  remain documented, not repaired by changing frozen inputs.",
        "- Even a positive result authorizes only extraction of a",
        "  **separate tiny master-based production candidate** with",
        "  independent four-platform CI, not merging this experiment.",
    ])
    return "\n".join(lines)+"\n",eligible


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--linux",required=True)
    p.add_argument("--windows",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    both={host:verify(getattr(a,host),host) for host in HOSTS}
    require(both["linux"][0]["head"]==both["windows"][0]["head"]
            and both["linux"][0]["original_source_sha256"]==
                both["windows"][0]["original_source_sha256"],
            "Original Git blob or tested candidate commit differed between OS")
    for key in CASE_ORDER:
        l,r=both["linux"][1][key],both["windows"][1][key]
        require(l["input_sha256"]==r["input_sha256"]
                and l["original_hashes"]==r["original_hashes"]
                and l["returned_bytes"]==r["returned_bytes"],
                f"Same original source/graph input changed between hosts: {key}")
    report,eligible=summarize(both)
    Path(a.output).write_text(report,encoding="utf-8")
    print(report,flush=True)
    print(f"Scientific completeness and exact semantic gates passed; "
          f"frozen memory/time promotion gate={'PASS' if eligible else 'NO-GO'}",
          flush=True)


if __name__=="__main__":
    main()
