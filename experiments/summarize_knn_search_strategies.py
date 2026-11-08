"""Strict fail-closed cross-host audit of *all* frozen sklearn engine trials."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

DATASETS=("iris","wine","breast_cancer","digits")
K_VALUES=(1,3,10)
METHODS=("auto","brute","kd_tree","ball_tree")
OTHERS=METHODS[1:]
HOSTS=("linux","windows")
PHASE_REPS=10
REPS=20
SYNTHETIC=("duplicate_pairs","rectangular_grid",
           "fortran_noncontiguous","constant_ties","equidistant_chain")
BACKENDS=("numba","cython")
MASTER="8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128"
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}


def check(condition,message):
    if not condition:
        raise AssertionError(message)


def sha(x,n=64):
    return isinstance(x,str) and len(x)==n and all(
        ch in "0123456789abcdef" for ch in x.lower())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_pair_metadata(x):
    return isinstance(x,dict) and set(x)=={"neighbors","degrees"} and all(
        sha(v) for v in x.values())


def timing(x):
    return type(x) in (int,float) and 0<x<1e12


def validate(root,host):
    root=Path(root)
    d=read(root/"results.json")
    versions=(root/"pip-freeze.txt").read_text(encoding="utf-8")
    check(len(versions)>80 and d.get("protocol")=="knn_search_strategy_v1",
          f"Missing original real run or dependency versions {host}")
    check(
        d.get("master")==MASTER and sha(d.get("head"),40)
        and d.get("datasets")==list(DATASETS)
        and d.get("k_values")==list(K_VALUES)
        and d.get("algorithms")==list(METHODS)
        and d.get("repetitions")==REPS
        and d.get("phase_repetitions")==PHASE_REPS
        and d.get("semantic_seed")==28109
        and d.get("semantic_cap")==1000
        and d.get("semantic_backends")==list(BACKENDS)
        and d.get("synthetic_fixtures")==list(SYNTHETIC)
        and set(d.get("source_sha256",{}))==set(METHODS)
        and all(sha(v) for v in d["source_sha256"].values()),
        f"Original master / workload / source provenance changed: {host}")
    env=d.get("host",{})
    check(
        isinstance(env.get("os"),str) and bool(env["os"]) and
        isinstance(env.get("python"),str) and bool(env["python"]) and
        all(env.get("env",{}).get(key)=="1" for key in
            ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
             "MKL_NUM_THREADS","NUMBA_NUM_THREADS")),
        f"Frozen threadpool environment missing: {host}")
    case_rows=d.get("real_cases",[])
    full=d.get("full_graph_pairs",[])
    phase=d.get("search_phase_rows",[])
    synthetics=d.get("synthetic_cases",[])
    native=d.get("native_cases",[])
    check(
        len(case_rows)==36 and len(full)==720 and
        len(phase)==480 and len(synthetics)==90 and len(native)==24,
        f"Partial experiment outcomes {host}")
    cases={}
    case_inputs={}
    original_sha={}
    for r in case_rows:
        ds,k,method=r.get("dataset"),r.get("k"),r.get("method")
        key=(ds,k,method)
        check(ds in DATASETS and k in K_VALUES and
              method in OTHERS and key not in cases and
              r.get("original_method") in ("brute","kd_tree","ball_tree")
              and r.get("candidate_method")==method and
              sha(r.get("input_sha256"))
              and isinstance(r.get("raw_neighbor_different_rows"),int)
              and 0<=r["raw_neighbor_different_rows"]<=
                  {"iris":150,"wine":178,"breast_cancer":569,
                   "digits":1797}[ds]
              and type(r.get("exact")) is bool
              and type(r.get("different_rows")) is int
              and type(r.get("missing_edges")) is int
              and type(r.get("extra_edges")) is int
              and 0<=r["different_rows"] and
              0<=r["missing_edges"] and 0<=r["extra_edges"]
              and exact_pair_metadata(r.get("reference"))
              and exact_pair_metadata(r.get("candidate"))
              and (r["exact"]==
                   (r["reference"]==r["candidate"])),
              f"Invalid recorded real graph comparison: {host}/{key}")
        cases[key]=r
        if ds in case_inputs:
            check(case_inputs[ds]==r["input_sha256"],
                  f"Input changed between k/algorithms {host}/{ds}")
        else:
            case_inputs[ds]=r["input_sha256"]
        dk=(ds,k)
        if dk in original_sha:
            check(original_sha[dk]==r["reference"],
                  f"Baseline graph differs between candidate methods {host}/{dk}")
        else:
            original_sha[dk]=r["reference"]
    check(set(cases)=={(d,k,a) for d in DATASETS
                      for k in K_VALUES for a in OTHERS},
          f"Unplanned/missing real candidate cases: {host}")

    graph_count=defaultdict(list)
    for r in full:
        ds,k,method,rep=(r.get("dataset"),r.get("k"),
                          r.get("method"),r.get("repeat"))
        key=(ds,k,method,rep)
        check((ds,k,method) in cases and type(rep)==int and
              0<=rep<REPS
              and r.get("first")==("auto" if rep%2==0 else method)
              and timing(r.get("reference_ns"))
              and timing(r.get("candidate_ns"))
              and r.get("output_exact")==cases[ds,k,method]["exact"]
              and r.get("reference_sha256")==
                  cases[ds,k,method]["reference"]
              and r.get("candidate_sha256")==
                  cases[ds,k,method]["candidate"],
              f"Unbalanced/mismatched full original graph timing: {host}/{key}")
        graph_count[key].append(r)
    check(
        set(graph_count)=={(d,k,m,i) for d in DATASETS
                          for k in K_VALUES for m in OTHERS
                          for i in range(REPS)}
        and all(len(v)==1 for v in graph_count.values()),
        f"Missing or duplicate 720 full reference/candidate pairs: {host}")

    phase_count=defaultdict(list)
    phase_resolved={}
    for r in phase:
        ds,k,m,rep=(r.get("dataset"),r.get("k"),
                     r.get("method"),r.get("repeat"))
        key=(ds,k,m,rep)
        check(ds in DATASETS and k in K_VALUES and m in METHODS
              and type(rep)==int and 0<=rep<PHASE_REPS
              and r.get("selected_method") in
                  ("brute","kd_tree","ball_tree")
              and timing(r.get("fit_ns"))
              and timing(r.get("query_ns")),
              f"Missing query/fit phase metadata: {host}/{key}")
        dk=(ds,k,m)
        if dk in phase_resolved:
            check(phase_resolved[dk]==r["selected_method"],
                  f"scikit method changed during identical trial: {host}/{dk}")
        else:
            phase_resolved[dk]=r["selected_method"]
        phase_count[key].append(r)
    check(
        set(phase_count)=={(d,k,m,i) for d in DATASETS
                          for k in K_VALUES for m in METHODS
                          for i in range(PHASE_REPS)}
        and all(len(v)==1 for v in phase_count.values()),
        f"Missing/repeated 480 search-phase samples: {host}")
    for (ds,k,m),r in cases.items():
        check(
            phase_resolved[ds,k,"auto"]==r["original_method"]
            and phase_resolved[ds,k,m]==r["candidate_method"],
            f"Resolved native sklearn fit method differs by phase: {host}/{ds}/{k}/{m}")

    synthetic_count=defaultdict(list)
    synthetic_inputs={}
    for r in synthetics:
        name,k,p,m=(r.get("name"),r.get("k"),
                     r.get("p"),r.get("method"))
        key=(name,k,p,m)
        check(name in SYNTHETIC and k in K_VALUES and
              p in (1,2) and m in OTHERS and
              sha(r.get("input_sha256")) and
              type(r.get("exact")) is bool and
              isinstance(r.get("different_rows"),int) and
              isinstance(r.get("missing_edges"),int) and
              isinstance(r.get("extra_edges"),int) and
              min(r["different_rows"],r["missing_edges"],
                  r["extra_edges"])>=0 and
              exact_pair_metadata(r.get("reference"))
              and exact_pair_metadata(r.get("candidate")) and
              r["exact"]==(r["reference"]==r["candidate"]),
              f"Synthetic safety case invalid: {host}/{key}")
        if name in synthetic_inputs:
            check(synthetic_inputs[name]==r["input_sha256"],
                  f"Synthetic input changed between engines {host}/{name}")
        else:
            synthetic_inputs[name]=r["input_sha256"]
        synthetic_count[key].append(r)
    check(
        set(synthetic_count)=={(name,k,p,m) for name in SYNTHETIC
                              for k in K_VALUES for p in (1,2)
                              for m in OTHERS}
        and all(len(v)==1 for v in synthetic_count.values()),
        f"Missing or duplicated 90 fixed tied-graph tests: {host}")

    native_count=defaultdict(list)
    for r in native:
        ds,m,b=(r.get("dataset"),r.get("method"),r.get("backend"))
        key=(ds,m,b)
        check(ds in DATASETS and m in OTHERS and b in BACKENDS
              and r.get("k")==10 and r.get("seed")==28109
              and type(r.get("graph_exact")) is bool and
              r["graph_exact"]==cases[ds,10,m]["exact"],
              f"Native original graph integrity mismatch: {host}/{key}")
        if not r["graph_exact"]:
            check(r.get("status")=="skipped_graph_mismatch"
                  and "state_hashes" not in r,
                  f"Mismatch graph was silently propagated: {host}/{key}")
        else:
            hashes=r.get("state_hashes",{})
            check(
                r.get("status")=="exact" and set(hashes)==STATES
                and all(sha(h) for h in hashes.values())
                and r.get("effective_iterations")==1000
                and r.get("stop_reason")=="max_iter"
                and 0<=r.get("accuracy_unlabeled",-1)<=1
                and r.get("backend_identity")=={
                    "numba":"pcc_numba.pcc_propagate_numba",
                    "cython":"pcc_step.pcc_propagate"}[b],
                f"Graph-equivalent native backend failed: {host}/{key}")
        native_count[key].append(r)
    check(set(native_count)=={(ds,m,b) for ds in DATASETS
                             for m in OTHERS for b in BACKENDS}
          and all(len(v)==1 for v in native_count.values()),
          f"Missing/duplicate native controls: {host}")
    return d,cases,synthetic_inputs,case_inputs


def quartiles(xs):
    seq=sorted(xs)
    def at(q):
        v=(len(seq)-1)*q
        i=int(v)
        return seq[i]*(1-(v-i))+seq[min(i+1,len(seq)-1)]*(v-i)
    return at(.25),at(.75)


def generate(hosts):
    lines=[
        "# Exact sklearn nearest-neighbor search strategies vs original auto",
        "",
        "Fully frozen observational study: **all 1,440 matched full original-vs-candidate graph builds**,",
        "**960 NN fit+query phase samples**, **180 synthetic tied/duplicate graph comparisons**,",
        "and **48 fixed native status controls** (same-backend exact or explicit graph-mismatch skip)",
        "completed on Linux and Windows. No production `pcc_graph.py`, PCA, labels,",
        "NumPy/Numba/Cython kernels, RNG or stopping behavior modified.",
        "",
        "## Original sklearn algorithm chosen by `auto`",
        "",
        "| Host | Dataset | k=1 | k=3 | k=10 |",
        "|---|---|---|---|---|",
    ]
    for host,(_,cases,_,_) in hosts.items():
        for ds in DATASETS:
            lines.append(
                f"| {host} | {ds} | "+" | ".join(
                    cases[ds,k,"brute"]["original_method"]
                    for k in K_VALUES)+" |")
    lines.extend(["",
        "## Whole production graph builder: paired candidate/reference ratios",
        "",
        "Each cell is the **median** of 20 paired complete graph-build times.",
        "A value below 1.0 indicates faster candidate. The `Exact` column",
        "requires *entire* ordered, padded `neib_list` and `neib_qt`,",
        "not merely identical undirected edge counts. Times in ms.",
        "",
        "| Host | Dataset | k | Alternative | Exact output | Original ms | Alternative ms | Candidate/original | Paired speed gain |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ])
    per={}
    for host,(d,cases,_,_) in hosts.items():
        for ds in DATASETS:
            for k in K_VALUES:
                for m in OTHERS:
                    vals=[x for x in d["full_graph_pairs"] if
                          x["dataset"]==ds and x["k"]==k and
                          x["method"]==m]
                    a=[x["reference_ns"]/1e6 for x in vals]
                    b=[x["candidate_ns"]/1e6 for x in vals]
                    ratio=median([x["candidate_ns"]/x["reference_ns"]
                                  for x in vals])
                    per[host,ds,k,m]=ratio
                    lines.append(
                        f"| {host} | {ds} | {k} | {m} | "
                        f"{'yes' if cases[ds,k,m]['exact'] else 'NO'} | "
                        f"{median(a):.3f} | {median(b):.3f} | "
                        f"{ratio:.3f}× | {100*(1-ratio):+.2f}% |")
    lines.extend(["",
        "## Search-only fit and query phase medians (ms)",
        "",
        "| Host | Dataset | k | Actual engine | Fit | Query |",
        "|---|---|---:|---|---:|---:|",
    ])
    for host,(d,_,_,_) in hosts.items():
        for ds in DATASETS:
            for k in K_VALUES:
                for m in METHODS:
                    vals=[x for x in d["search_phase_rows"]
                          if x["dataset"]==ds and x["k"]==k and
                          x["method"]==m]
                    lines.append(
                        f"| {host} | {ds} | {k} | "
                        f"{m} ({vals[0]['selected_method']}) | "
                        f"{median(x['fit_ns'] for x in vals)/1e6:.3f} | "
                        f"{median(x['query_ns'] for x in vals)/1e6:.3f} |")
    lines.extend(["",
        "## Adversarial tie/duplicate compatibility",
        "",
        "| Host | Engine | Exact out of 30 | Incompatible | First incompatible example |",
        "|---|---|---:|---:|---|",
    ])
    for host,(d,_,_,_) in hosts.items():
        for m in OTHERS:
            x=[v for v in d["synthetic_cases"] if v["method"]==m]
            ok=sum(z["exact"] for z in x)
            bad=next((z for z in x if not z["exact"]),None)
            reason=("-" if bad is None else
                    f"{bad['name']}, k={bad['k']}, p={bad['p']}, "
                    f"changed rows={bad['different_rows']}, "
                    f"missing={bad['missing_edges']}, extra={bad['extra_edges']}")
            lines.append(f"| {host} | {m} | {ok} | {30-ok} | {reason} |")
    lines.extend(["",
        "## Preregistered universal-replacement decision",
        "",
        "A universal default replacement must (i) match *all* complete real",
        "and synthetic graphs on *both* OS, (ii) save >=5% on the paired",
        "whole graph at k10 in >=3/4 datasets on each OS, and",
        "(iii) cause no >5% regression in any of 12 dataset/k cells/OS.",
        "No dataset-aware heuristics are inferred after observing results.",
        "",
        "| Engine | Exact all original real + synthetic on both OS | k10 improvements >=5% (Linux/Windows of 4) | Worst paired regression | Qualified |",
        "|---|---|---|---:|---|",
    ])
    eligible=[]
    for m in OTHERS:
        compatible=all(
            all(c[ds,k,m]["exact"] for ds in DATASETS for k in K_VALUES)
            and all(v["exact"] for v in d["synthetic_cases"] if
                    v["method"]==m)
            for (d,c,_,_) in hosts.values())
        counts=[
            sum(per[host,ds,10,m]<=.95 for ds in DATASETS)
            for host in HOSTS]
        worst=max(per[host,ds,k,m]-1
                  for host in HOSTS for ds in DATASETS for k in K_VALUES)
        qualified=compatible and all(n>=3 for n in counts) and worst<=.05
        if qualified:
            eligible.append(m)
        lines.append(
            f"| {m} | {'yes' if compatible else 'NO'} | "
            f"{counts[0]}/4, {counts[1]}/4 | {100*worst:+.2f}% | "
            f"{'YES' if qualified else 'NO'} |")
    lines.extend(["",
        "**Decision:** "+(
            "Candidates meeting frozen gate: "+", ".join(eligible)+
            "; still require a separate minimal production patch and review."
            if eligible else
            "**No universal alternative meets preregistered compatibility and speed gates.** "
            "Keep original sklearn `algorithm='auto'` in production.")
        ,"",
        "### Limits",
        "",
        "- Raw neighbor rankings and complete symmetric graphs need not be",
        "  identical when distance ties occur; only entire graph identity",
        "  can support a drop-in replacement without a semantic change.",
        "- The matched timing uses hosted, uncontrolled runners, not a",
        "  CPU-isolated local Windows workstation. Do not pool systems.",
        "- A 1000-iteration exact native graph check establishes same",
        "  tested graph/backends, not all future datasets or convergence.",
        "- Production code remains unchanged and PR is experimental",
        "  and unmerged; even a qualified method cannot be enabled silently.",
        "- The independent #192 CSR-offset improvement concerns dense",
        "  materialization rather than nearest-neighbor search and stays",
        "  draft pending an explicit production integration decision.",
    ])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux",required=True)
    parser.add_argument("--windows",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    hosts={}
    for host,path in (("linux",a.linux),("windows",a.windows)):
        hosts[host]=validate(path,host)
    check(hosts["linux"][0]["head"]==hosts["windows"][0]["head"],
          "Linux and Windows executed different commit SHA")
    check(hosts["linux"][0]["source_sha256"]==
          hosts["windows"][0]["source_sha256"],
          "Original Git blob or replacements differ across hosts")
    check(hosts["linux"][2]==hosts["windows"][2],
          "Adversarial synthetic inputs differ across host")
    check(hosts["linux"][3]==hosts["windows"][3],
          "Real input arrays differ across host")
    report=generate(hosts)
    Path(a.output).write_text(report,encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":
    main()
