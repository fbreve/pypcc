"""Mandatory all-conditions validation, paired timing summary and decision gate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

REAL=("iris","wine","breast_cancer","digits")
KS=(1,3,10)
SYNTH=("empty_1","empty_10000","isolated_30000_1000",
       "sparse_1000_1000","sparse_10000_1000",
       "medium_1000_3000","dense_1000_30000",
       "dense_10000_100000","mixed_loops_1000_2000",
       "repeated_reciprocal_1000_3000")
SPARSE=("empty_1","empty_10000","isolated_30000_1000",
        "sparse_1000_1000","sparse_10000_1000",
        "medium_1000_3000","repeated_reciprocal_1000_3000")
HOSTS=("linux","windows")
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
SEEDS=(28319,28447)
STATES={"predictions","dominance","positions","strengths","distances",
        "owndeg"}
MASTER="35dc81caf0388be8ea1729837ea6d54358b53964"
REPS=30


def require(ok,why):
    if not ok:
        raise AssertionError(why)


def sha256(v):
    return isinstance(v,str) and len(v)==64 and all(
        x in "0123456789abcdef" for x in v.lower())


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def validate(root,host):
    root=Path(root)
    d=read(root/"result.json")
    freeze=(root/"pip-freeze.txt").read_text(encoding="utf-8")
    require(len(freeze)>80 and d.get("schema")=="edge_index_diagonal_v1"
            and d.get("master")==MASTER
            and isinstance(d.get("head"),str) and len(d["head"])==40
            and sha256(d.get("original_source_sha256"))
            and d.get("real_datasets")==list(REAL)
            and d.get("k_values")==list(KS)
            and d.get("synthetic_cases")==list(SYNTH)
            and d.get("repeats")==REPS and
            d.get("label_seeds")==list(SEEDS) and
            d.get("backends")==list(BACKENDS) and
            d.get("modes")==list(MODES) and
            d.get("max_iter")==1000 and
            d.get("random_seed")==28311,
            f"Unfrozen or incomplete provenance: {host}")
    env=d.get("environment",{})
    require(all(env.get("thread_env",{}).get(k)=="1" for k in (
            "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
            "MKL_NUM_THREADS","NUMBA_NUM_THREADS"))
            and isinstance(env.get("os"),str)
            and isinstance(env.get("python"),str),
            f"Missing original single-thread host env: {host}")
    cases=d.get("conditions",[])
    pairs=d.get("graph_pairs",[])
    natives=d.get("native_pairs",[])
    ctrl=d.get("edge_controls",[])
    require(len(cases)==22 and len(pairs)==660 and len(natives)==48
            and len(ctrl)==8,
            f"Incomplete benchmark cohort on {host}")
    condition={}
    realnames={f"{name}-k{k}" for name in REAL for k in KS}
    ids=realnames|set(SYNTH)
    for row in cases:
        name=row.get("case")
        require(name in ids and name not in condition
                and row.get("kind")==(
                    "real" if name in realnames else "synthetic")
                and type(row.get("num_nodes"))==int
                and type(row.get("num_directed_input_edges"))==int
                and row["num_nodes"]>=1
                and row["num_directed_input_edges"]>=0
                and sha256(row.get("input_sha256"))
                and type(row.get("self_loops_present"))==bool
                and set(row.get("original_hashes",{}))=={
                    "neighbors","degrees"}
                and all(sha256(v) for v in row["original_hashes"].values()),
                f"Invalid exact case metadata: {host}/{name}")
        if name in realnames:
            ds,k=name.rsplit("-k",1)
            require(row.get("dataset")==ds and row.get("k")==int(k)
                    and sha256(row.get("data_sha256")) and
                    sha256(row.get("target_sha256"))
                    and row["self_loops_present"] is False,
                    f"Real original kNN fingerprint mismatch: {host}/{name}")
        if name in SYNTH:
            require(row["self_loops_present"]==(
                name=="mixed_loops_1000_2000"),
                f"Unexpected self loop in synthetic {host}/{name}")
        condition[name]=row
    require(set(condition)==ids,f"Missing expected case: {host}")
    groups=defaultdict(list)
    for row in pairs:
        name,rep=row.get("case"),row.get("repeat")
        key=(name,rep)
        require(name in condition and type(rep)==int and
                0<=rep<REPS and row.get("kind")==condition[name]["kind"]
                and row.get("order")==(
                    "reference_first" if rep%2==0 else "candidate_first")
                and type(row.get("reference_ns")) in (int,float)
                and type(row.get("candidate_ns")) in (int,float)
                and 0<row["reference_ns"]<1e12
                and 0<row["candidate_ns"]<1e12,
                f"Invalid paired graph elapsed time/order: {host}/{key}")
        groups[key].append(row)
    require(set(groups)=={(name,rep) for name in ids
                           for rep in range(REPS)}
            and all(len(x)==1 for x in groups.values()),
            f"Missing/duplicated 660 strict graph pairs: {host}")
    nativegroups=defaultdict(list)
    for row in natives:
        name,seed,b,mode=(row.get("dataset"),row.get("label_seed"),
                          row.get("backend"),row.get("mode"))
        key=(name,seed,b,mode)
        require(name in REAL and seed in SEEDS and
                b in BACKENDS and mode in MODES
                and row.get("iteration_cap")==1000
                and row.get("complete_exact_parity") is True
                and type(row.get("effective_iterations"))==int
                and 0<row["effective_iterations"]<=1000
                and row.get("stop_reason") in ("max_iter","early_stop")
                and row.get("backend_identity")=={
                    "numpy":"pcc_numpy.pcc_propagate_numpy",
                    "numba":"pcc_numba.pcc_propagate_numba",
                    "cython":"pcc_step.pcc_propagate"}[b]
                and set(row.get("state_hashes",{}))==STATES
                and all(sha256(z) for z in row["state_hashes"].values())
                and 0<=row.get("accuracy_unlabeled",-1)<=1,
                f"Native original full state or metadata invalid: {host}/{key}")
        nativegroups[key].append(row)
    require(set(nativegroups)=={
        (ds,seed,b,mode) for ds in REAL for seed in SEEDS
        for b in BACKENDS for mode in MODES}
        and all(len(v)==1 for v in nativegroups.values()),
        f"Missing/duplicate 48 full native state pairs: {host}")
    names=("empty_0","empty_0_with_edges","negative_nodes",
           "out_of_bounds","negative_index","wrong_shape",
           "nonintegral_cast","only_self_loops")
    require(tuple(x.get("case") for x in ctrl)==names and
            all(x.get("exact") is True and sha256(x.get("input_sha256"))
                and x.get("outcome",{}).get("status") in ("ok","error")
                for x in ctrl),
            f"Original invalid/empty edge behavior changed: {host}")
    return d,condition


def percentile(values,q):
    x=sorted(values)
    v=(len(x)-1)*q
    i=int(v)
    return x[i]*(1-(v-i))+x[min(i+1,len(x)-1)]*(v-i)


def report(both):
    d1=both["linux"][0]
    names=[f"{ds}-k{k}" for ds in REAL for k in KS]+list(SYNTH)
    lines=[
        "# Exact loop-free edge-index diagonal cleanup: two-OS paired benchmark",
        "",
        "**All 1,320 paired complete edge-index constructions** across two",
        "hosts, **96 native original-vs-candidate full-state checks** and",
        "**16 invalid/empty semantic controls** passed the exact graph",
        "and original PCC behavior gates. The original reference is",
        "the frozen production Git blob, not an approximate clone.",
        "No NumPy, Numba, Cython, sklearn kNN graph builder, graph topology",
        "or original native stopping logic was changed.",
        "",
        "## Real and synthetic full-builder timings",
        "",
        "Each row is based on 30 true alternating original/candidate",
        "paired complete builder repetitions, in the same host process.",
        "Original/candidate times are separate medians (milliseconds);",
        "improvement is 100×(1−median paired candidate/reference ratio).",
        "No outliers or regression conditions discarded.",
        "",
        "| OS | Condition | Self-loop input | Original ms | Candidate ms | Paired time reduction | Paired Q1–Q3 ratio |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    ratios={}
    for host,(d,conditions) in both.items():
        for name in names:
            rows=[x for x in d["graph_pairs"] if x["case"]==name]
            require(len(rows)==REPS,
                    "Missing before/after pair group")
            orig=[x["reference_ns"]/1e6 for x in rows]
            trial=[x["candidate_ns"]/1e6 for x in rows]
            paired=[x["candidate_ns"]/x["reference_ns"] for x in rows]
            m=median(paired)
            ratios[host,name]=m
            q1,q3=percentile(paired,.25),percentile(paired,.75)
            lines.append(
                f"| {host} | {name} | "
                f"{'yes' if conditions[name]['self_loops_present'] else 'no'} | "
                f"{median(orig):.3f} | {median(trial):.3f} | "
                f"{100*(1-m):+.2f}% | {q1:.3f}×–{q3:.3f}× |")
    sparse_count={host:sum(ratios[host,name]<=.90 for name in SPARSE)
                  for host in HOSTS}
    regressions=[
        (host,name,ratios[host,name])
        for host in HOSTS for name in names if ratios[host,name]>1.10]
    eligible=all(c>=5 for c in sparse_count.values()) and not regressions
    lines.extend([
        "",
        "## Frozen precommitted production adoption criteria",
        "",
        "Require >=10% paired median gain in >=5 of 7 explicitly",
        "prespecified loop-free sparse synthetic conditions on **each**",
        "OS, and no >10% regression for any 22 real/synthetic conditions",
        "on either OS. No new dataset-specific dispatch threshold or",
        "post-hoc filtering is allowed.",
        "",
        f"- Sparse cases >=10% improvement: Linux **{sparse_count['linux']}/7**, "
        f"Windows **{sparse_count['windows']}/7**.",
        f"- Median paired regressions >10%: **{len(regressions)}**.",
        f"- **Decision: {'CANDIDATE MEETS frozen host gates' if eligible else 'NO-GO under frozen gates'}**.",
        "",
        "This criterion is only a candidate-promotion gate. Even when",
        "met, the original benchmark/protocol/workflow stay on this",
        "independent experimental draft. A separate tiny master-based",
        "PR with only the source line(s) and permanent compatibility",
        "tests needs fresh four-platform regression and explicit review.",
        "",
        "### Scope and measurement limitations",
        "",
        "- The optimization only affects graph building from an",
        "  externally supplied `edge_index`; original `build_knn_graph`",
        "  and underlying `NearestNeighbors(algorithm='auto')` remain",
        "  unchanged, as do all compiled propagation algorithms.",
        "- Source self-loop inputs retain the exact original cleanup",
        "  path. Sparse graphs with many isolates may benefit more",
        "  than dense graphs; all named dense and loop-containing",
        "  regressions are shown above without hiding any.",
        "- Exact graph-array/degree comparisons include full padded",
        "  width, int64 dtypes, CSR order and isolated nodes. Native",
        "  six-final-state hashes and stopping metadata matched",
        "  within each same backend/label mask/stop mode; no cross-",
        "  backend RNG equality is claimed.",
        "- Times are 30 technical repeated calls on GitHub-hosted",
        "  Linux/Windows machines, not locally controlled workstation",
        "  performance. Do not pool host timings or infer universal",
        "  throughput. Native 1k cap is not convergence evidence.",
    ])
    return "\n".join(lines)+"\n",eligible


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--linux",required=True)
    p.add_argument("--windows",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    both={host:validate(getattr(a,host),host) for host in HOSTS}
    require(both["linux"][0]["head"]==both["windows"][0]["head"]
            and both["linux"][0]["original_source_sha256"]==
                both["windows"][0]["original_source_sha256"],
            "Original Git blob / experimental commit differs across OS")
    for name in [f"{ds}-k{k}" for ds in REAL for k in KS]+list(SYNTH):
        left=both["linux"][1][name]
        right=both["windows"][1][name]
        require(left["input_sha256"]==right["input_sha256"]
                and left["original_hashes"]==right["original_hashes"],
                f"Original same input graph differed across hosts: {name}")
    document,eligible=report(both)
    Path(a.output).write_text(document,encoding="utf-8")
    print(document,flush=True)
    print(f"STRICT 2-OS GRAPH/NATIVE PARITY PASS; "
          f"independent production candidate gate={'PASS' if eligible else 'NO-GO'}",
          flush=True)


if __name__=="__main__":
    main()
