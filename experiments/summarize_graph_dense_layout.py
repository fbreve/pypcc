"""Strict two-OS complete original-vs-candidate graph benchmark report."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median

DATASETS=("iris","wine","breast_cancer","digits")
K_VALUES=(1,3,10)
SEEDS=(27103,27211)
BACKENDS=("numpy","numba","cython")
MODES=("fixed","early_stopping")
HOSTS=("linux","windows")
ROUNDS=30
MASTER="8298ece05c4edf4bbdc871d0aa5b13d9b8ed4128"
EXPECTED_STATES={"predictions","dominance","positions","strengths",
                 "distances","owndeg"}


def require(ok,message):
    if not ok:
        raise AssertionError(message)


def hash256(x):
    return isinstance(x,str) and len(x)==64 and all(
        c in "0123456789abcdef" for c in x.lower())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def valid_time(x):
    return type(x) in (int,float) and 0<x<300000000000


def validate(root,host):
    root=Path(root)
    data=read(root/"result.json")
    pip=(root/"pip-freeze.txt").read_text(encoding="utf-8")
    require(len(pip)>100 and data.get("protocol")=="graph_dense_layout_v1",
            f"Missing original experimental provenance on {host}")
    require(
        data.get("master")==MASTER and
        isinstance(data.get("head"),str) and len(data["head"])==40 and
        hash256(data.get("original_graph_sha256")) and
        data.get("datasets")==list(DATASETS) and
        data.get("k_values")==list(K_VALUES) and
        data.get("rounds")==ROUNDS and
        data.get("label_seeds")==list(SEEDS) and
        data.get("backends")==list(BACKENDS) and
        data.get("modes")==list(MODES) and
        data.get("native_cap")==1000,
        f"Graph protocol changed or output missing: {host}")
    all_graphs=data.get("graph_pairs",[])
    phase=data.get("phase_rows",[])
    native=data.get("semantic_pairs",[])
    edge=data.get("edge_index_cases",[])
    require(
        len(all_graphs)==len(DATASETS)*len(K_VALUES)*ROUNDS and
        len(phase)==len(DATASETS)*ROUNDS and
        len(native)==len(DATASETS)*len(SEEDS)*len(BACKENDS)*len(MODES) and
        len(edge)==5 and
        data.get("completeness")=={
            "graph_pairs":360,"phase_rows":120,
            "semantic_pairs":48,"edge_index_cases":5,"all_equal":True},
        f"Native experiment incomplete: {host}")
    groups=defaultdict(list)
    graph_hashes={}
    for row in all_graphs:
        d,k,rep=row.get("dataset"),row.get("k"),row.get("repeat")
        key=(d,k,rep)
        require(d in DATASETS and k in K_VALUES and
                type(rep)==int and 0<=rep<ROUNDS and
                valid_time(row.get("reference_ns")) and
                valid_time(row.get("candidate_ns")) and
                row.get("first")==("reference" if rep%2==0 else "candidate"),
                f"Invalid or unbalanced graph timing {host}/{key}")
        hashes=row.get("hashes",{})
        require(
            set(hashes)=={"neighbors","degrees"} and
            all(hash256(x) for x in hashes.values()),
            f"Missing exact graph SHA256: {host}/{key}")
        grp=(d,k)
        if grp in graph_hashes:
            require(graph_hashes[grp]==hashes,
                    f"Original/candidate graph changed between repetitions: {host}/{grp}")
        else:
            graph_hashes[grp]=hashes
        groups[key].append(row)
    expected={(d,k,rep) for d in DATASETS for k in K_VALUES
              for rep in range(ROUNDS)}
    require(set(groups)==expected and all(len(v)==1 for v in groups.values()),
            f"Missing/repeated full graph construction: {host}")
    ph=defaultdict(list)
    for row in phase:
        d,k,rep=row.get("dataset"),row.get("k"),row.get("repeat")
        key=(d,k,rep)
        require(
            d in DATASETS and k==10 and
            type(rep)==int and 0<=rep<ROUNDS and
            all(valid_time(row.get(field))
                for field in ("neighbors_search_ns","sparse_reciprocal_ns",
                              "reference_scatter_ns","candidate_scatter_ns")) and
            row.get("hashes")==graph_hashes[d,10],
            f"Malformed phase trace or graph changed: {host}/{key}")
        ph[key].append(row)
    expectedph={(d,10,rep) for d in DATASETS for rep in range(ROUNDS)}
    require(set(ph)==expectedph and all(len(v)==1 for v in ph.values()),
            f"Missing or repeated isolated graph phases: {host}")
    sem=defaultdict(list)
    for row in native:
        d,seed,backend,mode=(row.get("dataset"),row.get("label_seed"),
                              row.get("backend"),row.get("mode"))
        key=(d,seed,backend,mode)
        require(
            d in DATASETS and seed in SEEDS and backend in BACKENDS and
            mode in MODES and
            row.get("iterations_cap")==1000 and
            1<=row.get("reference_effective_iterations",0)<=1000 and
            row.get("reference_effective_iterations")==
                row.get("candidate_effective_iterations") and
            row.get("reference_stop_reason")==
                row.get("candidate_stop_reason") and
            row.get("reference_stop_reason") in ("max_iter","early_stop") and
            row.get("native_info_exact") is True and
            row.get("six_state_hashes_exact") is True and
            row.get("reference_heldout_accuracy")==
                row.get("candidate_heldout_accuracy") and
            0<=row.get("reference_heldout_accuracy",-1)<=1 and
            row.get("reference_backend_identity")==
                row.get("candidate_backend_identity") and
            row.get("reference_backend_identity")==
                {"numpy":"pcc_numpy.pcc_propagate_numpy",
                 "numba":"pcc_numba.pcc_propagate_numba",
                 "cython":"pcc_step.pcc_propagate"}[backend] and
            set(row.get("state_hashes",{}))==EXPECTED_STATES and
            all(hash256(v) for v in row["state_hashes"].values()),
            f"Native reference/candidate state fails: {host}/{key}")
        sem[key].append(row)
    expectedsem={(d,seed,b,m) for d in DATASETS for seed in SEEDS
                 for b in BACKENDS for m in MODES}
    require(set(sem)==expectedsem and all(len(v)==1 for v in sem.values()),
            f"Missing/repeated native full-state control: {host}")
    ed={x.get("case"):x for x in edge}
    require(set(ed)==set(range(5)),
            f"Missing edge cases: {host}")
    for i in range(5):
        x=ed[i]
        require(x.get("num_nodes") in (0,3,5,8,9) and
                set(x.get("hashes",{}))=={"neighbors","degrees"} and
                all(hash256(z) for z in x["hashes"].values()),
                f"Bad edgelist baseline: {host}/{i}")
    h=data.get("host",{})
    require(isinstance(h.get("python"),str) and h.get("python") and
            isinstance(h.get("os"),str) and
            isinstance(h.get("numpy"),str) and
            all(h.get("thread_env",{}).get(k)=="1"
                for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                          "MKL_NUM_THREADS","NUMBA_NUM_THREADS")),
            f"Missing fixed environment: {host}")
    return data,graph_hashes


def quartiles(seq):
    s=sorted(seq)
    def p(q):
        pos=(len(s)-1)*q
        i=int(pos)
        return s[i]*(1-(pos-i))+s[min(i+1,len(s)-1)]*(pos-i)
    return p(.25),p(.75)


def summarize(os_data):
    lines=[
        "# Frozen original-versus-CSR-offset graph materialization audit",
        "",
        "**All 720 matched graph-build pairs** (360 per OS), **240 3-phase",
        "traces**, **96 native six-state reference/candidate backend pairs**",
        "(48 per OS) and 5 adversarial edge-index cases per OS passed exact",
        "original-graph equality/semantic controls. Original code source is",
        "the frozen master Git blob, not an approximate reimplementation.",
        "No benchmarks/runner backend, graph topology, RNG, label placement",
        "or original NumPy/Numba/Cython kernel semantics changed.",
        "",
        "## Complete original full kNN construction vs candidate",
        "",
        "Milliseconds, median of 30 **alternated paired** reference/candidate",
        "measurements within each OS/dataset/k. Positive improvement means",
        "the candidate was faster; median paired candidate/reference ratios",
        "are not necessarily ratio of medians. No outliers removed.",
        "",
        "| Host | Dataset | k | Original median (ms) | Candidate median (ms) | Median candidate/original | Improvement (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for host,d in os_data.items():
        for ds in DATASETS:
            for k in K_VALUES:
                rows=[z for z in d["graph_pairs"]
                      if z["dataset"]==ds and z["k"]==k]
                a=[x["reference_ns"]/1e6 for x in rows]
                b=[x["candidate_ns"]/1e6 for x in rows]
                paired=[x["candidate_ns"]/x["reference_ns"] for x in rows]
                m=median(paired)
                lines.append(
                    f"| {host} | {ds} | {k} | {median(a):.3f} | "
                    f"{median(b):.3f} | {m:.3f}× | {100*(1-m):+.2f}% |")
    lines.extend(["",
        "## Profiled original k=10 stages vs new CSR scatter",
        "",
        "| Host | Dataset | Original nearest-neighbor search (ms) | CSR symmetrization (ms) | Original ordered dense scatter (ms) | New offset scatter (ms) | Scatter candidate/original |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for host,d in os_data.items():
        for ds in DATASETS:
            rows=[x for x in d["phase_rows"] if x["dataset"]==ds]
            def med(field):
                return median([x[field]/1e6 for x in rows])
            ratio=median([x["candidate_scatter_ns"]/
                          x["reference_scatter_ns"] for x in rows])
            lines.append(
                f"| {host} | {ds} | {med('neighbors_search_ns'):.3f} | "
                f"{med('sparse_reciprocal_ns'):.3f} | "
                f"{med('reference_scatter_ns'):.3f} | "
                f"{med('candidate_scatter_ns'):.3f} | {ratio:.3f}× |")
    lines.extend(["",
        "## Requirements and decision",
        "",
        "- The primary endpoint is **the whole graph builder**, not",
        "  isolated scatter. If scatter improves but the original",
        "  `NearestNeighbors.fit/kneighbors` dominates, full function",
        "  speedup will be small; do not confuse these effects.",
        "- Every condition preserves the complete **ordered** padded",
        "  `neib_list` and `neib_qt` arrays byte-for-byte and their",
        "  graph hashes, including the actual production symmetrization.",
        "  Both original and candidate graphs yield identical full six",
        "  native PCC final array hashes and original stopping metadata",
        "  per backend/seed/mode on the tested fixed 1000 iterations.",
        "- Production Cython and Numba have distinct RNG systems;",
        "  cross-backend state equality was NOT required or asserted.",
        "- Windows/Linux runners have different CPU and BLAS/scipy",
        "  installations; their timings must not be pooled or treated",
        "  as local workstation performance. Technical repeats within",
        "  a job are not independent samples of all environments.",
        "- This is a **draft** candidate: no automatic merge even if",
        "  a mean speed gain appears. Decide independently whether",
        "  the full-builder gain is practically material and whether",
        "  any dataset/k suffers a material regression.",
        "- The former 216-fit full PCC performance baseline #190",
        "  already exists; this is a distinct test of the graph",
        "  **construction** stage rather than another kernel benchmark.",
    ])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux",type=Path,required=True)
    parser.add_argument("--windows",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    byhost={}
    graph_fingerprints={}
    for host,path in (("linux",args.linux),("windows",args.windows)):
        data,fps=validate(path,host)
        byhost[host]=data
        graph_fingerprints[host]=fps
    require(byhost["linux"]["head"]==byhost["windows"]["head"],
            "Both OS did not run identical candidate code")
    require(byhost["linux"]["original_graph_sha256"]==
            byhost["windows"]["original_graph_sha256"],
            "Original master Git source differed by platform")
    require(graph_fingerprints["linux"]==graph_fingerprints["windows"],
            "Original candidate graph/degree fingerprints differ across OS")
    require(
        {x["case"]:x["hashes"] for x in byhost["linux"]["edge_index_cases"]}==
        {x["case"]:x["hashes"] for x in byhost["windows"]["edge_index_cases"]},
        "Edge-index output changed across OS")
    document=summarize(byhost)
    args.output.write_text(document,encoding="utf-8")
    print(document,flush=True)


if __name__=="__main__":
    main()
