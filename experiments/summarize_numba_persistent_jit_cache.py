"""Mandatory independent both-OS CPU-for-disk Numba persistent-cache verdict."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("wine","breast_cancer","digits")
SEEDS=(66013,66117)
VARIANTS=("original","candidate")
PHASES=("cold","fresh_reuse")
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}


def must(test,explain):
    if not test:
        raise AssertionError(explain)


def sha(x):
    return isinstance(x,str) and len(x)==64 and all(
        ch in "0123456789abcdef" for ch in x)


def positive(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0


def compare(x,y,name):
    must(x==y,"native source/corpus/cache state changed "+name)


def validate_fit(f,classes):
    must(positive(f["cpu_ns"]) and positive(f["wall_ns"]),
         "missing fresh process CPU/wall timer")
    must(f["wall_ns"]>=f["cpu_ns"]*.6,
         "gross timer miscalibration")
    must(f["original_backend_name"]=="pcc_propagate_numba" and
         f["mode"]=="early_stopping" and f["iteration_cap"]==300,
         "not true Numba native production call")
    must(f["layout"]==("C" if classes>=8 else "F"),
         "Numba 8+ class C-order memory layout changed")
    must(set(f["state_hashes"])==STATES and
         all(sha(h) for h in f["state_hashes"].values()),
         "missing all six full native state arrays")
    must(type(f["native_rng_next"]) is float and
         math.isfinite(f["native_rng_next"]) and
         0<=f["native_rng_next"]<1,
         "missing identical native Numba RNG next draw")
    stop=f["stop_info"]
    must(stop["effective_iterations"]==300 and
         stop["stop_reason"]=="max_iter" and
         type(stop["checks"]) is int and stop["checks"]>=1 and
         type(stop["stalled_checks"]) is int and stop["stalled_checks"]>=0 and
         type(stop["stop_max"]) is int and stop["stop_max"]>=1,
         "original Numba stop/iteration/check behavior changed")


def validate(doc):
    must(doc["schema"]==1 and doc["base"]==BASE,
         "wrong pinned Git master")
    must(doc["platform"] in ("linux","win32") and
         len(doc["head"])==40,
         "invalid independent host or source head")
    must(sha(doc["source_sha256"]) and
         sha(doc["candidate_sha256"]) and
         doc["source_sha256"]!=doc["candidate_sha256"],
         "missing actual two Numba source module SHA")
    must(doc["thread_limit"]=="1" and
         len(doc["cases"])==6 and doc["numba"],
         "wrong native thread/Numba/case setup")
    cases={}
    for case in doc["cases"]:
        ds,seed=case["dataset"],case["seed"]
        key=(ds,seed)
        must(ds in DATASETS and seed in SEEDS and key not in cases,
             "missing or duplicate independent Numba case")
        must(case["phase_order"]==(
            ["original","candidate"] if seed==SEEDS[0]
            else ["candidate","original"]),
             "cold compiler variant order not counterbalanced")
        records=case["variants"]
        must(set(records)==set(VARIANTS),"missing source pair")
        expected_classes={"wine":3,"breast_cancer":2,"digits":10}[ds]
        expected_nodes={"wine":178,"breast_cancer":569,"digits":1797}[ds]
        original_ref=records["original"]["cold"]["first"]
        first_original=records["original"]["cold"]
        for variant in VARIANTS:
            entry=records[variant]
            must(entry["source_sha256"]==
                 doc["source_sha256" if variant=="original" else "candidate_sha256"],
                 "source module derivation SHA differs within OS")
            must(type(entry["cache_size_bytes"]) is int and
                 0<=entry["cache_size_bytes"]<=32*1024*1024,
                 "cold compiled Numba cache disk exceeds frozen budget")
            if variant=="original":
                must(entry["cache_size_bytes"]==0,
                     "source baseline unexpectedly persisted Numba code")
            else:
                must(entry["cache_size_bytes"]>0,
                     "candidate did not create any Numba machine code cache")
            for phase in PHASES:
                record=entry[phase]
                must(record["variant"]==variant and
                     record["phase"]==phase and
                     record["dataset"]==ds and
                     record["seed"]==seed and
                     record["case_id"]==f"{ds}-labels{seed}",
                     "wrong process identity")
                must(record["real_nodes"]==expected_nodes and
                     record["classes"]==expected_classes,
                     "sklearn dataset/class count changed")
                must(record["module_sha256"]==entry["source_sha256"],
                     "fresh interpreter imported an unexpected module")
                for name in ("fixture_sha","input_sha","labels_sha"):
                    must(sha(record[name]),"missing canonical native input SHA "+name)
                    compare(record[name],first_original[name],name)
                must(len(record["ordered_graph_sha"])==2 and
                     all(sha(h) for h in record["ordered_graph_sha"]),
                     "canonical undirected graph SHA missing")
                compare(record["ordered_graph_sha"],
                        first_original["ordered_graph_sha"],
                        "ordered dense int64 graph")
                must(type(record["cache_loaded_count"]) is int
                     and type(record["cache_saved_count"]) is int
                     and isinstance(record["fresh_process_cache_trace"],list),
                     "Numba cache provenance incomplete")
                if variant=="original":
                    must(record["cache_loaded_count"]==0 and
                         record["cache_saved_count"]==0 and
                         not record["fresh_process_cache_trace"],
                         "source original hit a compiled Numba disk cache")
                elif phase=="cold":
                    must(record["cache_saved_count"]>=1,
                         "new candidate failed initial native Numba cache save")
                else:
                    must(record["cache_loaded_count"]>=1,
                         "new candidate Python process failed actual Numba cache data hit")
                for kind in ("first","hot"):
                    validate_fit(record[kind],expected_classes)
                    for state in ("state_hashes","native_rng_next",
                                  "stop_info","layout"):
                        compare(record[kind][state],original_ref[state],
                                f"{variant}/{phase}/{kind}/{state}")
        cases[key]=case
    must(set(cases)=={(d,s) for d in DATASETS for s in SEEDS},
         "not all 6 independent fresh process case pairs present")
    return cases


def decide(docs,valid):
    lines=[
        "# Persistent native Numba JIT code cache: original vs four decorators",
        "",
        "Scientific completeness: **PASS**, 2 OS x 3 real sklearn datasets x",
        "2 label seeds x 2 Numba-source variants x 2 independent fresh",
        "processes = **48 brand-new Python interpreter launches**, each",
        "with two full native PCC fits = **96 complete fits**.",
        "All original/cached first and hot seeded native results are identical:",
        "six state arrays, native Numba RNG endpoint, stopping counters,",
        "native Numba class C/F order, labels and undirected graph.",
        "The second candidate interpreter must emit an actual Numba",
        "'[cache] data loaded' diagnostic for the candidate source.",
        "",
        "| OS | Dataset | Cached fresh-process first-fit CPU saved | Original new-process first-fit median ms | Cache-loaded new-process first-fit median ms | Cached cold-compile CPU overhead | Warmed same-process hot CPU change | Disk code cache per fixture, max MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows=[]
    go=True
    for doc,cases in zip(docs,valid):
        for ds in DATASETS:
            group=[cases[(ds,s)] for s in SEEDS]
            # CPU gate uses match within each exact OS and label seed.
            warm_saved=100*(1-statistics.median(
                c["variants"]["candidate"]["fresh_reuse"]["first"]["cpu_ns"]/
                c["variants"]["original"]["fresh_reuse"]["first"]["cpu_ns"]
                for c in group))
            cold_overhead=100*(statistics.median(
                c["variants"]["candidate"]["cold"]["first"]["cpu_ns"]/
                c["variants"]["original"]["cold"]["first"]["cpu_ns"]
                for c in group)-1)
            hot_delta=100*(statistics.median(
                c["variants"]["candidate"]["fresh_reuse"]["hot"]["cpu_ns"]/
                c["variants"]["original"]["fresh_reuse"]["hot"]["cpu_ns"]
                for c in group)-1)
            original_ms=statistics.median(
                c["variants"]["original"]["fresh_reuse"]["first"]["cpu_ns"]/1e6
                for c in group)
            loaded_ms=statistics.median(
                c["variants"]["candidate"]["fresh_reuse"]["first"]["cpu_ns"]/1e6
                for c in group)
            disk_mb=max(c["variants"]["candidate"]["cache_size_bytes"]
                        for c in group)/1048576
            if (warm_saved<25 or cold_overhead>30 or
                    hot_delta>5 or disk_mb>32):
                go=False
            rows.append(dict(os=doc["platform"],dataset=ds,
                             saved_cpu_pct=warm_saved,
                             cold_cpu_overhead_pct=cold_overhead,
                             hot_cpu_change_pct=hot_delta,
                             baseline_fresh_ms=original_ms,
                             candidate_loaded_ms=loaded_ms,
                             disk_cache_max_mib=disk_mb))
            lines.append(
                f"| {doc['platform']} | {ds} | {warm_saved:+.2f}% | "
                f"{original_ms:.2f} | {loaded_ms:.2f} | "
                f"{cold_overhead:+.2f}% | {hot_delta:+.2f}% | {disk_mb:.2f} |")
    lines+=["",
        f"**Frozen prospective verdict: {'GO' if go else 'NO-GO'}.**",
        "",
        "GO requires >=25% reduced *first-fit process CPU* from the",
        "verified disk-cache new-process launch vs an uncached source on",
        "ALL three datasets and BOTH OS, <=30% first cold compile CPU",
        "overhead, <=5% regression in same-process already-hot CPU,",
        "and <=32 MiB persistent native disk code per fixture.",
        "",
        "Cache loading is NOT hot-kernel acceleration. This measurement",
        "deliberately excludes Python import, sklearn graph-building,",
        "Numba installation, model construction and public set_graph;",
        "it quantifies CPU spent in the first PCC fit after imports.",
        "Performance in one long-lived Python interpreter may be unchanged.",
        "Hosted Windows and Linux CPU timing must not be pooled.",
        "Source change only @njit -> @njit(cache=True) at four decorators.",
        "The dense padded int64 neighbor list and all particle operations",
        "remain exactly unchanged. No production code is promoted by this",
        "scientific report. A separate minimal source PR and its own",
        "four-platform complete CI are required after GO.",
    ]
    return "\n".join(lines)+"\n",go,rows


def main(path):
    files=list(Path(path).glob("**/numba_persistent_cache_results.json"))
    must(len(files)==2,"mandatory Linux and Windows archived native corpora absent")
    ds=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    must({d["platform"] for d in ds}=={"linux","win32"},
         "both independent GitHub OS results required")
    ds.sort(key=lambda d:d["platform"])
    cases=[validate(d) for d in ds]
    for field in ("base","head","source_sha256","candidate_sha256"):
        must(len({d[field] for d in ds})==1,
             "cross-host canonical Git source identity changed "+field)
    for dsname in DATASETS:
        for seed in SEEDS:
            a,b=(c[(dsname,seed)] for c in cases)
            for attr in ("fixture_sha","input_sha","labels_sha",
                         "ordered_graph_sha"):
                x=a["variants"]["original"]["cold"][attr]
                y=b["variants"]["original"]["cold"][attr]
                compare(x,y,"cross-host real fixture/graph "+attr)
    report,go,rows=decide(ds,cases)
    Path("numba_persistent_cache_decision.json").write_text(
        json.dumps(dict(schema=1,complete=True,
                        verdict="GO" if go else "NO-GO",
                        python_processes=48,full_native_fits=96,
                        rows=rows),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    Path("numba_persistent_cache_report.md").write_text(report,encoding="utf-8")
    print(report,flush=True)
    print("STRICT COMPLETE both-host Numba persistent JIT code cache:",
          "GO" if go else "NO-GO",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
