"""Strict Linux/Windows exact 20-reused-fit original-vs-production cache study."""
import json
import math
from pathlib import Path
import statistics
import sys

OLD="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
PROD="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
DATASETS=("wine","breast_cancer","digits")
SEEDS=(70013,70117)
VARIANTS=("original","production")
PHASES=("cold","warm")
COUNTS=(1,2,5,10,20)
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}
THREADS=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
         "MKL_NUM_THREADS","NUMBA_NUM_THREADS")


def ensure(c,reason):
    if not c:
        raise AssertionError(reason)


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(
        c in "0123456789abcdef" for c in value)


def positive(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0


def verify_subprocess(x,dataset,seed,variant,phase,sha):
    ensure(x["schema"]==1 and x["dataset"]==dataset and
           x["first_seed"]==seed and x["variant"]==variant and
           x["phase"]==phase and x["source_sha256"]==sha,
           "wrong original science sub-process source/config")
    ensure(x["platform"] in ("linux","win32") and
           x["native_backend_name"]=="pcc_propagate_numba" and
           x["backend_identity_direct"] is True,
           "not actual original production Numba dispatch")
    ensure(x["real_nodes"]=={"wine":178,"breast_cancer":569,
                            "digits":1797}[dataset] and
           x["classes"]=={"wine":3,"breast_cancer":2,"digits":10}[dataset],
           "wrong original sklearn real-class cohort")
    ensure(x["k_nn"]==10 and x["labels_per_class"]==3 and
           x["max_iter"]==30000 and x["early_stop"] is True and
           x["propagation_seed"]==0,
           "original PCC propagation not frozen")
    ensure(x["model_graph_stable"] is True and
           x["original_graph_prepared_once"] is True,
           "public original PCC graph rebuilt or replaced during 20 fits")
    ensure(x["mask_order"]==[seed,next(s for s in SEEDS if s!=seed)],
           "masked public fits not alternating exactly")
    ensure(all(x["threads"][name]=="1" for name in THREADS),
           "native thread settings changed")
    ensure(valid_sha(x["input_sha"]) and
           len(x["graph_hashes"])==2 and
           all(valid_sha(v) for v in x["graph_hashes"]),
           "missing exact real dense kNN graph source")
    for name in ("labels_sha","fixture_sha"):
        ensure(set(x[name])=={str(s) for s in SEEDS} and
               all(valid_sha(v) for v in x[name].values()),
               "two real label-mask SHA source evidence missing")
    ensure(set(x["final_timed_state_hashes"])==STATES and
           all(valid_sha(s) for s in x["final_timed_state_hashes"].values())
           and valid_sha(x["final_timed_mapped_sha"]) and
           type(x["final_timed_rng_next"]) is float,
           "missing final timed Numba state/RNG evidence")
    samples=x["replay_outcomes"]
    ensure(len(samples)==20,"missing one or more exact native fit replays")
    reference_by_mask={}
    for i,row in enumerate(samples,1):
        expected_seed=x["mask_order"][(i-1)%2]
        ensure(row["index"]==i and row["label_seed"]==expected_seed,
               "wrong replay ordinal/label input mask")
        ensure(set(row["state_hashes"])==STATES and
               all(valid_sha(s) for s in row["state_hashes"].values())
               and valid_sha(row["mapped_sha"]) and
               type(row["numba_rng_next"]) is float and
               0<=row["numba_rng_next"]<1,
               "missing exact six array hashes or Numba random state")
        ensure(row["native_layout"]==("C" if x["classes"]>=8 else "F"),
               "Numba dominance must use genuine C/F production policy")
        stop=row["stop"]
        ensure(stop["stop_reason"] in ("early_stop","max_iter") and
               type(stop["effective_iterations"]) is int and
               0<stop["effective_iterations"]<=30000 and
               type(stop["checks"]) is int and stop["checks"]>=1 and
               type(stop["stalled_checks"]) is int and stop["stalled_checks"]>=0
               and type(stop["stop_max"]) is int and stop["stop_max"]>=1,
               "invalid original real stop metadata")
        ensure((stop["effective_iterations"]<30000)==(
            stop["stop_reason"]=="early_stop"),
            "early stopping reason and effective iteration count mismatch")
        identity=(row["state_hashes"],row["mapped_sha"],
                  row["numba_rng_next"],row["stop"],row["native_layout"])
        ensure(expected_seed not in reference_by_mask or
               reference_by_mask[expected_seed]==identity,
               "repeating same labels in same persistent PCC model changed state or RNG")
        reference_by_mask.setdefault(expected_seed,identity)
    ensure(len(reference_by_mask)==2,
           "only one label mask was actually used")
    ensure(x["final_timed_state_hashes"]==samples[-1]["state_hashes"] and
           x["final_timed_mapped_sha"]==samples[-1]["mapped_sha"] and
           x["final_timed_rng_next"]==samples[-1]["numba_rng_next"],
           "timed 20th fit differs from after-timing exact native replay")

    wall=x["app_wall_ns_at_fit_count"]
    cpu=x["app_cpu_ns_at_fit_count"]
    ends=x["checkpoints_end_ns"]
    cpus=x["checkpoints_process_cpu_ns"]
    expect={str(n) for n in COUNTS}
    ensure(set(wall)==set(cpu)==set(ends)==set(cpus)==expect,
           "missing 1,2,5,10,20 time-to-native-prediction checkpoints")
    ensure(all(type(v) is int and v>0 for v in wall.values()) and
           all(type(v) is int and v>0 for v in cpu.values()),
           "invalid complete application CPU/wall time")
    for a,b in zip(COUNTS,COUNTS[1:]):
        aa,bb=str(a),str(b)
        ensure(wall[aa]<wall[bb] and cpu[aa]<cpu[bb] and
               ends[aa]<ends[bb],
               "cumulative new-process timings not increasing with fits")
    ensure(x["script_entry_ns"]<x["imported_ns"]<=x["graph_start_ns"]<
           x["graph_done_ns"]<=x["model_ready_start_ns"]<=
           x["model_ready_ns"]<=x["first_fit_start_ns"]<
           ends["1"]<=ends["20"]<=x["fit20_done_ns"],
           "missing native source import/sklearn graph/model phases")
    ensure(x["fit20_done_ns"]-x["first_fit_start_ns"]<=wall["20"] and
           positive(x["postfit_replay_and_process_exit_wall_ns"]) and
           x["full_subprocess_wall_ns"]>wall["20"],
           "posttiming 20 exact native diagnostic replays charged to application")
    ensure(type(x["cache_data_loaded"]) is int and
           type(x["cache_data_saved"]) is int and
           isinstance(x["cache_trace"],list),
           "native Numba compiled code cache diagnostics missing")
    if variant=="original":
        ensure(not x["cache_trace"] and x["cache_data_loaded"]==
               x["cache_data_saved"]==0,
               "original control unexpectedly loaded a cache")
    elif phase=="cold":
        ensure(x["cache_data_saved"]>=1,"production first JIT did not save native code")
    else:
        ensure(x["cache_data_loaded"]>=1,
               "production independent warm new process didn't load Numba code")


def verify_host(doc):
    ensure(doc["schema"]==1 and doc["original"]==OLD
           and doc["production"]==PROD
           and doc["platform"] in ("linux","win32")
           and len(doc["branch_head"])==40,
           "wrong source/version/OS")
    ensure(valid_sha(doc["original_source_sha256"]) and
           valid_sha(doc["production_source_sha256"]) and
           doc["original_source_sha256"]!=doc["production_source_sha256"],
           "not exact pre/post-#216 source code")
    ensure(len(doc["cases"])==6,
           "incomplete fixed original 3 real datasets x 2 first mask orders")
    scenarios={}
    for row in doc["cases"]:
        dataset,seed=row["dataset"],row["first_seed"]
        ident=(dataset,seed)
        ensure(dataset in DATASETS and seed in SEEDS and ident not in scenarios,
               "duplicate/nonfrozen cohort")
        ensure(row["variant_order"]==(
            list(VARIANTS) if seed==SEEDS[0] else list(VARIANTS[::-1])),
            "cached variant order not counterbalanced")
        ensure(set(row["variants"])==set(VARIANTS),
               "both original and current production sources required")
        reference=None
        for variant in VARIANTS:
            entry=row["variants"][variant]
            sha=doc["original_source_sha256" if variant=="original"
                    else "production_source_sha256"]
            ensure(entry["source_sha256"]==sha,
                   "actual original or production Git source drifted")
            ensure(len(entry["launches"])==2,
                   "not independent cold and warm Python interpreters")
            disk=entry["disk_cache_bytes"]
            ensure(type(disk) is int and disk>=0 and disk<=32*1024*1024,
                   "Numba code disk cache exceeds preregistered bound")
            ensure((disk==0) if variant=="original" else (disk>0),
                   "unexpected source cache persistence")
            for phase,observed in zip(PHASES,entry["launches"]):
                verify_subprocess(observed,dataset,seed,variant,phase,sha)
                ensure(observed["platform"]==doc["platform"],
                       "child OS source differs from parent")
                fields=("input_sha","graph_hashes","labels_sha","fixture_sha",
                        "final_timed_state_hashes","final_timed_mapped_sha",
                        "final_timed_rng_next","replay_outcomes",
                        "original_graph_prepared_once","model_graph_stable")
                if reference is None:
                    reference={k:observed[k] for k in fields}
                for k in fields:
                    ensure(observed[k]==reference[k],
                           "20 repeated fits source/CPU cache/RNG parity failure: "+
                           repr((ident,variant,phase,k)))
        scenarios[ident]=row
    ensure(set(scenarios)=={(d,s) for d in DATASETS for s in SEEDS},
           "missing a full native reused-graph original case")
    return scenarios


def make_report(docs,collections):
    rows=[]
    md=[
        "# Post-#216 Numba persistent-process: amortizing native disk JIT CPU",
        "",
        "**Full original two-OS scientific verification PASSED.**",
        "48 actually distinct Python interpreters (24 Linux/24 Windows).",
        "Each builds exactly ONE real undirected kNN10 graph, installs it",
        "ONCE in an unchanged original PCC(impl='numba') instance, and",
        "performs 20 successive public fits alternating two distinct",
        "balanced labeled masks with original monitored 30k cap.",
        "All original native six states, NumPy labels/graph, next true",
        "Numba RNG samples and stopping metrics are exact at ALL 20",
        "post-timing replay steps across source versions and OS.",
        "960 native fits are timed; another 960 seeded native fits",
        "validate exact states outside the time-to-prediction boundary.",
        "",
        "## New Python process → Nth completed native PCC fit (graph reused)",
        "",
        "| OS | Dataset | Fits N | Uncached whole app wall ms | Cached whole app wall ms | Wall saved | CPU saved |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    useful=True
    for doc,cases in zip(docs,collections):
        for dataset in DATASETS:
            ordered=[cases[(dataset,s)] for s in SEEDS]
            for n in COUNTS:
                key=str(n)
                matched=[
                    (c["variants"]["original"]["launches"][1],
                     c["variants"]["production"]["launches"][1])
                    for c in ordered]
                wall=100*(1-statistics.median(
                    b["app_wall_ns_at_fit_count"][key]/
                    a["app_wall_ns_at_fit_count"][key] for a,b in matched))
                cpu=100*(1-statistics.median(
                    b["app_cpu_ns_at_fit_count"][key]/
                    a["app_cpu_ns_at_fit_count"][key] for a,b in matched))
                uncached_ms=statistics.median(
                    a["app_wall_ns_at_fit_count"][key] for a,b in matched)/1e6
                cached_ms=statistics.median(
                    b["app_wall_ns_at_fit_count"][key] for a,b in matched)/1e6
                cold=100*(statistics.median(
                    c["variants"]["production"]["launches"][0][
                        "app_wall_ns_at_fit_count"][key]/
                    c["variants"]["original"]["launches"][0][
                        "app_wall_ns_at_fit_count"][key]
                    for c in ordered)-1)
                if n==20 and (wall<10 or cpu<10 or cold>20):
                    useful=False
                rows.append(dict(os=doc["platform"],dataset=dataset,
                                 n_fits=n,uncached_wall_median_ms=uncached_ms,
                                 production_cached_wall_median_ms=cached_ms,
                                 wall_saved_pct=wall,cpu_saved_pct=cpu,
                                 cold_first_cache_wall_overhead_pct=cold))
                md.append(f"| {doc['platform']} | {dataset} | {n} | "
                          f"{uncached_ms:.2f} | {cached_ms:.2f} | "
                          f"{wall:+.2f}% | {cpu:+.2f}% |")
    verdict="USEFUL-AT-20" if useful else "AMORTIZED"
    md+=["",
        f"**Predeclared observational verdict: {verdict}.**",
        "",
        "USEFUL-AT-20 requires >=10% cumulative whole-program wall and CPU",
        "savings after 20 fits on ALL THREE datasets on EACH OS, cold first",
        "compilation cumulative wall overhead <=20%, exact 20-fit semantics",
        "and <=32 MiB disk code per fixture. Failure is AMORTIZED, not CI",
        "failure; no frozen criterion was altered after seeing data.",
        "",
        "This is deliberately NOT 20 separate interpreter startups or",
        "20 separately constructed graphs: the same public PCC model",
        "and ordered dense int64 kNN graph are reused throughout.",
        "Only the TWO alternate label-mask starting orders per dataset",
        "are independent observations; N=1/2/5/10/20 checkpoints are",
        "correlated within each interpreter, not technical repeats.",
        "The old and new Numba code differ in four disk-cache decorators",
        "only. Their already-compiled particle kernels are identical;",
        "cache avoids repeated new-process compilation, so the",
        "fixed startup saving is diluted across successive warm fits.",
        "Cython/NumPy backends, changing hardware, different kNN sizes",
        "and public 500k maximum were not covered by this 30k-capped",
        "native Numba test. Keep scientific PR DRAFT/UNMERGED.",
    ]
    return "\n".join(md)+"\n",verdict,rows


def main(folder):
    files=list(Path(folder).glob("**/post216_persistent_amortization_results.json"))
    ensure(len(files)==2,"both complete original OS raw persistent corpora required")
    docs=[json.loads(f.read_text(encoding="utf-8")) for f in files]
    ensure({d["platform"] for d in docs}=={"linux","win32"},
           "must have exact one Linux and one Windows corpus")
    docs.sort(key=lambda d:d["platform"])
    collections=[verify_host(d) for d in docs]
    for name in ("original","production","branch_head",
                 "original_source_sha256","production_source_sha256"):
        ensure(len({d[name] for d in docs})==1,
               "cross-host source canonical Git identity differs "+name)
    for ident in {(d,s) for d in DATASETS for s in SEEDS}:
        left,right=(v[ident] for v in collections)
        orig=left["variants"]["original"]["launches"][0]
        ref=right["variants"]["original"]["launches"][0]
        for key in ("input_sha","graph_hashes","labels_sha","fixture_sha",
                    "final_timed_state_hashes","final_timed_mapped_sha",
                    "final_timed_rng_next","replay_outcomes"):
            ensure(orig[key]==ref[key],
                   "Linux/Windows original persistent model state mismatch "+key)
    report,verdict,rows=make_report(docs,collections)
    Path("post216_persistent_amortization_report.md").write_text(
        report,encoding="utf-8")
    Path("post216_persistent_amortization_decision.json").write_text(
        json.dumps(dict(schema=1,complete=True,verdict=verdict,
                        distinct_python_processes=48,
                        timed_20_fit_sequences=48,
                        actual_timed_native_fits=960,
                        post_timing_replay_native_fits=960,
                        independent_first_mask_orders_per_dataset=2,
                        checkpoints=list(COUNTS),rows=rows),
                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(report,flush=True)
    print("STRICT COMPLETE two-host 48 actual persistent Python interpreters, "
          "960 original native fits timed + 960 full Numba original state/"
          "RNG/stop replay fits on one reused graph per interpreter PASS; "
          "observational verdict "+verdict,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
