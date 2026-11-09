"""Strict independent Linux/Windows real Python startup-to-prediction cache audit."""
import json
import math
from pathlib import Path
import statistics
import sys

PROD="b4c100bd22eeff1979a1a581bc73682b8c3705b8"
OLD="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("wine","breast_cancer","digits")
SEEDS=(68013,68117)
PHASES=("cold","warm1","warm2","warm3")
VARIANTS=("original","production")
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}
THREADS=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
         "MKL_NUM_THREADS","NUMBA_NUM_THREADS")


def check(ok,reason):
    if not ok:
        raise AssertionError(reason)


def sha(s):
    return isinstance(s,str) and len(s)==64 and all(
        c in "0123456789abcdef" for c in s)


def positive(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0


def validate_run(r,ds,seed,variant,phase,source):
    check(r["schema"]==1 and r["dataset"]==ds and r["seed"]==seed and
          r["variant"]==variant and r["phase"]==phase and
          r["case_id"]==f"{ds}-labels{seed}","wrong original fixture/phase")
    check(r["source_sha256"]==source and r["backend_module"]=="pcc_numba"
          and r["true_original_pcc_dispatch"] is True,
          "not real original production high-level Numba backend")
    check(r["platform"] in ("linux","win32"),"unrecognized OS")
    check(r["real_nodes"]=={"wine":178,"breast_cancer":569,"digits":1797}[ds]
          and r["classes"]=={"wine":3,"breast_cancer":2,"digits":10}[ds],
          "wrong real sklearn dataset/classes")
    check(r["real_k_nn"]==10 and r["labels_per_class"]==3
          and r["propagation_seed"]==0
          and r["full_fit_max_iter"]==30000
          and r["early_stop"] is True,"changed native PCC config")
    check(all(r["threads"].get(v)=="1" for v in THREADS),
          "uncontrolled CPU threading")
    check(all(sha(r[k]) for k in ("fixture_sha","input_hash",
                                  "label_hash","mapped_labels_hash")),
          "missing source, graph, label and mapping hashes")
    check(len(r["graph_hashes"])==2 and
          all(sha(v) for v in r["graph_hashes"]),
          "missing ordered original dense int64 graph SHA")
    check(set(r["state_hashes"])==STATES and
          all(sha(v) for v in r["state_hashes"].values()),
          "missing six original native full-state hashes")
    check(type(r["native_rng_endpoint"]) is float and
          0<=r["native_rng_endpoint"]<1,
          "native Numba RNG endpoint missing")
    t=r["phase_timing"]
    check(t["total_cpu_to_first_fit_ns"]==r["app_first_result_cpu_ns"]
          and positive(r["app_first_result_cpu_ns"])
          and positive(r["app_first_result_wall_ns"]),
          "missing true full-process CPU/wall to first prediction")
    check(positive(r["python_spawn_and_import_wall_ns"]) and
          r["python_spawn_and_import_wall_ns"]<r["app_first_result_wall_ns"],
          "Python process startup+imports omitted from wall timing")
    check(positive(r["postfit_verification_and_process_exit_ns"]) and
          r["total_process_including_postfit_checks_wall_ns"]>
          r["app_first_result_wall_ns"],
          "postfit hashing accidentally included in first-result wall")
    chain=("entry_ns","imported_ns","prep_start_ns","prep_end_ns",
           "model_start_ns","model_ready_ns","fit_start_ns","fit_end_ns")
    check(all(type(t[x]) is int for x in chain) and
          all(t[a]<=t[b] for a,b in zip(chain,chain[1:])),
          "bad original application stage chronology")
    for field in ("import_within_script_ns","prep_ns","graph_build_ns",
                  "data_load_ns","preprocessing_ns","model_and_set_graph_ns",
                  "fit_wall_ns","fit_cpu_ns"):
        check(positive(t[field]),"missing application phase "+field)
    check(t["fit_end_ns"]-t["fit_start_ns"]==t["fit_wall_ns"],
          "not the real unwrapped public fit_predict timer")
    check(t["graph_build_ns"]<=t["prep_ns"]*1.05 and
          t["data_load_ns"]+t["preprocessing_ns"]+t["graph_build_ns"]
          <=t["prep_ns"]*1.1,
          "source sklearn preparation phase accounting inconsistent")
    info=r["native_stop"]
    check(info["stop_reason"] in ("early_stop","max_iter")
          and type(info["effective_iterations"]) is int
          and 0<info["effective_iterations"]<=30000
          and type(info["checks"]) is int and info["checks"]>=1
          and type(info["stalled_checks"]) is int
          and info["stalled_checks"]>=0
          and type(info["stop_max"]) is int and info["stop_max"]>=1,
          "native exact stop metric incomplete")
    check((info["effective_iterations"]<30000)==
          (info["stop_reason"]=="early_stop"),
          "actual native monitored stop reason inconsistent")
    check(type(r["cache_load_count"]) is int and
          type(r["cache_save_count"]) is int and
          isinstance(r["cache_trace"],list),
          "no genuine Numba on-disk cache provenance")
    if variant=="original":
        check(not r["cache_trace"] and r["cache_load_count"]==0
              and r["cache_save_count"]==0,
              "no-cache original unexpectedly persists native machine code")
    elif phase=="cold":
        check(r["cache_save_count"]>=1,
              "first production run did not save new Numba machine code")
    else:
        check(r["cache_load_count"]>=1,
              "later fresh production interpreter did not load native code")


def verify_host(doc):
    check(doc["schema"]==1 and doc["production"]==PROD
          and doc["original"]==OLD and doc["platform"] in ("linux","win32")
          and len(doc["branch_head"])==40
          and doc["scientific_no_production_diff"] is True,
          "not original pinned post-216 source")
    check(sha(doc["original_source_sha256"]) and
          sha(doc["production_source_sha256"]) and
          doc["original_source_sha256"]!=doc["production_source_sha256"],
          "not canonical old vs cached source Git blobs")
    check(len(doc["cases"])==6,"missing one of 6 original case masks")
    cases={}
    for c in doc["cases"]:
        ds,seed=c["dataset"],c["seed"]
        key=(ds,seed)
        check(ds in DATASETS and seed in SEEDS and key not in cases,
              "incorrect duplicate/fake original case")
        expected_order=(list(VARIANTS) if seed==SEEDS[0] else
                        list(VARIANTS[::-1]))
        check(c["order"]==expected_order,"variant sequence not balanced by seed")
        check(set(c["variants"])==set(VARIANTS),"missing original/production sources")
        reference=None
        for variant in VARIANTS:
            data=c["variants"][variant]
            expected_sha=doc[
                "original_source_sha256" if variant=="original"
                else "production_source_sha256"]
            check(data["source_sha256"]==expected_sha,
                  "true Numba module source identity changed")
            check(len(data["launches"])==4,"missing independent fresh launches")
            cache_bytes=data["disk_code_cache_bytes"]
            check(type(cache_bytes) is int and
                  0<=cache_bytes<=32*1024*1024,"native compiled code cache too large")
            check(cache_bytes>0 if variant=="production" else cache_bytes==0,
                  "unexpected absence/presence of native Numba machine code disk cache")
            for phase,r in zip(PHASES,data["launches"]):
                validate_run(r,ds,seed,variant,phase,expected_sha)
                check(r["platform"]==doc["platform"],
                      "child original OS inconsistent with parent")
                fields=("fixture_sha","input_hash","graph_hashes",
                        "label_hash","mapped_labels_hash",
                        "state_hashes","native_rng_endpoint","native_stop",
                        "true_original_pcc_dispatch")
                if reference is None:
                    reference={k:r[k] for k in fields}
                for field in fields:
                    check(r[field]==reference[field],
                          "original vs cached native state/stop/RNG/graph diverged "
                          +repr((ds,seed,variant,phase,field)))
        cases[key]=c
    check(set(cases)=={(d,s) for d in DATASETS for s in SEEDS},
          "incomplete six independent original label masks")
    return cases


def summarize(docs,verified):
    lines=[
        "# Post-216: whole Python startup to first native Numba prediction",
        "",
        "**STRICT COMPLETE:** 96 independent new Python interpreters;",
        "real offline sklearn datasets and StandardScaler; original undirected",
        "union kNN10 graph built afresh in every process; original high-level",
        "PCC(impl='numba') dispatch; max_iter=30000 with monitoring on.",
        "Original Git Numba no-cache and actual post-216 cached Numba differ",
        "only by FOUR decorators; original full six states, class mapping,",
        "native Numba RNG and stopping exactly match across versions/hosts.",
        "Parent clock starts before Python creation; child fit return",
        "timestamp is sampled BEFORE all verification work and serialization.",
        "Three fully new cached interpreters genuinely load compiled code",
        "after first cache creation on private disk per dataset/mask/source.",
        "",
        "| OS | Dataset | Full app original wall ms | Full app cache wall ms | Whole app wall saved | Whole app CPU saved | First uncached app wall overhead | Cache app startup/import share | Cache app kNN build share | Cache app PCC fit share | Early-stopped seeds | Numba cache MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows=[]
    meaningful=True
    for d,cases in zip(docs,verified):
        for ds in DATASETS:
            two=[cases[(ds,s)] for s in SEEDS]
            paired=[
                (c["variants"]["original"]["launches"][i],
                 c["variants"]["production"]["launches"][i])
                for c in two for i in (1,2,3)]
            wall=100*(1-statistics.median(
                b["app_first_result_wall_ns"]/a["app_first_result_wall_ns"]
                for a,b in paired))
            cpu=100*(1-statistics.median(
                b["app_first_result_cpu_ns"]/a["app_first_result_cpu_ns"]
                for a,b in paired))
            original_ms=statistics.median(
                a["app_first_result_wall_ns"] for a,b in paired)/1e6
            production_ms=statistics.median(
                b["app_first_result_wall_ns"] for a,b in paired)/1e6
            cold=100*(statistics.median(
                c["variants"]["production"]["launches"][0][
                    "app_first_result_wall_ns"]/
                c["variants"]["original"]["launches"][0][
                    "app_first_result_wall_ns"] for c in two)-1)
            share={}
            for name,key in (("imports","python_spawn_and_import_wall_ns"),
                             ("graph","graph_build_ns"),("fit","fit_wall_ns")):
                share[name]=statistics.median(
                    100*(b[key] if key in b else b["phase_timing"][key])/
                    b["app_first_result_wall_ns"] for a,b in paired)
            stopped=sum(
                c["variants"]["original"]["launches"][0]["native_stop"][
                    "stop_reason"]=="early_stop" for c in two)
            disk=max(c["variants"]["production"]["disk_code_cache_bytes"]
                     for c in two)/1048576
            if wall<20 or cpu<20 or cold>20 or disk>32:
                meaningful=False
            rows.append(dict(
                os=d["platform"],dataset=ds,
                full_app_wall_saved_pct=wall,full_app_cpu_saved_pct=cpu,
                original_app_wall_median_ms=original_ms,
                production_app_wall_median_ms=production_ms,
                cold_first_app_wall_overhead_pct=cold,
                cache_startup_import_share_pct=share["imports"],
                cache_knn_graph_build_share_pct=share["graph"],
                cache_native_fit_share_pct=share["fit"],
                stopped_label_masks=stopped,disk_mib=disk))
            lines.append(
                f"| {d['platform']} | {ds} | {original_ms:.2f} | "
                f"{production_ms:.2f} | {wall:+.2f}% | {cpu:+.2f}% | "
                f"{cold:+.2f}% | {share['imports']:.1f}% | "
                f"{share['graph']:.1f}% | {share['fit']:.1f}% | "
                f"{stopped}/2 | {disk:.2f} |")
    verdict="MEANINGFUL" if meaningful else "LIMITED"
    lines.extend([
        "",
        f"**Preregistered OBSERVATIONAL verdict: {verdict}.**",
        "",
        "MEANINGFUL requires >=20% paired application-to-first-prediction",
        "wall AND CPU saved for each real dataset on each OS, plus <=20%",
        "additional cold first-launch wall and <=32MiB native disk code.",
        "LIMITED is a valid, complete scientific result, not failed CI.",
        "",
        "This is a full data-to-first-prediction **30k-cap** workload, not",
        "the public 500k default and not end-to-end process exit time.",
        "Diagnostic SHA, repeated state/stop replay, and native RNG probe",
        "were executed after the fit endpoint, so none is reported as a",
        "user's application startup cost. The native particle kernel",
        "did not become ~90% faster; compatible JIT recompilation is avoided.",
        "A cold first-ever cache compile can cost slightly more and the",
        "benefit depends on persistent writable compatible cached code.",
        "OS hosts differ in CPU/hardware and times must not be pooled;",
        "three warm launches are technical repetitions for each of only",
        "two independent label masks. Preserve original dense int64 graph.",
        "NO further production source merge is warranted; #216 is merged.",
    ])
    return "\n".join(lines)+"\n",verdict,rows


def main(root):
    paths=list(Path(root).glob("**/post216_full_app_cache_results.json"))
    check(len(paths)==2,"both original OS raw study artifacts required")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    check({d["platform"] for d in docs}=={"linux","win32"},
          "require real Linux and Windows independent original native runs")
    docs.sort(key=lambda x:x["platform"])
    valid=[verify_host(x) for x in docs]
    for key in ("production","original","branch_head",
                "original_source_sha256","production_source_sha256"):
        check(len({d[key] for d in docs})==1,
              "source drift across Linux and Windows: "+key)
    for ds in DATASETS:
        for seed in SEEDS:
            a,b=(v[(ds,seed)] for v in valid)
            x=a["variants"]["original"]["launches"][0]
            y=b["variants"]["original"]["launches"][0]
            for key in ("fixture_sha","input_hash","graph_hashes","label_hash",
                        "mapped_labels_hash","state_hashes",
                        "native_rng_endpoint","native_stop"):
                check(x[key]==y[key],
                      "OS exact original source/data/state/RNG mismatch: "+key)
    text,verdict,rows=summarize(docs,valid)
    Path("post216_full_app_cache_report.md").write_text(text,encoding="utf-8")
    Path("post216_full_app_cache_decision.json").write_text(
        json.dumps(dict(schema=1,complete=True,
                        observational_verdict=verdict,
                        fresh_python_processes=96,
                        first_real_native_fits=96,
                        postfit_diagnostic_replays=96,
                        independent_label_masks=2,
                        warmed_reuse_launches_per_case=3,
                        rows=rows),indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print(text,flush=True)
    print("STRICT BOTH OS COMPLETE: all 96 real native full process startup/"
          "sklearn kNN graph/fit outputs and cache source/stop/RNG exact, "
          "observational verdict:",verdict,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
