"""Strict cross-host observed stop-on/fixed full fit data validation and report."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("wine","breast_cancer","digits")
SEEDS=(45013,45117)
BACKENDS=("numpy","numba","cython")
MODES=("early_stopping","fixed")
EXPECTED={(ds,seed,backend,mode,rep)
          for ds in DATASETS for seed in SEEDS
          for backend in BACKENDS for mode in MODES
          for rep in range(3)}
WARMUPS={(ds,seed,backend,index)
         for ds in DATASETS for seed in SEEDS
         for backend in BACKENDS for index in range(2)}
PARITY={(ds,seed,backend)
        for ds in DATASETS for seed in SEEDS for backend in BACKENDS}
PHASES=("initialization_and_dispatch_s","propagation_s",
        "postprocessing_s","fit_predict_s")
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}
IDENTITY={"numpy":"pcc_numpy.pcc_propagate_numpy",
          "numba":"pcc_numba.pcc_propagate_numba",
          "cython":"pcc_step.pcc_propagate"}
PROTECTED=("pcc.py","pcc_numba.py","pcc_numpy.py","pcc_graph.py",
           "pcc_step.pyx","benchmarks/runner.py","benchmarks/run.py")


def check(cond,message):
    if not cond:
        raise AssertionError(message)


def is_sha(a):
    return isinstance(a,str) and len(a)==64 and all(v in "0123456789abcdef" for v in a)


def positive(a):
    return type(a) in (int,float) and math.isfinite(a) and a>0


def validate_mode(r,backend,mode,cap):
    check(r["backend_identity"]==IDENTITY[backend],"backend fallback")
    check(r["backend"]==backend and r["mode"]==mode and r["max_iter"]==cap,
          "mode/backend/native cap mismatch")
    count=r["effective_iterations"]
    check(type(count) is int and 0<count<=cap,"invalid effective native iteration count")
    check(r["stop_reason"] in ("early_stop","max_iter"),"unknown stop outcome")
    check(type(r["checks"]) is int and r["checks"]>=0,"bad check counter")
    check(type(r["stalled_checks"]) is int and r["stalled_checks"]>=0,
          "bad stalled check count")
    check(type(r["stop_max"]) is int and r["stop_max"]>=1,"bad original threshold")
    if mode=="fixed":
        check(count==cap and r["stop_reason"]=="max_iter",
              "fixed native fit prematurely stopped")
    else:
        check((count<cap)==(r["stop_reason"]=="early_stop"),
              "invalid early stopping reason versus observed iteration count")
    check(set(r["state_hashes"])==STATES and
          all(is_sha(v) for v in r["state_hashes"].values()),
          "six final native states missing")
    check(0<=r["accuracy_unlabeled"]<=1,"accuracy invalid")
    for name in PHASES:
        check(positive(r[name]),"missing phase elapsed time "+name)
    t=sum(r[k] for k in PHASES[:3])
    check(abs(t-r["fit_predict_s"])<max(1e-5,.01*r["fit_predict_s"]),
          "timestamp phase additivity drift")
    check(positive(r["propagation_s_per_iteration"]) and
          abs(r["propagation_s"]/count-r["propagation_s_per_iteration"])<1e-10,
          "per-iteration propagation metric malformed")
    check(r["propagation_seed_effective"]==(123456789 if backend=="cython" else 0),
          "real propagation RNG seed incorrect")


def verify_one(doc):
    check(doc["schema"]==1 and doc["base_commit"]==BASE,"wrong pinned master")
    check(doc["os"] in ("linux","win32"),"wrong OS")
    check(len(doc["branch_head"])==40,"missing branch source SHA")
    check(all(len(doc[k])==count for k,count in
              (("prepared",6),("layout_probes",3),("measured",108),
               ("warmups",36),("parity",18))),
          "incomplete frozen cohort")
    manifest=doc["manifest"]
    check(manifest["repository"]["commit"]==doc["branch_head"],
          "manifest source commit mismatch")
    check(manifest["configuration"]["base_commit"]==BASE,
          "manifest benchmark config source mismatch")
    check(manifest["configuration"]["iterations"]==30000 and
          manifest["configuration"]["es_chk"]==2000 and
          manifest["configuration"]["repeats"]==3 and
          manifest["configuration"]["modes"]==list(MODES),
          "unapproved observational parameters")
    check(manifest["configuration"]["seeds"]==list(SEEDS) and
          manifest["configuration"]["datasets"]==list(DATASETS) and
          manifest["configuration"]["backends"]==list(BACKENDS),
          "unapproved fixtures/backends")
    threads=manifest["environment"]["thread_settings"]
    check(all(threads.get(s)=="1" for s in (
        "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
        "MKL_NUM_THREADS","NUMBA_NUM_THREADS")),"native thread controls drifted")
    sources=manifest["repository"]["source_hashes"]
    check(all(is_sha(sources.get(s)) for s in PROTECTED),
          "missing native source provenance")
    extension=manifest["cython_extension"]
    check(is_sha(extension["sha256"]) and
          extension["path"].lower().endswith((".pyd",".so")),
          "Cython binary not identified")
    check(positive(doc["experiment_wall_seconds"]),"wall elapsed omitted")
    preps={}
    for i,p in enumerate(doc["prepared"]):
        key=(p["dataset"],p["seed"])
        check(key==(DATASETS[i//2],SEEDS[i%2]),"dataset/mask preparation order")
        check(key not in preps,"duplicate fixture")
        check(p["case_id"]==f"{key[0]}-labels{key[1]}", "invalid case id")
        check(p["graph_reused"]==(i%2==1),"wrong graph cache reuse")
        check(p["classes"]=={"wine":3,"breast_cancer":2,"digits":10}[key[0]],
              "unexpected number of known classes")
        check(all(is_sha(p[k]) for k in (
            "fixture_sha","neighbors_sha","degrees_sha","truth_sha","labels_sha",
            "input_sha")),"input/graph hashes missing")
        preps[key]=p
    layouts={}
    for row in doc["layout_probes"]:
        ds=row["dataset"]
        check(ds in DATASETS and ds not in layouts,"missing duplicate native layout probe")
        layouts[ds]=row["native_layout"]
        check(set(row["native_layout"])==set(BACKENDS),"unmeasured native layouts")
        for backend in BACKENDS:
            value=row["native_layout"][backend]
            check(value["backend_identity"]==IDENTITY[backend],"native layout fallback")
            check(value["dominance_order"]==(
                "C" if backend=="numba" and ds=="digits" else "F"),
                "incorrect post-#203 production Numba memory order")
    check(set(layouts)==set(DATASETS),"missing native C/F checks")

    timed={}
    for row in doc["measured"]:
        key=(row["dataset"],row["seed"],row["backend"],row["mode"],row["rep"])
        check(key in EXPECTED and key not in timed,"wrong duplicate timing identity")
        ds,seed,backend,mode,rep=key
        check(row["case_id"]==f"{ds}-labels{seed}" and
              row["fixture_sha"]==preps[(ds,seed)]["fixture_sha"],
              "row input provenance mismatch")
        check(row["backend_order"]==list(BACKENDS[rep:]+BACKENDS[:rep]) and
              row["mode_order"]==list(MODES if rep%2==0 else MODES[::-1]),
              "invalid randomized/balanced backend or stop schedule")
        check(row["backend_order"][row["backend_order_index"]]==backend and
              row["mode_order"][row["mode_order_index"]]==mode,
              "inconsistent schedule order")
        validate_mode(row,backend,mode,30000)
        if backend=="numpy":
            check(is_sha(row["numpy_rng_sha"]),"missing exact NumPy RNG endpoint")
        else:
            check(row["numpy_rng_sha"] is None,"unmeasurable native RNG overstated")
        timed[key]=row
    check(set(timed)==EXPECTED,"not all full real native fits present")
    for ds in DATASETS:
        for seed in SEEDS:
            for backend in BACKENDS:
                for mode in MODES:
                    same=[timed[(ds,seed,backend,mode,i)] for i in range(3)]
                    expected=(same[0]["state_hashes"],
                              same[0]["effective_iterations"],
                              same[0]["stop_reason"],
                              same[0]["checks"],same[0]["stalled_checks"],
                              same[0]["stop_max"],
                              same[0]["accuracy_unlabeled"],
                              same[0]["numpy_rng_sha"])
                    for r in same[1:]:
                        actual=(r["state_hashes"],r["effective_iterations"],
                                r["stop_reason"],r["checks"],r["stalled_checks"],
                                r["stop_max"],r["accuracy_unlabeled"],
                                r["numpy_rng_sha"])
                        check(actual==expected,
                              "same-seed repeated original native fit diverged")
    warms={}
    for w in doc["warmups"]:
        name=w["case_id"].rsplit("-labels",1)[0]
        seed=int(w["case_id"].rsplit("-labels",1)[1])
        key=(name,seed,w["backend"],w["index"])
        check(key in WARMUPS and key not in warms,"wrong warmup")
        validate_mode(w["info"],w["backend"],"fixed",2)
        if w["backend"]=="numpy":check(is_sha(w["numpy_rng_sha"]),"warm RNG absent")
        warms[key]=w
    check(set(warms)==WARMUPS,"missing warmups")

    prefix={}
    for p in doc["parity"]:
        ds,seed=p["case_id"].split("-labels",1)
        key=(ds,int(seed),p["backend"])
        check(key in PARITY and key not in prefix,"bad replay identity")
        check(p["original_native_prefix_parity"] is True,"prefix replay not verified")
        r=timed[(ds,int(seed),p["backend"],"early_stopping",0)]
        check(p["effective_iterations"]==r["effective_iterations"] and
              p["stop_reason"]==r["stop_reason"] and
              p["state_hashes"]==r["state_hashes"] and
              p["state_hashes"]==p["fixed_prefix_hashes"],
              "stopped/native deterministic prefix mismatch")
        check(p["reached_cap"]==(r["effective_iterations"]==30000),
              "incorrect capped label")
        check(p["method"]==("same_cap" if p["reached_cap"]
                          else "seeded_prefix_replay"),
              "missing actual independent native prefix replay")
        if p["method"]=="seeded_prefix_replay":
            check(positive(p["replay_only_elapsed_s"]),"missing replay timer")
        else:
            check(p["replay_only_elapsed_s"] is None,"unexpected replay at same cap")
        if p["backend"]=="numpy":
            check(p["numpy_rng_sha"]==r["numpy_rng_sha"]==
                  p["fixed_prefix_numpy_rng_sha"],
                  "NumPy RNG endpoint mismatch on prefix")
        else:
            check(p["numpy_rng_sha"] is None and
                  p["fixed_prefix_numpy_rng_sha"] is None,
                  "unsupported private Cython/Numba RNG state claim")
        prefix[key]=p
    check(set(prefix)==PARITY,"incomplete truncated-prefix/cap assessment")
    return preps,timed,prefix


def render(data):
    lines=[
        "# Post-#211 original PCC native early stopping vs fixed cap",
        "",
        "**COMPLETE observational baseline**: two independent OS hosts,",
        "216/216 original native full fits, 72/72 two-iteration warmups,",
        "36/36 true native seeded prefix/same-cap parity checks,",
        "six exact full-state array hashes and original early-stop counters.",
        "Graph is built once for each dataset, reused for both label masks.",
        "No source, kernel, cache policy, stop rule or dense graph changed.",
        "",
        "## Fit-time and stopping behavior, native backends, max_iter=30,000",
        "",
        "| Host | Dataset | Backend | Stopped masks / 2 | Median true effective iterations | Early-stop-on fit median ms | Fixed-cap fit median ms | Median paired fit CPU saving | Monitored prop. fraction |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d,(_,measurements,prefix) in data:
        for ds in DATASETS:
            for backend in BACKENDS:
                stop=[measurements[(ds,s,backend,"early_stopping",i)]
                      for s in SEEDS for i in range(3)]
                fixed=[measurements[(ds,s,backend,"fixed",i)]
                       for s in SEEDS for i in range(3)]
                paired=[(measurements[(ds,s,backend,"early_stopping",i)],
                         measurements[(ds,s,backend,"fixed",i)])
                        for s in SEEDS for i in range(3)]
                fraction=100*statistics.median(
                    r["propagation_s"]/r["fit_predict_s"] for r in stop)
                reduction=100*(1-statistics.median(
                    a["fit_predict_s"]/b["fit_predict_s"] for a,b in paired))
                stopped=sum(prefix[(ds,s,backend)]["method"]=="seeded_prefix_replay"
                            for s in SEEDS)
                lines.append(
                    f"| {d['os']} | {ds} | {backend} | {stopped}/2 | "
                    f"{statistics.median(r['effective_iterations'] for r in stop):.0f} | "
                    f"{statistics.median(r['fit_predict_s'] for r in stop)*1e3:.3f} | "
                    f"{statistics.median(r['fit_predict_s'] for r in fixed)*1e3:.3f} | "
                    f"{reduction:+.2f}% | {fraction:.1f}% |")
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- The 3 timed repetitions per mode/mask are technical repeats, not",
        "  three independent label seeds; only TWO independent masks/dataset.",
        "- When stopping occurred before the 30k cap, a separately reseeded",
        "  full run with early_stop=False and max_iter set exactly to that",
        "  stopped iteration reproduced all six arrays and the NumPy MT RNG",
        "  endpoint where measurable. Replay is not an uninterrupted observer;",
        "  this does not prove that a stopped fit is scientifically converged.",
        "- Cython production default potential row-max cache is ON when",
        "  early_stop=True and OFF when early_stop=False (#178). Thus even",
        "  same-cap stop/no-stop relative times include different native",
        "  specialization costs, not only potential-check overhead.",
        "- 30,000 iterations are a deliberate experimental cap, **not** the",
        "  public max_iter=500,000. A case that did not stop before 30k",
        "  is **censored**, not a case that could never stop.",
        "- The historic 3,600-fit 20-dataset stopping campaign in #17,",
        "  and alternative criterion studies #5/#7/#174 remain valid and",
        "  have not been repeated or supplanted.",
        "- Full fit times exclude graph construction, set_graph copying,",
        "  model instantiation, per-fit RNG seeding and Numba JIT compilation.",
        "- The four phase timings are measurements of the actual original",
        "  native backend via the corrected #203 runner timing wrapper;",
        "  layout is independently checked (Numba 10-class C-order).",
        "- Differences between Linux/Windows hosted CPU hardware and",
        "  compiled native binary builds cannot be called controlled speedups.",
        "- This report proposes **no production optimization**. Any future",
        "  code change requires its own prospectively frozen native original",
        "  versus candidate study, strict state parity and independent CI.",
    ]
    return "\n".join(lines)+"\n"


def main(root):
    paths=list(Path(root).glob("**/post211_early_stop_result.json"))
    check(len(paths)==2,f"expected complete two-OS run artifacts, got: {paths}")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    check({d["os"] for d in docs}=={"linux","win32"},"both OS required")
    docs.sort(key=lambda d:d["os"])
    results=[verify_one(d) for d in docs]
    check(len({d["branch_head"] for d in docs})==1,"source head differs across hosts")
    for ds in DATASETS:
        for seed in SEEDS:
            a=results[0][0][(ds,seed)]
            b=results[1][0][(ds,seed)]
            for name in ("fixture_sha","neighbors_sha","degrees_sha",
                         "truth_sha","labels_sha","input_sha"):
                check(a[name]==b[name],"cross-OS original input/graph mismatch "+name)
    for filename in PROTECTED:
        hashes={d["manifest"]["repository"]["source_hashes"][filename] for d in docs}
        check(len(hashes)==1,"cross-OS source differs "+filename)
    out=render(list(zip(docs,results)))
    Path("post211_early_stop_report.md").write_text(out,encoding="utf-8")
    print(out,flush=True)
    print("STRICT COMPLETE TWO-OS 216 original native early-stop/fixed fits + "
          "72 warmups + 36 state/RNG prefix parity checks PASS",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
