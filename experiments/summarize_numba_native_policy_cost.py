"""Strict two-host audit: original Numba policy and neighborhood CPU profile.

Observation only: varying movement policy changes trajectories, not isolated
instruction costs. Native-call wall time is NOT separately timed process CPU.
"""
import json
from pathlib import Path
import statistics
import sys

BASE = "4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
REAL = ("iris","wine","breast_cancer","digits")
GEOMETRIES = tuple((d,k) for d in REAL for k in (3,10)) + (("ring_8192_32",2),)
SEEDS = (89021,89119)
POLICIES = (0.,.5,1.)
CAPS = (1000,10000)
BATCH = {1000:64,10000:16}
REPS = 4
STATE = {"predictions","dominance","positions","strengths","distances",
         "owndeg","node_labels","mapped_labels"}
CLASSES = {"iris":3,"wine":3,"breast_cancer":2,"digits":10,"ring_8192_32":32}
NODES = {"iris":150,"wine":178,"breast_cancer":569,"digits":1797,"ring_8192_32":8192}


def demand(condition,message):
    if not condition:
        raise AssertionError(message)


def sha64(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)


def expected():
    return {(ds,k,seed,pgrd,cap) for ds,k in GEOMETRIES
            for seed in SEEDS for pgrd in POLICIES for cap in CAPS}


def check_state(values):
    demand(set(values)==STATE and all(sha64(h) for h in values.values()),
           "missing original eight-array hashes")


def checked(doc):
    demand(doc["schema"]==1 and doc["base"]==BASE and
           doc["native_original"] is True and doc["true_post_argmax"] is True,
           "protected original Numba source or real post-argmax missing")
    demand(doc["pairs"]==REPS and
           {int(k):int(v) for k,v in doc["cpu_batches"].items()}==BATCH,
           "frozen batch count or Windows CPU clock depth changed")
    demand(isinstance(doc["head"],str) and len(doc["head"])==40 and
           all(c in "0123456789abcdef" for c in doc["head"]),
           "not original Git commit SHA-1")
    demand(len(doc["source_hashes"])==6 and
           all(sha64(x) for x in doc["source_hashes"].values()),
           "original Git-source byte fingerprints missing")
    result={}
    for cohort in doc["cohorts"]:
        key=(cohort["dataset"],cohort["k"],cohort["seed"],
             cohort["p_grd"],cohort["cap"])
        demand(key in expected() and key not in result,"unregistered/duplicate graph-policy cohort")
        ds,k,seed,pgrd,cap=key
        demand(cohort["nodes"]==NODES[ds] and
               cohort["classes"]==CLASSES[ds] and
               cohort["particles"]==3*CLASSES[ds] and
               cohort["native_impl"]=="numba" and cohort["source_unchanged"],
               "original native graph/particle population invalid")
        demand(all(sha64(cohort[x]) for x in ("graph_sha","degree_sha",
               "labels_sha","fixture_sha","true_numba_argmax_sha")),
               "graph/output source fingerprint missing")
        demand(cohort["batch_size"]==BATCH[cap] and
               len(cohort["pairs"])==REPS,"frozen 4 CPU pairs per cohort missing")
        demand(len(cohort["short"])==2 and
               [x["cap"] for x in cohort["short"]]==[0,5],
               "original exact 0/5 native short-prefix missing")
        for short in cohort["short"]:
            check_state(short["states"])
            demand(type(short["next_rng"]) is float and
                   0<=short["next_rng"]<1,"short original native MT sample missing")
        for i,pair in enumerate(cohort["pairs"]):
            demand(pair["rep"]==i,"changed original paired sample order")
            a,b=pair["original"],pair["instrumented"]
            for tag,arm in (("original",a),("instrumented",b)):
                demand(type(arm["fit_cpu_ns"]) is int and arm["fit_cpu_ns"]>0 and
                       type(arm["fit_wall_ns"]) is int and arm["fit_wall_ns"]>0,
                       "Windows/native full fit process-clock zero")
                demand(len(arm["states"])==BATCH[cap],"not all 8-fit units recorded")
                for state in arm["states"]:
                    check_state(state)
                demand(arm["layout"]==[CLASSES[ds]>=8,CLASSES[ds]<8,True],
                       "original Numba dominance C/F layout or owndeg F lost")
                demand(type(arm["next_rng"]) is float and
                       0<=arm["next_rng"]<1,"native original next Numba MT draw missing")
                if tag=="original":
                    demand(arm["native_wall_ns"] is None,
                           "unwrapped arm cannot produce nested native wall")
                else:
                    demand(type(arm["native_wall_ns"]) is int and
                           0<arm["native_wall_ns"]<arm["fit_wall_ns"],
                           "native call wall not nested in actual original full fit")
            demand(a["states"]==b["states"] and a["layout"]==b["layout"] and
                   a["next_rng"]==b["next_rng"],
                   "original-vs-scoped-wall-observer state or MT draw mismatch")
        result[key]=cohort
    demand(set(result)==expected() and len(result)==108,
           "missing any of 108 preregistered genuine cohorts")
    return result


def med(x):
    return statistics.median(x)


def group_metrics(rows):
    out={}
    for ds,k in GEOMETRIES:
        for cap in CAPS:
            for pgrd in POLICIES:
                pairs=[p for seed in SEEDS
                       for p in rows[(ds,k,seed,pgrd,cap)]["pairs"]]
                demand(len(pairs)==8,"missing two independent masks × four technical repeats")
                n=BATCH[cap]
                out[(ds,k,cap,pgrd)]={
                    "baseline_cpu_ms":med([p["original"]["fit_cpu_ns"]/n/1e6
                                            for p in pairs]),
                    "baseline_wall_ms":med([p["original"]["fit_wall_ns"]/n/1e6
                                             for p in pairs]),
                    "profile_wall_ms":med([p["instrumented"]["fit_wall_ns"]/n/1e6
                                            for p in pairs]),
                    "backend_wall_ms":med([p["instrumented"]["native_wall_ns"]/n/1e6
                                            for p in pairs]),
                    "backend_wall_share_pct":med([
                        100*p["instrumented"]["native_wall_ns"]/
                        p["instrumented"]["fit_wall_ns"] for p in pairs]),
                    "wrapper_perturbation_pct":med([
                        100*(p["instrumented"]["fit_wall_ns"]/
                             p["original"]["fit_wall_ns"]-1) for p in pairs])
                }
    return out


def report(summary):
    lines=[
        "# Original-source Numba native policy and undirected degree cost attribution",
        "",
        "**OBSERVATIONAL COMPLETE; no source candidate, performance GO or production merge.**",
        "The actual original Git Numba kernel, graph, eight native state arrays,",
        "MT19937 draw and real Numba-only post-argmax path were unchanged.",
        "",
        "Actual native-call time is nested WALL time, not native process CPU.",
        "Whole fit uses process CPU in preregistered 64- or 16-fit batches",
        "to mitigate Windows quantization. Reported timing remainder also",
        "includes timer/wrapper overhead and should not be called pure Python CPU.",
        "",
        "Policies p_grd=0/0.5/1 change TRAJECTORIES and random draws;",
        "measured differences are observational, NOT causal greedy-instruction",
        "time. Original symmetric kNN3 vs kNN10 also changes actual graph",
        "topology. Dense padded int64 neighbor lists are preserved.",
        ""]
    flags=[]
    for host in ("linux","win32"):
        rows=summary[host]
        lines.extend(["## "+host,"",
            "| Corpus | k | Iterations | p_grd | Baseline full CPU ms/fit | Original native WALL ms/fit (observed) | Native WALL share | Wrapper wall perturbation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for ds,k in GEOMETRIES:
            for cap in CAPS:
                for policy in POLICIES:
                    x=rows[(ds,k,cap,policy)]
                    lines.append(f"| {ds} | {k} | {cap} | {policy:.1f} | "
                                 f"{x['baseline_cpu_ms']:.4f} | "
                                 f"{x['backend_wall_ms']:.4f} | "
                                 f"{x['backend_wall_share_pct']:.1f}% | "
                                 f"{x['wrapper_perturbation_pct']:+.2f}% |")
                    if ds=="digits" and k==10 and cap==10000:
                        flags.append(dict(host=host,policy=policy,
                            native_share=x["backend_wall_share_pct"],
                            perturbation=x["wrapper_perturbation_pct"],
                            native70=x["backend_wall_share_pct"]>=70,
                            low_perturbation=abs(x["wrapper_perturbation_pct"])<=5))
        lines.extend(["", "### Unmodified whole-fit CPU relative to p_grd=0.5", "",
            "| Corpus | k | Iterations | p_grd=0.0 relative CPU | p_grd=1.0 relative CPU |",
            "|---|---:|---:|---:|---:|"])
        for ds,k in GEOMETRIES:
            for cap in CAPS:
                baseline=rows[(ds,k,cap,.5)]["baseline_cpu_ms"]
                t0=rows[(ds,k,cap,0.)]["baseline_cpu_ms"]
                t1=rows[(ds,k,cap,1.)]["baseline_cpu_ms"]
                lines.append(f"| {ds} | {k} | {cap} | "
                             f"{100*(baseline-t0)/baseline:+.2f}% | "
                             f"{100*(baseline-t1)/baseline:+.2f}% |")
        lines.append("")
    lines.extend(["## Prespecified interpretation flags (NOT optimization GO)", ""])
    for item in flags:
        lines.append(f"- {item['host']} Digits k10 10000-step policy {item['policy']:.1f}: "
                     f"native WALL {item['native_share']:.1f}% of profiled full fit "
                     f"(>=70%: {item['native70']}), observer perturbation "
                     f"{item['perturbation']:+.2f}% (<=5%: {item['low_perturbation']})")
    lines.extend(["", "Prior source-changing greedy, direct-row, RNG and fused-rowmax",
                  "NO-GOs remain binding. A new actual source optimization would",
                  "require a separately preregistered whole-fit original-vs-candidate",
                  "Linux/Windows CPU gate, all states, native RNG and stopping.",
                  "Artifacts are finite-retention and need separate raw ZIP backup."])
    return "\n".join(lines)+"\n",flags


def main(root):
    files=sorted(Path(root).rglob("numba-native-policy-original-*.json"))
    demand(len(files)==2,f"expected exact Linux/Windows raw results, got {files}")
    docs=[json.loads(x.read_text(encoding="utf-8")) for x in files]
    both={x["platform"]:x for x in docs}
    demand(set(both)=={"linux","win32"},"both independent hosts missing")
    for k in ("head","base","source_hashes"):
        demand(both["linux"][k]==both["win32"][k],
               "original source different across hosts "+k)
    allrows={host:checked(x) for host,x in both.items()}
    for key in expected():
        a,b=allrows["linux"][key],allrows["win32"][key]
        for f in ("graph_sha","degree_sha","labels_sha",
                  "fixture_sha","true_numba_argmax_sha"):
            demand(a[f]==b[f],"source graph or label mismatch across hosts "+f)
        for i in range(REPS):
            for arm in ("original","instrumented"):
                x,y=a["pairs"][i][arm],b["pairs"][i][arm]
                demand(x["states"]==y["states"] and
                       x["next_rng"]==y["next_rng"],
                       "original native cross-host state or seeded RNG differed")
    metrics={host:group_metrics(rows) for host,rows in allrows.items()}
    report_text,flags=report(metrics)
    Path("numba_native_policy_cost_report.md").write_text(report_text,
                                                         encoding="utf-8")
    Path("numba_native_policy_cost_decision.json").write_text(
        json.dumps(dict(schema=1,base=BASE,
                        scientific="OBSERVATIONAL_COMPLETE",
                        optimization_go="NOT_APPLICABLE",
                        cohorts_per_host=108,timed_full_fits_per_host=34560,
                        original_graph_rng_and_eight_state_parity=True,
                        flags=flags,
                        groups={h:{str(k):v for k,v in rows.items()}
                                for h,rows in metrics.items()}),
                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(report_text,flush=True)
    print("STRICT ORIGINAL NUMBA POLICY AND DEGREE TWO HOSTS COMPLETE",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
