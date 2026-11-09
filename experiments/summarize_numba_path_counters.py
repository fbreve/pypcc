"""Independent exact-original two-OS Numba path-counter science aggregator.

Requires immutable original #231 raw Linux/Windows data as independently
collected baseline state fingerprints. No counter timing or CPU benchmark.
"""
import json
from pathlib import Path
import statistics
import sys

BASE = "4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
GEOM = tuple((name,k) for name in ("iris","wine","breast_cancer","digits")
             for k in (3,10)) + (("ring_8192_32",2),)
SEEDS = (89021,89119)
POLICIES = (0.,.5,1.)
MODES = (("fixed_1000",1000,False),("fixed_10000",10000,False),
         ("monitored_10000",10000,True))
NODES = {"iris":150,"wine":178,"breast_cancer":569,"digits":1797,
         "ring_8192_32":8192}
CLASSES = {"iris":3,"wine":3,"breast_cancer":2,"digits":10,
           "ring_8192_32":32}
STATE = ("predictions","dominance","positions","strengths","distances",
         "owndeg","node_labels","mapped_labels")
COUNTERS = ("attempts","invalid_current","no_degree","invalid_label",
            "greedy_attempt","greedy_infinite_weight","greedy_weighted",
            "greedy_fallback","direct_random","valid_dest","unlabeled_update",
            "owndeg_random_increment","accepted","rejected","invalid_dest",
            "labeled_dest","distance_relax","position_changed",
            "weighted_greedy_accepted","random_accepted","eligible")


def insist(value,why):
    if not value:
        raise AssertionError(why)


def is_sha256(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)


def state_ok(x):
    insist(isinstance(x,dict) and set(x)==set(STATE) and
           all(is_sha256(v) for v in x.values()),"not all eight SHA256 array states")


def expected():
    return {(ds,k,seed,policy,mode,cap,stop)
            for ds,k in GEOM for seed in SEEDS for policy in POLICIES
            for mode,cap,stop in MODES}


def check_counts(c,iterations,particles):
    insist(isinstance(c,dict) and set(c)==set(COUNTERS),"missing counter column")
    insist(all(type(x) is int and x>=0 for x in c.values()),"counter not nonnegative int")
    a=c
    insist(a["attempts"]==particles*iterations,"attempts must be actual iterations x particles")
    insist(a["attempts"]==
           a["invalid_current"]+a["no_degree"]+a["invalid_label"]+a["eligible"],
           "eligibility accounting")
    insist(a["eligible"]==a["greedy_attempt"]+a["direct_random"]==
           a["greedy_weighted"]+a["greedy_fallback"]+a["direct_random"],
           "genuine greedy/random choice accounting")
    insist(a["greedy_attempt"]==a["greedy_weighted"]+a["greedy_fallback"],
           "greedy weighted vs fallback mismatch")
    insist(a["eligible"]==a["valid_dest"]+a["invalid_dest"],
           "native destination validity accounting")
    insist(a["valid_dest"]==a["unlabeled_update"]+a["labeled_dest"]==
           a["accepted"]+a["rejected"],"node update/acceptance accounting")
    insist(a["accepted"]==
           a["weighted_greedy_accepted"]+a["random_accepted"],
           "move acceptance source accounting")
    insist(a["greedy_infinite_weight"]<=a["greedy_attempt"] and
           a["weighted_greedy_accepted"]<=a["greedy_weighted"] and
           a["random_accepted"]<=a["greedy_fallback"]+a["direct_random"],
           "weighted/rescaled/fallback move bounds")
    insist(a["owndeg_random_increment"]<=
           min(a["valid_dest"],a["greedy_fallback"]+a["direct_random"]),
           "true non-greedy owndeg increments")
    insist(a["distance_relax"]<=a["valid_dest"] and
           a["position_changed"]<=a["accepted"],"distance/movement bounds")


def check_pair(original,counted,cap,stop,particles):
    for data in (original,counted):
        state_ok(data["state"])
        insist(type(data["next_rng"]) is float and
               0<=data["next_rng"]<1,"next original native MT19937 draw")
        s=data["stop"]
        insist(type(s["effective_iterations"]) is int and
               0<=s["effective_iterations"]<=cap and
               s["stop_reason"] in ("early_stop","max_iter") and
               s["checks"]==((s["effective_iterations"]+9)//10 if stop else 0),
               "original real Numba stop control invalid")
        if not stop:
            insist(s["effective_iterations"]==cap and
                   s["stop_reason"]=="max_iter","fixed mode not really fixed")
    insist(original["state"]==counted["state"] and
           original["stop"]==counted["stop"] and
           original["next_rng"]==counted["next_rng"],
           "counter sidecar changed original Numba state, RNG or stopping")
    insist(original["counts"] is None,"uncounted original arm reported counters")
    check_counts(counted["counts"],counted["stop"]["effective_iterations"],particles)


def check_doc(d):
    insist(d["schema"]==1 and d["base"]==BASE and
           d["scoped_identity"] is True and
           d["no_native_timing"] is True,"original unmodified observer provenance")
    insist(is_sha256(d["candidate_sha"]) and
           set(d["source_hashes"]) ==
           {"pcc.py","pcc_numba.py","pcc_graph.py","pcc_numpy.py",
            "pcc_validation.py","pcc_step.pyx"} and
           all(is_sha256(v) for v in d["source_hashes"].values()),
           "exact real Git source byte guards missing")
    insist(isinstance(d["head"],str) and len(d["head"])==40 and
           all(c in "0123456789abcdef" for c in d["head"]),
           "original Git head is SHA1, not a 64-hex source hash")
    insist(tuple(d["sidecar"])==COUNTERS,"counter order/schema not frozen")
    insist(d["cohort_count"]==len(d["cohorts"])==162,"all 162 native cases required")
    insist(len(d["edge_probes"])==18,"zero/five genuine Numba probes missing")
    for probe in d["edge_probes"]:
        insist(probe["cap"] in (0,5) and
               probe["full_row_argmax_verified"] is True,
               "actual original postpropagation full argmax not executed")
        insist(probe["dataset"] in ("iris","digits","ring_8192_32") and
               probe["p_grd"] in POLICIES,"invalid source replay probe")
        insist(is_sha256(probe["graph_sha"]) and is_sha256(probe["label_sha"]),
               "probe original graph and label fingerprints missing")
        check_pair(probe["original"],probe["counter"],probe["cap"],False,
                   3*CLASSES[probe["dataset"]])
    rows={}
    for item in d["cohorts"]:
        key=(item["dataset"],item["k"],item["seed"],item["p_grd"],
             item["mode"],item["cap"],item["early_stop"])
        insist(key in expected() and key not in rows,"unknown/duplicate original graph-policy-mode")
        ds,k,seed,policy,mode,cap,stop=key
        insist(item["nodes"]==NODES[ds] and
               item["classes"]==CLASSES[ds] and
               item["particles"]==3*CLASSES[ds] and
               item["rollback_exact"] is True and
               item["post_argmax_active"] is True,
               "original graph class/rollback/Numba policy invalid")
        for field in ("graph_sha","degree_sha","label_sha","fixture_sha"):
            insist(is_sha256(item[field]),"original ordered graph or label hash")
        nrep=(2 if ds in ("digits","ring_8192_32") and
              mode=="fixed_10000" else 1)
        insist(len(item["results"])==nrep,"original reused-fit replay count missing")
        for rep in item["results"]:
            check_pair(rep["original"],rep["counter"],cap,stop,item["particles"])
        rows[key]=item
    insist(set(rows)==expected(),"missing or fabricated preregistered 162 cases")
    return rows


def check_231(ref,rows):
    insist(ref["schema"]==1 and ref["base"]==BASE and
           ref["native_original"] is True and
           ref["true_post_argmax"] is True,"original frozen #231 reference provenance")
    insist(ref["pairs"]==4 and len(ref["cohorts"])==108,
           "complete raw original 108-cohort reference missing")
    matched=0
    for earlier in ref["cohorts"]:
        ds,k,seed,policy,cap=(earlier[x] for x in
                              ("dataset","k","seed","p_grd","cap"))
        mode="fixed_"+str(cap)
        key=(ds,k,seed,policy,mode,cap,False)
        insist(key in rows,"science #231 source condition unavailable")
        current=rows[key]
        insist(current["graph_sha"]==earlier["graph_sha"] and
               current["degree_sha"]==earlier["degree_sha"] and
               current["label_sha"]==earlier["labels_sha"] and
               current["fixture_sha"]==earlier["fixture_sha"],
               "fixed reference undirected graph/masks changed since #231")
        baseline=earlier["pairs"][0]["original"]
        source_state=baseline["states"][0]
        ours=current["results"][0]["original"]
        insist(ours["state"]==source_state and
               ours["next_rng"]==baseline["next_rng"],
               "different actual current Numba reference output from #231")
        matched+=1
    insist(matched==108,"exact original #231 108 source replays missing")


def rate(n,d):
    return 100*n/d if d else 0.


def metrics(rows):
    summary={}
    for ds,k in GEOM:
        for pgrd in POLICIES:
            for mode,cap,stop in MODES:
                items=[rows[(ds,k,seed,pgrd,mode,cap,stop)]
                       for seed in SEEDS]
                counts=[item["results"][0]["counter"]["counts"] for item in items]
                total={key:sum(c[key] for c in counts) for key in COUNTERS}
                d=total["eligible"]; g=total["greedy_attempt"]; valid=total["valid_dest"]
                summary[(ds,k,pgrd,mode)]=dict(
                    attempts=total["attempts"],eligible=total["eligible"],
                    greedy_attempt_rate_pct=rate(g,d),
                    weighted_greedy_rate_pct=rate(total["greedy_weighted"],d),
                    fallback_given_greedy_pct=rate(total["greedy_fallback"],g),
                    random_branch_rate_pct=rate(total["direct_random"],d),
                    unlabeled_dest_pct=rate(total["unlabeled_update"],valid),
                    labeled_dest_pct=rate(total["labeled_dest"],valid),
                    ownership_updates_pct=rate(total["owndeg_random_increment"],valid),
                    accepted_pct=rate(total["accepted"],valid),
                    actual_position_changes_pct=rate(total["position_changed"],valid),
                    rejected_pct=rate(total["rejected"],valid),
                    distance_relax_pct=rate(total["distance_relax"],valid))
    return summary


def report(host_results):
    out=["# Original Numba particle-path mechanism — exact two-host science",
         "",
         "**Scientific complete observation only; no optimization candidate, "
         "CPU speedup or production GO.** The actual original Numba kernel and "
         "the counter-cloned kernel matched full eight-state arrays, native "
         "MT19937 next RNG, real stopping info, previous state ownership and "
         "original graph across Windows and Linux, and reproduced #231 original "
         "fixed-profile output SHA for all 108 matching inputs per host.",
         "",
         "The counters are **untimed** and measure how many branches, "
         "visits and accepted decisions occur; they are NOT exact instruction "
         "latencies, standalone CPU times, or memory bandwidth. Policies "
         "p_grd=0/0.5/1 lead to genuinely DIFFERENT trajectories; the "
         "whole-fit process-CPU comparison belongs only to the previously "
         "measured uninstrumented original source #231.",
         ""]
    for host,rows in host_results.items():
        out.extend(["## "+host,"",
            "| Data | k | Policy | Mode | Greedy attempts % | Fallback / greedy % | Unlabeled destinations % | Accepted % | Position changes % |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|"])
        for ds,k in GEOM:
            for p in POLICIES:
                for mode,cap,stop in MODES:
                    v=rows[(ds,k,p,mode)]
                    out.append(f"| {ds} | {k} | {p:.1f} | {mode} | "
                               f"{v['greedy_attempt_rate_pct']:.2f}% | "
                               f"{v['fallback_given_greedy_pct']:.2f}% | "
                               f"{v['unlabeled_dest_pct']:.2f}% | "
                               f"{v['accepted_pct']:.2f}% | "
                               f"{v['actual_position_changes_pct']:.2f}% |")
        out.append("")
    out.extend(["## Interpretation and next decision", "",
                "Compare the observed greedy attempt and fallback mixes "
                "against original uninstrumented source total-fit CPU from "
                "the #231 artifacts only. The difference between p_grd "
                "policies includes node/dominance trajectories, not merely "
                "branch costs. The next optimization, if any, must have a "
                "new preregistered source-exact whole-fit CPU GO with native "
                "RNG/stop/state and real Linux/Windows benefit.",
                "",
                "The intentional dense padded int64 neighbor layout is "
                "unchanged. Original scientific full output RAW JSON "
                "artifacts have finite GitHub Actions retention."])
    return "\n".join(out)+"\n"


def main(workdir):
    folder=Path(workdir)
    fresh=sorted(folder.rglob("numba-path-counts-native-*.json"))
    prior=sorted(folder.rglob("numba-native-policy-original-*.json"))
    insist(len(fresh)==2 and len(prior)==2,
           f"need frozen 2-host native and #231 original archives: {fresh}, {prior}")
    data={d["platform"]:d for d in
          (json.loads(f.read_text(encoding="utf-8")) for f in fresh)}
    previous={d["platform"]:d for d in
              (json.loads(f.read_text(encoding="utf-8")) for f in prior)}
    insist(set(data)==set(previous)=={"linux","win32"},
           "original native host evidence absent")
    for field in ("head","base","source_hashes","candidate_sha","sidecar"):
        insist(data["linux"][field]==data["win32"][field],
               "different source/sidecar between native hosts: "+field)
    verified={host:check_doc(doc) for host,doc in data.items()}
    for host,rows in verified.items():
        check_231(previous[host],rows)
    for key in expected():
        a,b=verified["linux"][key],verified["win32"][key]
        for f in ("graph_sha","degree_sha","label_sha","fixture_sha"):
            insist(a[f]==b[f],"cross-host input differs: "+f)
        for result_a,result_b in zip(a["results"],b["results"]):
            for arm in ("original","counter"):
                for part in ("state","next_rng","stop"):
                    insist(result_a[arm][part]==result_b[arm][part],
                           "cross-host original state/RNG/stop differs")
            insist(result_a["counter"]["counts"]==result_b["counter"]["counts"],
                   "cross-host original Numba native branch events differ")
    for a,b in zip(data["linux"]["edge_probes"],data["win32"]["edge_probes"]):
        insist(a==b,"different original native 0/5-step cross-host path controls")
    measured={host:metrics(rows) for host,rows in verified.items()}
    output=Path("numba_path_counters_report.md")
    output.write_text(report(measured),encoding="utf-8")
    Path("numba_path_counters_decision.json").write_text(json.dumps(dict(
        schema=1,base=BASE,scientific="COUNTERS_SOURCE_EXACT_TWO_HOST_COMPLETE",
        optimization="NONE_AND_NOT_REQUESTED",cohorts_per_host=162,
        original_231_fixed_replays_per_host=108,
        source_rng_stop_8states_cross_host_exact=True,
        counts={host:{str(key):value for key,value in rows.items()}
                for host,rows in measured.items()}
        ),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(output.read_text(encoding="utf-8"),flush=True)
    print("STRICT ORIGINAL NUMBA PATH COUNTERS TWO-HOST PASS",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
