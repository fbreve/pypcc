"""Independent original Numba greedy-weight composition and frozen #232 replay.

Strict observation-only two-host audit. No native CPU of instrumented code.
"""
import json
from pathlib import Path
import sys

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
GEOM=tuple((ds,k) for ds in ("iris","wine","breast_cancer","digits")
           for k in (3,10))+(("ring_8192_32",2),)
SEEDS=(89021,89119)
POLICIES=(0.0,0.5,1.0)
MODES=(("fixed_1000",1000,False),
       ("fixed_10000",10000,False),
       ("monitored_10000",10000,True))
NODES={"iris":150,"wine":178,"breast_cancer":569,"digits":1797,
       "ring_8192_32":8192}
CLASSES={"iris":3,"wine":3,"breast_cancer":2,"digits":10,
         "ring_8192_32":32}
STATES={"predictions","dominance","positions","strengths","distances",
        "owndeg","node_labels","mapped_labels"}
OLD_STAT=("attempts","invalid_current","no_degree","invalid_label",
          "greedy_attempt","greedy_infinite_weight","greedy_weighted",
          "greedy_fallback","direct_random","valid_dest","unlabeled_update",
          "owndeg_random_increment","accepted","rejected","invalid_dest",
          "labeled_dest","distance_relax","position_changed",
          "weighted_greedy_accepted","random_accepted","eligible")
EXTRA=("candidate_weight_evaluations","valid_weight_neighbor",
       "invalid_weight_neighbor","dominance_zero_valid",
       "dominance_one_valid","dominance_strict_fraction_valid",
       "dominance_other_valid","distance_saturated255_valid",
       "distance_zero_valid","distance_1_to_254_valid",
       "distance_weight_zero_valid","distance_weight_other_valid",
       "product_zero_any","product_other_any","product_zero_valid",
       "product_other_valid","distance255_dominance_zero",
       "distance255_dominance_other","prefix_stall","greedy_degree_sum")
STAT=OLD_STAT+EXTRA
PROTECTED={"pcc.py","pcc_numba.py","pcc_graph.py",
           "pcc_numpy.py","pcc_validation.py","pcc_step.pyx"}


def require(x,msg):
    if not x:
        raise AssertionError(msg)


def is_sha(v):
    return isinstance(v,str) and len(v)==64 and all(ch in "0123456789abcdef" for ch in v)


def state(s):
    require(isinstance(s,dict) and set(s)==STATES and
            all(is_sha(v) for v in s.values()),"eight original state hashes missing")


def expected():
    return {(ds,k,seed,p,mode,cap,stop)
            for ds,k in GEOM for seed in SEEDS for p in POLICIES
            for mode,cap,stop in MODES}


def validate_counts(c,iterations,particles):
    require(isinstance(c,dict) and set(c)==set(STAT),
            "frozen 41 counter names changed")
    require(all(type(v) is int and v>=0 for v in c.values()),
            "negative or noninteger Numba source-native event count")
    a=c
    require(a["attempts"]==iterations*particles,
            "true original particle iteration count mismatch")
    require(a["attempts"]==
            a["invalid_current"]+a["no_degree"]+a["invalid_label"]+a["eligible"],
            "particle eligibility accounting")
    require(a["eligible"]==a["greedy_attempt"]+a["direct_random"],
            "actual p_grd branch coverage")
    require(a["greedy_attempt"]==
            a["greedy_weighted"]+a["greedy_fallback"],
            "actual greedy weighted/fallback count")
    require(a["eligible"]==a["valid_dest"]+a["invalid_dest"],
            "original destination validity coverage")
    require(a["valid_dest"]==
            a["unlabeled_update"]+a["labeled_dest"]==
            a["accepted"]+a["rejected"],
            "original accepted/updated node coverage")
    require(a["accepted"]==
            a["weighted_greedy_accepted"]+a["random_accepted"],
            "original accepted movement modes")
    require(a["distance_relax"]<=a["valid_dest"] and
            a["position_changed"]<=a["accepted"],
            "original distance/move counts impossible")
    require(a["candidate_weight_evaluations"]==
            a["valid_weight_neighbor"]+a["invalid_weight_neighbor"]==
            a["product_zero_any"]+a["product_other_any"]==
            a["greedy_degree_sum"],"all greedy candidate neighbor evaluations missing")
    require(a["valid_weight_neighbor"]==
            a["dominance_zero_valid"]+a["dominance_one_valid"]+
            a["dominance_strict_fraction_valid"]+a["dominance_other_valid"],
            "original greedy dominance buckets incomplete")
    require(a["valid_weight_neighbor"]==
            a["distance_saturated255_valid"]+a["distance_zero_valid"]+
            a["distance_1_to_254_valid"],
            "original uint8 candidate distance buckets incomplete")
    require(a["valid_weight_neighbor"]==
            a["distance_weight_zero_valid"]+a["distance_weight_other_valid"]==
            a["product_zero_valid"]+a["product_other_valid"],
            "candidate valid weight/product categories incomplete")
    require(a["distance_saturated255_valid"]==
            a["distance255_dominance_zero"]+
            a["distance255_dominance_other"],
            "saturated distance dominance categories incomplete")
    require(a["product_zero_valid"]<=a["product_zero_any"] and
            a["product_other_valid"]<=a["product_other_any"] and
            a["prefix_stall"]<=a["candidate_weight_evaluations"],
            "valid product or prefix subset impossible")
    require(a["candidate_weight_evaluations"]>=a["greedy_attempt"],
            "original greedy loop has no active neighbor")


def check_pair(a,b,cap,stop,particles):
    for x in (a,b):
        state(x["state"])
        require(type(x["next_rng"]) is float and
                0<=x["next_rng"]<1,"genuine original seeded native MT19937 absent")
        info=x["stop"]
        require(type(info["effective_iterations"]) is int and
                0<=info["effective_iterations"]<=cap and
                info["stop_reason"] in ("max_iter","early_stop"),
                "true native Numba stop info missing")
        if not stop:
            require(info["effective_iterations"]==cap and
                    info["stop_reason"]=="max_iter","not actual original fixed loop")
    require(all(a[k]==b[k] for k in ("state","next_rng","stop")),
            "original native vs observation has changed actual output/stop/RNG")
    require(a["counts"] is None,"original uninstrumented counts must be absent")
    validate_counts(b["counts"],b["stop"]["effective_iterations"],particles)


def check(d):
    require(d["schema"]==1 and d["base"]==BASE and
            d["scoped_identity"] is True and d["no_native_timing"] is True,
            "original Numba source/dispatch/timing guard missing")
    require(d["cohort_count"]==len(d["cohorts"])==162 and
            len(d["edge_probes"])==18 and tuple(d["sidecar"])==STAT,
            "precommitted 162 cohorts / 18 edge tests / 41 fields missing")
    require(set(d["source_hashes"])==PROTECTED and
            all(is_sha(x) for x in d["source_hashes"].values()) and
            is_sha(d["candidate_sha"]),"original Git SHA or clone SHA missing")
    require(isinstance(d["head"],str) and len(d["head"])==40 and
            all(ch in "0123456789abcdef" for ch in d["head"]),
            "Git revision must be canonical SHA1")
    rows={}
    for x in d["cohorts"]:
        key=(x["dataset"],x["k"],x["seed"],x["p_grd"],x["mode"],
             x["cap"],x["early_stop"])
        require(key in expected() and key not in rows,
                "duplicate or missing preregistered Numba original cohort")
        ds,k,seed,p,mode,cap,stop=key
        require(x["nodes"]==NODES[ds] and
                x["classes"]==CLASSES[ds] and
                x["particles"]==3*CLASSES[ds] and
                x["rollback_exact"] is True and
                x["post_argmax_active"] is True,
                "original graph/class/rollback/argmax provenance incomplete")
        require(all(is_sha(x[j]) for j in
                    ("graph_sha","degree_sha","label_sha","fixture_sha")),
                "original graph and label sha missing")
        nrep=2 if ds in ("digits","ring_8192_32") and mode=="fixed_10000" else 1
        require(len(x["results"])==nrep,"repeated original fit source probes incomplete")
        for r in x["results"]:
            check_pair(r["original"],r["counter"],cap,stop,x["particles"])
        rows[key]=x
    require(set(rows)==expected(),"not all 162 genuine original Numba cohorts")
    for edge in d["edge_probes"]:
        require(edge["dataset"] in ("iris","digits","ring_8192_32") and
                edge["p_grd"] in POLICIES and edge["cap"] in (0,5) and
                edge["full_row_argmax_verified"] is True,
                "genuine original native zero/five-step #228 path absent")
        check_pair(edge["original"],edge["counter"],edge["cap"],
                   False,3*CLASSES[edge["dataset"]])
    return rows


def frozen_232(prior,now):
    require(prior["schema"]==1 and prior["base"]==BASE and
            len(prior["cohorts"])==162 and
            tuple(prior["sidecar"])==OLD_STAT,
            "previous complete 162-cohort exact native #232 baseline missing")
    prior_map={}
    for c in prior["cohorts"]:
        key=(c["dataset"],c["k"],c["seed"],c["p_grd"],
             c["mode"],c["cap"],c["early_stop"])
        require(key not in prior_map,"duplicate old #232 case")
        prior_map[key]=c
    require(set(prior_map)==expected(),"previous #232 162 unique cases missing")
    for key in expected():
        original=prior_map[key]
        new=now[key]
        for f in ("graph_sha","degree_sha","label_sha","fixture_sha",
                  "rollback_exact","post_argmax_active"):
            require(original[f]==new[f],
                    "changed original graph/labels/stop/rollback from #232: "+f)
        require(len(original["results"])==len(new["results"]),
                "number of repeated same original native fits differs")
        for o,n in zip(original["results"],new["results"]):
            for branch in ("original","counter"):
                for part in ("state","next_rng","stop"):
                    require(o[branch][part]==n[branch][part],
                            "altered original Numba trajectory vs #232 "+part)
            for field in OLD_STAT:
                require(o["counter"]["counts"][field]==
                        n["counter"]["counts"][field],
                        "changed original Numba branch behavior vs #232: "+field)
    require(len(prior["edge_probes"])==18,"#232 short source proofs absent")
    for o,n in zip(prior["edge_probes"],now["__edges"]):
        for field in ("dataset","k","seed","p_grd","cap",
                      "graph_sha","label_sha"):
            require(o[field]==n[field],"zero/five original graph/probe mismatch")
        for b in ("original","counter"):
            for field in ("state","next_rng","stop"):
                require(o[b][field]==n[b][field],
                        "original 0/5 source replay mismatch")
        for field in OLD_STAT:
            require(o["counter"]["counts"][field]==n["counter"]["counts"][field],
                    "original edge branch counter changed vs #232")


def rate(n,d):
    return 100.*n/d if d else None


def summarize(rows):
    scores={}
    for ds,k in GEOM:
        for mode,cap,stop in MODES:
            for p in POLICIES:
                values=[rows[(ds,k,seed,p,mode,cap,stop)]["results"][0]["counter"]["counts"]
                        for seed in SEEDS]
                agg={key:sum(v[key] for v in values) for key in STAT}
                n=agg["candidate_weight_evaluations"]
                valid=agg["valid_weight_neighbor"]
                scores[(ds,k,mode,p)]=dict(
                    greedy_attempts=agg["greedy_attempt"],
                    candidates=n,
                    valid_candidates=valid,
                    invalid_pct=rate(agg["invalid_weight_neighbor"],n),
                    zero_dom_pct=rate(agg["dominance_zero_valid"],valid),
                    one_dom_pct=rate(agg["dominance_one_valid"],valid),
                    fractional_dom_pct=rate(agg["dominance_strict_fraction_valid"],valid),
                    anomalous_dom_pct=rate(agg["dominance_other_valid"],valid),
                    saturated_distance_pct=rate(agg["distance_saturated255_valid"],valid),
                    zero_distance_pct=rate(agg["distance_zero_valid"],valid),
                    zero_product_pct=rate(agg["product_zero_valid"],valid),
                    zero_distance_weight_pct=rate(agg["distance_weight_zero_valid"],valid),
                    saturated_and_zero_dom_pct=rate(
                        agg["distance255_dominance_zero"],valid),
                    prefix_stall_pct=rate(agg["prefix_stall"],n),
                    # Absolute native counts retained for scientific reuse.
                    counts=agg)
    return scores


def format_percent(v):
    return "N/A" if v is None else f"{v:.2f}%"


def report(summaries):
    lines=["# Original Numba greedy weight zero/saturation source census",
           "",
           "**Scientific observation complete; NOT an optimization candidate.**",
           "Original Numba production kernel, actual native MT19937 draws,",
           "all eight full states and the original pcc.py/Numba #228 path",
           "matched the separately compiled sidecar exactly across Linux",
           "and Windows. All prior 21 event counters matched #232 bytewise.",
           "",
           "Counts below are real *untimed* Numba original-loop event counts;",
           "NO claim of CPU savings, memory bandwidth or instruction cycles.",
           "A single p_grd change modifies native RNG choices, dominance",
           "and visited nodes; percentages are observational conditions.",
           "Percentages have denominators VALID greedy neighbor weight",
           "evaluations, except prefix-stall percentage over all greedy",
           "candidates. For p_grd=0 there are no greedy candidates:",
           "N/A means no observation, not zero probability.",
           ""]
    for host in ("linux","win32"):
        scores=summaries[host]
        lines += ["## "+host,"",
                  "| Dataset | k | Mode | p_grd | Greedy neighbors | Zero dominance | Distance 255 | Zero product | Prefix stalls |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        for ds,k in GEOM:
            for mode,cap,stop in MODES:
                for p in POLICIES:
                    c=scores[(ds,k,mode,p)]
                    lines.append(f"| {ds} | {k} | {mode} | {p:.1f} | "
                                 f"{c['candidates']:,} | "
                                 f"{format_percent(c['zero_dom_pct'])} | "
                                 f"{format_percent(c['saturated_distance_pct'])} | "
                                 f"{format_percent(c['zero_product_pct'])} | "
                                 f"{format_percent(c['prefix_stall_pct'])} |")
        lines.append("")
    lines += ["## Scientific interpretation / future hypothesis", "",
              "Zero candidate dominance does not itself authorize skipping",
              " multiplication: native low-level buffers, signed zero and",
              " nonfinite combinations can change NaN/sums/roulette strict",
              " comparisons, and original C/F row and neighbor ordering",
              " must remain exact. A high fraction of distance=255",
              " reflects saturation, NOT that the original uint8 distance",
              " read could be safely omitted without knowing its value.",
              "",
              "Older fused greedy #107/#109, bound-load #114, scratch",
              " reduction #116, saturated-read #112, layout #100/#104",
              " and binary prefix #234 decisions are binding history.",
              " Further source changes need a DISTINCT frozen paired",
              " original-vs-candidate complete-fit CPU gate in both OS.",
              "",
              "The deliberate padded dense int64 neighbor graph remains",
              " unchanged. Raw GitHub Actions artifacts have finite retention."]
    return "\n".join(lines)+"\n"


def main(root):
    folder=Path(root)
    fresh=sorted(folder.rglob("numba-weight-composition-native-*.json"))
    old=sorted(folder.rglob("numba-path-counts-native-*.json"))
    require(len(fresh)==len(old)==2,
            f"exact complete 2 host raw data and frozen #232 needed: {fresh}, {old}")
    def load(files):
        docs=[json.loads(path.read_text(encoding="utf-8")) for path in files]
        by={x["platform"]:x for x in docs}
        require(set(by)=={"linux","win32"},"independent Linux/Windows host data missing")
        return by
    data=load(fresh)
    previous=load(old)
    for field in ("head","base","source_hashes","candidate_sha","sidecar"):
        require(data["linux"][field]==data["win32"][field],
                "different native source provenance between OS: "+field)
    checked={}
    for host,d in data.items():
        cases=check(d)
        with_edges={**cases,"__edges":d["edge_probes"]}
        frozen_232(previous[host],with_edges)
        checked[host]=cases
    for key in expected():
        a,b=checked["linux"][key],checked["win32"][key]
        for field in ("graph_sha","degree_sha","label_sha","fixture_sha"):
            require(a[field]==b[field],"cross-OS original graph or label mismatch")
        for ar,br in zip(a["results"],b["results"]):
            for arm in ("original","counter"):
                for field in ("state","stop","next_rng"):
                    require(ar[arm][field]==br[arm][field],
                            "cross-OS source Numba state/seed/stop divergence")
            require(ar["counter"]["counts"]==br["counter"]["counts"],
                    "cross-OS complete 41 integer weighted events differ")
    require(data["linux"]["edge_probes"]==data["win32"]["edge_probes"],
            "cross-OS original 0/5 native edge path differ")
    scores={host:summarize(rows) for host,rows in checked.items()}
    text=report(scores)
    Path("numba_weight_composition_report.md").write_text(text,encoding="utf-8")
    Path("numba_weight_composition_decision.json").write_text(json.dumps(dict(
        schema=1,base=BASE,science="ORIGINAL_NUMBA_WEIGHT_COUNTERS_COMPLETE",
        source_candidate=False,performance_go="NOT_APPLICABLE",
        cohorts_per_host=162,original_232_path_replays_per_host=162,
        original_232_edge_replays_per_host=18,source_state_rng_stop_exact=True,
        counters={host:{str(k):v for k,v in rows.items()}
                  for host,rows in scores.items()}),
        indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(text,flush=True)
    print("STRICT ORIGINAL NUMBA WEIGHT COMPOSITION TWO HOSTS COMPLETE",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
