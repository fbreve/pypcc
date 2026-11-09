"""Independent Linux/Windows source-exact Numba greedy oracle sensitivity audit.

The oracle is not a production optimization: it requires a complete previously
recorded trajectory. CPU differences are hypothetical, not an achievable GO.
"""
import json
from pathlib import Path
import statistics
import sys

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
GEOM=(("breast_cancer",10),("digits",3),("digits",10),
      ("ring_8192_32",2))
SEEDS=(89021,89119)
POLICIES=(0.,.5,1.)
MODES=("fixed","monitored")
STATE={"predictions","dominance","positions","strengths","distances",
       "owndeg","node_labels","mapped_labels"}
PROTECTED={"pcc.py","pcc_numba.py","pcc_graph.py","pcc_numpy.py",
           "pcc_validation.py","pcc_step.pyx"}
CAP=10000
BATCH=8
REPS=4


def must(cond,reason):
    if not cond:
        raise AssertionError(reason)


def sha(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)


def saved(a,b):
    return 100*(a-b)/a


def state_valid(s):
    must(set(s)==STATE and all(sha(x) for x in s.values()),
         "eight original full-array SHA fingerprints missing")


def expect():
    return {(ds,k,seed,p,mode) for ds,k in GEOM
            for seed in SEEDS for p in POLICIES for mode in MODES}


def row_valid(row):
    ds,k,seed,p,mode=(row[x] for x in
                       ("dataset","k","seed","p_grd","stop_mode"))
    must((ds,k,seed,p,mode) in expect() and row["cap"]==CAP,
         "nonpreregistered geometry/policy/stop")
    must(row["rollback_exact"] is True and
         row["prior_fit_independent"] is True,"exception or old state invariant")
    for field in ("graph_sha","degree_sha","label_sha","fixture_sha"):
        must(sha(row[field]),"source graph/labels hash invalid")
    for key in ("baseline","recorder","oracle"):
        x=row[key]
        state_valid(x["state"])
        must(type(x["next_rng"]) is float and
             0<=x["next_rng"]<1,"true native next MT19937 not captured")
        inf=x["stop"]
        must(inf["effective_iterations"]>0 and
             inf["effective_iterations"]<=CAP and
             inf["stop_reason"] in ("max_iter","early_stop"),
             "invalid actual native original stop")
        if mode=="fixed":
            must(inf["effective_iterations"]==CAP and
                 inf["stop_reason"]=="max_iter","fixed path not full cap")
    # Keep the source kNN neighborhood degree 'k' unchanged while
    # checking the actual full state/stop/native RNG fields.
    for field in ("state","stop","next_rng"):
        must(row["baseline"][field]==row["recorder"][field]==row["oracle"][field],
             "oracle or recorder changed original full fit / MT state: "+field)
    tape=row["tape"]
    must(type(tape["events"]) is int and type(tape["weighted"]) is int and
         type(tape["fallback"]) is int and
         0<=tape["weighted"]<=tape["events"] and
         tape["weighted"]+tape["fallback"]==tape["events"],
         "incomplete genuine greedy tape")
    must(tape["events"]>0 if p>0 else tape["events"]==0,
         "wrong number of recorded actual greedy attempts")
    # Symmetric kNN union degrees can exceed k; never impose an
    # invented global upper bound such as k=10 or 22.
    must(0<=tape["max_degree"]<=8192 and
         tape["tape_allocated_bytes"]>=tape["tape_used_bytes"]>0,
         "invalid real original tape size/degree")
    for field in ("event_meta_sha","event_total_sha",
                  "event_uniform_sha","event_weighted_sha"):
        must(sha(tape[field]),"missing immutable original event tape SHA")
    must(len(row["pairs"])==REPS,"4 preregistered original/oracle pairs missing")
    for n,pair in enumerate(row["pairs"]):
        must(pair["rep"]==n,"counterbalanced trial index mismatch")
        orig,oracle=pair["original"],pair["oracle"]
        for arm in (orig,oracle):
            must(type(arm["cpu_ns"]) is int and arm["cpu_ns"]>0 and
                 type(arm["wall_ns"]) is int and arm["wall_ns"]>0,
                 "missing actual Windows batch process CPU or wall")
            must(len(arm["states"])==len(arm["stops"])==BATCH,
                 "not all eight timed complete original PCC fits")
            for out_state,stop in zip(arm["states"],arm["stops"]):
                state_valid(out_state)
                must(out_state==row["baseline"]["state"] and
                     stop==row["baseline"]["stop"],
                     "repeated graph fit changed original state/stop")
            must(type(arm["next_rng"]) is float and
                 0<=arm["next_rng"]<1,"complete-fit native next RNG missing")
        for name in ("states","stops","next_rng"):
            must(orig[name]==oracle[name],
                 "timed original-vs-oracle source outcomes differ: "+name)
    return ds,k,seed,p,mode


def check_document(d):
    must(d["schema"]==1 and d["base"]==BASE and
         d["oracle_cannot_run_without_prior_tape"] is True and
         d["original_pcc_source_unchanged"] is True and
         d["no_achievable_optimization_go"] is True,
         "original source or counterfactual caveat missing")
    must(set(d["source_hashes"])==PROTECTED and
         all(sha(v) for v in d["source_hashes"].values()) and
         len(d["clone_hashes"])==2 and
         all(sha(x) for x in d["clone_hashes"]),
         "separate source derived original recorder/oracle missing")
    must(len(d["head"])==40 and
         all(c in "0123456789abcdef" for c in d["head"]),
         "Git source commit is not SHA-1 length")
    must(d["batch"]==BATCH and d["repetitions"]==REPS and
         len(d["cohorts"])==48,"frozen complete-fit sampling parameters missing")
    must(len(d["short_controls"])==12,"zero/5 step original controls missing")
    for control in d["short_controls"]:
        must(control["cap"] in (0,5) and
             control["p_grd"] in POLICIES and
             control["dataset"] in ("digits","ring_8192_32"),
             "invalid original 0/5 native controls")
        state_valid(control["state"])
        # Short controls record exact original graph ID/seed and full
        # resulting eight-state/RNG/stop; input SHA values live in each
        # frozen full cohort. Do not require unrecorded short-only keys.
        must((control["dataset"],control["k"],control["seed"]) in
             (("digits",10,89021),("ring_8192_32",2,89119)),
             "unexpected original 0/5 input graph/seed")
        must(type(control["next_rng"]) is float and
             0<=control["next_rng"]<1 and
             0<=control["stop"]["effective_iterations"]<=control["cap"],
             "missing short-control native RNG/stop")
        must(control["tape_count"]>=0 and
             (control["tape_count"]==0 if
              control["p_grd"]==0 or control["cap"]==0 else True),
             "illegal greedy event count for zero/direct-random")
    rows={}
    for row in d["cohorts"]:
        key=row_valid(row)
        must(key not in rows,"duplicate source condition")
        rows[key]=row
    must(set(rows)==expect(),"missing 48 original frozen conditions")
    return rows


def match_231(prior,rows):
    must(prior["schema"]==1 and prior["base"]==BASE and
         prior["native_original"] is True and
         len(prior["cohorts"])==108,"wrong frozen native #231 archive")
    match=0
    for raw in prior["cohorts"]:
        ds,k,seed,p,cap=(raw[x] for x in
                         ("dataset","k","seed","p_grd","cap"))
        if (ds,k) not in GEOM or cap!=CAP:
            continue
        original=rows[(ds,k,seed,p,"fixed")]
        must(all(original[a]==raw[b] for a,b in
                 (("graph_sha","graph_sha"),("degree_sha","degree_sha"),
                  ("label_sha","labels_sha"),("fixture_sha","fixture_sha"))),
             "original input drift from #231")
        frozen=raw["pairs"][0]["original"]
        must(original["baseline"]["state"]==frozen["states"][0] and
             original["baseline"]["next_rng"]==frozen["next_rng"],
             "original state/RNG differs from #231 frozen native output")
        match+=1
    must(match==24,"not all original #231 fixed 24 matching cases")


def match_232(prior,rows):
    must(prior["schema"]==1 and prior["base"]==BASE and
         len(prior["cohorts"])==162,"wrong frozen Numba counters #232 archive")
    match=0
    for raw in prior["cohorts"]:
        ds,k,seed,p,mode,cap,stop=(raw[x] for x in
                ("dataset","k","seed","p_grd","mode","cap","early_stop"))
        if (ds,k) not in GEOM or cap!=CAP:
            continue
        selected=rows[(ds,k,seed,p,"monitored" if stop else "fixed")]
        baseline=raw["results"][0]["original"]
        counts=raw["results"][0]["counter"]["counts"]
        must(selected["baseline"]["state"]==baseline["state"] and
             selected["baseline"]["next_rng"]==baseline["next_rng"] and
             selected["baseline"]["stop"]==baseline["stop"],
             "current trajectory not equal frozen source #232")
        tape=selected["tape"]
        must(tape["events"]==counts["greedy_attempt"] and
             tape["weighted"]==counts["greedy_weighted"] and
             tape["fallback"]==counts["greedy_fallback"],
             "oracle tape not exact real weighted greedy original events #232")
        match+=1
    must(match==48,"not all source #232 native event cohorts replayed")


def compute(rows):
    out={}
    for ds,k in GEOM:
        for policy in POLICIES:
            for mode in MODES:
                pairs=[pair for seed in SEEDS for pair in
                       rows[(ds,k,seed,policy,mode)]["pairs"]]
                must(len(pairs)==8,"two masks x four paired process CPU batches missing")
                def ratios(key):
                    return [saved(p["original"][key],p["oracle"][key]) for p in pairs]
                out[(ds,k,policy,mode)]=dict(
                    original_cpu_ms=statistics.median(
                        [p["original"]["cpu_ns"]/BATCH/1e6 for p in pairs]),
                    oracle_cpu_ms=statistics.median(
                        [p["oracle"]["cpu_ns"]/BATCH/1e6 for p in pairs]),
                    cpu_sensitivity_pct=statistics.median(ratios("cpu_ns")),
                    wall_sensitivity_pct=statistics.median(ratios("wall_ns")),
                    actual_recorded_events=statistics.median(
                        [rows[(ds,k,seed,policy,mode)]["tape"]["events"] for seed in SEEDS]),
                    actual_recorded_fallback=statistics.median(
                        [rows[(ds,k,seed,policy,mode)]["tape"]["fallback"] for seed in SEEDS]),
                    tape_allocated_bytes=max(
                        rows[(ds,k,seed,policy,mode)]["tape"]["tape_allocated_bytes"]
                        for seed in SEEDS))
    return out


def render(scores):
    lines=["# Numba greedy-choice CPU sensitivity with an identical actual trajectory",
           "",
           "**Counterfactual diagnostic only — no source optimization, no code GO.**",
           "All original weighted-greedy choices, complete particle state, real",
           "Numba MT19937 RNG draws and stopping were held exactly equal by",
           "a prerecorded genuine full original-run event tape. The oracle",
           "supplies actual original weight totals and neighbor choices and",
           "skips their computation, but still consumes the same Numba random",
           "draws. The tape is unavailable for a genuine future classification;",
           "the observed CPU difference is NOT an attainable production",
           "speedup, nor a strict bound on possible optimization.",
           "",
           "Whole-fit process CPU was measured on full public PCC fits,",
           "not kernel-only timers. The original vs oracle CPU differences",
           "include sidecar lookup/guard overhead and different unused scratch",
           "writes, while the expensive *creation* of the oracle tape is",
           "excluded from the timer. Different p_grd values retain genuinely",
           "different original trajectories; comparisons are only WITHIN",
           "each frozen policy/graph/mask and then shown side-by-side.",
           ""]
    for host in ("linux","win32"):
        lines += ["## "+host,"",
                  "| Graph | p_grd | Stop mode | Original CPU ms/fit | Oracle CPU ms/fit | Hypothetical full CPU difference | Tape capacity bytes |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for ds,k in GEOM:
            for policy in POLICIES:
                for mode in MODES:
                    r=scores[host][(ds,k,policy,mode)]
                    lines.append(f"| {ds} k{k} | {policy:.1f} | {mode} | "
                                 f"{r['original_cpu_ms']:.4f} | "
                                 f"{r['oracle_cpu_ms']:.4f} | "
                                 f"{r['cpu_sensitivity_pct']:+.2f}% | "
                                 f"{r['tape_allocated_bytes']:,} |")
        lines.append("")
    lines += ["## Readout and future work","",
              "The direct-random-only p_grd=0 condition has an EMPTY tape;",
              "its measured difference is a background control for",
              "scientific wrapper/JIT differences and process-clock noise.",
              "The p_grd=0.5 and 1.0 sensitivities isolate the contrast",
              "of original scoring versus lookup+uniform replay *conditional",
              "on the exact same original movement choices*. They do not",
              "provide a production shortcut to reproduce those choices.",
              "All 48 original vs genuine recorder vs oracle pairs per",
              "host matched the full eight state hashes/MT/stop, and",
              "the 24 original #231 fixed and 48 original #232 counted",
              "source conditions per host were independently replayed.",
              "The deliberate dense padded int64 graph was not changed.",
              "Do not merge this experiment, lower previous performance",
              "GO gates, or remove the unobserved fallback branch.",
              "Raw GitHub Actions artifacts expire; the PR/report is only",
              "a compact summary, not raw-byte archival."]
    return "\n".join(lines)+"\n"


def main(path):
    root=Path(path)
    fresh=sorted(root.rglob("numba-greedy-oracle-native-*.json"))
    old231=sorted(root.rglob("numba-native-policy-original-*.json"))
    old232=sorted(root.rglob("numba-path-counts-native-*.json"))
    must(len(fresh)==len(old231)==len(old232)==2,
         "need 2 original hosts plus 2 archived #231 and 2 #232 raw artifacts")
    def load(files):
        docs=[json.loads(p.read_text(encoding="utf-8")) for p in files]
        result={d["platform"]:d for d in docs}
        must(set(result)=={"linux","win32"},"original two-host provenance missing")
        return result
    now,oldtiming,oldcounts=map(load,(fresh,old231,old232))
    for field in ("head","base","source_hashes","clone_hashes"):
        must(now["linux"][field]==now["win32"][field],
             "original/counterfactual source differs across OS: "+field)
    verified={host:check_document(d) for host,d in now.items()}
    for host in ("linux","win32"):
        match_231(oldtiming[host],verified[host])
        match_232(oldcounts[host],verified[host])
    for key in expect():
        a,b=verified["linux"][key],verified["win32"][key]
        for field in ("graph_sha","degree_sha","label_sha","fixture_sha",
                      "baseline","recorder","oracle","tape"):
            must(a[field]==b[field],
                 "different original input/tape/state/MT cross-host: "+field)
        for original,replay in zip(a["pairs"],b["pairs"]):
            for side in ("original","oracle"):
                for field in ("states","stops","next_rng"):
                    must(original[side][field]==replay[side][field],
                         "cross-host actual complete fit divergence")
    must(now["linux"]["short_controls"]==now["win32"]["short_controls"],
         "cross-host original zero/five-step condition mismatched")
    summary={host:compute(rows) for host,rows in verified.items()}
    report=render(summary)
    Path("numba_greedy_oracle_report.md").write_text(report,encoding="utf-8")
    Path("numba_greedy_oracle_decision.json").write_text(
        json.dumps(dict(schema=1,base=BASE,science="EXACT_TRAJECTORY_ORACLE_COMPLETE",
                        no_implementable_optimization=True,cohorts_per_host=48,
                        timed_fits_per_host=3072,
                        matches_frozen_231_per_host=24,
                        matches_frozen_232_per_host=48,
                        source_8_state_mt_rng_stopping_exact=True,
                        summary={h:{str(key):v for key,v in d.items()}
                                 for h,d in summary.items()}),
                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(report,flush=True)
    print("STRICT ORIGINAL NUMBA GREEDY ORACLE BOTH HOSTS COMPLETE",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
