"""Strict frozen Linux/Windows two-host Numba fused rowmax science."""
import json
from pathlib import Path
import statistics
import sys

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
DATASETS=("iris","wine","breast_cancer","digits","ring_8192_32")
SEEDS=(87523,87697)
MODES=("monitored","fixed")
STATE={"predictions","dominance","positions","strengths","distances",
       "owndeg","node_labels","mapped_labels"}
CLASSES={"iris":3,"wine":3,"breast_cancer":2,"digits":10,"ring_8192_32":32}
NODES={"iris":150,"wine":178,"breast_cancer":569,"digits":1797,"ring_8192_32":8192}


def require(value,reason):
    if not value:
        raise AssertionError(reason)


def hash64(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def median(seq):
    return statistics.median(seq)


def saving(original,candidate):
    return 100.*(original-candidate)/original


def state_valid(state):
    require(set(state)==STATE and all(hash64(x) for x in state.values()),
            "missing complete eight-array native state fingerprint")


def validate_doc(d):
    require(d["schema"]==1 and d["base"]==BASE and
            d["source_protected"] and d["true_numba_argmax_active"],
            "frozen original production source or existing real Numba path missing")
    require(d["batch"]==8 and d["repetitions"]==5 and len(d["cases"])==10,
            "preregistered source paired sample count missing")
    require(all(hash64(d[x]) for x in ("head","original_sha","candidate_sha")),
            "original Git/candidate provenance missing")
    result={}
    for c in d["cases"]:
        key=(c["dataset"],c["seed"])
        require(key[0] in DATASETS and key[1] in SEEDS and key not in result,
                "unexpected duplicated/missing dataset label mask")
        require(c["warmup_complete"] and
                all(hash64(x) for x in c["graph_shas"]+[c["label_sha"],c["fixture_sha"]]),
                "JIT warmup or original undirected graph/mask missing")
        tests=c["tests"]
        require(len(tests)==2 and {t["mode"] for t in tests}==set(MODES),
                "fixed/monitored original propagation missing")
        for t in tests:
            mode=t["mode"]
            require(len(t["pairs"])==5,"missing original-vs-candidate 5 paired CPU batches")
            for rep,p in enumerate(t["pairs"]):
                require(p["rep"]==rep,"changed counterbalanced pair order")
                for name in ("original","candidate"):
                    b=p[name]
                    require(type(b["cpu_ns"]) is int and b["cpu_ns"]>0 and
                            type(b["wall_ns"]) is int and b["wall_ns"]>0,
                            "missing batched process CPU or wall")
                    require(len(b["states"])==len(b["infos"])==
                            len(b["layouts"])==8,"not eight actual full native fits")
                    require(type(b["next_rng"]) is float and
                            0<=b["next_rng"]<1,"native RNG endpoint absent")
                    for st,info,layout in zip(b["states"],b["infos"],b["layouts"]):
                        state_valid(st)
                        require(type(info["effective_iterations"]) is int and
                                0<info["effective_iterations"]<=10000 and
                                info["stop_reason"] in ("max_iter","early_stop"),
                                "invalid native stop information")
                        if mode=="fixed":
                            require(info["effective_iterations"]==10000 and
                                    info["stop_reason"]=="max_iter",
                                    "original fixed path unexpectedly stopped")
                        require(len(layout)==3 and layout[0] is (CLASSES[key[0]]>=8)
                                and layout[1] is (CLASSES[key[0]]<8)
                                and layout[2] is True,
                                "Numba dominance/ownership memory layout changed")
                require(all(p["original"][f]==p["candidate"][f]
                            for f in ("states","infos","layouts","next_rng")),
                        "source exact state/stop/RNG mismatch inside paired full fit")
        require(len(c["memory"])==2 and
                {m["mode"] for m in c["memory"]}==set(MODES),
                "independent traced full-fit peaks missing")
        for mem in c["memory"]:
            require(len(mem["samples"])==3,"three independent memory samples missing")
            for idx,p in enumerate(mem["samples"]):
                require(p["rep"]==idx and
                        type(p["baseline"]) is int and p["baseline"]>0 and
                        type(p["candidate"]) is int and p["candidate"]>0,
                        "invalid original/candidate tracked memory sample")
        require(len(c["prefix"])==1 and c["prefix"][0]["steps"]==2001,
                "native monitored 2001-step prefix missing")
        state_valid(c["prefix"][0]["state"]["state"])
        require(len(c["forced"])==4 and
                {(f["p_grd"],f["steps"]) for f in c["forced"]} ==
                {(g,n) for g in (0.,1.) for n in (2,1001)},
                "Numba forced greedy/random true native prefixes missing")
        for f in c["forced"]:
            state_valid(f["state"]["state"])
            require(0<f["state"]["stop"]["effective_iterations"]<=f["steps"],
                    "invalid forced native iteration count")
        result[key]=c
    require(set(result)=={(ds,seed) for ds in DATASETS for seed in SEEDS},
            "not all original independent test conditions")
    return result


def compute(hostcases):
    rows={}
    for ds in DATASETS:
        for mode in MODES:
            paired=[]
            for seed in SEEDS:
                test=next(t for t in hostcases[(ds,seed)]["tests"]
                          if t["mode"]==mode)
                paired+=test["pairs"]
            require(len(paired)==10,"not two mask seeds times five repeats")
            rows[(ds,mode)]=dict(
                cpu=median([saving(p["original"]["cpu_ns"],
                                   p["candidate"]["cpu_ns"]) for p in paired]),
                wall=median([saving(p["original"]["wall_ns"],
                                    p["candidate"]["wall_ns"]) for p in paired]),
                original_ms=median([p["original"]["cpu_ns"]/8e6 for p in paired]),
                candidate_ms=median([p["candidate"]["cpu_ns"]/8e6 for p in paired]))
    return rows


def gates(scores, cases):
    checks=[]
    for host,rows in scores.items():
        for (ds,mode),r in rows.items():
            if ds in ("digits","ring_8192_32"):
                minimum=6. if ds=="digits" else 8.
                checks.append(dict(host=host,group="main_full_cpu",dataset=ds,
                                   mode=mode,value=r["cpu"],minimum=minimum,
                                   passed=r["cpu"]>=minimum))
                checks.append(dict(host=host,group="main_no_wall_regression",
                                   dataset=ds,mode=mode,value=r["wall"],
                                   minimum=-5.,passed=r["wall"]>=-5.))
            else:
                checks.append(dict(host=host,group="other_real_no_cpu_regression",
                                   dataset=ds,mode=mode,value=r["cpu"],
                                   minimum=-5.,passed=r["cpu"]>=-5.))
        for ds in DATASETS:
            for seed in SEEDS:
                for mem in cases[host][(ds,seed)]["memory"]:
                    for sample in mem["samples"]:
                        budget=sample["baseline"]+8*NODES[ds]+1048576
                        checks.append(dict(host=host,group="bounded_peak",
                                           dataset=ds,mode=mem["mode"],
                                           seed=seed,value=sample["candidate"],
                                           minimum=budget,passed=sample["candidate"]<=budget))
    return checks


def report(scores,checks,decision):
    lines=["# Original-source Numba fused row-max complete-fit CPU decision",
           "", "**Frozen scientific verdict: "+decision+"**", "",
           "Unmodified original master from commit "+BASE+
           " versus a separately compiled true Numba native kernel with"
           " the only guarded dominance/max loop edit. Original RNG, stopping,"
           " dense graph, layout, all eight array states, prior-fit ownership,"
           " and the effective Numba postpropagation path are verified.",
           "",
           "CPU and wall from 8 consecutive full fits per arm, original"
           " source paired in five alternating batches per seed/mode/host."
           " Two independent mask seeds per dataset. No trimmed pairs or"
           " pooled host timings.", ""]
    for host,rows in scores.items():
        lines+=["## "+host,"",
                "| Corpus | Mode | Original CPU ms/fit | Candidate CPU ms/fit | CPU saved | Wall saved |",
                "|---|---|---:|---:|---:|---:|"]
        for ds in DATASETS:
            for mode in MODES:
                z=rows[(ds,mode)]
                lines.append(f"| {ds} | {mode} | {z['original_ms']:.4f} |"
                             f" {z['candidate_ms']:.4f} | {z['cpu']:+.2f}% |"
                             f" {z['wall']:+.2f}% |")
        lines.append("")
    for group in ("main_full_cpu","main_no_wall_regression",
                  "other_real_no_cpu_regression","bounded_peak"):
        rows=[x for x in checks if x["group"]==group]
        failed=[x for x in rows if not x["passed"]]
        lines += [f"### {group}: {len(rows)-len(failed)}/{len(rows)} PASS"]
        for x in failed:
            lines.append(f"- FAIL {x['host']} {x['dataset']} {x['mode']} "
                         f"observed {x['value']:.4f}, frozen limit {x['minimum']:.4f}")
        lines.append("")
    lines += ["The earlier Cython fusion experiments #65/#82 are distinct"
              " and do not validate this Numba optimization. Global"
              " potential-cache #213 and prior RNG/neighbor NO-GOs remain",
              "binding. GitHub-hosted Windows process CPU is quantized;"
              " batch timing is precommitted, not a posthoc correction.",
              "Tracemalloc covers only Python/NumPy-tracked allocations,"
              " not native RSS.", "",
              "**Do not merge this scientific branch** regardless of"
              " performance decision. A production change requires an"
              " independent reviewed minimal PR, with separate four-platform CI."
             ]
    return "\n".join(lines)+"\n"


def main(root):
    files=sorted(Path(root).rglob("numba-fused-rowmax-native-*.json"))
    require(len(files)==2,f"need exactly two host artifacts: {files}")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    by_os={x["platform"]:x for x in docs}
    require(set(by_os)=={"linux","win32"},"Linux and Windows both required")
    for field in ("head","base","original_sha","candidate_sha"):
        require(by_os["linux"][field]==by_os["win32"][field],
                "source/experiment differed across OS: "+field)
    parsed={host:validate_doc(d) for host,d in by_os.items()}
    for key in {(ds,s) for ds in DATASETS for s in SEEDS}:
        l,w=parsed["linux"][key],parsed["win32"][key]
        require(l["graph_shas"]==w["graph_shas"] and
                l["label_sha"]==w["label_sha"] and
                l["fixture_sha"]==w["fixture_sha"],"graph or labels differ between hosts")
        for mode in MODES:
            lt=next(t for t in l["tests"] if t["mode"]==mode)
            wt=next(t for t in w["tests"] if t["mode"]==mode)
            for a,b in zip(lt["pairs"],wt["pairs"]):
                require(a["original"]["states"]==b["original"]["states"] and
                        a["candidate"]["states"]==b["candidate"]["states"] and
                        a["original"]["infos"]==b["original"]["infos"],
                        "cross-host full original native state/stop differs")
    scores={host:compute(rows) for host,rows in parsed.items()}
    checks=gates(scores,parsed)
    decision="GO" if all(x["passed"] for x in checks) else "NO-GO"
    result=report(scores,checks,decision)
    Path("numba_fused_rowmax_report.md").write_text(result,encoding="utf-8")
    Path("numba_fused_rowmax_decision.json").write_text(
        json.dumps(dict(schema=1,baseline=BASE,decision=decision,
                        full_fits_per_host=1600,cohorts_per_host=20,
                        source_exact=True,native_parity=True,
                        checks=checks),indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print(result,flush=True)
    print("STRICT NUMBA FUSED ROWMAX BOTH HOSTS SCIENCE COMPLETE:",decision,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
