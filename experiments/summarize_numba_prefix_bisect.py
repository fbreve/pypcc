"""Strict frozen original-source Numba greedy prefix bisect two-host CPU gate."""
import json
from pathlib import Path
import statistics
import sys

BASE="4c1c410d8eb04547421f72ee1d3ca2eeb5763ff2"
DATASETS=("iris_k10","wine_k10","breast_cancer_k10","digits_k3","digits_k10","ring4096_d2","ring4096_d32")
POLICIES=(0.0,0.5,1.0)
SEEDS=(89021,89119)
MODES=("monitored","fixed")
STATE={"predictions","dominance","positions","strengths","distances",
       "owndeg","node_labels","mapped_labels"}
CLASSES={"iris_k10":3,"wine_k10":3,"breast_cancer_k10":2,
         "digits_k3":10,"digits_k10":10,"ring4096_d2":32,
         "ring4096_d32":32}
NODES={"iris_k10":150,"wine_k10":178,"breast_cancer_k10":569,
       "digits_k3":1797,"digits_k10":1797,"ring4096_d2":4096,
       "ring4096_d32":4096}


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
            d["source_protected"] is True and
            d["true_numba_argmax_active"] is True,
            "original current production source/Numba argmax missing")
    require(d["batch"]==8 and d["repetitions"]==4 and len(d["cases"])==42,
            "frozen 7 graphs x2 masks x3 policies not present")
    require(isinstance(d["head"],str) and len(d["head"])==40 and
            all(ch in "0123456789abcdef" for ch in d["head"]) and
            all(hash64(d[x]) for x in ("original_sha","candidate_sha")),
            "Git SHA40 or independent source SHA256 missing")
    ch=d["characterization"]
    require(ch["cases"]==1500 and ch["original_equals_binary"] is True and
            hash64(ch["choice_sha"]),
            "frozen selector plateau/boundary/nonmonotone characterization incomplete")
    result={}
    for case in d["cases"]:
        key=(case["dataset"],case["seed"],case["p_grd"])
        require(key[0] in DATASETS and key[1] in SEEDS and
                key[2] in POLICIES and key not in result,
                "unknown/duplicate frozen graph-mask-policy")
        require(case["warmup_complete"] is True and
                len(case["graph_shas"])==2 and
                all(hash64(x) for x in case["graph_shas"]+
                    [case["label_sha"],case["fixture_sha"]]),
                "genuine undirected graph/mask source hashes missing")
        require(len(case["tests"])==2 and
                {x["mode"] for x in case["tests"]}==set(MODES),
                "true fixed/monitored source modes missing")
        for t in case["tests"]:
            require(len(t["pairs"])==4,"not four precommitted paired CPU samples")
            for rep,pair in enumerate(t["pairs"]):
                require(pair["rep"]==rep,"counterbalanced pair index invalid")
                for arm in ("original","candidate"):
                    b=pair[arm]
                    require(type(b["cpu_ns"]) is int and b["cpu_ns"]>0 and
                            type(b["wall_ns"]) is int and b["wall_ns"]>0,
                            "missing complete-fit process CPU/wall")
                    require(len(b["states"])==len(b["infos"])==
                            len(b["layouts"])==8,"not eight actual full native fits")
                    require(type(b["next_rng"]) is float and
                            0<=b["next_rng"]<1,"true native MT19937 endpoint missing")
                    for state,info,layout in zip(b["states"],b["infos"],b["layouts"]):
                        state_valid(state)
                        require(type(info["effective_iterations"]) is int and
                                0<=info["effective_iterations"]<=10000 and
                                info["stop_reason"] in ("max_iter","early_stop"),
                                "original native iterations/stop invalid")
                        if t["mode"]=="fixed":
                            require(info["effective_iterations"]==10000 and
                                    info["stop_reason"]=="max_iter",
                                    "fixed native 10000-step fit unexpectedly stopped")
                        require(layout==[CLASSES[key[0]]>=8,
                                         CLASSES[key[0]]<8, True],
                                "original C/F dominance or original F ownership lost")
                require(all(pair["original"][field]==pair["candidate"][field]
                            for field in ("states","infos","layouts","next_rng")),
                        "binary candidate changed source state/stop/MT")
        require(len(case["memory"])==2 and
                {x["mode"] for x in case["memory"]}==set(MODES),
                "three paired per-mode process memory samples missing")
        for mem in case["memory"]:
            require(len(mem["samples"])==3,"3 complete-fit traced peaks absent")
            for i,x in enumerate(mem["samples"]):
                require(x["rep"]==i and
                        type(x["baseline"]) is int and x["baseline"]>0 and
                        type(x["candidate"]) is int and x["candidate"]>0,
                        "invalid original/fast traced peak")
        require(len(case["prefix"])==1 and
                case["prefix"][0]["steps"]==2001,
                "actual original monitored prefix absent")
        state_valid(case["prefix"][0]["state"]["state"])
        require(len(case["forced"])==2 and
                [x["steps"] for x in case["forced"]]==[0,5],
                "original zero/five-step genuine native replay absent")
        for x in case["forced"]:
            state_valid(x["state"]["state"])
            require(x["p_grd"]==key[2] and
                    0<=x["state"]["stop"]["effective_iterations"]<=x["steps"],
                    "actual original short prefix/RNG invalid")
        result[key]=case
    require(set(result)=={(ds,seed,p) for ds in DATASETS
                         for seed in SEEDS for p in POLICIES},
            "42 original source cases missing")
    return result

def compute(hostcases):
    result={}
    for ds in DATASETS:
        for pgrd in POLICIES:
            for mode in MODES:
                paired=[]
                for seed in SEEDS:
                    case=hostcases[(ds,seed,pgrd)]
                    paired.extend(next(x for x in case["tests"]
                                       if x["mode"]==mode)["pairs"])
                require(len(paired)==8,"two independent label masks x4 pairs missing")
                result[(ds,pgrd,mode)]=dict(
                    cpu=median([saving(x["original"]["cpu_ns"],
                                       x["candidate"]["cpu_ns"]) for x in paired]),
                    wall=median([saving(x["original"]["wall_ns"],
                                        x["candidate"]["wall_ns"]) for x in paired]),
                    original_ms=median([x["original"]["cpu_ns"]/8e6 for x in paired]),
                    candidate_ms=median([x["candidate"]["cpu_ns"]/8e6 for x in paired]))
    return result

def gates(scores,cases):
    checks=[]
    for host,rows in scores.items():
        for (ds,pgrd,mode),value in rows.items():
            primary=(ds in ("digits_k10","ring4096_d32") and pgrd==0.5)
            if primary:
                checks.append(dict(host=host,group="primary_full_cpu",
                                   dataset=ds,pgrd=pgrd,mode=mode,
                                   value=value["cpu"],minimum=5.,
                                   passed=value["cpu"]>=5.))
            else:
                checks.append(dict(host=host,group="nonprimary_no_cpu_regression",
                                   dataset=ds,pgrd=pgrd,mode=mode,
                                   value=value["cpu"],minimum=-5.,
                                   passed=value["cpu"]>=-5.))
            checks.append(dict(host=host,group="no_full_wall_regression",
                               dataset=ds,pgrd=pgrd,mode=mode,
                               value=value["wall"],minimum=-5.,
                               passed=value["wall"]>=-5.))
        for ds in DATASETS:
            for seed in SEEDS:
                for pgrd in POLICIES:
                    for mem in cases[host][(ds,seed,pgrd)]["memory"]:
                        for x in mem["samples"]:
                            budget=x["baseline"]+8*NODES[ds]+1048576
                            checks.append(dict(host=host,group="bounded_traced_peak",
                                               dataset=ds,seed=seed,pgrd=pgrd,
                                               mode=mem["mode"],value=x["candidate"],
                                               minimum=budget,passed=x["candidate"]<=budget))
    return checks

def report(scores,checks,decision):
    lines=["# Original-source Numba greedy prefix bisect 2-host whole-fit CPU",
           "",
           "**Frozen scientific verdict: "+decision+"**",
           "",
           "Only actual original sequential Numba has been compared to the",
           " uniquely source-derived binary first-prefix-greater selector.",
           " The already-calculated cumulative floating prefixes and true",
           " native RNG/stop/neighbor order are preserved. Degree <8 and",
           " nonmonotonic prefix paths retain the original linear scan.",
           " Cython #93 and fused Numba #107/#114 are PREVIOUS distinct",
           " experiments and are not being rerun.",
           "",
           "True full public PCC.fit_predict CPU/wall measured on 8-fit",
           " process-time batches, 4 alternating pairs for each of two",
           " actual label masks, p_grd 0/.5/1, genuine early-stop vs fixed",
           " 10000 steps. Percent saved is pairwise median and positive",
           " means binary source candidate was faster.",
           ""]
    for host in ("linux","win32"):
        lines+=["## "+host,"",
                "| Graph | p_grd | Stop mode | Original CPU ms/fit | Candidate CPU ms/fit | CPU saved | Wall saved |",
                "|---|---:|---|---:|---:|---:|---:|"]
        for ds in DATASETS:
            for pgrd in POLICIES:
                for mode in MODES:
                    x=scores[host][(ds,pgrd,mode)]
                    lines.append(f"| {ds} | {pgrd:.1f} | {mode} | "
                                 f"{x['original_ms']:.4f} | "
                                 f"{x['candidate_ms']:.4f} | "
                                 f"{x['cpu']:+.2f}% | {x['wall']:+.2f}% |")
        lines.append("")
    for group in ("primary_full_cpu","nonprimary_no_cpu_regression",
                  "no_full_wall_regression","bounded_traced_peak"):
        selected=[x for x in checks if x["group"]==group]
        failed=[x for x in selected if not x["passed"]]
        lines.append(f"### {group}: {len(selected)-len(failed)}/{len(selected)} PASS")
        for x in failed:
            lines.append(f"- FAIL {x['host']} {x['dataset']} p={x['pgrd']} "
                         f"{x['mode']} observed {x['value']:+.4f}, "
                         f"prospective minimum {x['minimum']:+.4f}")
        lines.append("")
    lines+=["**Binding result: "+decision+".** Do not merge this experiment;",
             " a real GO needs a separate minimal production PR with its",
             " own fresh four-platform package regression and maintainer",
             " review. A NO-GO leaves original Numba and Cython untouched.",
             " Windows process CPU timing may be quantized; frozen 8-fit",
             " batches, separate OS and untrimmed pairs were retained.",
             " Memory peaks are traced Python/NumPy only, not native RSS.",
             " Raw Actions archive ZIPs expire under provider retention."]
    return "\n".join(lines)+"\n

def main(root):
    files=sorted(Path(root).rglob("numba-prefix-bisect-native-*.json"))
    require(len(files)==2,f"need 2 complete original OS JSON: {files}")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    by_os={x["platform"]:x for x in docs}
    require(set(by_os)=={"linux","win32"},"missing Linux or Windows raw native experiment")
    for field in ("head","base","original_sha","candidate_sha",
                  "characterization"):
        require(by_os["linux"][field]==by_os["win32"][field],
                "original source or selector characterization differed across hosts: "+field)
    checked={host:validate_doc(doc) for host,doc in by_os.items()}
    for key in {(ds,seed,p) for ds in DATASETS
                for seed in SEEDS for p in POLICIES}:
        left,right=checked["linux"][key],checked["win32"][key]
        require(left["graph_shas"]==right["graph_shas"] and
                left["label_sha"]==right["label_sha"] and
                left["fixture_sha"]==right["fixture_sha"],
                "original ordered undirected graph/labels differ between OS")
        for mode in MODES:
            a=next(x for x in left["tests"] if x["mode"]==mode)
            b=next(x for x in right["tests"] if x["mode"]==mode)
            for orig,other in zip(a["pairs"],b["pairs"]):
                require(orig["original"]["states"]==other["original"]["states"] and
                        orig["candidate"]["states"]==other["candidate"]["states"] and
                        orig["original"]["infos"]==other["original"]["infos"] and
                        orig["candidate"]["infos"]==other["candidate"]["infos"] and
                        orig["original"]["next_rng"]==other["original"]["next_rng"],
                        "genuine Numba cross-host state/RNG/stop mismatch")
    values={host:compute(cases) for host,cases in checked.items()}
    checks=gates(values,checked)
    decision="GO" if all(x["passed"] for x in checks) else "NO-GO"
    text=report(values,checks,decision)
    Path("numba_prefix_bisect_report.md").write_text(text,encoding="utf-8")
    Path("numba_prefix_bisect_decision.json").write_text(
        json.dumps(dict(schema=1,base=BASE,decision=decision,
                        cohorts_per_host=84,full_timed_fits_per_host=5376,
                        source_exact=True,selection_edge_cases=1500,
                        eight_state_native_rng_and_stop_parity=True,
                        criteria=checks),indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print(text,flush=True)
    print("STRICT BOTH-OS ORIGINAL NUMBA PREFIX BISECT SCIENCE COMPLETE:",decision,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
