"""Mandatory cross-platform Numba dense-neighbor indexing exact CPU decision."""
import json
import math
from pathlib import Path
import statistics
import sys

BASE="d25d5446b7fa51cc0264a496b7eea87dbb11f1fc"
DATASETS=("iris","wine","breast_cancer","digits")
SEEDS=(64013,64117)
MODES=("monitored","fixed")
STATES={"predictions","dominance","positions","strengths","distances","owndeg"}
CASES={(d,s) for d in DATASETS for s in SEEDS}
VARIANTS=("original","candidate")


def must(x,m):
    if not x:
        raise AssertionError(m)


def sha(x):
    return isinstance(x,str) and len(x)==64 and all(
        c in "0123456789abcdef" for c in x)


def pos(x):
    return type(x) in (float,int) and math.isfinite(x) and x>0


def validate_fit(x,mode,cap,traced=False):
    must(x["iteration_cap"]==cap and x["mode"]==mode,"wrong original native mode")
    it=x["effective_iterations"]
    must(type(it) is int and 0<it<=cap,"invalid Numba iterations")
    must(x["stop_reason"] in ("max_iter","early_stop"),"invalid Numba stop reason")
    must(type(x["stop_max"]) is int and x["stop_max"]>=1,"invalid original threshold")
    must(type(x["checks"]) is int and x["checks"]>=0,"invalid check count")
    must(type(x["stalled_checks"]) is int and x["stalled_checks"]>=0,
         "invalid original stalled check count")
    if mode=="fixed":
        must(it==cap and x["stop_reason"]=="max_iter" and x["checks"]==0,
             "fixed native Numba was stopped/monitored")
    elif it<cap:
        must(x["stop_reason"]=="early_stop","premature monitored exit")
    must(pos(x["elapsed_ns"]) and pos(x["propagation_elapsed_ns"]) and
         x["propagation_elapsed_ns"]<=x["elapsed_ns"]*1.01,
         "invalid complete fit and native timing")
    must(set(x["state_hashes"])==STATES and
         all(sha(v) for v in x["state_hashes"].values()),
         "six native array hashes incomplete")
    must(x["predicted_hash"]==x["state_hashes"]["predictions"] and
         sha(x["mapped_labels_hash"]),"missing output/mapping hash")
    must(type(x["rng_next"]) is float and math.isfinite(x["rng_next"])
         and 0<=x["rng_next"]<1,"missing original Numba next RNG sample")
    if traced:
        must(pos(x["peak_bytes"]),"missing separate traced peak")
    else:
        must(x["peak_bytes"] is None,"CPU measurement incorrectly traced")


def equal(a,b,where):
    for k in ("state_hashes","rng_next","mapped_labels_hash",
              "effective_iterations","stop_reason","checks",
              "stalled_checks","stop_max"):
        must(a[k]==b[k],"original vs candidate native state/RNG/stop "+repr((where,k)))


def verify_doc(d):
    must(d["schema"]==1 and d["base"]==BASE,"source not pinned to original master")
    must(d["platform"] in ("linux","win32"),"unknown platform")
    must(len(d["branch_head"])==40 and sha(d["source_sha256"]) and
         sha(d["candidate_sha256"]),"canonical Git source identity missing")
    must(all(d["threads"].get(v)=="1" for v in (
        "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
        "MKL_NUM_THREADS","NUMBA_NUM_THREADS")),"CPU thread settings differ")
    must(len(d["cases"])==8,"incomplete 4 datasets×2 independent seeds")
    cases={}
    for c in d["cases"]:
        ds,seed=c["dataset"],c["seed"]
        key=(ds,seed)
        must(key in CASES and key not in cases,"duplicate/missing fixture identity")
        must(c["case_id"]==f"{ds}-labels{seed}","original class case mismatch")
        must(c["nodes"]=={"iris":150,"wine":178,"breast_cancer":569,
                         "digits":1797}[ds],"real dataset rows changed")
        must(c["classes"]=={"iris":3,"wine":3,"breast_cancer":2,
                           "digits":10}[ds],"real class count changed")
        must(c["graph_reused"]==(seed==SEEDS[1]),"graph was not reused")
        must(all(sha(c[n]) for n in ("labels_sha","fixture_sha","input_sha"))
             and len(c["graph_sha"])==2
             and all(sha(h) for h in c["graph_sha"]),"missing input/graph hashes")
        must(len(c["samples"])==12 and len(c["peaks"])==6 and
             len(c["warmups"])==4 and len(c["forced_choices"])==2,
             "incomplete original CPU/peak/warmup/forced cohort")
        warm=set()
        for w in c["warmups"]:
            mode=w["mode"]
            variant=w["variant"]
            key2=(variant,mode)
            must(variant in VARIANTS and mode in MODES and key2 not in warm,
                 "duplicate original/candidate native warmup")
            validate_fit(w,mode,2)
            warm.add(key2)
        must(len(warm)==4,"not all true JIT modes warmed")
        measured={}
        original_determinism={}
        for p in c["samples"]:
            mode=p["mode"]
            r=p["rep"]
            key2=(mode,r)
            must(mode in MODES and type(r) is int and 0<=r<6 and
                 key2 not in measured,"incomplete original/candidate timing pair")
            must(p["order"]==(["candidate","original"] if r%2==0
                              else ["original","candidate"]),
                 "alternating source order changed")
            for var in VARIANTS:
                validate_fit(p[var],mode,30000)
            equal(p["original"],p["candidate"],(key,mode,r))
            orig=p["original"]
            ref=(orig["state_hashes"],orig["rng_next"],
                 orig["effective_iterations"],orig["stop_reason"],
                 orig["checks"],orig["stalled_checks"],orig["stop_max"])
            must(mode not in original_determinism or
                 original_determinism[mode]==ref,
                 "original repeated native Numba trajectory changed")
            original_determinism.setdefault(mode,ref)
            measured[key2]=p
        must(set(measured)=={(m,r) for m in MODES for r in range(6)},
             "missing a timing repeat or entire stop mode")
        tr=set()
        for p in c["peaks"]:
            mode,r=p["mode"],p["rep"]
            key2=(mode,r)
            must(mode in MODES and type(r) is int and 0<=r<3 and key2 not in tr,
                 "wrong tracemalloc sample")
            for var in VARIANTS:validate_fit(p[var],mode,30000,True)
            equal(p["original"],p["candidate"],(key,mode,"traced",r))
            tr.add(key2)
        must(tr=={(m,r) for m in MODES for r in range(3)},
             "missing peak repeats")
        sp=c["short_prefix"]
        must(sp["cap"]==2001,"short stopped prefix not original 2001")
        validate_fit(sp["original"],"monitored",2001)
        validate_fit(sp["candidate"],"monitored",2001)
        equal(sp["original"],sp["candidate"],(key,"2001-step"))
        forced=set()
        for p in c["forced_choices"]:
            policy=p["p_grd"]
            must(type(policy) is float and policy in (0.,1.)
                 and policy not in forced,"missing forced greedy/random policy")
            validate_fit(p["original"],"fixed",2)
            validate_fit(p["candidate"],"fixed",2)
            equal(p["original"],p["candidate"],(key,"forced",policy))
            forced.add(policy)
        must(forced=={0.,1.},"not both direct Numba neighbor selection paths")
        cases[key]=c
    must(set(cases)==CASES,"not all independent dataset/mask cells")
    return cases


def report(docs,verified):
    lines=[
        "# Current-master Numba direct neighbor indexing — preregistered science",
        "",
        "**Complete audit:** exact frozen original Numba Git source vs a separately",
        "compiled direct-dense-index candidate. 192 full-fit source-paired CPU",
        "observations (6 alternating × 4 datasets × 2 masks × 2 modes × 2 OS),",
        "96 separately traced memory peak pairs, 32 two-path JIT warmup pairs,",
        "16 additional exact 2001-step monitored prefix pairs, and",
        "32 independent greedy-only/random-only choice pairs. Exact six",
        "terminal state hashes, Numba next RNG, true native stopping/check",
        "metadata, source graph order and output label maps were identical.",
        "",
        "| Host | Dataset | Mode | Paired whole-fit CPU saved | Original full median ms | Candidate full median ms | Extra traced Python peak bytes |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    go=True
    rows=[]
    for doc,cases in zip(docs,verified):
        host=doc["platform"]
        for ds in DATASETS:
            for mode in MODES:
                cpu=[p for seed in SEEDS for p in cases[(ds,seed)]["samples"]
                     if p["mode"]==mode]
                traced=[p for seed in SEEDS for p in cases[(ds,seed)]["peaks"]
                        if p["mode"]==mode]
                must(len(cpu)==12 and len(traced)==6,
                     "pooled repeats unexpectedly missing")
                saving=100*(1-statistics.median([
                    p["candidate"]["elapsed_ns"]/p["original"]["elapsed_ns"]
                    for p in cpu]))
                orig=statistics.median(p["original"]["elapsed_ns"] for p in cpu)/1e6
                cand=statistics.median(p["candidate"]["elapsed_ns"] for p in cpu)/1e6
                extra=statistics.median([
                    p["candidate"]["peak_bytes"]-p["original"]["peak_bytes"]
                    for p in traced])
                n={"iris":150,"wine":178,"breast_cancer":569,"digits":1797}[ds]
                if saving< -5 or extra>8*n+1048576:go=False
                if ds in ("breast_cancer","digits") and saving<5:go=False
                lines.append(f"| {host} | {ds} | {mode} | {saving:+.2f}% | "
                             f"{orig:.3f} | {cand:.3f} | {extra:+.0f} |")
                rows.append(dict(host=host,dataset=ds,mode=mode,
                                 full_fit_cpu_saved_pct=saving,
                                 baseline_median_ms=orig,candidate_median_ms=cand,
                                 extra_traced_peak_bytes=extra))
    lines.extend([
        "",
        f"**Immutable prospective decision: {'GO' if go else 'NO-GO'}**.",
        "GO requires >=5% paired FULL-fit savings in BOTH Breast Cancer and",
        "Digits in EACH stop mode and EACH OS, no cohort slowdowns >5%,",
        "plus bounded extra Python tracemalloc peak <=8*n+1MiB.",
        "Scientific completeness SUCCESS does not mean CPU performance GO.",
        "This is not a graph representation optimization: original",
        "dense padded int64 neib_list and active degree remain unchanged.",
        "",
        "The experiment does not separately claim the amount of cost",
        "attributable to creating a slice view inside LLVM. These are",
        "actual full native fit timings from independent JIT-compiled",
        "source on hosted CPUs. Tracemalloc is not process RSS, and",
        "technical timing repeats are not statistically independent masks.",
        "No original model, output, RNG, stop threshold or backend was",
        "modified. Do not merge this scientific PR. A prospective GO",
        "would still require a separately reviewed production PR with",
        "new independent four-platform regression.",
    ])
    return "\n".join(lines)+"\n",go,rows


def main(root):
    files=list(Path(root).glob("**/numba_neighbor_direct_results.json"))
    must(len(files)==2,"require exactly both complete Linux/Windows raw corpora")
    docs=[json.loads(x.read_text(encoding="utf-8")) for x in files]
    must({d["platform"] for d in docs}=={"linux","win32"},
         "require one independent host per platform")
    docs.sort(key=lambda d:d["platform"])
    verified=[verify_doc(doc) for doc in docs]
    for key in CASES:
        x,y=verified[0][key],verified[1][key]
        for name in ("labels_sha","graph_sha","input_sha","fixture_sha"):
            must(x[name]==y[name],"cross-host original corpus SHA changed "+name)
    for name in ("base","branch_head","source_sha256","candidate_sha256"):
        must(len({json.dumps(doc[name]) for doc in docs})==1,
             "the original/candidate Git source differs across hosts "+name)
    report_text,go,rows=report(docs,verified)
    Path("numba_neighbor_direct_decision.json").write_text(
        json.dumps(dict(schema=1,completeness="PASS",decision="GO" if go else "NO-GO",
            full_cpu_pairs=192,peak_pairs=96,warmup_pairs=32,
            prefix_pairs=16,forced_choice_pairs=32,
            rows=rows),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    Path("numba_neighbor_direct_report.md").write_text(report_text,encoding="utf-8")
    print(report_text,flush=True)
    print("COMPLETE two-host Numba dense neighbor original-source audit:",
          "GO" if go else "NO-GO",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
